#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

from . import frame_budget
from . import finefs_events
from .finefs_element_descriptions import ELEMENT_DESCRIPTIONS
from .finefs_goe_rules import detailed_rules
from .scoreboard_mask import VIDEO_SUBDIR, mask_regions, require_masked

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "FineFS"


DEFAULT_FPS = frame_budget.sampling_fps(180.0)


LAST_SHORT_PROGRAM = 728

SEGMENTS = {"SP": "Short Program", "FS": "Free Skating"}


COMPONENTS = (
    ("skating_skills", "Skating Skills"),
    ("transitions", "Transitions"),
    ("performance", "Performance"),
    ("composition", "Composition"),
    ("interpretation", "Interpretation of the Music"),
)


CRITERIA = {
    "skating_skills": (
        "Defined by overall cleanness and sureness, edge control and flow over the ice "
        "surface demonstrated by a command of the skating vocabulary (edges, steps, "
        "turns etc.), the clarity of technique and the use of effortless power to "
        "accelerate and vary speed.",
        ["Use of deep edges, steps and turns",
         "Balance, rhythmic knee action and precision of foot placement",
         "Flow and glide",
         "Varied use of power, speed and acceleration",
         "Use of multi directional skating",
         "Use of one foot skating"]),
    "transitions": (
        "The varied and purposeful use of intricate footwork, positions, movements and "
        "holds that link all elements.",
        ["Continuity of movements from one element to another",
         "Variety", "Difficulty", "Quality"]),
    "performance": (
        "Involvement of the Skater physically, emotionally and intellectually as they "
        "deliver the intent of the music and composition.",
        ["Physical, emotional intellectual involvement and projection",
         "Carriage and clarity of movement",
         "Variety and contrast of movements and energy",
         "Individuality / personality"]),
    "composition": (
        "An intentionally developed and/or original arrangement of all types of movements "
        "according to the principles of musical phrase, space, pattern, and structure.",
        ["Purpose (idea, concept, vision, mood)",
         "Pattern / ice coverage",
         "Multidimensional use of space and design of movements",
         "Phrase and form (movements and parts structured to match the musical phrase)",
         "Originality of the composition"]),
    "interpretation": (
        "The personal, creative, and genuine translation of the rhythm, character and "
        "content of music to movement on ice.",
        ["Movement and steps in time to the music",
         "Expression of the music's character / feeling and rhythm, when clearly "
         "identifiable",
         "Use of finesse to reflect the details and nuances of the music"]),
}


BANDS = [
    "10.00 outstanding", "9.00-9.75 excellent", "8.00-8.75 very good",
    "7.00-7.75 good", "6.00-6.75 above average", "5.00-5.75 average",
    "4.00-4.75 fair", "3.00-3.75 weak", "2.00-2.75 poor",
    "1.00-1.75 very poor", "0.25-0.75 extremely poor",
]

SYSTEM_HEADER = (
    "You are the judging panel at an ISU single skating competition. Watch the whole "
    "{segment} and award the five Program Components.\n\n"
    "Each Judge marks every component from 0.25 to 10.00 in 0.25 point increments, "
    "against these criteria (ISU Rule 504).\n")

SYSTEM_FOOTER = (
    "A mark falls in one of these bands:\n{bands}\n\n"
    "Report the panel's mark for each component, which is the trimmed mean of the "
    "Judges' marks and therefore need not land on a quarter point. Return only this "
    "JSON object and nothing else, without Markdown or code fences. Use the five "
    "component names as top-level keys and add a top-level \"PCS\" key for their sum. "
    "Copy every component name verbatim and use numbers rounded to two decimals. "
    "Do not include explanations or any other keys.")


USER_PROMPT = "<video>"


EVENT_PREFIX = re.compile(r"^(?:fs|sp)_")


def event_key(competition: str) -> str:
    stem = EVENT_PREFIX.sub("", competition.lower()).replace("_", "")
    return re.sub(r"(\d{4})", r"_\1_", stem)


DEFAULT_TEST_EVENTS = (
    "4cc_2019_men", "4cc_2019_women", "4cc_2020_men", "4cc_2020_women",
    "4cc_2022_men", "coc_2019_men", "gpf_2018_women", "idf_2019_men",
    "rc_2019_men", "sc_2019_men",
)


def build_system(segment: str) -> str:
    blocks = []
    for index, (key, name) in enumerate(COMPONENTS, 1):
        definition, bullets = CRITERIA[key]
        lines = [f"{index}. {name}: {definition}", "   The following must be considered:"]
        lines += [f"   - {b}" for b in bullets]
        blocks.append("\n".join(lines))
    return (SYSTEM_HEADER.format(segment=SEGMENTS[segment]) + "\n"
            + "\n".join(blocks) + "\n\n"
            + SYSTEM_FOOTER.format(bands="\n".join(f"  {b}" for b in BANDS)))


def build_target(marks: list) -> str:
    target = {name: round(float(mark), 2)
              for (_, name), mark in zip(COMPONENTS, marks)}
    target["PCS"] = round(sum(float(mark) for mark in marks), 2)
    return json.dumps(target, ensure_ascii=False)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def read_sample(path: Path, video_root: Path, stats: Counter):
    index = int(path.stem)
    annotation = json.loads(path.read_text(encoding="utf-8"))
    video = video_root / f"{index}.mp4"

    require_masked(video, "finefs")
    panel = annotation.get("program_component") or {}
    marks = []
    for key, _ in COMPONENTS:
        component = panel.get(key) or {}
        mark = component.get("score_of_pannel")
        if not isinstance(mark, (int, float)):
            stats["component_missing"] += 1
            return None
        if not 0.25 <= mark <= 10.0:
            stats["component_out_of_isu_range"] += 1
            return None
        marks.append(float(mark))

    segment = "SP" if index <= LAST_SHORT_PROGRAM else "FS"
    header_sum = annotation.get("total_program_component_score_not_factored")
    judges = [len((panel[key].get("judge_score") or [])) for key, _ in COMPONENTS]
    return {
        "index": index,
        "video": str(video),
        "segment": segment,
        "competition": annotation.get("competition"),
        "event": event_key(annotation.get("competition") or ""),
        "marks": marks,
        "pcs": round(sum(marks), 2),
        "pcs_sum_matches_header": (isinstance(header_sum, (int, float))
                                   and abs(sum(marks) - header_sum) <= 0.011),
        "n_judges": judges[0] if len(set(judges)) == 1 else judges,
        "n_elements": len(annotation.get("executed_element") or {}),
        "tes": annotation.get("total_element_score"),
        "total_segment_score": annotation.get("total_segment_score"),
        "deductions": annotation.get("deductions"),
        "judge_marks": {key: panel[key].get("judge_score") for key, _ in COMPONENTS},
    }


def build_row(sample: dict, fps: float | None) -> dict:
    return {
        "messages": [
            {"role": "system", "content": build_system(sample["segment"])},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": build_target(sample["marks"])},
        ],
        "videos": [sample["video"]],
        "chat_template_kwargs": {
            "fps": frame_budget.resolve_fps(sample["duration"], fps)},


        "sample_id": str(sample["index"]),
        "segment": sample["segment"],
        "competition": sample["competition"],
        "event": sample["event"],
        "duration": sample["duration"],
        "component_marks": {name: round(mark, 2)
                            for (_, name), mark in zip(COMPONENTS, sample["marks"])},
        "pcs": sample["pcs"],
        "pcs_sum_matches_header": sample["pcs_sum_matches_header"],
        "n_judges": sample["n_judges"],
        "n_elements": sample["n_elements"],
        "tes": sample["tes"],
        "total_segment_score": sample["total_segment_score"],
        "deductions": sample["deductions"],
        "judge_marks": sample["judge_marks"],
    }


def solve_split(samples: list, want_sp: int, want_fs: int):
    counts = defaultdict(lambda: [0, 0])
    for sample in samples:
        counts[sample["event"]][0 if sample["segment"] == "SP" else 1] += 1
    reachable = {(0, 0): ()}
    for event in sorted(counts):
        short, free = counts[event]
        nxt = dict(reachable)
        for (a, b), chosen in reachable.items():
            state, candidate = (a + short, b + free), chosen + (event,)
            if state not in nxt or candidate < nxt[state]:
                nxt[state] = candidate
        reachable = nxt
    return sorted(reachable.items(),
                  key=lambda kv: (abs(kv[0][0] - want_sp) + abs(kv[0][1] - want_fs),
                                  len(kv[1]), kv[1]))


def summarise(split: str, rows: list) -> None:
    durations = sorted(r["duration"] for r in rows)
    print(f"  {split}: {len(rows)} rows from {len({r['event'] for r in rows})} events, "
          f"{sum(1 for r in rows if r['segment'] == 'SP')} short program / "
          f"{sum(1 for r in rows if r['segment'] == 'FS')} free skating")
    print(f"    duration: min {durations[0]:.0f}s median "
          f"{durations[len(durations) // 2]:.0f}s max {durations[-1]:.0f}s "
          f"(128 frames spread over the whole program)")
    print(f"    sampling: {frame_budget.fps_histogram(rows)}")
    for _, name in COMPONENTS:
        marks = sorted(r["component_marks"][name] for r in rows)
        mean = sum(marks) / len(marks)
        print(f"    {name:<28} min {marks[0]:5.2f} median "
              f"{marks[len(marks) // 2]:5.2f} mean {mean:5.2f} max {marks[-1]:5.2f}")
    pcs = sorted(r["pcs"] for r in rows)
    print(f"    PCS sum                      min {pcs[0]:5.2f} median "
          f"{pcs[len(pcs) // 2]:5.2f} max {pcs[-1]:5.2f}")
    mismatched = sum(1 for r in rows if not r["pcs_sum_matches_header"])
    print(f"    components disagree with the header sum: {mismatched} rows")
    print(f"    distinct system prompts: {len({r['messages'][0]['content'] for r in rows})}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="FineFS root (default: %(default)s)")
    parser.add_argument("--video-dir", default=VIDEO_SUBDIR,
                        help="scoreboard-masked programs under the dataset root (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "sports",
                        help="where to write the metadata (default: %(default)s)")
    parser.add_argument("--prefix", default="finefs_pcs",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--suffix", default="",
                        help="appended to the output filenames (default: none)")
    parser.add_argument("--test-events", default=",".join(DEFAULT_TEST_EVENTS),
                        help="comma-separated event keys held out for test")
    parser.add_argument("--solve-split", action="store_true",
                        help="print the event subsets closest to the released test sizes "
                             "and exit")
    parser.add_argument("--fps", type=float, default=None,
                        help="pin one sampling fps for every row instead of "
                             "data/preprocess/frame_budget.py's duration ladder (default: the "
                             f"ladder, which spreads {frame_budget.FRAME_CAP} frames "
                             f"over the whole program, ~{DEFAULT_FPS:g} fps)")
    parser.add_argument("--jobs", type=int, default=16,
                        help="concurrent ffprobe calls (default: %(default)s)")
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    return parser.parse_args(argv)


def pcs_main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir = args.dataset_root / "annotation"
    video_root = args.dataset_root / args.video_dir
    if not annotation_dir.is_dir():
        sys.exit(f"missing annotations: {annotation_dir}")
    if not video_root.is_dir():
        sys.exit(f"rescaled programs not found: {video_root}\n"
                 "run the masking pass first; see data/README.md")

    print("task: ISU Program Components scoring, five marks plus their sum")
    print("rubric: ISU Rule 504 five-component definitions, quoted in the system prompt")
    print("target: panel marks (trimmed mean of the judges), summed per row")

    paths = sorted(annotation_dir.glob("*.json"), key=lambda p: int(p.stem))
    print(f"read {len(paths)} annotations")

    stats = Counter()
    samples = [s for s in (read_sample(p, video_root, stats) for p in paths)
               if s is not None]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        durations = pool.map(lambda s: probe_duration(Path(s["video"])), samples)
        for sample, duration in zip(samples, durations):
            sample["duration"] = round(duration, 2)
    kept = [s for s in samples if s["duration"] > 0.0]
    stats["video_unreadable"] += len(samples) - len(kept)
    samples = kept
    for reason, count in sorted(stats.items()):
        print(f"    dropped {count} ({reason})")
    if not samples:
        sys.exit("no usable programs")

    if args.solve_split:
        for (short, free), events in solve_split(samples, 146, 88)[:5]:
            print(f"  SP {short:3d} FS {free:3d}  events {len(events):2d}  "
                  f"{','.join(events)}")
        return 0

    test_events = {e.strip() for e in args.test_events.split(",") if e.strip()}
    known = {s["event"] for s in samples}
    unknown = sorted(test_events - known)
    if unknown:
        sys.exit(f"unknown test events: {unknown}\nknown: {sorted(known)}")

    rows_by_split = {"train": [], "test": []}
    for sample in sorted(samples, key=lambda s: s["index"]):
        split = "test" if sample["event"] in test_events else "train"
        rows_by_split[split].append(build_row(sample, args.fps))

    print(f"split: {len(known)} events, {len(test_events)} held out for test "
          f"(released sizes: 146 short programs and 88 free skatings)")
    for split in ("train", "test"):
        rows = rows_by_split[split]
        if not rows:
            sys.exit(f"no rows for split {split}")
        out_path = args.out_dir / f"{args.prefix}_{split}{args.suffix}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(split)
        summarise(split, rows)
        print(f"    -> {out_path}")

    events = {s: {r["event"] for r in rows_by_split[s]} for s in rows_by_split}
    shared = events["train"] & events["test"]
    print(f"events: {len(events['train'])} train / {len(events['test'])} test, "
          f"shared {len(shared)}")
    if shared:
        print(f"  WARNING: {sorted(shared)} appear in both splits")
    return 0


def main(argv=None) -> int:
    import sys
    if argv is None:
        argv = sys.argv[1:]
    dispatcher = argparse.ArgumentParser(add_help=False)
    dispatcher.add_argument("--task", choices=("goe", "pcs"), default="goe")
    dispatcher.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    if any(arg in ('-h', '--help') for arg in argv):
        dispatcher.print_help()
        return 0
    args, remaining = dispatcher.parse_known_args(argv)
    encode_videos(['--dataset-root', str(args.dataset_root)])
    mask_regions('finefs', args.dataset_root,
                 REPO_ROOT / 'data' / 'annotations' / 'finefs_scoreboard.json')
    if args.task == "pcs":
        return pcs_main(remaining + ['--dataset-root', str(args.dataset_root)])
    return goe_main(remaining + ['--dataset-root', str(args.dataset_root)])


CLIP_LONG_SIDE = 800


def encode_probe(path: Path) -> tuple:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames", "-of", "default=nw=1", str(path)],
        capture_output=True, text=True)
    fields = dict(line.split("=", 1) for line in result.stdout.strip().splitlines()
                  if "=" in line)
    try:
        return int(fields["width"]), int(fields["height"]), int(fields["nb_frames"])
    except (KeyError, ValueError):
        return 0, 0, 0


def encode_program(source: Path, target: Path, long_side: int, overwrite: bool) -> dict:
    width, height, frames = encode_probe(source)
    if frames == 0:
        return {"source": source, "status": "unreadable"}

    if target.is_file() and not overwrite:
        _, _, existing = encode_probe(target)
        if existing == frames:
            return {"source": source, "status": "cached", "frames": frames}


    scale = f"scale={long_side}:-2" if width >= height else f"scale=-2:{long_side}"

    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vf", scale,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-an", str(target)],
        capture_output=True, text=True)
    if result.returncode != 0:
        return {"source": source, "status": "ffmpeg_failed",
                "error": (result.stderr.strip().splitlines() or [""])[-1]}

    out_width, out_height, out_frames = encode_probe(target)
    status = "encoded" if out_frames == frames else "frame_count_changed"
    return {"source": source, "status": status, "frames": frames,
            "out_frames": out_frames, "geometry": f"{out_width}x{out_height}"}


def encode_parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="FineFS root (default: %(default)s)")
    parser.add_argument("--video-subdir", default="video",
                        help="source videos under the root (default: %(default)s)")
    parser.add_argument("--out-subdir", default="processed/videos",
                        help="written under the dataset root (default: %(default)s)")
    parser.add_argument("--long-side", type=int, default=CLIP_LONG_SIDE,
                        help="pixels on the longer edge (default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=12,
                        help="concurrent ffmpeg processes (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-encode even when a correct output already exists")
    return parser.parse_args(argv)


def encode_videos(argv=None) -> int:
    args = encode_parse_args(argv)
    source_root = args.dataset_root / args.video_subdir
    out_root = args.dataset_root / args.out_subdir
    if not source_root.is_dir():
        sys.exit(f"missing video directory: {source_root}")

    sources = sorted(source_root.glob("*.mp4"), key=lambda p: int(p.stem))
    if not sources:
        sys.exit(f"no videos found under {source_root}")
    print(f"{len(sources)} programs -> {out_root}")
    print(f"long side {args.long_side}px, frame rate and frame count preserved",
          flush=True)

    targets = [out_root / source.name for source in sources]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(
            lambda pair: encode_program(pair[0], pair[1], args.long_side, args.overwrite),
            zip(sources, targets)))

    counts, geometries = {}, {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        if "geometry" in result:
            geometries[result["geometry"]] = geometries.get(result["geometry"], 0) + 1
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    for geometry in sorted(geometries):
        print(f"  {geometry}: {geometries[geometry]}")

    bad = [r for r in results
           if r["status"] in ("unreadable", "ffmpeg_failed", "frame_count_changed")]
    for result in bad[:10]:
        print(f"  FAILED {result['source'].name}: {result['status']} "
              f"{result.get('frames')} -> {result.get('out_frames')} "
              f"{result.get('error', '')}")
    if bad:
        print(f"{len(bad)} programs unusable; process_finefs.py will not see them")
        return 1
    return 0


GOE_SYSTEM_PROMPT = (
    "You are a figure skating judge. Assess the visible execution quality of the "
    "specified element and give one GOE grade on a scale from -5 to 5 in half-point "
    "increments. Negative grades indicate execution faults, zero is neutral, "
    "and positive grades indicate better execution.\n"
    "Judge control, precision, stability and flow using the visual criteria supplied "
    "for this element. Do not award extra credit for base difficulty. Do not infer "
    "music matching or musical expression from this silent video. Unavailable music "
    "and whole-program requirements do not earn credit or incur deductions.\n"
)

GOE_OUTPUT_FORMAT = (
    'Return only this JSON object and nothing else: '
    '{"goe_grade": <number from -5.0 to 5.0 with one decimal place>}. '
    "Do not include an explanation or any other text."
)

ELEMENT_CATEGORIES = frozenset(("jump", "spin", "sequence"))


def goe_build_system(category, element, season="2018/19", segment="sp"):
    return (GOE_SYSTEM_PROMPT + "\n" + detailed_rules(category, element, season, segment)
            + "\n\n" + GOE_OUTPUT_FORMAT)


def clean_element_code(code):

    code = code.replace("1EU", "1Eu").replace("CCosp", "CCoSp")
    code = code.replace("1Eut", "1Eu+").rstrip(",")
    code = re.sub(r"[<>!*]", "", code)
    code = re.sub(r"([1-4](?:A|T|S|Lo|F|Lz|Eu))[qe]", r"\1", code)
    return "+".join(part for part in code.split("+")
                    if part and part not in ("REP", "COM", "COMBO"))


def median_grade(grades) -> float:
    if not isinstance(grades, list) or len(grades) < 3:
        raise ValueError("at least three judge grades required")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or not -5 <= x <= 5 or x != int(x) for x in grades):
        raise ValueError("judge grades must be integers in [-5, 5]")
    return float(median(grades))


def parse_interval(value: str) -> tuple[float, float]:
    def seconds(text):
        match = re.fullmatch(r"(\d+)-(\d+(?:\.\d+)?)", text.strip())
        if not match or float(match[2]) >= 60:
            raise ValueError(f"invalid timestamp: {text!r}")
        return int(match[1]) * 60 + float(match[2])
    if not isinstance(value, str) or value.count(",") != 1:
        raise ValueError(f"invalid interval: {value!r}")
    start, end = map(seconds, value.split(","))
    if end <= start:
        raise ValueError(f"nonpositive interval: {value!r}")
    return start, end


def goe_probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True)
    value = float(result.stdout.strip())
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid video duration: {path}")
    return value


def prepare_element(program_id, annotation, element_id, element, source, duration):
    if not re.fullmatch(r"element\d+", element_id):
        raise ValueError(f"invalid element id: {element_id!r}")
    category = element.get("coarse_class")
    if category not in ELEMENT_CATEGORIES:
        raise ValueError(f"unsupported element category: {category!r}")
    code = element.get("element")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("missing element name")

    clean_code = clean_element_code(code.strip())
    if clean_code not in ELEMENT_DESCRIPTIONS:
        raise ValueError(f"no action description for {clean_code!r}")
    if ELEMENT_DESCRIPTIONS[clean_code][0] != category:
        raise ValueError(f"element code/category mismatch: {clean_code!r}, {category!r}")
    start, end = parse_interval(element.get("time"))
    if end > duration:
        raise ValueError("annotated element extends beyond the source video")
    grade = median_grade(element.get("judge_score"))
    competition = annotation.get("competition")
    if not isinstance(competition, str) or not competition:
        raise ValueError("missing competition for event-based split")
    return {
        "sample_id": f"{program_id}_{element_id}", "program_id": str(program_id),
        "element_id": element_id, "element": clean_code, "category": category,
        "source_video": str(source.resolve()), "competition": competition,
        **finefs_events.context(program_id, competition),
        "element_start": start, "element_end": end,
        "clip_start": start, "clip_end": end,
        "judge_grades": element["judge_score"], "goe_grade": grade,
        "stored_goe_points": element.get("goe"), "base_value": element.get("bv"),
        "stored_element_score": element.get("score_of_pannel"),
    }


def goe_build_row(sample, fps=None):
    duration = sample["element_end"] - sample["element_start"]
    category, name, expected = ELEMENT_DESCRIPTIONS[sample["element"]]
    if category != sample["category"]:
        raise ValueError("element code/category mismatch")
    question = (
        f"<video>\nCompetition: {finefs_events.describe(sample)}\n"
        f"Element: {sample['element']}\nAction: {name}\n\n"
        f"Expected execution:\n{expected}\n\n"
        "Score the execution quality of this element."
    )
    return {
        "messages": [{"role": "system", "content": goe_build_system(
            category, sample["element"], sample["season"], sample["segment"])},
                     {"role": "user", "content": question},
                     {"role": "assistant", "content": json.dumps(
                         {"goe_grade": round(float(sample["goe_grade"]), 1)},
                         ensure_ascii=False)}],
        "videos": [sample["source_video"]],
        "chat_template_kwargs": {
            "video_start": sample["element_start"],
            "video_end": sample["element_end"],
            "fps": frame_budget.resolve_fps(duration, fps),
        },
        **sample, "duration": duration,
        "target_definition": "median_raw_judge_goe_grades",
        "rubric": "season_visual_goe_v2",
    }


def goe_main(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--video-dir", default=VIDEO_SUBDIR)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "sports")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--jobs", type=int, default=8, help="concurrent source-video probes")
    parser.add_argument("--test-events", default=",".join(DEFAULT_TEST_EVENTS))
    parser.add_argument("--example", type=int, help="generate only this program's selected element")
    parser.add_argument("--element", default="element1", help="element id for --example")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("jobs must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("fps must be finite and positive")
    annotation_dir = args.dataset_root / "annotation"
    paths = ([annotation_dir / f"{args.example}.json"] if args.example is not None
             else sorted(annotation_dir.glob("*.json"), key=lambda p: int(p.stem)))
    if not paths:
        parser.error("no annotations found")

    for path in paths:
        finefs_events.verify_annotation(path)
    def read_program(path):
        annotation = json.loads(path.read_text())
        source = args.dataset_root / args.video_dir / f"{path.stem}.mp4"
        require_masked(source, "finefs")
        program_samples, program_skipped = [], Counter()
        try:
            duration = goe_probe_duration(source)
        except (ValueError, subprocess.CalledProcessError):
            if args.example is not None:
                raise
            return [], Counter({"unreadable_video": 1})
        elements = annotation.get("executed_element") or {}
        if args.example is not None:
            elements = {args.element: elements[args.element]}
        for element_id, element in elements.items():
            try:
                program_samples.append(prepare_element(path.stem, annotation, element_id,
                                                       element, source, duration))
            except ValueError as error:
                if args.example is not None:
                    raise
                program_skipped[str(error)] += 1
        return program_samples, program_skipped

    samples, skipped = [], Counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for program_samples, program_skipped in pool.map(read_program, paths):
            samples.extend(program_samples)
            skipped.update(program_skipped)
    if not samples:
        parser.error(f"no usable elements: {dict(skipped)}")
    test_events = set(args.test_events.split(","))
    if args.example is None:
        unknown = test_events - {s["event"] for s in samples}
        if unknown:
            parser.error(f"unknown test events: {sorted(unknown)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "test": []}
    for sample in samples:
        row = goe_build_row(sample, args.fps)
        split = "test" if sample["event"] in test_events else "train"
        rows[split].append(row)
    print(f"prepared {len(samples)} source-video windows; no clips encoded", flush=True)
    if args.example is not None:
        row = next(r for subset in rows.values() for r in subset)
        output = args.out_dir / "example.json"
        output.write_text(json.dumps(row, indent=2) + "\n")
        print(output)
        print(json.dumps(row, indent=2))
    else:
        for split, subset in rows.items():
            output = args.out_dir / f"finefs_goe_{split}.jsonl"
            output.write_text("".join(json.dumps(row) + "\n" for row in subset))
            print(f"{split}: {len(subset)} elements -> {output}")
    print(f"skipped: {dict(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
