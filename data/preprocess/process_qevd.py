#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

from . import frame_budget
from . import loss_scale
from .feedback_style import sentence_count

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "QEVD"

FIT300K_DIR = "QEVD-FIT-300k"
COACH_DIRS = {"train": "QEVD-FIT-COACH", "test": "QEVD-FIT-COACH-Benchmark"}


FIT300K_SPLITS = ("train",)


QA_SECTIONS = ("high_level", "fine_grain")


TYPE_ONLY_QUERIES = {
    "what exercise is the user doing?",
    "what is the user doing?",
}


MIN_PREFIX_SEC = 2.0


FIT300K_SOURCE_FPS = 30.0
DEFAULT_FPS = frame_budget.sampling_fps(8.0)
DEFAULT_WINDOW_SEC = 8.0


HIGH_LEVEL_SYSTEM_PROMPT = (
    "You analyze workout videos and briefly describe the exercise and its execution. "
    "Respond in one to three concise sentences."
)

FINE_GRAIN_SYSTEM_PROMPT = (
    "You answer specific questions about visible movements in workout videos. "
    "Respond in one to three concise sentences."
)

YES_NO_START = re.compile(r"^\s*(?:yes|no)\b", re.I)


def fit300k_system_prompt(section: str, response: str) -> str:
    if section == "high_level":
        if sentence_count(response) > 3:
            return HIGH_LEVEL_SYSTEM_PROMPT.replace("one to three", "one to five")
        return HIGH_LEVEL_SYSTEM_PROMPT
    if section != "fine_grain":
        raise ValueError(f"Unknown FIT-300K question family: {section}")
    return FINE_GRAIN_SYSTEM_PROMPT + (
        " Begin with Yes or No." if YES_NO_START.match(response) else ""
    )


HIGH_LEVEL_QUERY = "What exercise is the user doing, and how are they performing it?"

COACH_SYSTEM_PROMPT = (
    "You are a workout coach. Give one brief, timely, and useful comment based on the "
    "exercise shown so far."
)

COACH_HINT = "What would you say now?"

def read_json(path: Path):
    if not path.is_file():
        sys.exit(f"not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def collect_fit300k(root: Path, clip_dir: str, max_per_clip: int,
                    one_per_clip: bool = True, prefer_family: str = "fine_grain",
                    family_ratio: float = 1.0):
    entries = read_json(root / FIT300K_DIR / "questions.json")
    clips = root / FIT300K_DIR / clip_dir
    if not clips.is_dir():
        sys.exit(f"clip directory not found: {clips}")
    on_disk = {name for name in clips.iterdir() if name.suffix == ".mp4"}
    on_disk = {path.name for path in on_disk}

    per_split = {"train": [], "test": []}
    stats = Counter()
    for entry in entries:
        split = entry.get("split")
        if split not in per_split:
            stats[f"unknown_split:{split}"] += 1
            continue
        name = Path(entry["video_path"]).name
        if name not in on_disk:
            stats["clip_missing_on_disk"] += 1
            continue

        pairs, seen = [], set()
        for section in QA_SECTIONS:
            block = entry.get(section) or {}
            for query, response in zip(block.get("query") or [],
                                       block.get("response") or []):
                query, response = (query or "").strip(), (response or "").strip()
                if not query or not response:
                    stats["blank_qa_pair"] += 1
                    continue
                if query.casefold() in TYPE_ONLY_QUERIES:
                    stats["type_only_question"] += 1
                    continue
                if (query, response) in seen:
                    stats["duplicate_qa_pair"] += 1
                    continue
                seen.add((query, response))
                pairs.append((section, query, response))
        if not pairs:
            stats["clip_without_qa"] += 1
            continue
        if one_per_clip:
            stats["extra_qa_per_clip"] += len(pairs) - 1
            preferred = [p for p in pairs if p[0] == prefer_family]
            pairs = [random.Random(name).choice(preferred or pairs)]
        elif max_per_clip:
            pairs = pairs[:max_per_clip]
        per_split[split].append({"clip": name, "video": str(clips / name), "pairs": pairs})


    return per_split, stats


def balance_families(samples, ratio: float, stats: Counter):
    fine = [s for s in samples if s["pairs"][0][0] == "fine_grain"]
    high = [s for s in samples if s["pairs"][0][0] != "fine_grain"]
    keep = round(len(fine) * ratio)
    if keep >= len(high):
        return samples
    high.sort(key=lambda s: random.Random(s["clip"]).random())
    stats["high_level_over_ratio"] += len(high) - keep
    return fine + high[:keep]


def family_of(sample: dict) -> str:
    return sample["pairs"][0][0]


def fit300k_records(sample: dict, fps: float | None):
    row_fps = frame_budget.resolve_fps(sample["duration"], fps)
    return [{
        "messages": [
            {"role": "system",
             "content": fit300k_system_prompt(section, response)},
            {"role": "user",
             "content": "<video>\n" + (HIGH_LEVEL_QUERY if section == "high_level"
                                       else query)},
            {"role": "assistant", "content": response},
        ],
        "videos": [sample["video"]],
        "chat_template_kwargs": {"fps": row_fps},
    } for section, query, response in sample["pairs"]]


def video_duration(path: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def segment_announcements(base: Path, entries) -> dict:
    per_video = {Path(entry["long_range_video_file"]).stem: entry["feedbacks"]
                 for entry in read_json(base / "feedbacks_long_range.json")}
    out, missing = {}, 0
    for entry in entries:
        frames = per_video.get(Path(entry["src_video_path"]).stem) or []
        index = entry["src_start_idx"]
        text = (frames[index] or "").strip() if index < len(frames) else ""
        if not text:
            missing += 1
        out[entry["video_path"]] = text
    if missing:
        print(f"    no announcement recovered for {missing}/{len(entries)} segments; "
              f"their history starts at the first feedback")
    return out


def collect_coach(root: Path, split: str):
    base = root / COACH_DIRS[split]
    entries = read_json(base / "feedbacks_segment_new.json")
    announcements = segment_announcements(base, entries)

    videos = [base / entry["video_path"] for entry in entries]
    with ThreadPoolExecutor(16) as pool:
        durations = dict(zip(videos, pool.map(video_duration, videos)))

    samples = []
    stats = Counter()
    for entry, video in zip(entries, videos):
        if not video.is_file():
            stats["segment_video_missing"] += 1
            continue
        duration = durations.get(video)
        if not duration:
            stats["duration_unreadable"] += 1
            continue
        exercise = (entry.get("exercise_name") or "").strip()
        if not exercise:
            stats["missing_exercise_name"] += 1
            continue

        responses = [
            {"time": round(float(t), 2), "text": (text or "").strip()}
            for text, t in zip(entry.get("responses") or [],
                               entry.get("response_timestamps") or [])
            if (text or "").strip()
        ]
        responses.sort(key=lambda r: r["time"])
        if not responses:
            stats["segment_without_feedback"] += 1
            continue
        samples.append({
            "segment": Path(entry["video_path"]).stem,
            "exercise": exercise,
            "announcement": announcements.get(entry["video_path"], ""),
            "duration": round(duration, 2),
            "responses": responses,
            "video": str(video),
        })
    return samples, stats


def history_block(feedbacks, announcement: str = "") -> str:
    lines = [f"<0.0 seconds> {announcement}"] if announcement else []
    lines += [f"<{f['time']:.1f} seconds> {f['text']}" for f in feedbacks]
    if not lines:
        return "Previous feedback: None\n\n"
    body = "\n".join(lines)
    return f"Previous feedback:\n{body}\n\n"


def coach_records(sample: dict, fps: float | None, window_sec: float = 0.0):
    records = []

    announcement = sample["announcement"]
    for index, feedback in enumerate(sample["responses"]):

        end = min(max(feedback["time"], MIN_PREFIX_SEC), sample["duration"])
        start = max(0.0, end - window_sec) if window_sec > 0 else 0.0


        history = history_block(sample["responses"][:index], announcement)
        records.append({
            "messages": [
                {"role": "system", "content": COACH_SYSTEM_PROMPT},
                {"role": "user", "content": f"<video>\n{history}{COACH_HINT}"},
                {"role": "assistant", "content": feedback["text"]},
            ],
            "videos": [sample["video"]],
            "chat_template_kwargs": {"video_start": round(start, 2),
                                     "video_end": round(end, 2),
                                     "fps": frame_budget.resolve_fps(end - start, fps)},


            "segment": sample["segment"],
            "step": index,
            "feedback_time": feedback["time"],
        })
    return records


def quantiles(values):
    values = sorted(values)
    return (values[len(values) // 2], values[int(len(values) * 0.9)], values[-1])


def write(out_path: Path, records) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="QEVD root (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "fitness",
                        help="where to write the metadata (default: %(default)s)")
    parser.add_argument("--prefix", default="qevd",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--subsets", default="fit300k,fit-coach",
                        help="comma-separated subsets to build (default: %(default)s)")
    parser.add_argument("--clip-dir", default="short_clips",
                        help="FIT-300K clip directory (default: %(default)s)")
    parser.add_argument("--max-qa-per-clip", type=int, default=0,
                        help="keep at most N pairs per clip, 0 = all, ignored while "
                             "--one-qa-per-clip is on (default: %(default)s)")
    parser.add_argument("--one-qa-per-clip", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="keep one randomly chosen FIT-300K question per clip, "
                             "seeded by clip name (default: %(default)s)")
    parser.add_argument("--prefer-family", default="fine_grain",
                        choices=("fine_grain", "high_level", "none"),
                        help="which question family that single pick favours when the "
                             "clip has both, none = uniform (default: %(default)s)")
    parser.add_argument("--family-ratio", type=float, default=1.0,
                        help="high_level rows per fine_grain row, surplus high_level "
                             "clips are dropped, 0 = keep every clip "
                             "(default: %(default)s)")
    parser.add_argument("--fps", type=float, default=None,
                        help="pin one sampling fps for every row instead of "
                             "data/preprocess/frame_budget.py's duration ladder (default: "
                             "the ladder)")
    parser.add_argument("--jobs", type=int, default=64,
                        help="concurrent ffprobe calls for the FIT-300K clip durations; "
                             "the result is cached (default: %(default)s)")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC,
                        help="streaming samples see only the last N seconds ending at "
                             "the feedback timestamp, 0 = the whole prefix [0, t] "
                             "(default: %(default)s)")
    parser.add_argument("--balance", action="store_true",
                        help="duplicate the minority question family up to the majority count "
                             "; opt-in experiment, the pipeline uses loss_scale weights instead")
    parser.add_argument("--seed", type=int, default=42,
                        help="upsampling seed when --balance is set (default: %(default)s)")
    return parser.parse_args(argv)


def build_fit300k(args) -> None:
    per_split, stats = collect_fit300k(args.dataset_root, args.clip_dir,
                                       args.max_qa_per_clip, args.one_qa_per_clip,
                                       args.prefer_family, args.family_ratio)
    fps = args.fps
    print("fit300k (questions.json, split field), sampling "
          + (f"pinned at {fps:g} fps" if fps else
             "per clip, data/preprocess/frame_budget.py's duration ladder")
          + f" (source {FIT300K_SOURCE_FPS:g} fps)"
          + (f", one question per clip, favouring {args.prefer_family}"
             if args.one_qa_per_clip else "")
          + (f", high_level:fine_grain {args.family_ratio:g}:1"
             if args.one_qa_per_clip and args.family_ratio > 0 else ""))
    for reason, count in sorted(stats.items()):
        print(f"    skipped {count} ({reason})")
    for split in FIT300K_SPLITS:
        samples = per_split[split]
        if not samples:
            sys.exit(f"no usable FIT-300K samples for split {split}")
        samples.sort(key=lambda s: s["clip"])


        durations = frame_budget.probe_durations([s["video"] for s in samples],
                                                workers=args.jobs)
        for sample in samples:
            sample["duration"] = durations[sample["video"]]
        weights = {}
        if split == "train" and getattr(args, "balance", False):

            fams = {}
            for s in samples:
                fams.setdefault(s["pairs"][0][0], []).append(s)
            if len(fams) == 2 and all(fams.values()):
                target = max(len(v) for v in fams.values())
                rng = random.Random(args.seed)
                balanced = []
                for fam in sorted(fams):
                    items = fams[fam]
                    grown = list(items)
                    while len(grown) < target:
                        need = target - len(grown)
                        grown.extend(items if need >= len(items) else rng.sample(items, need))
                    balanced.extend(grown)
                samples = sorted(balanced, key=lambda s: s["clip"])
        elif split == "train":


            weights = loss_scale.balanced_weights(samples, lambda s: "fit300k", family_of)
        records = [loss_scale.attach(r, weights.get(("fit300k", family_of(s)), 1.0))
                   for s in samples for r in fit300k_records(s, fps)]
        sections = Counter(section for s in samples for section, _, _ in s["pairs"])
        median, p90, top = quantiles([len(s["pairs"]) for s in samples])
        print(f"  {split}: {len(records)} samples from {len(samples)} clips")
        print(f"    pairs per clip: median {median}, p90 {p90}, max {top}")
        print(f"    question family {dict(sections.most_common())}")
        if weights:
            print("    loss_scale: " + loss_scale.summarise(
                samples, lambda s: "fit300k", family_of, weights))
        print(f"    sampling: {frame_budget.fps_histogram(records)}")
        out_path = args.out_dir / f"{args.prefix}_fit300k_{split}.jsonl"
        write(out_path, records)
        if split == "train":
            for stale in args.out_dir.glob(f"{args.prefix}_fit300k_train_*.jsonl"):
                stale.unlink()
        print(f"    -> {out_path}")


def build_coach(args) -> None:
    print("fit-coach (segment level, train=QEVD-FIT-COACH, "
          "test=QEVD-FIT-COACH-Benchmark), sampling "
          + (f"pinned at {args.fps:g} fps" if args.fps else
             "per window, data/preprocess/frame_budget.py's duration ladder")
          + (f", window {args.window_sec:g}s "
             f"({frame_budget.frames_for(args.window_sec, frame_budget.resolve_fps(args.window_sec, args.fps))} frames)"
             if args.window_sec > 0 else ", full prefix [0, t]"))
    print("    exercise name withheld from the prompt; the opening announcement is "
          "still quoted as the first history line")
    for split in COACH_DIRS:
        samples, stats = collect_coach(args.dataset_root, split)
        if not samples:
            sys.exit(f"no usable FIT-COACH samples for split {split}")
        samples.sort(key=lambda s: s["segment"])
        records = [r for s in samples
                   for r in coach_records(s, args.fps, args.window_sec)]
        out_path = args.out_dir / f"{args.prefix}_fitcoach_{split}.jsonl"
        write(out_path, records)

        for reason, count in sorted(stats.items()):
            print(f"    skipped {count} ({reason})")
        steps = quantiles([len(s["responses"]) for s in samples])
        lengths = quantiles([s["duration"] for s in samples])
        prefixes = quantiles([r["chat_template_kwargs"]["video_end"]
                              - r["chat_template_kwargs"]["video_start"] for r in records])
        exercises = Counter(s["exercise"] for s in samples)
        feedbacks = sum(len(s["responses"]) for s in samples)
        print(f"  {split}: {len(records)} samples from {len(samples)} segments "
              f"({len(exercises)} exercises, {feedbacks} feedbacks)")
        print(f"    feedbacks per segment: median {steps[0]}, p90 {steps[1]}, max {steps[2]}")
        print(f"    segment seconds: median {lengths[0]:.0f}, "
              f"p90 {lengths[1]:.0f}, max {lengths[2]:.0f}")
        print(f"    video clip seconds: median {prefixes[0]:.1f}, "
              f"p90 {prefixes[1]:.1f}, max {prefixes[2]:.1f}")
        print(f"    sampling: {frame_budget.fps_histogram(records)}")
        print(f"    -> {out_path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    subsets = {s.strip() for s in args.subsets.split(",") if s.strip()}
    unknown = subsets - {"fit300k", "fit-coach"}
    if unknown:
        sys.exit(f"unknown subsets: {sorted(unknown)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "fit300k" in subsets:
        build_fit300k(args)
    if "fit-coach" in subsets:
        build_coach(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
