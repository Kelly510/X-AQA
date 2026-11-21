#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

from . import frame_budget

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "KERAAL"

CLASSES = ("correct", "error1", "error2", "error3")
LETTERS = ("A", "B", "C", "D")


DEFAULT_FPS = frame_budget.sampling_fps(32.0)
DEFAULT_WINDOW_SEC = 32.0


EXERCISES = {
    "RTK": ("torso rotation",
            "rotate the trunk to the left, then the same rotation to the right"),
    "ELK": ("flank stretch",
            "bend the trunk laterally to the left and then to the right"),
    "CTK": ("hiding face",
            "a breathing exercise with the upper limbs flexed 90 degrees at the "
            "shoulder and at the elbow"),
}


VERDICTS = {
    "RTK": {
        "correct": "correct execution",
        "error1": "arms are not raised enough",
        "error2": "the torso's rotation is not sufficient",
        "error3": "the body is leaned on the side",
    },
    "ELK": {
        "correct": "correct execution",
        "error1": "the opposite arm is not along the body",
        "error2": "the body is not tilted",
        "error3": "the above arm is not bent",
    },
    "CTK": {
        "correct": "correct execution",
        "error1": "the arms are not raised enough",
        "error2": "the arms are not outspread enough",
        "error3": "the elbows are not bent enough",
    },
}

SYSTEM_PROMPT = (
    "You classify how a low-back-pain rehabilitation exercise was performed. "
    'Return only this JSON object and nothing else: '
    '{"answer": "<selected option letter>"}. '
    "The value must be one of the option letters shown for the current sample."
)


CLIP_RE = re.compile(r"^(?P<group>3|1A|2A)-(?P<exercise>[A-Z]{3})-(?P<subject>\w+)-"
                     r"(?P<tail>\w+)-(?P<rec>\d+)\.mp4$")


GROUP3_LABEL_RE = re.compile(r"^(?:C|E(?P<error>\d)B(?P<bodypart>\d))$")

ERROR_TYPES = {"error1": "error1", "error2": "error2", "error3": "error3"}


def read_anvil(path: Path) -> dict:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text, count=1).strip()
    root = ET.fromstring(text)

    out = {"evaluation": None, "type": None, "bodypart": None, "severity": None,
           "temporal_elements": 0}
    for track in root.iter("track"):
        name = track.get("name") or ""
        elements = list(track.iter("el"))
        if name == "Global evaluation":
            for element in elements:
                for attribute in element.iter("attribute"):
                    if (attribute.get("name") or "").lower() == "evaluation":
                        out["evaluation"] = (attribute.text or "").strip() or None
                        break
                if out["evaluation"]:
                    break
        elif name == "Global error":
            for element in elements:
                for attribute in element.iter("attribute"):
                    key = (attribute.get("name") or "")
                    value = (attribute.text or "").strip() or None
                    if key == "type" and out["type"] is None:
                        out["type"] = value
                    elif key == "bodyPart" and out["bodypart"] is None:
                        out["bodypart"] = value
                    elif key == "Evaluation" and out["severity"] is None:
                        out["severity"] = value
        elif name == "Temporal error":
            out["temporal_elements"] = len(elements)
    return out


def anvil_class(annotation: dict):
    evaluation = (annotation.get("evaluation") or "").lower()
    error = (annotation.get("type") or "").lower()
    if not evaluation:
        return None, "empty_annotation"
    if evaluation == "correct":

        return "correct", None
    if evaluation in ("incomplete", "motionless"):
        return None, f"global_{evaluation}"
    if evaluation != "incorrect":
        return None, f"unexpected_evaluation:{evaluation}"
    if error in CLASSES[1:]:
        return error, None
    if error in ("", "none"):
        return None, "incorrect_without_error_type"
    return None, f"error_type:{error}"


def probe_duration(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(probe.stdout.strip())
    except ValueError:
        return 0.0


def windows(duration: float, window_sec: float):
    if duration <= window_sec + 1e-6:
        return [(0.0, max(duration, 0.0))]
    spans = []
    start = 0.0
    while duration - start >= window_sec - 1e-6:
        spans.append((start, start + window_sec))
        start += window_sec
    if duration - start > 1e-6:
        spans.append((duration - window_sec, duration))
    return spans


def option_order(key: str) -> tuple:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    rng = random.Random(int(digest, 16))
    order = list(CLASSES)
    rng.shuffle(order)
    return tuple(order)


def build_records(sample: dict, fps: float | None, window_sec: float):
    exercise = sample["exercise"]
    name, description = EXERCISES[exercise]
    verdicts = VERDICTS[exercise]
    order = option_order(sample["clip"])
    options = {letter: klass for letter, klass in zip(LETTERS, order)}
    gold_letter = LETTERS[order.index(sample["gold"])]

    lines = [f"Exercise: {name} - {description}.", "Possible verdicts:"]
    lines += [f"{letter}. {verdicts[options[letter]]}" for letter in LETTERS]
    question = "<video>\n" + "\n".join(lines)

    spans = windows(sample["duration"], window_sec)
    records = []
    for index, (start, end) in enumerate(spans):
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": json.dumps(
                    {"answer": gold_letter}, ensure_ascii=False)},
            ],
            "videos": [sample["video"]],
            "chat_template_kwargs": {"video_start": round(start, 2),
                                     "video_end": round(end, 2),
                                     "fps": frame_budget.resolve_fps(end - start, fps)},


            "exercise": exercise,
            "exercise_name": name,
            "group": sample["group"],
            "subject": sample["subject"],
            "options": options,
            "gold_letter": gold_letter,
            "gold_code": sample["gold"],
            "window_index": index,
            "n_windows": len(spans),
        })
    return records


def group3_sample(match, clip: Path) -> dict:
    label = GROUP3_LABEL_RE.match(match["tail"])
    subject = match["subject"]
    return {
        "clip": clip.name,
        "video": str(clip),
        "group": "3",
        "exercise": match["exercise"],
        "subject": subject,
        "person": subject.split("T")[0],
        "recording": match["rec"],
        "gold": "correct" if label["error"] is None else f"error{label['error']}",
        "bodypart": None if label["bodypart"] is None else f"B{label['bodypart']}",
        "severity": None,
        "agree_evaluation": None,
        "agree_class": None,
        "temporal_elements": 0,
    }


def patient_sample(match, clip: Path, root: Path, annotator: str, stats: Counter):
    group = match["group"]
    stem = (f"G{group}-{match['exercise']}-{match['subject']}-"
            f"{match['tail']}-{match['rec']}")
    primary = root / f"group{group}" / f"annotator{annotator}" / f"{stem}.anvil"
    if not primary.is_file():
        stats["annotation_missing"] += 1
        return None
    annotation = read_anvil(primary)
    gold, reason = anvil_class(annotation)
    if gold is None:
        stats[reason] += 1
        return None

    other = "A" if annotator == "B" else "B"
    other_path = root / f"group{group}" / f"annotator{other}" / f"{stem}.anvil"
    agree_evaluation = agree_class = None
    if other_path.is_file():
        second = read_anvil(other_path)
        agree_evaluation = ((second.get("evaluation") or "").lower()
                            == (annotation.get("evaluation") or "").lower())
        agree_class = anvil_class(second)[0] == gold
    return {
        "clip": clip.name,
        "video": str(clip),
        "group": group,
        "exercise": match["exercise"],
        "subject": f"{match['subject']}-{match['tail']}",
        "person": f"{match['subject']}-{match['tail']}",
        "recording": match["rec"],
        "gold": gold,
        "bodypart": annotation.get("bodypart"),
        "severity": annotation.get("severity"),
        "agree_evaluation": agree_evaluation,
        "agree_class": agree_class,
        "temporal_elements": annotation.get("temporal_elements", 0),
    }


def collect(root: Path, video_dir: str, annotator: str, jobs: int):
    clip_dir = root / "processed" / video_dir
    if not clip_dir.is_dir():
        sys.exit(f"encoded clips not found: {clip_dir}\n"
                 "run process_keraal.py")

    samples, stats = defaultdict(list), Counter()
    for clip in sorted(clip_dir.glob("*.mp4")):
        match = CLIP_RE.match(clip.name)
        if match is None:
            stats["unexpected_clip_name"] += 1
            continue
        if match["exercise"] not in EXERCISES:
            stats[f"unknown_exercise:{match['exercise']}"] += 1
            continue
        if match["group"] == "3":
            if GROUP3_LABEL_RE.match(match["tail"]) is None:
                stats["unexpected_group3_label"] += 1
                continue
            samples["train"].append(group3_sample(match, clip))
        else:
            sample = patient_sample(match, clip, root, annotator, stats)
            if sample is not None:
                samples["test"].append(sample)

    flat = [s for group in samples.values() for s in group]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for sample, duration in zip(flat, pool.map(lambda s: probe_duration(Path(s["video"])), flat)):
            sample["duration"] = duration
    for split in list(samples):
        kept = [s for s in samples[split] if s["duration"] > 0.0]
        stats["clip_unreadable"] += len(samples[split]) - len(kept)
        samples[split] = kept
    return samples, stats


def summarise(name: str, samples, records) -> None:
    durations = sorted(s["duration"] for s in samples)
    print(f"  {name}: {len(records)} windows from {len(samples)} recordings "
          f"({len(records) / max(len(samples), 1):.2f} windows each)")
    print(f"    duration: min {durations[0]:.1f}s median "
          f"{durations[len(durations) // 2]:.1f}s max {durations[-1]:.1f}s")
    print(f"    sampling: {frame_budget.fps_histogram(records)}")
    for exercise in sorted(EXERCISES):
        group = [s for s in samples if s["exercise"] == exercise]
        if not group:
            continue
        counts = Counter(s["gold"] for s in group)
        majority = max(counts.values()) / len(group)
        print(f"    {exercise} ({EXERCISES[exercise][0]}): {len(group)} recordings "
              f"{{{', '.join(f'{k} {counts.get(k, 0)}' for k in CLASSES)}}} "
              f"majority {majority:.3f}")
    joint = Counter((r["gold_letter"], r["gold_code"]) for r in records)
    letters = Counter(letter for letter, _ in joint.elements())
    print(f"    gold letters: {dict(sorted(letters.items()))} "
          f"({len(joint)}/16 (letter, class) cells non-empty)")
    agreed = [s for s in samples if s["agree_class"] is not None]
    if agreed:
        same = sum(1 for s in agreed if s["agree_class"])
        evaluation = sum(1 for s in agreed if s["agree_evaluation"])
        print(f"    annotator agreement on the kept recordings: {same}/{len(agreed)} "
              f"same class, {evaluation}/{len(agreed)} same global evaluation")


def write_meta(path: Path, samples_by_split: dict, records_by_split: dict) -> None:
    meta = {}
    for split, samples in samples_by_split.items():
        windows_per_clip = Counter(r["videos"][0].rsplit("/", 1)[-1]
                                  for r in records_by_split[split])
        for sample in samples:
            meta[sample["clip"]] = {
                "split": split,
                "group": sample["group"],
                "exercise": sample["exercise"],
                "subject": sample["subject"],
                "person": sample["person"],
                "recording": sample["recording"],
                "gold_code": sample["gold"],
                "bodypart": sample["bodypart"],
                "severity": sample["severity"],
                "agree_evaluation": sample["agree_evaluation"],
                "agree_class": sample["agree_class"],
                "temporal_elements": sample["temporal_elements"],
                "duration": round(sample["duration"], 2),
                "n_windows": windows_per_clip[sample["clip"]],
            }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=1, sort_keys=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="KERAAL root (default: %(default)s)")
    parser.add_argument("--video-dir", default="videos",
                        help="encoded clip directory under processed/ "
                             "(default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "rehabilitation",
                        help="where to write the metadata (default: %(default)s)")
    parser.add_argument("--prefix", default="keraal",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--suffix", default="",
                        help="appended to the output filenames (default: none)")
    parser.add_argument("--annotator", choices=["A", "B"], default="B",
                        help="which medical annotator provides the test labels "
                             "(default: %(default)s, the more complete one)")
    parser.add_argument("--fps", type=float, default=None,
                        help="pin one sampling fps for every row instead of "
                             "data/preprocess/frame_budget.py's duration ladder (default: the "
                             f"ladder, {DEFAULT_FPS:g} fps for a full window)")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC,
                        help="window length in seconds (default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=16,
                        help="concurrent ffprobe calls (default: %(default)s)")
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    encode_videos(['--dataset-root', str(args.dataset_root)])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    window_fps = frame_budget.resolve_fps(args.window_sec, args.fps)
    print("task: four-class rehabilitation error classification (paper Sec. IV-C)")
    print(f"split: train = group3, test = group1A + group2A (annotator{args.annotator})")
    print(f"windows: {args.window_sec:g}s = "
          f"{frame_budget.frames_for(args.window_sec, window_fps)} frames at "
          f"{window_fps:g} fps, last window anchored at the end")

    samples, stats = collect(args.dataset_root, args.video_dir, args.annotator, args.jobs)
    for split in ("train", "test"):
        if not samples.get(split):
            sys.exit(f"no usable recordings for split {split}")
    for reason, count in sorted(stats.items()):
        if count:
            print(f"    dropped {count} ({reason})")

    records_by_split, persons = {}, {}
    for split in ("train", "test"):
        group = sorted(samples[split], key=lambda s: s["clip"])
        records = [r for s in group for r in build_records(s, args.fps, args.window_sec)]
        out_path = args.out_dir / f"{args.prefix}_{split}{args.suffix}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        samples[split] = group
        records_by_split[split] = records
        persons[split] = {s["person"] for s in group}
        print(split)
        summarise(split, group, records)
        print(f"    -> {out_path}")

    meta_path = args.out_dir / f"{args.prefix}_meta{args.suffix}.json"
    write_meta(meta_path, {s: samples[s] for s in ("train", "test")}, records_by_split)
    print(f"meta -> {meta_path}")

    shared = persons["train"] & persons["test"]
    print(f"persons: {len(persons['train'])} train / {len(persons['test'])} test, "
          f"shared {len(shared)}")
    if shared:
        print(f"  WARNING: {sorted(shared)} appear in both splits")
    return 0


def probe_frames(path: Path) -> int:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    written = probe.stdout.strip()
    return int(written) if written.isdigit() else -1


ENC_GROUPS = ("group3", "group1A", "group2A")

ENC_EXERCISES = ("CTK", "ELK", "RTK")

CANVAS = 448
CLIP_CRF = 18


GROUP3_RE = re.compile(
    r"^G3-Anon-(?P<exercise>[A-Z]{3})-(?P<subject>P\d+T\d+)-\w+-(?P<label>C|E\dB\d)-(?P<rec>\d+)\.\w+$")

PATIENT_RE = re.compile(
    r"^G(?P<group>1A|2A)-Anon-(?P<exercise>[A-Z]{3})-(?P<subject>[RS]\d+)-(?P<site>[A-Za-z]+)-(?P<rec>\d+)\.\w+$")


def out_name(source: Path) -> str | None:
    match = GROUP3_RE.match(source.name)
    if match:
        return (f"3-{match['exercise']}-{match['subject']}-"
                f"{match['label']}-{match['rec']}.mp4")
    match = PATIENT_RE.match(source.name)
    if match:
        return (f"{match['group']}-{match['exercise']}-{match['subject']}-"
                f"{match['site']}-{match['rec']}.mp4")
    return None


def plan_jobs(root: Path, out_dir: Path, args):
    jobs, skipped, stale = [], Counter(), []
    for group in args.groups:
        video_dir = root / group / "videos"
        if not video_dir.is_dir():
            sys.exit(f"videos not found: {video_dir}")
        for source in sorted(video_dir.iterdir()):
            if source.name.startswith(".") or not source.is_file():
                continue
            name = out_name(source)
            if name is None:
                skipped["unexpected_source_name"] += 1
                continue
            if args.exercises and name.split("-")[1] not in args.exercises:
                continue
            frames = probe_frames(source)
            if frames < 1:


                skipped["source_unreadable"] += 1
                continue
            out = out_dir / name
            if out.is_file() and not args.overwrite:
                if not args.verify or probe_frames(out) == frames:
                    skipped["exists"] += 1
                    continue
                stale.append(name)
            jobs.append({"source": source, "out": out, "frames": frames})
    return jobs, skipped, stale


def encode(job: dict):
    scale = (f"scale={CANVAS}:{CANVAS}:force_original_aspect_ratio=decrease,"
             f"pad={CANVAS}:{CANVAS}:(ow-iw)/2:(oh-ih)/2")
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(job["source"]), "-vf", scale, "-c:v", "libx264",
               "-crf", str(CLIP_CRF), "-pix_fmt", "yuv420p", "-an", str(job["out"])]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return job, (result.stderr.strip().splitlines() or ["ffmpeg failed"])[-1]
    written = probe_frames(job["out"])
    if written != job["frames"]:
        return job, f"wrote {written} frames, source has {job['frames']}"
    return job, None


def encode_parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="KERAAL root (default: %(default)s)")
    parser.add_argument("--video-dir", default="videos",
                        help="output directory under processed/ (default: %(default)s)")
    parser.add_argument("--groups", default=",".join(ENC_GROUPS),
                        help=f"comma-separated subset of {','.join(ENC_GROUPS)} "
                             "(default: %(default)s)")
    parser.add_argument("--exercises", default="all",
                        help=f"comma-separated subset of {','.join(ENC_EXERCISES)}, or all "
                             "(default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=16,
                        help="concurrent ffmpeg processes (default: %(default)s)")
    parser.add_argument("--verify", action="store_true",
                        help="probe outputs that already exist and re-encode any whose "
                             "frame count differs from the source")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-encode recordings that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be encoded")
    args = parser.parse_args(argv)

    args.groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())
    unknown = [g for g in args.groups if g not in ENC_GROUPS]
    if unknown:
        parser.error(f"unknown group(s) {unknown}, pick from {list(ENC_GROUPS)}")
    if args.exercises == "all":
        args.exercises = ()
    else:
        args.exercises = tuple(e.strip().upper() for e in args.exercises.split(",")
                               if e.strip())
        bad = [e for e in args.exercises if e not in ENC_EXERCISES]
        if bad:
            parser.error(f"unknown exercise(s) {bad}, pick from {list(ENC_EXERCISES)}")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    return args


def encode_videos(argv=None) -> int:
    args = encode_parse_args(argv)
    out_dir = args.dataset_root / "processed" / args.video_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs, skipped, stale = plan_jobs(args.dataset_root, out_dir, args)
    print(f"{len(jobs)} recordings -> {out_dir}")
    print(f"  groups {','.join(args.groups)}, "
          f"exercises {','.join(args.exercises) if args.exercises else 'all'}")
    print(f"  letterboxed to {CANVAS}x{CANVAS}, native fps kept, crf {CLIP_CRF}")
    if stale:
        print(f"  {len(stale)} existing outputs have the wrong frame count and get "
              f"re-encoded: {', '.join(sorted(stale)[:5])}"
              f"{' ...' if len(stale) > 5 else ''}")
    if skipped:
        print(f"  skipped {dict(sorted(skipped.items()))}")
    if args.dry_run or not jobs:
        return 0

    failures, done = [], 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for job, error in pool.map(encode, jobs):
            done += 1
            if error:
                failures.append((job["out"].name, error))
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} encoded, {len(failures)} failed", flush=True)

    for name, error in failures[:20]:
        print(f"  FAILED {name}: {error}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} recordings failed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
