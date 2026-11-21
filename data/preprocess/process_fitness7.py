#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace
import random
import re
import subprocess

from . import frame_budget

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = "https://data.mendeley.com/datasets/kgbb3yn47p/3"
TEST_SUBJECTS = frozenset({"001", "004", "009", "021", "024"})
EXERCISES = {
    "abs": "Standing dumbbell side bend",
    "back": "Bent-over dumbbell row",
    "bicep_curl": "Alternating dumbbell bicep curl",
    "push_up": "Standard push-up",
    "shoulder": "Standing dumbbell shoulder press",
    "squat": "Bodyweight squat",
    "tricep": "Overhead dumbbell triceps extension",
}
VIEWS = {"front", "side", "diagonal"}

FILENAME_EXCEPTIONS = {
    "bicep_curl/good/side/subject_003_bicep_good_side.mp4":
        "subject_003_bicep_curl_good_side.mp4",
}


def system_prompt(exercise: str) -> str:
    return (
        "Assess the exercise form in the video.\n"
        f"Exercise: {EXERCISES[exercise]}.\n"
        "Determine whether the demonstrated exercise form is good or bad.\n"
        'Return only this JSON object and nothing else: '
        '{"exercise_form": "good" or "bad"}.'
    )


def parse_video(path: Path, root: Path) -> dict:
    parts = path.relative_to(root).parts
    if len(parts) != 4:
        raise ValueError(f"Unexpected layout: {path}")
    exercise, quality, view, filename = parts
    filename = FILENAME_EXCEPTIONS.get(path.relative_to(root).as_posix(), filename)
    if exercise not in EXERCISES or quality not in {"good", "bad"} or view not in VIEWS:
        raise ValueError(f"Unknown exercise, quality or view: {path}")
    match = re.fullmatch(
        r"subject_(\d{3})_" + re.escape(exercise) + r"_(good|bad)_(front|side|diagonal)\.mp4",
        filename,
    )
    if not match or match[2] != quality or match[3] != view:
        raise ValueError(f"Filename/directory label or view disagreement: {path}")
    subject = match[1]
    if not 1 <= int(subject) <= 26:
        raise ValueError(f"Unknown subject: {path}")
    return {
        "sample_id": f"fitness7_subject_{subject}_{exercise}_{quality}_{view}",
        "subject_id": subject,
        "exercise": exercise,
        "exercise_name": EXERCISES[exercise],
        "gold_verdict": quality,
        "view": view,
        "split": "test" if subject in TEST_SUBJECTS else "train",
        "source_relative_path": path.relative_to(root).as_posix(),
        "path": str(path.resolve()),
    }


def inventory(root: Path) -> list[dict]:
    actions = {p.name for p in root.iterdir() if p.is_dir()}
    if actions != set(EXERCISES):
        raise ValueError(f"Unexpected exercise directories: {actions ^ set(EXERCISES)}")
    samples = [parse_video(p, root) for p in sorted(root.rglob("*.mp4"))]
    subjects = sorted({s["subject_id"] for s in samples})
    if subjects != [f"{i:03}" for i in range(1, 27)]:
        raise ValueError("Expected all 26 released subject IDs")
    if set(random.Random(42).sample(subjects, 5)) != TEST_SUBJECTS:
        raise ValueError("Fixed subject split differs from seed 42")
    if len(samples) != 1042 or len({s["sample_id"] for s in samples}) != 1042:
        raise ValueError("Expected exactly 1,042 unique released samples")
    if len({s["path"] for s in samples}) != 1042:
        raise ValueError("Duplicate resolved media paths")
    return samples


def probe_video(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
         "-of", "json", path], capture_output=True, text=True,
    )
    try:
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        has_video = any(s["codec_type"] == "video" for s in info["streams"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid media probe: {path}: {result.stderr.strip()}") from exc
    if result.returncode or not has_video or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Unreadable video or invalid duration: {path}")
    return duration


def build_record(sample: dict, duration: float) -> dict:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid duration for {sample['sample_id']}")
    return {
        "messages": [
            {"role": "system", "content": system_prompt(sample["exercise"])},
            {"role": "user", "content": "<video>"},
            {"role": "assistant", "content": json.dumps(
                {"exercise_form": sample["gold_verdict"]}, ensure_ascii=False)},
        ],
        "videos": [sample["path"]],
        "chat_template_kwargs": {"fps": frame_budget.sampling_fps(duration),
                                 "use_audio_in_video": False},
        **{key: value for key, value in sample.items() if key != "path"},
        "duration": duration,
        "label_granularity": "whole_video",
        "audio_in_model_input": False,
        "prompt_version": "fitness7_task_only_v1",
        "source_url": SOURCE,
    }


def validate(rows: list[dict], views: set) -> None:
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    if len({r["videos"][0] for r in rows}) != len(rows):
        raise ValueError("Duplicate video paths")
    if {r["view"] for r in rows} != views:
        raise ValueError("Exported views differ from the requested ones")
    counts = Counter((r["split"], r["gold_verdict"]) for r in rows)


    if (counts[("train", "good")] != counts[("train", "bad")]
            or counts[("test", "good")] != counts[("test", "bad")]
            or not counts[("test", "good")]):
        raise ValueError("Unexpected split/label distribution")
    for split in ("train", "test"):
        if {r["exercise"] for r in rows if r["split"] == split} != set(EXERCISES):
            raise ValueError(f"Missing exercise in {split}")
    for row in rows:
        expected_split = "test" if row["subject_id"] in TEST_SUBJECTS else "train"
        if row["split"] != expected_split:
            raise ValueError("Subject assigned to the wrong split")
        sample = parse_video(Path(row["source_relative_path"]), Path("."))
        for key in ("sample_id", "subject_id", "exercise", "exercise_name", "gold_verdict", "view"):
            if row[key] != sample[key]:
                raise ValueError(f"Source metadata mismatch: {key}")
        if row["messages"] != [
            {"role": "system", "content": system_prompt(row["exercise"])},
            {"role": "user", "content": "<video>"},
            {"role": "assistant", "content": json.dumps(
                {"exercise_form": row["gold_verdict"]}, ensure_ascii=False)},
        ]:
            raise ValueError("Unexpected prompt/target")
        duration = row["duration"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid duration")
        if row["chat_template_kwargs"] != {
            "fps": frame_budget.sampling_fps(duration), "use_audio_in_video": False,
        }:
            raise ValueError("Unexpected sampling/audio configuration")
        frames = frame_budget.frames_for(duration, row["chat_template_kwargs"]["fps"])
        if not frame_budget.FRAME_FLOOR <= frames <= frame_budget.FRAME_CAP:
            raise ValueError("Frame budget exceeded")


def main() -> None:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path,
                        default=workspace() / "datasets" / "Fitness-7")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "fitness")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--views", default="front",
                        help="camera angles to export, comma-separated or all; the release "
                             "films every execution from front / side / diagonal and one "
                             "angle per execution is enough (default: %(default)s)")
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    views = VIEWS if args.views == "all" else {v.strip() for v in args.views.split(",") if v.strip()}
    if views - VIEWS:
        parser.error(f"unknown view(s) {sorted(views - VIEWS)}, pick from {sorted(VIEWS)}")
    samples = [s for s in inventory(args.dataset_root) if s["view"] in views]
    print(f"views: {','.join(sorted(views))} -> {len(samples)} of 1,042 released clips")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        durations = list(pool.map(probe_video, [s["path"] for s in samples]))
    rows = [build_record(s, d) for s, d in zip(samples, durations)]
    validate(rows, views)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        selected = [r for r in rows if r["split"] == split]
        path = args.out_dir / f"fitness7_{split}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected),
                             encoding="utf-8")
        restored = [json.loads(line) for line in temporary.read_text(encoding="utf-8").splitlines()]
        if restored != selected:
            raise ValueError(f"Serialization mismatch: {path}")
        temporary.replace(path)
        print(f"{path}: {len(selected)} videos; {dict(Counter(r['gold_verdict'] for r in selected))}")
    print(f"Test subjects (seed 42): {', '.join(sorted(TEST_SUBJECTS))}")
    print(f"Sampling: {frame_budget.fps_histogram(rows)}; no audio input")


if __name__ == "__main__":
    main()
