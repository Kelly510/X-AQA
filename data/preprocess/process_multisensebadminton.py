#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace
import re

from . import frame_budget

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = workspace() / "datasets" / "MultiSenseBadminton"
MERGED_FEEDBACK = workspace() / "data" / "annotations" / "multisensebadminton_feedback.jsonl"
SYSTEM_PROMPT = (
    "You are a badminton coach assessing a player's overall skill in the stroke "
    "specified by the user. Base the assessment on the performance visible in the video.\n\n"
    "Assessment criteria:\n"
    "Evaluate movement technique: grip, posture, swing mechanics, coordination "
    "of the torso, arm and wrist, footwork, and contact point. Evaluate shot "
    "quality: accuracy, distance, direction, speed, and consistency across the "
    "visible attempts. Use only criteria that the camera view allows you to "
    "assess; an obscured detail is not evidence of an execution error.\n\n"
    "Classify the player's skill level as Beginner, Intermediate, or Expert.\n\n"
    "Briefly explain the observed strengths and weaknesses and give relevant "
    "execution advice. Describe the player's performance objectively, without "
    "addressing the player in the second person.\n\n"
    "Output format:\n"
    "Return only this JSON object and nothing else, without Markdown or code fences:\n"
    '{"skill_level": "<skill level>", "feedback": "<coaching feedback>"}\n'
    'Use exactly these two fields. The skill_level must be "Beginner", "Intermediate", '
    'or "Expert". The feedback must be a string.'
)
SKILL_LEVELS = ("Beginner", "Intermediate", "Expert")
STROKES = {"Forehand Clear": "forehand clear", "Backhand Driving": "backhand drive"}


LTO_TEST_SUBJECTS = (
    ("Sub00", "Sub09", "Sub11"),
    ("Sub01", "Sub15", "Sub14"),
    ("Sub02", "Sub18", "Sub19"),
    ("Sub03", "Sub21", "Sub20"),
    ("Sub04", "Sub22", "Sub24"),
    ("Sub04", "Sub09", "Sub11"),
    ("Sub07", "Sub15", "Sub14"),
    ("Sub17", "Sub18", "Sub19"),
    ("Sub23", "Sub21", "Sub20"),
    ("Sub00", "Sub22", "Sub24"),
)


def load_annotations(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    seen = set()
    for row in rows:
        key = (row["subject"], row["stroke_type"])
        if key in seen:
            raise ValueError(f"Duplicate annotation: {key}")
        seen.add(key)
        if not re.fullmatch(r"Sub\d{2}", row["subject"]):
            raise ValueError(f"Invalid subject: {row['subject']}")
        if row["stroke_type"] not in STROKES or row["annotation_scope"] != "subject_stroke":
            raise ValueError(f"Unsupported annotation scope/stroke: {key}")
        if row.get("original_skill_level") not in SKILL_LEVELS:
            raise ValueError(f"Missing or invalid original skill level: {key}")
        if not isinstance(row.get("feedback"), str) or not row["feedback"].strip():
            raise ValueError(f"Missing or empty feedback: {key}")
    if not rows:
        raise ValueError("No annotations")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate record IDs")
    for subject in {r["subject"] for r in rows}:
        if {stroke for sub, stroke in seen if sub == subject} != set(STROKES):
            raise ValueError(f"Incomplete stroke coverage: {subject}")
    return sorted(rows, key=lambda r: r["id"])


def front_video(root: Path, row: dict) -> Path:
    names = (["Forehand Clear Front Video.mov"] if row["stroke_type"] == "Forehand Clear"
             else ["Backhand Drive Front Video.mov", "Backhand Driving Front Video.mov"])
    matches = [root / row["subject"] / name for name in names
               if (root / row["subject"] / name).is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one front video for {row['id']}, found {matches}")
    return matches[0].resolve()


def build_record(annotation: dict, video: Path, duration: float,
                 start: float, end: float, window_index: int,
                 fps: float | None = None) -> dict:
    level = annotation["original_skill_level"]
    if level not in SKILL_LEVELS:
        raise ValueError(f"Invalid original skill level: {level}")
    provenance = {k: v for k, v in annotation.items()
                  if k not in ("score", "skill_level", "feedback", "score_aggregation")}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\nAssess the player's overall "
             f"{STROKES[annotation['stroke_type']]} skill level and provide coaching feedback."},
            {"role": "assistant", "content": json.dumps(
                {"skill_level": level, "feedback": annotation["feedback"]}, ensure_ascii=False)},
        ],
        "videos": [str(video)],
        "chat_template_kwargs": {
            "video_start": start, "video_end": end,
            "fps": frame_budget.resolve_fps(end - start, fps)},
        **provenance,
        "id": f"{annotation['id']}_window{window_index:04d}",
        "source_id": annotation["id"], "window_index": window_index,
        "skill_level": level,
        "target_definition": "original_skill_level_v1",
        "output_format": "skill_level_feedback_v1",
        "source_median_score": annotation.get("score"),
        "feedback": annotation["feedback"],
        "duration_seconds": end - start, "source_duration_seconds": duration,
        "video_view": "front", "video_scope": "window",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--annotations", type=Path, help="Default: data/annotations/multisensebadminton_feedback.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "sports")
    parser.add_argument("--fold", type=int, choices=range(1, 11), default=1,
                        help="Paper LTO iteration (1–10); default: 1")
    parser.add_argument("--fps", type=float, help="Override the repository's duration-based sampling FPS")
    parser.add_argument("--window-sec", type=float, default=32.0,
                        help="Window length and stride in seconds; final window ends at video end")
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    args = parser.parse_args()
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive")
    if not math.isfinite(args.window_sec) or args.window_sec <= 0:
        parser.error("--window-sec must be finite and positive")
    root = args.dataset_root.resolve()
    annotations_path = args.annotations or MERGED_FEEDBACK
    annotations = load_annotations(annotations_path)
    subjects = {r["subject"] for r in annotations}
    if subjects != {f"Sub{i:02d}" for i in range(25)}:
        raise ValueError("Paper LTO requires all 25 subjects, Sub00 through Sub24")
    videos = [front_video(root, r) for r in annotations]
    with ThreadPoolExecutor(max_workers=8) as pool:
        durations = list(pool.map(frame_budget.probe_duration, videos))
    records = []
    for annotation, video, duration in zip(annotations, videos, durations):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Could not probe a positive video duration: {video}")
        for index, (start, end) in enumerate(frame_budget.windows(duration, args.window_sec)):
            records.append(build_record(annotation, video, duration, start, end, index, args.fps))
    test_subjects = set(LTO_TEST_SUBJECTS[args.fold - 1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        selected = [r for r in records
                    if (r["subject"] in test_subjects) == (split == "test")]
        path = args.output_dir / f"multisensebadminton_qa_{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected),
                        encoding="utf-8")
        print(f"LTO {args.fold}: {path}: {len(selected)} records")


if __name__ == "__main__":
    main()
