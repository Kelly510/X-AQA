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
import hashlib
import json
import math
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace
from statistics import median
import subprocess

from . import frame_budget

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = workspace() / "datasets" / "T3Set"
MERGED_SUMMARIES = workspace() / "data" / "annotations" / "t3set_merged_summaries.jsonl"
SPLITS = ("train", "val", "test")
TECHNIQUES = {
    "attack": "forehand drive",
    "topspin": "forehand topspin",
    "pendulum": "forehand pendulum serve",
    "block": "forehand block",
    "flick": "forehand flick",
    "push": "forehand push",
    "short": "forehand short touch",
}
VIDEO_SUFFIXES = {".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v"}
SYSTEM_PROMPT = (
    "You are a table tennis coach. Assess the player's execution from the video "
    "and give specific, actionable feedback. Address the player directly in the "
    'second person. Respond in a few concise English sentences within a single paragraph.'
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_splits(root: Path) -> tuple[dict[str, str], dict]:
    assignments, provenance = {}, {}
    for split in SPLITS:
        path = root / f"{split}.json"
        data = load_json(path)
        if data["split"] != split or data["num_rounds"] != len(data["rounds"]):
            raise ValueError(f"Invalid split manifest: {path}")
        provenance[split] = {
            "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "generated_from": data.get("generated_from"),
        }
        for row in data["rounds"]:
            key = f"{row['player_id']}/{row['tech_name']}/{row['round_id']}"
            if row["path"] != key or key in assignments:
                raise ValueError(f"Invalid or overlapping round assignment: {key}")
            assignments[key] = split
    return assignments, provenance


def load_annotations(root: Path, assignments: dict[str, str]) -> list[dict]:
    path = MERGED_SUMMARIES
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, sources, rounds = set(), set(), set()
    for row in rows:
        key = f"{row['player_id']}/{row['technique']}/{row['round_id']}"
        expected_source = f"{key}/coach_suggestions/summary_suggestions/{row['coach_id']}.json"
        if row["technique"] not in TECHNIQUES or row["coach_id"] not in {"coach_comment_1", "coach_comment_2"}:
            raise ValueError(f"Unknown technique or coach: {row['id']}")
        if row["id"] != f"{key}/{row['coach_id']}" or row["id"] in seen:
            raise ValueError(f"Invalid or duplicate annotation ID: {row['id']}")
        if assignments.get(key) != row["split"] or row["source_file"] != expected_source:
            raise ValueError(f"Annotation provenance mismatch: {row['id']}")
        if hashlib.sha256((root / expected_source).read_bytes()).hexdigest() != row["source_sha256"]:
            raise ValueError(f"Original annotation has changed: {expected_source}")
        if row["status"] not in {"merged", "needs_review", "empty"}:
            raise ValueError(f"Unknown status: {row['id']}")
        text = row["comment_en"]
        if not isinstance(text, str) or text != text.strip() or "\n" in text:
            raise ValueError(f"Invalid paragraph: {row['id']}")
        if bool(text) != (row["status"] != "empty"):
            raise ValueError(f"Status/text mismatch: {row['id']}")
        seen.add(row["id"])
        sources.add(expected_source)
        rounds.add(key)
    actual_sources = {
        str(path.relative_to(root)) for path in root.glob(
            "anonymous_*/*/round_*/coach_suggestions/summary_suggestions/coach_comment_*.json")
    }
    if not rows or sources != actual_sources or rounds != set(assignments):
        raise ValueError("Merged annotation coverage differs from the source files or split manifests")
    return sorted(rows, key=lambda row: row["id"])


def select_video(root: Path, round_key: str) -> tuple[Path | None, str | None]:
    candidates = {"remote": [], "right": []}
    for path in sorted((root / round_key / "video").glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        view = path.parent.name.lower()
        if view == "romote":
            view = "remote"
        if view not in candidates:
            raise ValueError(f"Unknown video view: {path}")
        candidates[view].append(path)
    for view in ("remote", "right"):
        if len(candidates[view]) > 1:
            raise ValueError(f"Ambiguous videos for {round_key}/{view}: {candidates[view]}")
        if candidates[view]:
            return candidates[view][0].resolve(), view
    return None, None


def probe_video(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration:stream_tags=DURATION:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    info = json.loads(result.stdout)
    if not info.get("streams"):
        raise ValueError(f"No video stream: {path}")
    stream = info["streams"][0]
    values = [stream.get("duration")]
    tag = {k.upper(): v for k, v in stream.get("tags", {}).items()}.get("DURATION")
    if tag:
        hours, minutes, seconds = map(float, tag.split(":"))
        values.append(hours * 3600 + minutes * 60 + seconds)
    values.append(info.get("format", {}).get("duration"))
    for value in values:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            return duration
    raise ValueError(f"No positive video duration: {path}")


def build_record(annotation: dict, video: Path, view: str, duration: float) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n"
             f"Technique: {TECHNIQUES[annotation['technique']]} ({annotation['technique']}).\n"
             "Assess the player's execution across this practice round and provide coaching feedback."},
            {"role": "assistant", "content": annotation["comment_en"]},
        ],
        "videos": [str(video)],
        "chat_template_kwargs": {"fps": frame_budget.sampling_fps(duration)},
        "id": f"t3set/{annotation['id']}",
        "source": "t3set_comment",
        "split": annotation["split"],
        "player_id": annotation["player_id"],
        "technique": annotation["technique"],
        "round_id": annotation["round_id"],
        "coach_id": annotation["coach_id"],
        "source_id": annotation["id"],
        "source_file": annotation["source_file"],
        "source_sha256": annotation["source_sha256"],
        "annotation_status": annotation["status"],
        "review_notes": annotation["review_notes"],
        "video_view": view,
        "video_scope": "full_round",
        "duration_seconds": duration,
    }


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "sports")
    parser.add_argument("--include-needs-review", action="store_true",
                        help="Explicitly include nonempty annotations with unresolved review notes")
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    args = parser.parse_args()
    root, output = args.dataset_root.resolve(), args.output_dir.resolve()
    assignments, provenance = load_splits(root)
    annotations = load_annotations(root, assignments)
    selected = {key: select_video(root, key) for key in sorted(assignments)}
    paths = sorted({video for video, _ in selected.values() if video is not None})
    with ThreadPoolExecutor(max_workers=8) as pool:
        durations = dict(zip(paths, pool.map(probe_video, paths)))
    records = {split: [] for split in SPLITS}
    excluded = []
    for annotation in annotations:
        key = f"{annotation['player_id']}/{annotation['technique']}/{annotation['round_id']}"
        reasons = []
        if annotation["status"] == "empty":
            reasons.append("empty_comment")
        elif annotation["status"] == "needs_review" and not args.include_needs_review:
            reasons.append("needs_review")
        video, view = selected[key]
        if video is None:
            reasons.append("missing_video")
        if reasons:
            excluded.append({"source_id": annotation["id"], "split": annotation["split"],
                             "reasons": reasons, "review_notes": annotation["review_notes"]})
            continue
        records[annotation["split"]].append(build_record(annotation, video, view, durations[video]))
    report = {
        "input_annotations": len(annotations),
        "dataset_root": str(root),
        "include_needs_review": args.include_needs_review,
        "video_policy": "Full round; remote preferred, right fallback; video only",
        "split_provenance": provenance,
        "splits": {}, "excluded": excluded,
        "excluded_reason_counts": dict(Counter(reason for row in excluded for reason in row["reasons"])),
    }
    for split, rows in records.items():
        unique = {row["videos"][0]: row["duration_seconds"] for row in rows}
        ds = list(unique.values())
        report["splits"][split] = {
            "qa_rows": len(rows), "videos": len(unique),
            "coaches": dict(Counter(row["coach_id"] for row in rows)),
            "views_by_qa": dict(Counter(row["video_view"] for row in rows)),
            "duration_seconds_by_video": {
                "min": min(ds), "median": median(ds), "max": max(ds),
            } if ds else {},
        }
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        (output / f"t3set_comment_{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"splits": report["splits"], "excluded": report["excluded_reason_counts"]}, indent=2))


if __name__ == "__main__":
    main()
