#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
from collections import Counter, defaultdict
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
import subprocess

from . import frame_budget

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = workspace() / "datasets" / "CricketVision"
FPS = 16.0
PARTS = (("2", "Head"), ("3", "Shoulder"), ("4", "Hands"),
         ("5", "Hips"), ("6", "Feet"))
PHASES = ("Buildup", "Execution", "FollowThrough")


DEFAULT_BOUNDARY_TOLERANCE = 0.12

SYSTEM_PROMPT = """You assess cricket batting technique. The video shows one batting action. Score the Execution phase, around the instant immediately before bat-ball contact; use the preparation and follow-through only as context.

Assess these five body parts independently:
1. Head: judge how well the head aligns with the front foot and the reference alignment through the head. Greater angular departure lowers quality.
2. Shoulder: judge the side-on stance and alignment of the two shoulders. Greater angular departure from the appropriate alignment lowers quality.
3. Hands: judge hand placement relative to the vertical line beneath the head and eyes. Greater displacement from the appropriate position lowers quality.
4. Hips: judge hip alignment and the side-on stance. Greater displacement from the appropriate alignment lowers quality.
5. Feet: judge alignment of the feet with the head. Greater angular departure lowers quality.

Apply the alignment appropriate to the stroke: on-drive and square-drive/cut can require different shoulder alignment; square-drive/cut can require different hand placement; cut and hook allow different hip orientation. Do not penalize these stroke-specific adjustments as ordinary alignment errors.

For head, shoulder and feet, the published guide uses nonzero angular deviations below 7 degrees as the higher scoring band (5-10) and deviations between 7 and 15 degrees as the lower band (0-5); deviations beyond 15 degrees represent severe misalignment. These are reference bands, not a specified formula for each integer score. Hand and hip placement are judged by relative positional deviation; smaller departures receive higher scores.

For this task, return only this JSON object and nothing else, without Markdown or code fences:
{"head": <integer 1-10>, "shoulder": <integer 1-10>, "hands": <integer 1-10>, "hips": <integer 1-10>, "feet": <integer 1-10>}"""


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def decode_scores(av: dict, attributes: dict) -> dict[str, int]:
    scores = {}
    for key, name in PARTS:
        value = attributes[key]["options"][av[key]]
        score = int(value)
        if str(score) != str(value) or not 1 <= score <= 10:
            raise ValueError(f"Invalid decoded {name} score: {value!r}")
        scores[name] = score
    return scores


def load_strokes(path: Path, root: Path, tolerance: float,
                 counts: Counter) -> list[dict]:
    data = read_json(path)

    views = data.get("view", {})
    segments = defaultdict(list)
    points = defaultdict(list)
    seen_segments = set()
    for annotation_id, annotation in data["metadata"].items():
        av = annotation.get("av", {})
        z = annotation.get("z", [])
        phase = av.get("1")
        view_id = str(annotation["vid"])
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool)
                   and math.isfinite(t) and t >= 0 for t in z):
            counts["invalid_timestamp"] += 1
            continue
        if len(z) == 2 and phase in PHASES:
            if z[1] <= z[0]:
                counts["invalid_interval"] += 1
                continue
            signature = (view_id, phase, *z)
            if signature in seen_segments:
                counts["duplicate_phase_interval"] += 1
                continue
            seen_segments.add(signature)
            segments[view_id].append({"id": annotation_id, "phase": phase,
                                      "start": z[0], "end": z[1]})
        elif len(z) == 1 and phase == "Execution":
            counts["execution_points"] += 1
            try:
                scores = decode_scores(av, data["attribute"])
            except (KeyError, ValueError, TypeError):
                counts["incomplete_or_invalid_execution_scores"] += 1
                continue
            points[view_id].append({"id": annotation_id, "time": z[0], "scores": scores})
        elif len(z) == 1 and phase not in PHASES and any(k in av for k, _ in PARTS):
            counts["unknown_phase_scoring_point"] += 1

    strokes = []
    for view_id, spans in segments.items():
        spans.sort(key=lambda s: (s["start"], s["end"], s["id"]))
        file_ids = views.get(view_id, {}).get("fid_list", [])
        if not file_ids and view_id in data["file"]:
            file_ids = [view_id]
        if len(file_ids) != 1 or str(file_ids[0]) not in data["file"]:
            raise ValueError(f"Expected single-file view: {path.name}, {view_id}")
        filename = Path(data["file"][str(file_ids[0])]["fname"]).name
        video = (root / "Videos" / filename).resolve()
        executions = [s for s in spans if s["phase"] == "Execution"]
        assigned = defaultdict(list)
        for point in points[view_id]:
            matches = [s for s in executions if s["start"] <= point["time"] <= s["end"]]
            if len(matches) != 1:
                counts["unmatched_or_ambiguous_execution_point"] += 1
                continue
            assigned[matches[0]["id"]].append(point)
        for i, execution in enumerate(spans):
            if execution["phase"] != "Execution":
                continue
            counts["execution_intervals"] += 1
            labels = assigned[execution["id"]]
            if not labels:
                counts["execution_interval_without_valid_scores"] += 1
                continue
            if len({tuple(p["scores"].values()) for p in labels}) != 1:
                counts["conflicting_execution_scores"] += 1
                continue
            if i == 0 or i + 1 == len(spans):
                counts["missing_adjacent_phases"] += 1
                continue
            buildup, follow = spans[i - 1], spans[i + 1]
            if buildup["phase"] != "Buildup" or follow["phase"] != "FollowThrough":
                counts["missing_adjacent_phases"] += 1
                continue
            if not (buildup["start"] < execution["start"] < follow["start"]
                    and buildup["end"] < execution["end"] < follow["end"]
                    and abs(execution["start"] - buildup["end"]) <= tolerance
                    and abs(follow["start"] - execution["end"]) <= tolerance):
                counts["invalid_phase_continuity"] += 1
                continue
            counts["identical_extra_score_points"] += len(labels) - 1
            strokes.append({
                "id": f"cricketvision_{path.stem}_{execution['id']}",
                "video": str(video), "source_video": filename,
                "annotation_file": str(path.resolve()),
                "annotation_ids": [p["id"] for p in labels],
                "score_timestamps": [p["time"] for p in labels],
                "phase_intervals": {s["phase"]: [s["start"], s["end"]]
                                    for s in (buildup, execution, follow)},
                "start": buildup["start"], "end": follow["end"],
                "scores": labels[0]["scores"],
            })
    counts["candidate_strokes"] += len(strokes)
    return strokes


def source_split(strokes: list[dict], ratio: float, seed: int) -> dict[str, str]:
    names = sorted({s["source_video"] for s in strokes})
    if ratio > 0 and len(names) < 2:
        raise ValueError("A nonempty test split requires at least two source videos; use --test-ratio 0")
    n_test = min(len(names) - 1, max(1, round(len(names) * ratio))) if ratio else 0
    held_out = set(random.Random(seed).sample(names, n_test))
    return {n: "test" if n in held_out else "train" for n in names}


def probe_video(path: str) -> dict:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration,avg_frame_rate,nb_frames:format=duration",
             "-of", "json", path], capture_output=True, text=True, timeout=30)
        if result.returncode:
            return {"ok": False, "error": result.stderr.strip()[:500]}
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        duration = float(stream.get("duration", data.get("format", {}).get("duration", 0)))
        numerator, denominator = stream["avg_frame_rate"].split("/")
        rate = float(numerator) / float(denominator)
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Expected positive finite duration and frame rate")
        return {"ok": True, "duration": duration, "source_fps": rate}
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError,
            ZeroDivisionError) as exc:
        return {"ok": False, "error": str(exc)}


def build_record(stroke: dict, split: str, media_status: str) -> dict:
    duration = stroke["end"] - stroke["start"]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\nAssess the Execution phase of this cricket batting action."},
            {"role": "assistant", "content": json.dumps(
                {name.lower(): stroke["scores"][name] for _, name in PARTS},
                ensure_ascii=False)},
        ],
        "videos": [stroke["video"]],
        "chat_template_kwargs": {
            "video_start": stroke["start"], "video_end": stroke["end"],
            "fps": FPS,
        },
        "id": stroke["id"], "split": split, "task": "execution_body_part_scoring",
        "source_video": stroke["source_video"], "video_scope": "single_stroke",
        "target_phase": "Execution", "duration_seconds": round(duration, 8),
        "scores": stroke["scores"], "phase_intervals": stroke["phase_intervals"],
        "annotation_file": stroke["annotation_file"],
        "annotation_ids": stroke["annotation_ids"],
        "score_timestamps": stroke["score_timestamps"],
        "media_status": media_status,
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--boundary-tolerance", type=float, default=DEFAULT_BOUNDARY_TOLERANCE)
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    args = parser.parse_args(argv)
    if not 0 <= args.test_ratio < 1:
        parser.error("--test-ratio must be in [0, 1)")
    if args.jobs < 1 or not math.isfinite(args.boundary_tolerance) or args.boundary_tolerance < 0:
        parser.error("--jobs must be positive and --boundary-tolerance finite and nonnegative")
    root = args.dataset_root.resolve()
    files = sorted((root / "JSON").glob("*.json"))
    if not files:
        parser.error(f"No annotation JSON files under {root / 'JSON'}")
    counts = Counter()
    strokes = [s for path in files for s in load_strokes(path, root, args.boundary_tolerance, counts)]
    if not strokes:
        parser.error("No valid stroke/Execution score pairs")
    if len({s["id"] for s in strokes}) != len(strokes):
        raise ValueError("Duplicate stroke IDs")


    splits = source_split(strokes, args.test_ratio, args.seed)
    output = args.output_dir or (REPO_ROOT / "data" / "exports" / "original" / "sports")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "custom source-video-disjoint split; not official or verified player-disjoint",
        "input": "one complete stroke; Buildup through FollowThrough",
        "target": "five Execution scores decoded using VIA attribute options",
        "annotations_only": args.annotations_only, "seed": args.seed,
        "test_ratio": args.test_ratio, "source_splits": splits,
        "annotation_files": len(files), "boundary_tolerance": args.boundary_tolerance,
        "fps": FPS, "counts": counts,
    }
    probes = {}
    if not args.annotations_only:
        paths = sorted({s["video"] for s in strokes})
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            probes = dict(zip(paths, pool.map(probe_video, paths)))
        manifest["media"] = probes
        counts["probed_sources"] = len(probes)
        counts["readable_sources"] = sum(p["ok"] for p in probes.values())
        selected = []
        for stroke in strokes:
            probe = probes[stroke["video"]]
            if not probe["ok"]:
                counts["strokes_with_unreadable_source"] += 1
            elif stroke["end"] > probe["duration"]:
                counts["stroke_exceeds_video_duration"] += 1
            elif (stroke["end"] - stroke["start"]) * probe["source_fps"] < frame_budget.FRAME_FLOOR:
                counts["stroke_with_too_few_source_frames"] += 1
            else:
                selected.append(stroke)
        strokes = selected
    records = [build_record(s, splits[s["source_video"]],
                            "unverified" if args.annotations_only else "probe_passed") for s in strokes]
    counts["output_records"] = len(records)
    counts["train_records"] = sum(r["split"] == "train" for r in records)
    counts["test_records"] = sum(r["split"] == "test" for r in records)
    manifest["status"] = ("unverified_candidates" if args.annotations_only else
                          "no_usable_media" if not records else "exported")
    suffix = "candidates_manifest" if args.annotations_only else "manifest"
    write_json(output / f"cricketvision_score_{suffix}.json", manifest)
    print(json.dumps(dict(sorted(counts.items())), indent=2))
    if not records:
        print(f"No usable media; wrote audit to {output}. No training JSONL was written.")
        return 1
    if args.annotations_only:
        write_jsonl(output / "cricketvision_score_candidates.jsonl", records)
        write_json(output / "cricketvision_score_candidate_example.json", records[0])
        print(f"Wrote UNVERIFIED candidates to {output}; do not use for training.")
    else:
        for split in ("train", "test"):
            write_jsonl(output / f"cricketvision_score_{split}.jsonl",
                        [r for r in records if r["split"] == split])
        write_json(output / "cricketvision_score_example.json", records[0])
        print(f"Wrote scoring QA to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
