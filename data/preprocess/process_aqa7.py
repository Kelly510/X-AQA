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
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from . import frame_budget
from .aqa7_rules import rubric
from .paths import workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION = "aqa7_historical_judge_and_panel_v1"

SPORTS = {
    1: ("diving", 370, 300, 70),
    2: ("gym_vault", 176, 120, 56),
    3: ("ski_big_air", 175, 120, 55),
    4: ("snowboard_big_air", 206, 140, 66),
    5: ("sync_diving_3m", 88, 60, 28),
    6: ("sync_diving_10m", 91, 63, 28),
}


def quantize(value: Decimal, increment: Decimal) -> Decimal:
    return (value / increment).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * increment


def read_vector(path: Path, key: str, count: int) -> np.ndarray:
    data = loadmat(path)[key]
    if data.ndim != 2 or 1 not in data.shape or data.size != count:
        raise ValueError(f"Unexpected {key} shape in {path}: {data.shape}")
    values = data.reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {path}")
    return values


def load_annotations(root: Path, source_annotations_root: Path) -> list[dict]:
    samples = []
    seen = set()
    for split, column, size in (("train", 2, 803), ("test", 3, 303)):
        relative = f"Split_4/split_4_{split}_list.mat"
        array = loadmat(root / relative)[f"consolidated_{split}_list"]
        if array.shape != (size, 3) or not np.isfinite(array).all():
            raise ValueError(f"Invalid annotation matrix: {relative}")
        if not np.equal(array[:, :2], np.floor(array[:, :2])).all():
            raise ValueError(f"Noninteger class/sample ID: {relative}")
        counts = Counter()
        for row_index, (class_id, index, score) in enumerate(array, 1):
            class_id, index = int(class_id), int(index)
            if class_id not in SPORTS or not 1 <= index <= SPORTS[class_id][1]:
                raise ValueError(f"Unknown class/sample: {class_id}/{index}")
            key = (class_id, index)
            if key in seen:
                raise ValueError(f"Duplicate or cross-split ID: {key}")
            seen.add(key)
            counts[class_id] += 1
            samples.append(dict(sport=SPORTS[class_id][0], class_id=class_id,
                                index=index, original=Decimal(str(float(score))), split=split,
                                annotation_file=relative, annotation_row_1based=row_index))
        if counts != Counter({c: cfg[column] for c, cfg in SPORTS.items()}):
            raise ValueError(f"Unexpected split counts: {counts}")
    expected = {(c, i) for c, cfg in SPORTS.items() for i in range(1, cfg[1] + 1)}
    if seen != expected:
        raise ValueError("Official six-sport split does not cover all IDs")


    for class_id, folder, prefix in ((1, "diving", "diving"), (2, "gymnastic_vault", "gym_vault")):
        count = SPORTS[class_id][1]
        base = source_annotations_root / folder
        score_file = base / f"{prefix}_overall_scores.mat"
        difficulty_file = base / f"{prefix}_difficulty_level.mat"
        scores = read_vector(score_file, "overall_scores", count)
        difficulties = read_vector(difficulty_file, "difficulty_level", count)
        for sample in (s for s in samples if s["class_id"] == class_id):
            i = sample["index"] - 1
            if sample["original"] != Decimal(str(float(scores[i]))):
                raise ValueError(f"AQA-7/UNLV score mismatch: {folder}/{i + 1}")
            difficulty = Decimal(str(float(difficulties[i])))
            if difficulty <= 0:
                raise ValueError(f"Nonpositive difficulty: {difficulty_file}/{i + 1}")
            sample.update(difficulty=difficulty, difficulty_annotation_file=str(difficulty_file.resolve()),
                          overlap_group=f"unlv_{folder}_{i + 1:03d}")

    for sample in samples:
        video = root / "Actions" / sample["sport"] / f"{sample['index']:03d}.avi"
        if not video.is_file():
            raise FileNotFoundError(video)
        if sample["original"] < 0:
            raise ValueError(f"Negative score: {video}")
        sample["video"] = str(video.resolve())
    return samples


def probe(sample: dict) -> dict:
    completed = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=duration,nb_frames,r_frame_rate", "-of", "json", sample["video"],
    ], check=True, capture_output=True, text=True, timeout=60)
    stream = json.loads(completed.stdout)["streams"][0]
    duration = float(stream["duration"])
    frame_count = int(stream["nb_frames"])
    expected = 103
    if not math.isfinite(duration) or duration <= 0 or frame_count != expected:
        raise ValueError(f"Unexpected normalized video: {sample['video']}: {stream}")
    return dict(duration_seconds=duration, source_frame_count=frame_count,
                source_frame_rate=stream["r_frame_rate"])


def build_record(sample: dict, fps: float | None) -> dict:
    sport, original = sample["sport"], sample["original"]
    prompt, question, provenance = rubric(sport, sample["index"])
    extra = {}
    if sport in ("diving", "gym_vault"):
        difficulty = sample["difficulty"]
        raw = original / (3 * difficulty) if sport == "diving" else original - difficulty
        step = Decimal("0.5" if sport == "diving" else "0.1")
        target = quantize(raw, step)
        if not 0 <= raw <= 10:
            raise ValueError(f"Invalid execution proxy: {sample['video']}: {raw}")
        rebuilt = 3 * difficulty * target if sport == "diving" else difficulty + target
        key = "execution_score" if sport == "gym_vault" else "execution_award"
        answer = json.dumps({key: float(target)}, ensure_ascii=False)
        target_type = "single_judge_award_proxy" if sport == "diving" else "single_judge_execution_proxy"
        extra = dict(difficulty=float(difficulty), difficulty_source="UNLV-Olympic sport_and_id_join_score_verified",
                     difficulty_annotation_file=sample["difficulty_annotation_file"],
                     cross_dataset_overlap_group=sample["overlap_group"],
                     target_increment=float(step), target_rounding="ROUND_HALF_UP", unrounded_target=float(raw),
                     quantization_error=float(target - raw), reconstructed_original_score=float(rebuilt),
                     original_score_reconstruction_error=float(rebuilt - original),
                     neutral_penalty_status="unseparated_in_source" if sport == "gym_vault" else "not_applicable",
                     label_source="overall_score / (3 * difficulty)" if sport == "diving" else "overall_score - difficulty")
        resolution = step
    else:
        target = original
        digits = 0 if sport in ("ski_big_air", "snowboard_big_air") else 2
        resolution = Decimal(10) ** -digits
        if target % resolution or (digits == 0 and target > 50):
            raise ValueError(f"Invalid released score resolution/range: {sample['video']}: {target}")
        field = ("overall_score" if sport in ("ski_big_air", "snowboard_big_air")
                 else "dive_total" if sport in ("sync_diving_3m", "sync_diving_10m")
                 else "panel_score")

        answer = f'{{{json.dumps(field)}: {target:.{digits}f}}}'
        target_type = "single_attempt_panel_score" if digits == 0 else "dive_panel_total"
        extra = dict(label_source="released_score", difficulty=None, individual_judge_increment=None,
                     target_increment=float(resolution) if digits == 0 else None, target_rounding="none",
                     missing_components=["individual_judge_awards"] if digits == 0 else
                     ["difficulty", "execution_awards", "synchronization_awards"],
                     neutral_penalty_status="unseparated_in_source")
    return {
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": "<video>\n" + question},
                     {"role": "assistant", "content": answer}],
        "videos": [sample["video"]],
        "chat_template_kwargs": {"fps": frame_budget.resolve_fps(sample["duration_seconds"], fps)},
        "dataset": "AQA-7", "id": f"aqa7_{sport}_{sample['index']:03d}_score",
        "sport": sport, "class_id": sample["class_id"], "clip": f"{sample['index']:03d}",
        "split": sample["split"], "official_split": "Split-4",
        "split_status": "official",
        "target": target_type, "score": float(target), "original_score": float(original),
        "label_is_proxy": sport in ("diving", "gym_vault"),
        "individual_judge_ground_truth_available": False,
        "target_resolution": float(resolution), "audio_in_model_input": False,
        "label_granularity": "whole_attempt",
        "duration_seconds": sample["duration_seconds"], "source_frame_count": sample["source_frame_count"],
        "source_frame_rate": sample["source_frame_rate"], "temporally_normalized": True,
        "annotation_file": sample["annotation_file"], "annotation_row_1based": sample["annotation_row_1based"],
        "rubric_version": VERSION, **extra, **provenance,
    }


def verify(rows: list[dict], samples: list[dict], fps: float | None) -> None:
    if len(rows) != len(samples) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Row count or uniqueness mismatch")
    for row, sample in zip(rows, samples):
        if row != build_record(sample, fps):
            raise ValueError(f"Export/source mismatch: {row['id']}")
        if row["messages"][1]["content"].count("<video>") != len(row["videos"]):
            raise ValueError("Video placeholder mismatch")
        answer = json.loads(row["messages"][-1]["content"])
        if len(answer) != 1 or Decimal(str(next(iter(answer.values())))) != Decimal(str(row["score"])):
            raise ValueError("Answer/score mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=workspace() / "datasets/AQA-7")
    parser.add_argument("--source-annotations-root", "--unlv-root", dest="source_annotations_root",
                        type=Path, help="Original diving/vault MAT annotations; defaults to DATASET_ROOT/source_annotations/unlv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "sports")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fps", type=float)
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    args = parser.parse_args()
    if args.workers < 1 or (args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0)):
        parser.error("workers and fps must be positive; fps must be finite")
    dataset_root = args.dataset_root.resolve()
    source_annotations_root = (args.source_annotations_root.resolve() if args.source_annotations_root
                               else dataset_root / "source_annotations" / "unlv")
    samples = load_annotations(dataset_root, source_annotations_root)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for sample, info in zip(samples, pool.map(probe, samples)):
            sample.update(info)
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["split"]].append(sample)
    exports = []
    for split, group in groups.items():
        rows = [build_record(s, args.fps) for s in group]
        verify(rows, group, args.fps)
        exports.append((f"aqa7_score_{split}.jsonl", rows, group))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows, group in exports:
        path = args.out_dir / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
        verify([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()], group, args.fps)
        print(f"{name}: {len(rows)} rows; {frame_budget.fps_histogram(rows)}")
    for sport, *_ in SPORTS.values():
        for split in ("train", "test"):
            (args.out_dir / f"aqa7_{sport}_score_{split}.jsonl").unlink(missing_ok=True)
    (args.out_dir / "aqa7_trampoline_score_unsplit.jsonl").unlink(missing_ok=True)
    print(f"Verified {len(samples)} rows: 803 train, 303 test. Trampoline excluded: no official split.")


if __name__ == "__main__":
    main()
