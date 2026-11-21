#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
import csv
import json
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "REHAB24-6"


SOURCE_FPS = 30.0


DEFAULT_FPS = frame_budget.sampling_fps(8.0)

DEFAULT_TEST_PERSONS = (2, 3, 6)


EXERCISES = {
    1: ("Arm abduction", "sideway raising of the straightened right arm"),
    2: ("Arm VW", "fluent transition of arms between V (arms straight up) and W "
                  "(elbows down, hands up) shape"),
    3: ("Push-ups", "push-ups with hands on a table"),
    4: ("Leg abduction", "sideway raising of the straightened leg"),
    5: ("Leg lunge", "pushing a knee of the back leg down while keeping a right angle "
                     "on the front knee"),
    6: ("Squats", "squats"),
}


CRITERIA = {
    1: ["The arm stays straight at the elbow through the whole repetition.",
        "The arm travels sideways in the frontal plane, not forward across the body.",
        "The arm is raised at least to shoulder level.",
        "The trunk stays upright, without leaning to the opposite side or shrugging the "
        "shoulder.",
        "The arm is lowered back to the side under control."],
    2: ["In the V position both arms are straight and raised overhead.",
        "In the W position the elbows come down to about shoulder level while the hands "
        "stay up.",
        "The transition between the two shapes is fluent and continuous, without pauses "
        "or jerks.",
        "Both arms move symmetrically.",
        "The trunk stays upright and the lower back does not arch."],
    3: ["The hands rest on the table about shoulder-width apart.",
        "The body holds a straight line from head to heels, with the hips neither "
        "sagging nor piked.",
        "The elbows bend until the chest comes close to the table.",
        "The arms extend fully at the top.",
        "Both arms bend and extend at the same rate."],
    4: ["The moving leg stays straight at the knee.",
        "The leg travels sideways in the frontal plane, not forward or backward.",
        "The trunk stays upright, without leaning away from the moving leg.",
        "The hips stay level and square to the front, and the leg does not rotate "
        "outward.",
        "The leg is lowered back under control."],
    5: ["The front knee holds close to a right angle and stays above the ankle, not "
        "travelling past the toes.",
        "The back knee is pushed down toward the floor.",
        "The trunk stays upright over the hips.",
        "The feet stay in line with the hips square to the front, and balance is held.",
        "The starting position is recovered under control."],
    6: ["The feet stay flat with the heels down.",
        "The knees track over the feet, neither collapsing inward nor flaring out.",
        "The hips descend at least until the thighs are parallel to the floor.",
        "The back stays straight and the chest up, without rounding forward.",
        "Descent and ascent are symmetric and controlled."],
}

SYSTEM_TEMPLATE = (
    "You judge how well one repetition of a rehabilitation exercise was performed, as a "
    "physiotherapist would.\n\n"
    "Exercise: {exercise}\n{side}\n"
    "A correct repetition meets every one of these criteria:\n"
    "{criteria}\n\n"
    'Return only this JSON object and nothing else: '
    '{{"rehabilitation_form": "correct" or "incorrect"}}.')


USER_PROMPT = "<video>"


VERDICTS = {"1": "correct", "0": "incorrect"}


CAMERA_FILES = {
    "17": "{video_id}-Camera17-30fps.mp4",
    "18": "{video_id}-Camera18-30fps-transposed.mp4",
}


VIEWS = {
    "front": ("front", "side"),
    "half-profile": ("half-profile", "half-profile"),
    "profile": ("side", "front"),
}

EXTRA_PERSON = {
    "0": "none", "1": "negligible", "2": "noticeable", "3": "large",
}


def read_segmentation(path: Path) -> list:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    if not rows:
        sys.exit(f"no repetitions read from {path}")
    return rows


def probe(path: Path) -> tuple:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames,duration", "-of", "default=nw=1", str(path)],
        capture_output=True, text=True)
    fields = dict(line.split("=", 1) for line in result.stdout.strip().splitlines()
                  if "=" in line)
    try:
        return int(fields["nb_frames"]), float(fields["duration"])
    except (KeyError, ValueError):
        return 0, 0.0


def build_system(exercise_id: int, subtype: str) -> str:
    name, description = EXERCISES[exercise_id]
    criteria = "\n".join(f"{i}. {c}"
                         for i, c in enumerate(CRITERIA[exercise_id], 1))
    return SYSTEM_TEMPLATE.format(
        exercise=f"{name} ({description})",
        side=f"Performed with: {subtype}\n" if subtype else "",
        criteria=criteria)


def build_rows(repetitions: list, cameras: tuple, fps: float | None, stats: Counter) -> list:
    rows = []
    for repetition in repetitions:
        exercise_id = int(repetition["exercise_id"])
        orientation = repetition["cam17_orientation"]
        if orientation not in VIEWS:
            stats[f"unknown_orientation:{orientation}"] += 1
            continue
        first, last = int(repetition["first_frame"]), int(repetition["last_frame"])
        start = (first - 1) / SOURCE_FPS

        for camera in cameras:
            media = repetition["media"][camera]
            if media["nb_frames"] == 0:
                stats["video_unreadable"] += 1
                continue


            if last > media["nb_frames"]:
                stats["last_frame_clamped_to_video_end"] += 1
            end = min(last, media["nb_frames"]) / SOURCE_FPS
            row_fps = frame_budget.resolve_fps(end - start, fps)
            if end - start < 1.0 / row_fps:
                stats["window_shorter_than_one_frame"] += 1
                continue
            view = VIEWS[orientation][0 if camera == "17" else 1]
            sample_id = (f"{repetition['video_id']}-Camera{camera}-"
                         f"r{int(repetition['repetition_number']):02d}")
            extra_key = f"extra_person_in_cam{camera}"
            rows.append({
                "messages": [
                    {"role": "system",
                     "content": build_system(exercise_id,
                                             repetition["exercise_subtype"])},
                    {"role": "user", "content": USER_PROMPT},
                    {"role": "assistant",
                     "content": json.dumps(
                         {"rehabilitation_form": VERDICTS[repetition["correctness"]]},
                         ensure_ascii=False)},
                ],
                "videos": [media["path"]],
                "chat_template_kwargs": {"video_start": round(start, 3),
                                         "video_end": round(end, 3), "fps": row_fps},


                "sample_id": sample_id,
                "video_id": repetition["video_id"],
                "camera": camera,
                "exercise_id": exercise_id,
                "exercise_name": EXERCISES[exercise_id][0],
                "exercise_subtype": repetition["exercise_subtype"],
                "person_id": int(repetition["person_id"]),
                "repetition_number": int(repetition["repetition_number"]),
                "first_frame": first,
                "last_frame": last,
                "duration": round(end - start, 3),
                "view": view,
                "gold_verdict": VERDICTS[repetition["correctness"]],
                "lights_on": repetition["lights_on"] == "1",
                "extra_person": EXTRA_PERSON.get(repetition[extra_key], "unknown"),
                "mocap_erroneous": repetition["mocap_erroneous"] == "1",
            })
    return rows


def summarise(split: str, rows: list) -> None:
    durations = sorted(r["duration"] for r in rows)
    correct = sum(1 for r in rows if r["gold_verdict"] == "correct")
    print(f"  {split}: {len(rows)} rows, {correct} correct / {len(rows) - correct} "
          f"incorrect (correct rate {correct / len(rows):.3f})")
    print(f"    repetition length: min {durations[0]:.2f}s median "
          f"{durations[len(durations) // 2]:.2f}s p90 "
          f"{durations[int(len(durations) * 0.9)]:.2f}s max {durations[-1]:.2f}s")
    print(f"    sampling: {frame_budget.fps_histogram(rows)}")
    over = sum(1 for d in durations if d > 8.0)
    print(f"    longer than 8 s (frame-capped at 128 by train/sft_lora_full.sh): {over} "
          f"({over / len(rows):.1%})")
    for exercise_id in sorted(EXERCISES):
        group = [r for r in rows if r["exercise_id"] == exercise_id]
        if not group:
            continue
        counts = Counter(r["gold_verdict"] for r in group)
        majority = max(counts.values()) / len(group)
        print(f"    Ex{exercise_id} ({EXERCISES[exercise_id][0]}): {len(group)} rows "
              f"{{correct: {counts['correct']}, incorrect: {counts['incorrect']}}} "
              f"majority {majority:.3f}")
    for camera in sorted({r["camera"] for r in rows}):
        group = [r for r in rows if r["camera"] == camera]
        views = Counter(r["view"] for r in group)
        print(f"    Camera{camera}: {len(group)} rows, views "
              f"{dict(sorted(views.items()))}")
    distractor = sum(1 for r in rows if r["extra_person"] in ("noticeable", "large"))
    print(f"    extra person noticeable or larger: {distractor} rows; "
          f"lights off: {sum(1 for r in rows if not r['lights_on'])} rows")
    systems = {r["messages"][0]["content"] for r in rows}
    print(f"    distinct system prompts: {len(systems)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="REHAB24-6 root (default: %(default)s)")
    parser.add_argument("--video-dir", default="processed/videos",
                        help="rescaled recordings under the dataset root, written by "
                             "this script's rescale pass (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "rehabilitation",
                        help="where to write the metadata (default: %(default)s)")
    parser.add_argument("--prefix", default="rehab246",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--suffix", default="",
                        help="appended to the output filenames (default: none)")
    parser.add_argument("--cameras", default="17",
                        help="cameras to emit rows for (default: %(default)s)")
    parser.add_argument("--test-persons", default=",".join(map(str, DEFAULT_TEST_PERSONS)),
                        help="subject ids held out for test (default: %(default)s)")
    parser.add_argument("--fps", type=float, default=None,
                        help="pin one sampling fps for every row instead of "
                             "data/preprocess/frame_budget.py's duration ladder (default: the "
                             "ladder)")
    parser.add_argument("--jobs", type=int, default=16,
                        help="concurrent ffprobe calls (default: %(default)s)")
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    encode_videos(['--dataset-root', str(args.dataset_root)])
    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    unknown = [c for c in cameras if c not in CAMERA_FILES]
    if unknown:
        sys.exit(f"unknown cameras: {unknown} (known: {sorted(CAMERA_FILES)})")
    test_persons = {int(p) for p in args.test_persons.split(",") if p.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("task: rubric-conditioned correctness judgement with JSON rehabilitation_form")
    print("classes: two, because correctness is the only quality label in the release")
    print(f"cameras: {', '.join('Camera' + c for c in cameras)}")
    print(f"split: subject-disjoint, test = persons {sorted(test_persons)}")

    repetitions = read_segmentation(args.dataset_root / "Segmentation.csv")
    print(f"read {len(repetitions)} annotated repetitions")


    video_root = args.dataset_root / args.video_dir
    if not video_root.is_dir():
        sys.exit(f"rescaled recordings not found: {video_root}\n"
                 "the rescale pass did not run")
    media_paths = {}
    for repetition in repetitions:
        for camera in cameras:
            name = CAMERA_FILES[camera].format(video_id=repetition["video_id"])
            media_paths[(repetition["video_id"], camera)] = (
                video_root / f"Ex{repetition['exercise_id']}" / name)
    missing = sorted(str(p) for p in media_paths.values() if not p.is_file())
    if missing:
        sys.exit(f"{len(missing)} recordings missing under {video_root}, first: "
                 f"{missing[0]}\nthe rescale pass did not complete")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        probed = dict(zip(media_paths, pool.map(probe, media_paths.values())))
    print(f"probed {len(probed)} recordings across {len(cameras)} camera(s)")

    for repetition in repetitions:
        repetition["media"] = {
            camera: {"path": str(media_paths[(repetition["video_id"], camera)]),
                     "nb_frames": probed[(repetition["video_id"], camera)][0]}
            for camera in cameras}

    stats = Counter()
    rows = build_rows(repetitions, cameras, args.fps, stats)
    for reason, count in sorted(stats.items()):
        print(f"    {reason}: {count}")
    print(f"built {len(rows)} rows from {len(repetitions)} repetitions x "
          f"{len(cameras)} camera(s)")

    rows_by_split = {"train": [], "test": []}
    for row in rows:
        rows_by_split["test" if row["person_id"] in test_persons
                      else "train"].append(row)

    for split in ("train", "test"):
        split_rows = sorted(rows_by_split[split], key=lambda r: r["sample_id"])
        if not split_rows:
            sys.exit(f"no rows for split {split}")
        rows_by_split[split] = split_rows
        out_path = args.out_dir / f"{args.prefix}_{split}{args.suffix}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(split)
        summarise(split, split_rows)
        print(f"    -> {out_path}")

    persons = {s: {r["person_id"] for r in rows_by_split[s]} for s in rows_by_split}
    shared = persons["train"] & persons["test"]
    print(f"persons: {len(persons['train'])} train / {len(persons['test'])} test, "
          f"shared {len(shared)}")
    if shared:
        print(f"  WARNING: {sorted(shared)} appear in both splits")
    videos = {s: {r["video_id"] for r in rows_by_split[s]} for s in rows_by_split}
    if videos["train"] & videos["test"]:
        print(f"  WARNING: recordings in both splits: "
              f"{sorted(videos['train'] & videos['test'])}")
    missing_cells = [(s, e) for s in ("train", "test") for e in EXERCISES
                     if not any(r["exercise_id"] == e for r in rows_by_split[s])]
    if missing_cells:
        print(f"  WARNING: exercises absent from a split: {missing_cells}")
    return 0


EXERCISE_DIRS = tuple(f"Ex{i}" for i in range(1, 7))


DEFAULT_LONG_SIDE = 800


def encode_probe(path: Path) -> tuple:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames", "-of", "default=nw=1", str(path)],
        capture_output=True, text=True)
    fields = dict(line.split("=", 1) for line in result.stdout.strip().splitlines()
                  if "=" in line)
    try:
        return (int(fields["width"]), int(fields["height"]),
                int(fields["nb_frames"]))
    except (KeyError, ValueError):
        return 0, 0, 0


def encode(source: Path, target: Path, long_side: int, overwrite: bool) -> dict:
    width, height, frames = encode_probe(source)
    if frames == 0:
        return {"source": source, "status": "unreadable"}

    if target.is_file() and not overwrite:
        _, _, existing = encode_probe(target)
        if existing == frames:
            return {"source": source, "status": "cached", "frames": frames}


    if width >= height:
        scale = f"scale={long_side}:-2"
    else:
        scale = f"scale=-2:{long_side}"

    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vf", scale,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-an", str(target)],
        capture_output=True, text=True)
    if result.returncode != 0:
        return {"source": source, "status": "ffmpeg_failed",
                "error": result.stderr.strip().splitlines()[-1:] or [""]}

    out_width, out_height, out_frames = encode_probe(target)
    status = "encoded" if out_frames == frames else "frame_count_changed"
    return {"source": source, "status": status, "frames": frames,
            "out_frames": out_frames, "geometry": f"{out_width}x{out_height}"}


def encode_parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="REHAB24-6 root (default: %(default)s)")
    parser.add_argument("--out-subdir", default="processed/videos",
                        help="written under the dataset root (default: %(default)s)")
    parser.add_argument("--long-side", type=int, default=DEFAULT_LONG_SIDE,
                        help="pixels on the longer edge (default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=8,
                        help="concurrent ffmpeg processes (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-encode even when a correct output already exists")
    return parser.parse_args(argv)


def encode_videos(argv=None) -> int:
    args = encode_parse_args(argv)
    out_root = args.dataset_root / args.out_subdir

    sources = []
    for name in EXERCISE_DIRS:
        directory = args.dataset_root / name
        if not directory.is_dir():
            sys.exit(f"missing exercise directory: {directory}")
        sources += sorted(directory.glob("*.mp4"))
    if not sources:
        sys.exit(f"no recordings found under {args.dataset_root}")

    print(f"{len(sources)} recordings -> {out_root}")
    print(f"long side {args.long_side}px, frame rate and frame count preserved")

    targets = [out_root / source.parent.name / source.name for source in sources]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(
            lambda pair: encode(pair[0], pair[1], args.long_side, args.overwrite),
            zip(sources, targets)))

    counts = {}
    geometries = {}
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
              f"{result.get('frames')} -> {result.get('out_frames')}"
              f"{result.get('error', '')}")
    if bad:
        print(f"{len(bad)} recordings unusable; process_rehab246.py will not see them")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
