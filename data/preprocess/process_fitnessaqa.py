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
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

from . import frame_budget
from . import loss_scale

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "FitnessAQA"


SOURCE_FPS = 30.0


DEFAULT_FPS = frame_budget.sampling_fps(8.0)


KP_SYSTEM_PROMPT = (
    "You evaluate whether an exercise execution satisfies a given technical point. "
    'Return only this JSON object and nothing else: '
    '{"technical_point_satisfied": "yes" or "no"}.'
)

VERDICT = {True: "yes", False: "no"}

OHP_EXERCISE = "Overhead Press"
SQUAT_EXERCISE = "Back Squat"


OHP_POINTS = {


    "knees": "Keep the knees straight and fully extended throughout the lift, "
             "without bending them.",

    "elbows": "Keep the elbows in the correct position under the bar, "
              "without flaring them out.",
}

SS_POINT = ("Squat down to full depth, with the hips dropping below the knees at the "
            "bottom of the repetition.")


SQUAT_POINTS = {

    "inward": "Keep the knees tracking outward in line with the toes throughout the "
              "squat, without letting them cave inward.",

    "forward": "Keep the shins close to vertical, without the knees traveling far "
               "forward past the toes.",
}


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


OHP_LABEL_FILES = {error: f"error_{error}.json" for error in OHP_POINTS}
SQUAT_LABEL_FILES = {"inward": "error_knees_inward.json",
                     "forward": "error_knees_forward.json"}


def _load_video_errors(labelled: Path, points: dict, files: dict, name: str) -> dict:
    spans = {error: read_json(labelled / "Labels" / files[error])
             for error in points}
    videos = labelled / "videos"

    splits, dropped = {}, Counter()
    for split in ("train", "val", "test"):
        units = []
        for key in read_json(labelled / "Splits" / f"{split}_keys.json"):
            if not (videos / f"{key}.mp4").exists():
                dropped[f"{split}_video_missing"] += 1
                continue
            errors = {}
            for error in points:
                if key not in spans[error]:
                    dropped[f"{split}_label_missing"] += 1
                    continue
                errors[error] = bool(spans[error][key])
            if len(errors) == len(points):
                units.append((key, errors))
        splits[split] = units
    if dropped:
        print(f"{name}: dropped {dict(sorted(dropped.items()))}")
    return splits


def load_ohp(root: Path) -> dict:
    return _load_video_errors(root / "OHP" / "Labeled_Dataset", OHP_POINTS,
                              OHP_LABEL_FILES, "OHP")


def load_squat(root: Path) -> dict:
    return _load_video_errors(root / "Squat" / "Labeled_Dataset", SQUAT_POINTS,
                              SQUAT_LABEL_FILES, "Squat KIE/KFE")


def load_ss(root: Path) -> dict:
    error_dir = root / "Squat" / "Labeled_Dataset" / "Shallow_Squat_Error_Dataset"
    labels = read_json(error_dir / "labels_shallow_depth.json")
    videos = root / "Squat" / "Labeled_Dataset" / "videos"

    splits, dropped = {}, Counter()
    for split in ("train", "val", "test"):
        units = []
        for key in read_json(error_dir / "splits" / f"{split}_ids.json"):
            if key not in labels:
                dropped[f"{split}_label_missing"] += 1
                continue
            video_id, rep, frame = key.rsplit("_", 2)


            video = f"{video_id}_{rep}"
            if not (videos / f"{video}.mp4").exists():
                dropped[f"{split}_video_missing"] += 1
                continue
            units.append((key, video, int(frame), labels[key] == 1))
        splits[split] = units
    if dropped:
        print(f"SS: dropped {dict(sorted(dropped.items()))}")
    return splits


def is_image(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip().count(",") == 1


def extract_frames(units: list, source_dir: Path, out_dir: Path, quality: int,
                   workers: int) -> dict:
    by_video = defaultdict(dict)
    for key, video, frame, _ in units:
        by_video[video][frame] = key

    jobs = []
    for video, frames in sorted(by_video.items()):
        cuts = []
        for frame, key in sorted(frames.items()):
            out = out_dir / f"{key}.jpg"
            if out.exists() and is_image(out):
                continue
            cuts.append({"frame": frame, "out": out})
        if cuts:
            jobs.append({"source": source_dir / f"{video}.mp4", "cuts": cuts})

    out_dir.mkdir(parents=True, exist_ok=True)
    if not jobs:
        print(f"SS: all {len(by_video)} source videos already extracted under {out_dir}")
        return {key: out_dir / f"{key}.jpg"
                for key, _, _, _ in units}

    def encode(job):


        graph = ["[0:v]split=%d%s" % (len(job["cuts"]),
                                      "".join(f"[s{i}]" for i in range(len(job["cuts"]))))]
        for index, cut in enumerate(job["cuts"]):
            graph.append(f"[s{index}]select=eq(n\\,{cut['frame']})[v{index}]")
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                   "-i", str(job["source"]),
                   "-filter_complex", ";".join(graph)]
        for index, cut in enumerate(job["cuts"]):
            command += ["-map", f"[v{index}]", "-q:v", str(quality),
                        "-frames:v", "1", str(cut["out"])]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            reason = (result.stderr.strip().splitlines() or ["ffmpeg failed"])[-1]
            return [(cut["out"].name, reason) for cut in job["cuts"]]
        return [(cut["out"].name,
                 None if is_image(cut["out"]) else "not a readable image")
                for cut in job["cuts"]]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [item for batch in pool.map(encode, jobs) for item in batch]
    failures = [(name, reason) for name, reason in results if reason]
    if failures:
        for name, reason in failures[:10]:
            print(f"SS extraction failed: {name}: {reason}", file=sys.stderr)
        sys.exit(f"SS extraction: {len(failures)} images failed, "
                 f"{len(results) - len(failures)} written")
    print(f"SS: wrote {len(results)} single-frame images "
          f"({len(jobs)} source videos) under {out_dir}")
    return {key: out_dir / f"{key}.jpg" for key, _, _, _ in units}


def build_kp_record(unit: dict, fps: float | None) -> dict:
    tag = "<image>" if unit.get("is_image") else "<video>"
    question = (f"{tag}\nExercise: {unit['exercise']}\n"
                f"Technical point: {unit['point']}")
    record = {
        "messages": [
            {"role": "system", "content": KP_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": json.dumps(
                {"technical_point_satisfied": VERDICT[not unit["has_error"]]},
                ensure_ascii=False)},
        ],
    }
    if unit.get("is_image"):
        record["images"] = [str(unit["media"])]
    else:
        record["videos"] = [str(unit["media"])]
        record["chat_template_kwargs"] = {
            "fps": frame_budget.resolve_fps(unit["duration"], fps)}
    return record


def error_units(units, exercise: str, points: dict, videos: Path, cell_prefix: str):
    return [{"media": videos / f"{key}.mp4", "exercise": exercise,
             "point": points[error], "has_error": has_error, "is_image": False,
             "cell": f"{cell_prefix}_{error}"}
            for key, errors in units
            for error, has_error in sorted(errors.items())]


def ss_units(units, image_of: dict):
    return [{"media": image_of[key], "exercise": SQUAT_EXERCISE,
             "point": SS_POINT, "has_error": has_error, "is_image": True,
             "cell": "squat_ss"}
            for key, _, _, has_error in units]


def report(name: str, units: list) -> None:
    per_point = Counter()
    errors = Counter()
    surfaces = Counter()
    for unit in units:
        per_point[unit["point"]] += 1
        surfaces["image" if unit.get("is_image") else "video"] += 1
        if unit["has_error"]:
            errors[unit["point"]] += 1
    total = len(units)
    print(f"  {name}: {total} rows ({dict(sorted(surfaces.items()))})")
    for point, count in sorted(per_point.items()):
        print(f"    {point[:48]}...: {count} rows, error {errors[point]} "
              f"({errors[point] / count:.1%})")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="FitnessAQA root (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "fitness",
                        help="where to write the metadata (default: %(default)s)")
    parser.add_argument("--prefix", default="fitnessaqa",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--tasks", default="ohp,ss,squat",
                        help="comma-separated subset of {ohp, ss, squat} "
                             "(default: %(default)s)")
    parser.add_argument("--fps", type=float, default=None,
                        help=f"pin one sampling fps for every video row instead of "
                             f"data/preprocess/frame_budget.py's duration ladder; source is "
                             f"{SOURCE_FPS:g} fps (default: the ladder)")
    parser.add_argument("--no-merge-val", action="store_true",
                        help="train on the official train split only; by default val "
                             "is merged into train, as the UMAP'25 LMM protocol does")
    parser.add_argument("--no-extract", action="store_true",
                        help="skip the SS frame extraction; every image must already "
                             "exist under <dataset-root>/Squat/processed/ss_frames")
    parser.add_argument("--ss-dir", type=Path, default=None,
                        help="SS image directory (default: "
                             "<dataset-root>/Squat/processed/ss_frames)")
    parser.add_argument("--jpeg-quality", type=int, default=2,
                        help="ffmpeg -q:v for the extracted frames, 2 is near-lossless "
                             "(default: %(default)s)")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent ffmpeg processes (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="per-category training upsampling seed")
    parser.add_argument("--balance", action="store_true",
                        help="duplicate the minority verdict up to the majority count per "
                             "cell; opt-in experiment, the pipeline uses loss_scale weights instead")
    args = parser.parse_args(argv)

    args.tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    bad = [t for t in args.tasks if t not in ("ohp", "ss", "squat")]
    if bad:
        parser.error(f"--tasks accepts ohp, ss, squat; got {bad}")
    if args.ss_dir is None:
        args.ss_dir = args.dataset_root / "Squat" / "processed" / "ss_frames"
    return args


def downsample_train(units: list, seed: int) -> list:
    cells = {}
    for unit in units:
        cells.setdefault(unit['cell'], {True: [], False: []})[unit['has_error']].append(unit)
    selected = []
    for cell, classes in sorted(cells.items()):
        size = min(map(len, classes.values()))
        if not size:
            raise ValueError(f'Cannot balance training category with a missing class: {cell}')
        all_media = [str(u['media']) for group in classes.values() for u in group]
        if len(set(all_media)) != len(all_media):
            raise ValueError(f'Duplicate or conflicting training media in category {cell}')
        for has_error, group in sorted(classes.items()):
            ordered = sorted(group, key=lambda u: str(u['media']))
            chosen = (random.Random(f'{seed}:{cell}:{has_error}').sample(ordered, size)
                      if len(ordered) > size else ordered)
            selected.extend(chosen)
        print(f'    {cell}: {sum(map(len, classes.values()))} -> {2 * size} rows '
              f'({size} yes / {size} no)')
    return sorted(selected, key=lambda u: (u['point'], str(u['media'])))


def _resize_up(items: list, target: int, rng) -> list:
    out = list(items)
    while len(out) < target:
        need = target - len(out)
        out.extend(items if need >= len(items) else rng.sample(items, need))
    return out


def upsample_train(units: list, seed: int) -> list:
    cells = {}
    for unit in units:
        cells.setdefault(unit['cell'], {True: [], False: []})[unit['has_error']].append(unit)
    selected = []
    for cell, classes in sorted(cells.items()):
        if min(map(len, classes.values())) == 0:
            raise ValueError(f'Cannot balance training category with a missing class: {cell}')
        size = max(map(len, classes.values()))
        for has_error, group in sorted(classes.items()):
            ordered = sorted(group, key=lambda u: str(u['media']))
            rng = random.Random(f'{seed}:{cell}:{has_error}')
            selected.extend(_resize_up(ordered, size, rng))
        print(f'    {cell}: -> {2 * size} rows ({size} yes / {size} no, minority upsampled)')
    return sorted(selected, key=lambda u: (u['point'], str(u['media'])))


def dump_units(units: list, path, args, weights=None) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for unit in units:
            record = build_kp_record(unit, args.fps)
            if weights:
                loss_scale.attach(record, weights[(unit["cell"], unit["has_error"])])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_split(name: str, units: list, args) -> dict:
    cell_of, class_of = (lambda u: u["cell"]), (lambda u: u["has_error"])
    weights = None
    if name == "train":
        if getattr(args, "balance", False):
            units = upsample_train(units, args.seed)
        else:
            weights = loss_scale.balanced_weights(units, cell_of, class_of)
            print("    loss_scale: "
                  + loss_scale.summarise(units, cell_of, class_of, weights))
    path = args.out_dir / f"{args.prefix}_kp_{name}.jsonl"
    dump_units(units, path, args, weights)
    if name == "train":

        for stale in args.out_dir.glob(f"{args.prefix}_kp_train_*.jsonl"):
            stale.unlink()
    return {path: len(units)}


def main(argv=None) -> int:
    args = parse_args(argv)
    print("sampling fps: " + (f"{args.fps:g}, pinned" if args.fps else
                              "per clip, data/preprocess/frame_budget.py's duration ladder")
          + f" (source {SOURCE_FPS:g}); tasks: {','.join(args.tasks)}")

    ohp = load_ohp(args.dataset_root) if "ohp" in args.tasks else None
    ss = load_ss(args.dataset_root) if "ss" in args.tasks else None
    squat = load_squat(args.dataset_root) if "squat" in args.tasks else None

    if ss is not None:
        expected = {"train": 2542, "val": 529, "test": 540}
        for split, units in ss.items():
            if len(units) != expected[split]:
                print(f"warning: SS {split} holds {len(units)} samples, "
                      f"the public release has {expected[split]}")
    if ohp is not None:
        expected = {"train": 1582, "val": 339, "test": 339}
        for split, units in ohp.items():
            if len(units) != expected[split]:
                print(f"warning: OHP {split} holds {len(units)} videos, "
                      f"the public release has {expected[split]}")
    if squat is not None:
        expected = {"train": 1136, "val": 243, "test": 244}
        for split, units in squat.items():
            if len(units) != expected[split]:
                print(f"warning: Squat KIE/KFE {split} holds {len(units)} videos, "
                      f"the public release has {expected[split]}")

    image_of = None
    if ss is not None:
        if args.no_extract:
            image_of = {key: args.ss_dir / f"{key}.jpg"
                        for key, _, _, _ in ss["train"] + ss["val"] + ss["test"]}
            missing = [str(p) for p in image_of.values() if not p.exists()]
            if missing:
                sys.exit(f"{len(missing)} SS images missing under {args.ss_dir}, "
                         f"first: {missing[0]}")
        else:
            every = ss["train"] + ss["val"] + ss["test"]
            source = args.dataset_root / "Squat" / "Labeled_Dataset" / "videos"
            image_of = extract_frames(every, source, args.ss_dir, args.jpeg_quality,
                                      args.workers)

    train_units = {"ohp": [], "ss": [], "squat": []}
    test_units = {"ohp": [], "ss": [], "squat": []}

    train_side = ("train",) if args.no_merge_val else ("train", "val")
    if ohp is not None:
        train_units["ohp"] = [u for s in train_side for u in ohp[s]]
        test_units["ohp"] = ohp["test"]
    if ss is not None:
        train_units["ss"] = [u for s in train_side for u in ss[s]]
        test_units["ss"] = ss["test"]
    if squat is not None:
        train_units["squat"] = [u for s in train_side for u in squat[s]]
        test_units["squat"] = squat["test"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test"):
        units = []
        if ohp is not None:
            units += error_units(train_units["ohp"] if name == "train" else test_units["ohp"],
                                 OHP_EXERCISE, OHP_POINTS,
                                 args.dataset_root / "OHP" / "Labeled_Dataset" / "videos",
                                 "ohp")
        if ss is not None:
            units += ss_units(train_units["ss"] if name == "train" else test_units["ss"],
                              image_of)
        if squat is not None:
            units += error_units(
                train_units["squat"] if name == "train" else test_units["squat"],
                SQUAT_EXERCISE, SQUAT_POINTS,
                args.dataset_root / "Squat" / "Labeled_Dataset" / "videos",
                "squat")
        if not units:
            sys.exit(f"{name} split is empty")
        units.sort(key=lambda u: (u["point"], str(u["media"])))


        durations = frame_budget.probe_durations(
            [str(u["media"]) for u in units if not u.get("is_image")],
            workers=args.workers)
        for unit in units:
            unit["duration"] = 0.0 if unit.get("is_image") else \
                durations[str(unit["media"])]
        report(name, units)
        print("    sampling: " + frame_budget.fps_histogram(
            [{} if u.get("is_image") else
             {"chat_template_kwargs":
              {"fps": frame_budget.resolve_fps(u["duration"], args.fps)}}
             for u in units]))
        for path, rows in write_split(name, units, args).items():
            print(f"    -> {path} ({rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
