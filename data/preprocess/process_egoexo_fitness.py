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
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

from . import frame_budget
from . import loss_scale
from .feedback_style import feedback_person_phrase, sentence_count

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "EgoExo-Fitness"


ACTION_KEY_RE = re.compile(r"^(?P<record_id>.+)_action_(?P<action_index>\d+)$")


EGO_VIEWS = ("ego_l", "ego_m", "ego_r")
EXO_VIEWS = ("exo_m", "exo_l", "exo_r")
ALL_VIEWS = EGO_VIEWS + EXO_VIEWS


SOURCE_FPS = 30.0


DEFAULT_FPS = frame_budget.sampling_fps(8.0)


RELATIVE_PREFIX = Path("data/EgoExo-Fitness")

SCORE_MIN, SCORE_MAX = 1, 5


KP_SYSTEM_PROMPT = (
    "You evaluate whether an exercise execution satisfies a given technical point. "
    'Return only this JSON object and nothing else: '
    '{"technical_point_satisfied": "yes" or "no"}.'
)

SCORE_SYSTEM_PROMPT = (
    "You rate overall exercise execution quality using the provided evaluation "
    f'results. Return only this JSON object and nothing else: '
    f'{{"action_quality_score": integer from {SCORE_MIN} to {SCORE_MAX}}}.'
)

COMMENT_SYSTEM_PROMPT = (
    "You give concise, concrete feedback on exercise technique. Assess the execution "
    "visible in the video against the provided technical points. Respond in one or "
    "two sentences {person}."
)


VERDICT = {True: "yes", False: "no"}

TASKS = ("kp", "score", "comment")


DEFAULT_TASKS = ("kp", "comment")

def read_json(path: Path):
    if not path.is_file():
        sys.exit(f"annotation not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    sys.exit(f"unexpected keypoint verdict: {value!r}")


def record_views(meta: dict, views: tuple) -> dict:
    records = meta["records"]
    return {
        record_id: [v for v in views if v in records[index].get("views", [])]
        for record_id, index in meta["record_index"].items()
    }


def load_samples(dataset_root: Path, video_dir: str, video_path_style: str,
                 views: tuple, sample_one: bool = False, seed: int = 42):
    raw = dataset_root / "raw_annotations"
    judgements = read_json(raw / "interpretable_action_judgement.json")
    views_of = record_views(read_json(raw / "meta_records.json"), views)

    samples = []
    skipped_incomplete = 0
    missing_clips = 0
    unknown_records = 0
    for action_key in sorted(judgements):
        match = ACTION_KEY_RE.match(action_key)
        if match is None:
            sys.exit(f"unexpected action key: {action_key!r}")
        record_id = match.group("record_id")
        if record_id not in views_of:
            unknown_records += 1
            continue

        annotations = judgements[action_key].get("annotations") or []
        if not annotations:
            skipped_incomplete += 1
            continue

        annotation = annotations[0]
        verification = annotation.get("key_point_verification") or []
        score = annotation.get("action_quality_score")
        comment = (annotation.get("comment") or "").strip()

        valid_score = isinstance(score, int) and not isinstance(score, bool) \
            and SCORE_MIN <= score <= SCORE_MAX
        if not verification or not comment or not valid_score:
            skipped_incomplete += 1
            continue
        labels = {point: parse_flag(value) for point, value in verification}

        available = []
        for view in views_of[record_id]:
            clip = f"{action_key}_{view}.mp4"
            resolved = dataset_root / "processed" / video_dir / clip
            if not resolved.is_file():
                missing_clips += 1
                continue
            available.append((view, clip, resolved))
        if sample_one and available:
            available = [random.Random(f"{seed}:{action_key}").choice(available)]

        for view, clip, resolved in available:
            video = str(resolved) if video_path_style == "abs" else \
                str(RELATIVE_PREFIX / "processed" / video_dir / clip)
            samples.append({
                "record_id": record_id,
                "action_index": int(match.group("action_index")),
                "view": view,
                "action": annotation.get("action_name"),
                "labels": labels,
                "score": score,
                "comment": comment,
                "video": video,


                "abs_video": str(resolved),
            })

    if unknown_records:
        print(f"skipped {unknown_records} actions whose record is absent from meta_records.json")
    if skipped_incomplete:
        print(f"skipped {skipped_incomplete} actions missing keypoints, a valid score or a comment")
    if missing_clips:
        print(f"skipped {missing_clips} (action, view) pairs with no encoded clip under "
              f"processed/{video_dir}/")
    if not samples:
        sys.exit(f"no usable samples built from {raw}")
    return samples


def point_listing(sample: dict) -> str:
    return "\n".join(f"{text} \u2014 {VERDICT[ok]}"
                     for text, ok in sample["labels"].items())


def wrap(sample: dict, fps: float | None, system: str, question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "videos": [sample["video"]],
        "chat_template_kwargs": {
            "fps": frame_budget.resolve_fps(sample["duration"], fps)},
    }


def header(sample: dict) -> str:
    return f"<video>\nExercise: {sample['action']}\n"


def build_kp_record(unit: dict, fps: float) -> dict:
    sample = unit["sample"]
    question = f"{header(sample)}Technical point: {unit['text']}"
    return wrap(sample, fps, KP_SYSTEM_PROMPT, question, json.dumps(
        {"technical_point_satisfied": VERDICT[unit["label"]]}, ensure_ascii=False))


def build_score_record(sample: dict, fps: float) -> dict:
    question = (f"{header(sample)}"
                f"Technique evaluation:\n{point_listing(sample)}\n\n"
                "Rate the overall execution quality.")
    return wrap(sample, fps, SCORE_SYSTEM_PROMPT, question, json.dumps(
        {"action_quality_score": sample["score"]}, ensure_ascii=False))


def build_comment_record(sample: dict, fps: float) -> dict:
    points = "\n".join(sample["labels"])
    question = (f"{header(sample)}"
                f"Technical points:\n{points}\n\n"
                "Comment on the execution.")
    system = COMMENT_SYSTEM_PROMPT.format(person=feedback_person_phrase(sample["comment"]))
    if sentence_count(sample["comment"]) > 2:
        system = system.replace("one or two sentences", "one to five sentences")
    return wrap(sample, fps, system, question, sample["comment"])


def kp_units(samples):
    return [
        {"sample": sample, "index": index, "text": text, "label": ok}
        for sample in samples
        for index, (text, ok) in enumerate(sample["labels"].items(), 1)
    ]


def resize(items, target: int, rng):
    if target == len(items):
        return list(items)
    if target < len(items):
        return rng.sample(items, target)
    out = items * (target // len(items))
    out.extend(rng.sample(items, target % len(items)))
    return out


def balance_kp(units, ratio: float, mode: str, seed: int, max_dup: float = 0.0, single_class_cap: int = 5):
    if single_class_cap < 0:
        raise ValueError("single_class_cap must be nonnegative")
    if mode == "none":
        return list(units)
    cells = {}
    for unit in units:
        key = (unit["sample"]["action"], unit["text"])
        cells.setdefault(key, {True: [], False: []})[unit["label"]].append(unit)

    out, untouched, capped = [], 0, 0

    for key in sorted(cells):
        pos, neg = (sorted(cells[key][label], key=lambda u: u["sample"]["video"])
                    for label in (True, False))
        rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
        if not pos or not neg:
            untouched += 1
            if mode == "downsample":
                out.extend(resize(pos + neg, min(len(pos + neg), single_class_cap), rng))
            else:
                out.extend(pos + neg)
            continue
        if mode == "downsample":
            keep_pos = min(len(pos), round(ratio * len(neg)))
            keep_neg = min(len(neg), round(len(pos) / ratio))
        else:
            keep_pos = max(len(pos), round(ratio * len(neg)))
            keep_neg = max(len(neg), round(len(pos) / ratio))
            if max_dup:
                limit_pos = int(len(pos) * max_dup)
                limit_neg = int(len(neg) * max_dup)
                capped += keep_pos > limit_pos or keep_neg > limit_neg
                keep_pos, keep_neg = min(keep_pos, limit_pos), min(keep_neg, limit_neg)
        out.extend(resize(pos, max(keep_pos, 1), rng))
        out.extend(resize(neg, max(keep_neg, 1), rng))
    if untouched:
        print(f"    {untouched}/{len(cells)} (action, point) cells hold a single class "
              + (f"and were capped at {single_class_cap} rows each" if mode == "downsample" else "and were left untouched"))
    if capped:
        print(f"    {capped}/{len(cells)} cells hit the {max_dup:g}x duplication cap and "
              f"stop short of {ratio:g}:1")
    return out


def balance_score(samples, mode: str, max_dup: float, seed: int):
    if mode == "none":
        return list(samples)
    actions = {}
    for sample in samples:
        actions.setdefault(sample["action"], {}).setdefault(sample["score"], []).append(sample)

    out = []
    for ordinal, action in enumerate(sorted(actions)):
        buckets = actions[action]
        largest = max(len(v) for v in buckets.values())
        rng = random.Random(seed + ordinal)
        for score in sorted(buckets):
            bucket = buckets[score]
            target = min(largest, int(len(bucket) * max_dup))
            out.extend(resize(bucket, max(target, len(bucket)), rng))
    return out


def split_by_subject(samples, test_ratio: float, seed: int, restarts: int):
    per_record = Counter(s["record_id"] for s in samples)
    total = sum(per_record.values())
    target = total * test_ratio
    records = sorted(per_record)

    best = None
    for restart in range(restarts):
        rng = random.Random(seed + restart)
        shuffled = records[:]
        rng.shuffle(shuffled)
        test_records, test_count = [], 0
        for record in shuffled:
            size = per_record[record]

            if test_count + size <= target or not test_records:
                test_records.append(record)
                test_count += size
        gap = abs(test_count - target)
        if best is None or gap < best[0]:
            best = (gap, set(test_records), test_count)

    _, test_records, test_count = best
    print(f"subject-disjoint split: {len(test_records)}/{len(records)} records in test, "
          f"{test_count}/{total} clips ({test_count / total:.1%}, target {test_ratio:.0%})")
    return test_records


def summarise(name: str, samples, fps: float | None = None) -> dict:
    actions = Counter(s["action"] for s in samples)
    views = Counter(s["view"] for s in samples)
    cameras = Counter(s["view"].split("_")[0] for s in samples)
    scores = Counter(s["score"] for s in samples)
    records = sorted({s["record_id"] for s in samples})
    positives = sum(v for s in samples for v in s["labels"].values())
    points = sum(len(s["labels"]) for s in samples)
    mean_score = sum(s["score"] for s in samples) / len(samples)
    print(f"  {name}: {len(samples)} clips, {len(records)} records, "
          f"{len({(s['record_id'], s['action_index']) for s in samples})} actions, "
          f"{points} keypoints ({positives / points:.1%} True)")
    print(f"    views {dict(sorted(views.items()))} -> {dict(sorted(cameras.items()))}")
    print(f"    scores {dict(sorted(scores.items()))} (mean {mean_score:.2f})")


    print("    sampling (per clip): " + frame_budget.fps_histogram(
        [{"chat_template_kwargs": {"fps": frame_budget.resolve_fps(s["duration"], fps)}}
         for s in samples]))
    return {
        "num_clips": len(samples),
        "num_records": len(records),
        "records": records,
        "views": dict(sorted(views.items())),
        "cameras": dict(sorted(cameras.items())),
        "actions": dict(sorted(actions.items())),
        "keypoint_positive_rate": round(positives / points, 4),
        "score_distribution": dict(sorted(scores.items())),
        "score_mean": round(mean_score, 4),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="EgoExo-Fitness root (default: %(default)s)")
    parser.add_argument("--video-dir", default="videos",
                        help="clip directory under processed/ (default: %(default)s)")
    parser.add_argument("--video-path-style", choices=["abs", "relative"], default="abs",
                        help=f"absolute paths (default) or {RELATIVE_PREFIX}/... relative paths")
    parser.add_argument("--views", default="random-exo",
                        help="random-exo (default: one exocentric camera per action "
                             "instance, drawn with --seed), all / ego / exo, or a "
                             f"comma-separated subset of {','.join(ALL_VIEWS)}")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "fitness",
                        help="where to write the metadata (default: %(default)s)")
    parser.add_argument("--prefix", default="egoexo_fitness",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--test-ratio", type=float, default=0.3,
                        help="target share of clips in the test set (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="split seed (default: %(default)s)")
    parser.add_argument("--balance", action="store_true",
                        help="duplicate the minority class up to the majority count per cell "
                             "(kp) / per action (score); opt-in experiment, the pipeline uses "
                             "loss_scale weights instead; "
                             "default: no resampling")
    parser.add_argument("--fps", type=float, default=None,
                        help=f"pin one sampling fps for every row instead of "
                             f"data/preprocess/frame_budget.py's duration ladder; source is "
                             f"{SOURCE_FPS:g} fps (default: the ladder)")
    parser.add_argument("--jobs", type=int, default=32,
                        help="concurrent ffprobe calls for the clip durations "
                             "(default: %(default)s)")
    parser.add_argument("--restarts", type=int, default=200,
                        help="random restarts when packing records (default: %(default)s)")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                        help=f"comma-separated subset of {'/'.join(TASKS)} to emit "
                             "(default: %(default)s)")
    parser.add_argument("--kp-single-class-cap", type=int, default=5,
                        help="maximum rows per single-class kp train cell under --balance; "
                             "0 omits these cells (default: %(default)s)")
    args = parser.parse_args(argv)
    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in args.tasks if t not in TASKS]
    if unknown:
        parser.error(f"unknown task(s) {unknown}, pick from {list(TASKS)}")
    groups = {"all": ALL_VIEWS, "ego": EGO_VIEWS, "exo": EXO_VIEWS,
              "random-exo": EXO_VIEWS}
    args.sample_one_view = args.views == "random-exo"
    if args.views in groups:
        args.views = groups[args.views]
    else:
        args.views = tuple(v.strip() for v in args.views.split(",") if v.strip())
        unknown_views = [v for v in args.views if v not in ALL_VIEWS]
        if unknown_views:
            parser.error(f"unknown view(s) {unknown_views}, pick from {list(ALL_VIEWS)}")
    if args.kp_single_class_cap < 0:
        parser.error("--kp-single-class-cap must be nonnegative")
    return args


def write_rows(path: Path, rows) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _slug(value) -> str:
    out = "".join(c if c.isalnum() else "_" for c in str(value)).strip("_").lower()
    while "__" in out:
        out = out.replace("__", "_")
    return out or "x"


def emit_kp(subset, name: str, out_dir: Path, prefix: str, fps: float, args) -> list:
    units = kp_units(subset)
    cell_of = lambda u: (u["sample"]["action"], u["text"])
    class_of = lambda u: u["label"]
    if name == "train" and getattr(args, "balance", False):
        units = balance_kp(units, 1.0, "upsample", args.seed, max_dup=0.0,
                           single_class_cap=args.kp_single_class_cap)
        weights = {}
    else:

        weights = (loss_scale.balanced_weights(units, cell_of, class_of)
                   if name == "train" else {})
    labels = Counter(u["label"] for u in units)
    total = sum(labels.values())
    print(f"  kp/{name}: {total} rows from {len(subset)} clips, "
          f"true {labels[True]} ({labels[True] / total:.1%}) / "
          f"false {labels[False]} ({labels[False] / total:.1%}), "
          f"{len({(u['sample']['action'], u['index']) for u in units})} cells")
    if weights:
        print("    loss_scale: " + loss_scale.summarise(units, cell_of, class_of, weights))
    path = out_dir / f"{prefix}_kp_{name}.jsonl"
    write_rows(path, [loss_scale.attach(build_kp_record(u, fps),
                                       weights.get((cell_of(u), class_of(u)), 1.0))
                      for u in units])
    if name == "train":
        for stale in out_dir.glob(f"{prefix}_kp_train_*.jsonl"):
            stale.unlink()
    return [path]


def emit_score(subset, name: str, out_dir: Path, prefix: str, fps: float, args) -> list:
    rows = subset
    cell_of = lambda s: s["action"]
    class_of = lambda s: s["score"]
    if name == "train" and getattr(args, "balance", False):
        rows = balance_score(rows, "uniform", float(max(len(rows), 1)), args.seed)
        weights = {}
    else:
        weights = (loss_scale.balanced_weights(rows, cell_of, class_of)
                   if name == "train" else {})
    scores = Counter(s["score"] for s in rows)
    print(f"  score/{name}: {len(rows)} rows from {len(subset)} clips, "
          f"scores {dict(sorted(scores.items()))}")
    if weights:
        print("    loss_scale: " + loss_scale.summarise(rows, cell_of, class_of, weights))
    path = out_dir / f"{prefix}_score_{name}.jsonl"
    write_rows(path, [loss_scale.attach(build_score_record(s, fps),
                                       weights.get((cell_of(s), class_of(s)), 1.0))
                      for s in rows])
    if name == "train":
        for stale in out_dir.glob(f"{prefix}_score_train_*.jsonl"):
            stale.unlink()
    return [path]


def main(argv=None) -> int:
    args = parse_args(argv)
    encode_videos(['--dataset-root', str(args.dataset_root)])
    samples = load_samples(args.dataset_root, args.video_dir, args.video_path_style,
                           args.views, args.sample_one_view, args.seed)
    fps = args.fps


    durations = frame_budget.probe_durations([s["abs_video"] for s in samples],
                                             workers=args.jobs)
    for sample in samples:
        sample["duration"] = durations[sample["abs_video"]]
    print(f"built {len(samples)} clips over views {','.join(args.views)}"
          + (f", one per action instance drawn with seed {args.seed}"
             if args.sample_one_view else "")
          + f" from {args.dataset_root / 'raw_annotations'}")
    lengths = sorted(s["duration"] for s in samples)
    print(f"clip seconds: min {lengths[0]:.1f} median {lengths[len(lengths) // 2]:.1f} "
          f"max {lengths[-1]:.1f} (source {SOURCE_FPS:g} fps)")
    print("sampling fps: " + (f"{fps:g}, pinned" if fps else
                              "per clip, data/preprocess/frame_budget.py's duration ladder"))

    test_records = split_by_subject(samples, args.test_ratio, args.seed, args.restarts)
    splits = {"train": [], "test": []}
    for sample in samples:
        splits["test" if sample["record_id"] in test_records else "train"].append(sample)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, subset in splits.items():

        subset.sort(key=lambda s: (s["record_id"], s["action_index"], s["view"]))
        stats[name] = summarise(name, subset, fps)

        for task in args.tasks:
            if task == "kp":
                paths = emit_kp(subset, name, args.out_dir, args.prefix, fps, args)
            elif task == "score":
                paths = emit_score(subset, name, args.out_dir, args.prefix, fps, args)
            else:
                out_path = args.out_dir / f"{args.prefix}_{task}_{name}.jsonl"
                write_rows(out_path, [build_comment_record(s, fps) for s in subset])
                paths = [out_path]
            for path in paths:
                print(f"    -> {path}")

    assert not (set(stats["train"]["records"]) & set(stats["test"]["records"])), \
        "record leaked across splits"
    return 0


CLIP_FPS = 30
CLIP_CRF = 18

FRAME_RE = re.compile(r"^frame_(\d+)\.jpg$")


def frame_index(frame_dir: Path) -> set:
    out = set()
    for name in frame_dir.iterdir():
        match = FRAME_RE.match(name.name)
        if match:
            out.add(int(match.group(1)))
    return out


def clip_frames(path: Path) -> int:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    written = probe.stdout.strip()
    return int(written) if written.isdigit() else -1


def plan_clips(dataset_root: Path, views: tuple, out_dir: Path, args):
    raw = dataset_root / "raw_annotations"
    meta = read_json(raw / "meta_records.json")
    judgements = read_json(raw / "interpretable_action_judgement.json")
    index = meta["record_index"]

    jobs, skipped, dropped, present = [], Counter(), [], {}
    existing = []
    for action_key in sorted(judgements):
        record_id = action_key.rsplit("_action_", 1)[0]
        if record_id not in index:
            skipped["unknown_record"] += 1
            continue
        start, end = judgements[action_key]["st_ed_frame"]
        record = meta["records"][index[record_id]]
        for view in views:
            frames = record.get("frames", {}).get(view)
            if view not in record.get("views", []) or not frames:
                continue
            frame_dir = dataset_root / "frames_open" / record_id / view
            if not frame_dir.is_dir():
                skipped["no_frame_dir"] += 1
                continue
            if end > frames["num_frames"]:
                skipped["range_past_end"] += 1
                continue
            out = out_dir / f"{action_key}_{view}.mp4"

            if frame_dir not in present:
                present[frame_dir] = frame_index(frame_dir)
            have = [i for i in range(start, end + 1) if i in present[frame_dir]]
            count = end - start + 1
            missing = (count - len(have)) / count
            if missing > args.max_missing:


                stale = out.is_file() and clip_frames(out) != count
                dropped.append((out.name, count - len(have), count, stale))
                skipped["too_many_missing_frames"] += 1
                if stale and args.prune:
                    out.unlink()
                continue

            job = {"view": view, "frame_dir": frame_dir, "start": start, "count": count,
                   "out": out, "have": have if len(have) != count else None}
            if out.is_file() and not args.overwrite:

                if not args.verify or clip_frames(out) == count:
                    skipped["exists"] += 1
                    continue
                existing.append(out.name)
            jobs.append(job)

    return jobs, skipped, dropped, existing


def encode(job: dict) -> tuple[dict, str | None]:
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    listing = None
    if job["have"] is None:

        command += ["-framerate", str(CLIP_FPS),
                    "-start_number", str(job["start"]),
                    "-i", str(job["frame_dir"] / "frame_%010d.jpg")]
    else:


        listing = tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False,
                                              encoding="utf-8")
        stops = job["have"] + [job["start"] + job["count"]]
        listing.write("ffconcat version 1.0\n")
        for position, frame in enumerate(job["have"]):
            begin = job["start"] if position == 0 else frame
            listing.write(f"file '{job['frame_dir']}/frame_{frame:010d}.jpg'\n")
            listing.write(f"duration {(stops[position + 1] - begin) / CLIP_FPS:.6f}\n")
        listing.close()
        command += ["-f", "concat", "-safe", "0", "-i", listing.name,
                    "-r", str(CLIP_FPS), "-fps_mode", "cfr"]


    command += ["-frames:v", str(job["count"])]

    if job["view"].startswith("exo"):
        command += ["-vf", "transpose=1"]
    command += ["-c:v", "libx264", "-crf", str(CLIP_CRF), "-pix_fmt", "yuv420p",
                str(job["out"])]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
    finally:
        if listing is not None:
            Path(listing.name).unlink(missing_ok=True)
    if result.returncode != 0:
        return job, (result.stderr.strip().splitlines() or ["ffmpeg failed"])[-1]

    written = clip_frames(job["out"])
    if written != job["count"]:
        return job, f"wrote {written} frames, expected {job['count']}"
    return job, None


def encode_parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="EgoExo-Fitness root (default: %(default)s)")
    parser.add_argument("--video-dir", default="videos",
                        help="clip directory under processed/ (default: %(default)s)")
    parser.add_argument("--views", default="all",
                        help="all / ego / exo, or a comma-separated subset of "
                             f"{','.join(ALL_VIEWS)} (default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=16,
                        help="concurrent ffmpeg processes (default: %(default)s)")
    parser.add_argument("--max-missing", type=float, default=0.2,
                        help="drop a clip whose annotated range is missing more than "
                             "this share of its frames (default: %(default)s)")
    parser.add_argument("--verify", action="store_true",
                        help="probe clips that already exist and re-encode any whose "
                             "frame count is short of the annotated range")
    parser.add_argument("--prune", action="store_true",
                        help="delete short clips that --max-missing refuses to rebuild")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-encode clips that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be encoded")
    args = parser.parse_args(argv)
    groups = {"all": ALL_VIEWS, "ego": EGO_VIEWS, "exo": EXO_VIEWS}
    if args.views in groups:
        args.views = groups[args.views]
    else:
        args.views = tuple(v.strip() for v in args.views.split(",") if v.strip())
        unknown = [v for v in args.views if v not in ALL_VIEWS]
        if unknown:
            parser.error(f"unknown view(s) {unknown}, pick from {list(ALL_VIEWS)}")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if not 0 <= args.max_missing <= 1:
        parser.error("--max-missing must be between 0 and 1")
    return args


def encode_videos(argv=None) -> int:
    args = encode_parse_args(argv)
    out_dir = args.dataset_root / "processed" / args.video_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs, skipped, dropped, stale = plan_clips(args.dataset_root, args.views,
                                               out_dir, args)
    per_view = Counter(j["view"] for j in jobs)
    holed = sum(1 for j in jobs if j["have"] is not None)
    print(f"{len(jobs)} clips to encode into {out_dir}")
    print(f"  per view {dict(sorted(per_view.items()))}")
    if holed:
        print(f"  {holed} of them have holes in the frame range and are muxed with "
              f"held frames")
    if stale:
        print(f"  {len(stale)} existing clips are short of their range and get "
              f"re-encoded: {', '.join(sorted(stale)[:5])}"
              f"{' ...' if len(stale) > 5 else ''}")
    if skipped:
        print(f"  skipped {dict(sorted(skipped.items()))}")
    for name, missing, count, stale in dropped:
        state = "short clip on disk" if stale else "no clip on disk"
        print(f"  DROPPED {name}: {missing}/{count} frames missing "
              f"(> --max-missing {args.max_missing:g}), {state}")
    if args.dry_run or not jobs:
        return 0

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for done, (job, error) in enumerate(pool.map(encode, jobs), 1):
            if error:
                failures.append((job["out"].name, error))
            if done % 200 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} encoded, {len(failures)} failed", flush=True)

    for name, error in failures[:20]:
        print(f"  FAILED {name}: {error}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} clips failed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
