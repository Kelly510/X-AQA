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
import math
import random
import re
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
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "FLAG3D"

AVATARS = ("M001", "M002", "M003", "M004")


SOURCE_FPS = 30.0


DEFAULT_FPS = frame_budget.sampling_fps(8.0)


RELATIVE_PREFIX = Path("data/FLAG3D")


KP_SYSTEM_PROMPT = (
    "You evaluate whether an exercise execution satisfies a given technical point. "
    'Return only this JSON object and nothing else: '
    '{"technical_point_satisfied": "yes" or "no"}.'
)

VERDICT = {True: "yes", False: "no"}


DELTA = 6.0

DEFAULT_TEST_PERSONS = ("P009", "P010")

ANNOTATION_RE = re.compile(r"^(?P<action>A\d+)(?P<person>P\d+)(?P<rep>R\d+)\.csv$")
CLIP_STEM_RE = re.compile(r"^(?P<avatar>M\d+)(?P<person>P\d+)(?P<action>A\d+)"
                          r"(?P<rep>R\d+)_s(?P<segment>\d+)$")
POINT_RE = re.compile(r"^Technical point: (?P<text>.*)$", re.MULTILINE)


def g(mean: float) -> float:
    return 2.0 * mean - 1.0


def parse_flag(value: str) -> bool:
    text = value.strip()
    if text in ("1", "0"):
        return text == "1"
    sys.exit(f"unexpected key score: {value!r}")


def action_names(root: Path) -> dict:
    out = {}
    for path in sorted((root / "annotations").glob("*I001.txt")):
        lines = [l.strip() for l in path.read_text(encoding="utf-8",
                                                   errors="replace").splitlines()]
        text = next((l for l in lines if l), "")
        if text:
            out[path.name[:4]] = text
    return out


def load_sequences(root: Path, video_dir: str, video_path_style: str, avatars: tuple,
                   keep_action_column: bool):
    annotation_dir = root / "FLAG3D++" / "annotation"
    if not annotation_dir.is_dir():
        sys.exit(f"annotations not found: {annotation_dir}")
    names = action_names(root)
    clip_dir = root / "processed" / video_dir

    sequences, notes = [], Counter()
    for csv_path in sorted(annotation_dir.glob("*.csv")):
        match = ANNOTATION_RE.match(csv_path.name)
        if match is None:
            notes["unexpected_annotation_name"] += 1
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        header, body = rows[0], [r for r in rows[1:] if len(r) > 3]
        if not body:
            notes["empty_annotation"] += 1
            continue

        action_id = match["action"]
        action = names.get(action_id) or header[-1].strip()
        criteria = [c.strip() for c in header[3:]]
        if not keep_action_column:

            if criteria[-1].lower() == action.lower():
                criteria = criteria[:-1]
            else:
                notes["action_column_not_last"] += 1
        notes[f"{len(criteria)}_standards"] += 1

        for avatar in avatars:
            stem = f"{avatar}{match['person']}{action_id}{match['rep']}"
            segments = []
            for index, row in enumerate(body):
                clip = clip_dir / f"{stem}_s{index:02d}.mp4"
                if not clip.is_file():
                    notes["missing_clip"] += 1
                video = str(clip) if video_path_style == "abs" else \
                    str(RELATIVE_PREFIX / "processed" / video_dir / clip.name)
                segments.append({
                    "index": index,
                    "labels": [parse_flag(v) for v in row[3:3 + len(criteria)]],
                    "video": video,


                    "abs_video": str(clip),
                    "present": clip.is_file(),
                })
            sequences.append({
                "sequence": stem,
                "avatar": avatar,
                "person": match["person"],
                "action_id": action_id,
                "action": action,
                "rep": match["rep"],
                "criteria": criteria,
                "segments": segments,
            })

    if not sequences:
        sys.exit(f"no usable sequences built from {annotation_dir}")
    print(f"loaded {len(sequences)} sequences "
          f"({sum(len(s['segments']) for s in sequences)} repetitions) from "
          f"{annotation_dir}")
    print(f"  clips read from {clip_dir}")
    if notes:
        print(f"  notes {dict(sorted(notes.items()))}")
    return sequences


def sequence_score(sequence: dict, segments=None) -> float:
    segments = sequence["segments"] if segments is None else segments
    if not segments:
        sys.exit(f"{sequence['sequence']}: no repetitions to score")
    total = DELTA
    for index in range(len(sequence["criteria"])):
        hits = sum(1 for seg in segments if seg["labels"][index])
        total += g(hits / len(segments))
    return total


def kp_units(sequences):
    return [
        {"sequence": sequence, "segment": segment, "index": index, "text": text,
         "label": segment["labels"][index]}
        for sequence in sequences
        for segment in sequence["segments"] if segment["present"]
        for index, text in enumerate(sequence["criteria"])
    ]


def build_kp_record(unit: dict, fps: float | None) -> dict:
    question = (f"<video>\nExercise: {unit['sequence']['action']}\n"
                f"Technical point: {unit['text']}")
    return {
        "messages": [
            {"role": "system", "content": KP_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": json.dumps(
                {"technical_point_satisfied": VERDICT[unit["label"]]},
                ensure_ascii=False)},
        ],
        "videos": [unit["segment"]["video"]],
        "chat_template_kwargs": {
            "fps": frame_budget.resolve_fps(unit["segment"]["duration"], fps)},
    }


def resize(items, target: int, rng):
    if target == len(items):
        return list(items)
    if target < len(items):
        return rng.sample(items, target)
    out = items * (target // len(items))
    out.extend(rng.sample(items, target % len(items)))
    return out


def balance_kp(units, ratio: float, mode: str, seed: int, max_dup: float = 0.0):
    if mode == "none":
        return list(units)
    cells = {}
    for unit in units:
        key = (unit["sequence"]["action"], unit["index"])
        cells.setdefault(key, {True: [], False: []})[unit["label"]].append(unit)

    out, untouched, capped = [], 0, 0

    for ordinal, key in enumerate(sorted(cells)):
        pos, neg = cells[key][True], cells[key][False]
        if not pos or not neg:
            untouched += 1
            out.extend(pos + neg)
            continue
        rng = random.Random(seed + ordinal)
        if mode == "downsample":
            keep_pos = min(len(pos), round(ratio * len(neg)))
            keep_neg = min(len(neg), round(len(pos) / ratio))
        else:
            keep_pos = max(len(pos), round(ratio * len(neg)))
            keep_neg = max(len(neg), round(len(pos) / ratio))
            if max_dup:
                limit_pos, limit_neg = int(len(pos) * max_dup), int(len(neg) * max_dup)
                capped += keep_pos > limit_pos or keep_neg > limit_neg
                keep_pos, keep_neg = min(keep_pos, limit_pos), min(keep_neg, limit_neg)
        out.extend(resize(pos, max(keep_pos, 1), rng))
        out.extend(resize(neg, max(keep_neg, 1), rng))
    if untouched:
        print(f"    {untouched}/{len(cells)} (action, standard) cells hold a single "
              f"verdict and were left untouched")
    if capped:
        print(f"    {capped}/{len(cells)} cells hit the {max_dup:g}x duplication cap and "
              f"stop short of {ratio:g}:1")
    return out


def verify_scores(rows, sequences) -> dict:
    observed = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        stem = Path(row["videos"][0]).stem
        match = CLIP_STEM_RE.match(stem)
        if match is None:
            sys.exit(f"clip name is not parseable back into a sequence: {stem}")
        point = POINT_RE.search(row["messages"][1]["content"])
        if point is None:
            sys.exit(f"no technical point in the prompt of {stem}")
        key = f"{match['avatar']}{match['person']}{match['action']}{match['rep']}"
        observed[key][point["text"]][int(match["segment"])] = \
            json.loads(row["messages"][-1]["content"]).get(
                "technical_point_satisfied") == "yes"

    scores = {}
    for sequence in sequences:
        key = sequence["sequence"]
        per_point = observed.get(key)
        if per_point is None:
            sys.exit(f"{key}: emitted no rows, so its score cannot be rebuilt")
        rebuilt = DELTA
        for text in sequence["criteria"]:
            verdicts = per_point.get(text)
            if verdicts is None:
                sys.exit(f"{key}: standard {text!r} is missing from the rows")
            if len(verdicts) != len(sequence["segments"]):
                sys.exit(f"{key}: {len(verdicts)} repetitions in the rows against "
                         f"{len(sequence['segments'])} in the annotation, so Eq. (1) "
                         f"would use a different denominator than the paper")
            rebuilt += g(sum(verdicts.values()) / len(verdicts))
        expected = sequence_score(sequence)
        if abs(rebuilt - expected) > 1e-9:
            sys.exit(f"{key}: Eq. (1) rebuilt from the rows is {rebuilt:.6f}, "
                     f"annotation says {expected:.6f}")
        scores[key] = expected
    return scores


def summarise(name: str, sequences) -> None:
    segments = [seg for s in sequences for seg in s["segments"] if seg["present"]]
    missing = sum(1 for s in sequences for seg in s["segments"] if not seg["present"])
    verdicts = Counter(label for s in sequences
                       for seg in s["segments"] if seg["present"]
                       for label in seg["labels"])
    total = sum(verdicts.values()) or 1
    print(f"  {name}: {len(sequences)} sequences, {len(segments)} repetitions, "
          f"{len({s['person'] for s in sequences})} persons, "
          f"{len({s['avatar'] for s in sequences})} avatars, "
          f"{len({s['action'] for s in sequences})} actions")
    print(f"    verdicts {total} ({verdicts[False] / total:.1%} no)"
          + (f", {missing} repetitions without a clip" if missing else ""))


def parse_avatars(parser, value: str, flag: str) -> tuple:
    if value == "all":
        return AVATARS
    chosen = tuple(a.strip() for a in value.split(",") if a.strip())
    unknown = [a for a in chosen if a not in AVATARS]
    if unknown:
        parser.error(f"{flag}: unknown avatar(s) {unknown}, pick from {list(AVATARS)}")
    return chosen


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="FLAG3D root (default: %(default)s)")
    parser.add_argument("--video-dir", default="segments",
                        help="clip directory under processed/, written by "
                             "this script's clip pass (default: %(default)s)")
    parser.add_argument("--video-path-style", choices=["abs", "relative"], default="abs",
                        help=f"absolute paths (default) or {RELATIVE_PREFIX}/... relative")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "fitness",
                        help="where to write the metadata (default: %(default)s)")
    parser.add_argument("--prefix", default="flag3d",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--test-persons", default=",".join(DEFAULT_TEST_PERSONS),
                        help="persons held out for test; the paper uses 8:2 by person "
                             "and picks them at random (default: %(default)s)")
    parser.add_argument("--train-avatars", default="M001",
                        help="avatars kept on the training side, comma-separated or all. "
                             "Each extra avatar is another render of the same MoCap take, "
                             "i.e. augmentation (default: %(default)s)")
    parser.add_argument("--test-avatars", default="M001",
                        help="avatars kept on the test side. Stay at one: four renders of "
                             "one take are four correlated items, and the paper's unit is "
                             "the take (default: %(default)s)")
    parser.add_argument("--fps", type=float, default=None,
                        help=f"pin one sampling fps for every row instead of "
                             f"data/preprocess/frame_budget.py's duration ladder; source is "
                             f"{SOURCE_FPS:g} fps (default: the ladder)")
    parser.add_argument("--jobs", type=int, default=32,
                        help="concurrent ffprobe calls for the clip durations "
                             "(default: %(default)s)")
    parser.add_argument("--kp-ratio", type=float, default=1.0,
                        help="target yes:no ratio per (action, standard) cell in the "
                             "training split (default: %(default)s)")
    parser.add_argument("--kp-balance", choices=["upsample", "downsample", "none"],
                        default="upsample",
                        help="how to reach --kp-ratio: duplicate the minority verdict "
                             "(default), trim the majority one, or leave it alone")
    parser.add_argument("--kp-max-dup", type=float, default=0.0,
                        help="cap on how many times one training row may be repeated, "
                             "0 to disable (default: %(default)s)")
    parser.add_argument("--keep-action-column", action="store_true",
                        help="keep the annotation's last label column, which repeats the "
                             "activity name and is 1 in 1783 of 1784 sequences")
    parser.add_argument("--seed", type=int, default=42,
                        help="recording split and resampling seed (default: %(default)s)")
    parser.add_argument("--val-ratio", type=float, default=0.01,
                        help="fraction of original training recordings held out before balancing")
    parser.add_argument("--balance", action="store_true",
                        help="duplicate the minority verdict up to the majority count per "
                             "(action, standard) cell; opt-in experiment, the pipeline uses "
                             "loss_scale weights instead; "
                             "default: no resampling")
    args = parser.parse_args(argv)

    args.train_avatars = parse_avatars(parser, args.train_avatars, "--train-avatars")
    args.test_avatars = parse_avatars(parser, args.test_avatars, "--test-avatars")
    args.test_persons = tuple(p.strip() for p in args.test_persons.split(",") if p.strip())
    bad = [p for p in args.test_persons if not re.fullmatch(r"P\d{3}", p)]
    if bad:
        parser.error(f"--test-persons look like P001..P010, got {bad}")
    if not args.test_persons:
        parser.error("--test-persons must name at least one person")
    if args.kp_ratio <= 0:
        parser.error("--kp-ratio must be positive")
    if not math.isfinite(args.val_ratio) or not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be finite and strictly between 0 and 1")
    if args.kp_max_dup and args.kp_max_dup < 1:
        parser.error("--kp-max-dup must be 0 (disabled) or >= 1")
    return args


def write_rows(path: Path, rows) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def recording_id(sequence):
    return sequence['person'] + sequence['action_id'] + sequence['rep']


def split_validation(splits, ratio, seed):
    if not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError('Validation ratio must be finite and between 0 and 1')
    groups = defaultdict(list)
    for sequence in splits['train']:
        groups[recording_id(sequence)].append(sequence)

    complete = sorted(key for key, group in groups.items()
                      if all(seg['present'] for s in group for seg in s['segments']))
    nval = max(1, round(len(groups) * ratio))
    if nval >= len(groups) or nval > len(complete):
        raise ValueError('Not enough complete recordings for nonempty train/val splits')
    heldout = set(random.Random(seed).sample(complete, nval))
    result = {'train': [], 'val': [], 'test': list(splits['test'])}
    for key, group in sorted(groups.items()):
        result['val' if key in heldout else 'train'].extend(sorted(group, key=lambda s: s['sequence']))
    ids = {name: {recording_id(s) for s in group} for name, group in result.items()}
    if any(ids[a] & ids[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Original recording overlaps across splits')
    return result


def split_sequences(sequences, args):
    splits = {"train": [], "test": []}
    dropped = Counter()
    for sequence in sequences:
        held_out = sequence["person"] in args.test_persons
        wanted = args.test_avatars if held_out else args.train_avatars
        if sequence["avatar"] not in wanted:
            dropped["avatar_not_requested"] += 1
            continue
        if held_out and any(not seg["present"] for seg in sequence["segments"]):


            dropped["test_sequence_incomplete"] += 1
            continue
        splits["test" if held_out else "train"].append(sequence)
    if dropped:
        print(f"dropped {dict(sorted(dropped.items()))}")
    assert not ({s["person"] for s in splits["train"]}
                & {s["person"] for s in splits["test"]}), "person leaked across splits"
    return splits


def main(argv=None) -> int:
    args = parse_args(argv)
    encode_videos(['--dataset-root', str(args.dataset_root)])
    avatars = tuple(a for a in AVATARS
                    if a in set(args.train_avatars) | set(args.test_avatars))
    sequences = load_sequences(args.dataset_root, args.video_dir, args.video_path_style,
                               avatars, args.keep_action_column)
    print(f"split 8:2 by person -> test {','.join(args.test_persons)}; "
          f"avatars train {','.join(args.train_avatars)} / "
          f"test {','.join(args.test_avatars)}")


    present = [seg for s in sequences for seg in s["segments"] if seg["present"]]
    durations = frame_budget.probe_durations([seg["abs_video"] for seg in present],
                                            workers=args.jobs)
    for segment in present:
        segment["duration"] = durations[segment["abs_video"]]
    lengths = sorted(seg["duration"] for seg in present)
    print(f"repetition seconds: min {lengths[0]:.1f} median "
          f"{lengths[len(lengths) // 2]:.1f} max {lengths[-1]:.1f} "
          f"(source {SOURCE_FPS:g} fps)")
    print("sampling fps: " + (f"{args.fps:g}, pinned" if args.fps else
                              "per clip, data/preprocess/frame_budget.py's duration ladder"))

    splits = split_validation(split_sequences(sequences, args), args.val_ratio, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'version': 1, 'unit': 'person_action_recording_without_avatar',
                'seed': args.seed, 'val_ratio': args.val_ratio,
                'test_persons': args.test_persons, 'train_avatars': args.train_avatars,
                'test_avatars': args.test_avatars,
                'balance': {'mode': args.kp_balance, 'ratio': args.kp_ratio,
                            'max_dup': args.kp_max_dup, 'applies_to': 'train'},
                'splits': {}}
    for name in ("train", "val", "test"):
        subset = sorted(splits[name], key=lambda s: s["sequence"])
        if not subset:
            sys.exit(f"{name} split is empty")
        summarise(name, subset)

        units = kp_units(subset)
        cell_of = lambda u: (u["sequence"]["action"], u["index"])
        class_of = lambda u: u["label"]
        if name == "train" and getattr(args, "balance", False):


            units = balance_kp(units, args.kp_ratio, "upsample", args.seed, args.kp_max_dup)
            weights = {}
        else:


            weights = (loss_scale.balanced_weights(units, cell_of, class_of)
                       if name == "train" else {})
        rows = [loss_scale.attach(build_kp_record(unit, args.fps),
                                 weights.get((cell_of(unit), class_of(unit)), 1.0))
                for unit in units]
        verdicts = Counter(unit["label"] for unit in units)
        total = sum(verdicts.values())
        print(f"    kp/{name}: {total} rows, yes {verdicts[True]} "
              f"({verdicts[True] / total:.1%}) / no {verdicts[False]} "
              f"({verdicts[False] / total:.1%}), "
              f"{len({(u['sequence']['action'], u['index']) for u in units})} cells")
        if weights:
            print("    loss_scale: "
                  + loss_scale.summarise(units, cell_of, class_of, weights))
        print(f"    sampling: {frame_budget.fps_histogram(rows)}")

        if name in ("val", "test"):
            scores = verify_scores(rows, subset)
            values = sorted(scores.values())
            span = values[-1] - values[0]
            print(f"    Eq. (1) rebuilt from these rows matches the annotation for all "
                  f"{len(scores)} sequences")
            print(f"    score: min {values[0]:.3f} / mean "
                  f"{sum(values) / len(values):.3f} / max {values[-1]:.3f}, "
                  f"{len(set(values))} distinct values")
            print(f"    R-l2 denominator s_max - s_min = {span:.3f} "
                  f"(Eq. (1) allows [3, 9], i.e. 6)")

        out_path = args.out_dir / f"{args.prefix}_kp_{name}.jsonl"
        write_rows(out_path, rows)
        if name == "train":
            for stale in args.out_dir.glob(f"{args.prefix}_kp_train_*.jsonl"):
                stale.unlink()
        manifest['splits'][name] = {
            'recordings': sorted({recording_id(s) for s in subset}),
            'sequences': [s['sequence'] for s in subset],
            'raw_qa_rows': len(kp_units(subset)), 'exported_rows': len(rows),
            'file': out_path.name}
        print(f"    -> {out_path}")
    path = args.out_dir / f'{args.prefix}_kp_split_manifest.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(manifest, indent=2) + '\n')
    tmp.replace(path)
    return 0


def read_ranges(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return [(int(row[1]), int(row[2])) for row in rows[1:] if len(row) >= 3]


def clip_frames(path: Path) -> int:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    written = probe.stdout.strip()
    return int(written) if written.isdigit() else -1


def plan_clips(root: Path, out_dir: Path, args):
    annotation_dir = root / "FLAG3D++" / "annotation"
    if not annotation_dir.is_dir():
        sys.exit(f"annotations not found: {annotation_dir}")

    jobs, skipped, stale = [], Counter(), []
    for csv_path in sorted(annotation_dir.glob("*.csv")):
        match = ANNOTATION_RE.match(csv_path.name)
        if match is None:
            skipped["unexpected_annotation_name"] += 1
            continue
        if args.persons and match["person"] not in args.persons:
            continue
        ranges = read_ranges(csv_path)
        if not ranges:
            skipped["empty_annotation"] += 1
            continue

        for avatar in args.avatars:
            stem = f"{avatar}{match['person']}{match['action']}{match['rep']}"
            source = root / "subsetvideo" / "subset_video" / f"{stem}.mp4"
            if not source.is_file():
                skipped["no_source_video"] += 1
                continue
            cuts = []
            for index, (start, end) in enumerate(ranges):
                out = out_dir / f"{stem}_s{index:02d}.mp4"
                count = end - start
                if count < 1:
                    skipped["empty_range"] += 1
                    continue
                if out.is_file() and not args.overwrite:


                    if not args.verify or clip_frames(out) == count:
                        skipped["exists"] += 1
                        continue
                    stale.append(out.name)
                cuts.append({"start": start, "count": count, "out": out})
            if cuts:
                jobs.append({"source": source, "cuts": cuts})

    return jobs, skipped, stale


def encode(job: dict):
    cuts = job["cuts"]
    graph = ["[0:v]split=%d%s" % (len(cuts), "".join(f"[s{i}]" for i in range(len(cuts))))]
    for index, cut in enumerate(cuts):

        graph.append(f"[s{index}]trim=start_frame={cut['start']}:"
                     f"end_frame={cut['start'] + cut['count']},"
                     f"setpts=PTS-STARTPTS[v{index}]")
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(job["source"]), "-filter_complex", ";".join(graph)]
    for index, cut in enumerate(cuts):
        command += ["-map", f"[v{index}]", "-c:v", "libx264", "-crf", str(CLIP_CRF),
                    "-pix_fmt", "yuv420p", "-r", str(CLIP_FPS),
                    "-frames:v", str(cut["count"]), str(cut["out"])]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        reason = (result.stderr.strip().splitlines() or ["ffmpeg failed"])[-1]
        return job, [(cut["out"].name, reason) for cut in cuts]

    errors = []
    for cut in cuts:
        written = clip_frames(cut["out"])
        if written != cut["count"]:
            errors.append((cut["out"].name,
                           f"wrote {written} frames, expected {cut['count']}"))
    return job, errors


CLIP_FPS = 30
CLIP_CRF = 18


def encode_parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="FLAG3D root (default: %(default)s)")
    parser.add_argument("--video-dir", default="segments",
                        help="clip directory under processed/ (default: %(default)s)")
    parser.add_argument("--avatars", default="M001",
                        help=f"comma-separated subset of {','.join(AVATARS)}, or all "
                             "(default: %(default)s)")
    parser.add_argument("--persons", default="all",
                        help="comma-separated subset of P001..P010, or all "
                             "(default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=16,
                        help="concurrent ffmpeg processes (default: %(default)s)")
    parser.add_argument("--verify", action="store_true",
                        help="probe clips that already exist and re-cut any whose frame "
                             "count is short of its annotated range")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-cut clips that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be cut")
    args = parser.parse_args(argv)

    if args.avatars == "all":
        args.avatars = AVATARS
    else:
        args.avatars = tuple(a.strip() for a in args.avatars.split(",") if a.strip())
        unknown = [a for a in args.avatars if a not in AVATARS]
        if unknown:
            parser.error(f"unknown avatar(s) {unknown}, pick from {list(AVATARS)}")
    if args.persons == "all":
        args.persons = ()
    else:
        args.persons = tuple(p.strip() for p in args.persons.split(",") if p.strip())
        bad = [p for p in args.persons if not re.fullmatch(r"P\d{3}", p)]
        if bad:
            parser.error(f"persons look like P001..P010, got {bad}")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    return args


def encode_videos(argv=None) -> int:
    args = encode_parse_args(argv)
    out_dir = args.dataset_root / "processed" / args.video_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs, skipped, stale = plan_clips(args.dataset_root, out_dir, args)
    clips = sum(len(job["cuts"]) for job in jobs)
    print(f"{clips} clips from {len(jobs)} source videos into {out_dir}")
    print(f"  avatars {','.join(args.avatars)}, "
          f"persons {','.join(args.persons) if args.persons else 'all'}")
    if stale:
        print(f"  {len(stale)} existing clips are short of their range and get re-cut: "
              f"{', '.join(sorted(stale)[:5])}{' ...' if len(stale) > 5 else ''}")
    if skipped:
        print(f"  skipped {dict(sorted(skipped.items()))}")
    if args.dry_run or not jobs:
        return 0

    failures, done = [], 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for job, errors in pool.map(encode, jobs):
            failures.extend(errors)
            done += len(job["cuts"])
            if done % 2000 < len(job["cuts"]) or done == clips:
                print(f"  {done}/{clips} cut, {len(failures)} failed", flush=True)

    for name, error in failures[:20]:
        print(f"  FAILED {name}: {error}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} clips failed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
