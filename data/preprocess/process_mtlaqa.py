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
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

import numpy as np

from . import frame_budget
from .mtlaqa_dive_descriptions import DIVE_REQUIREMENTS
from .mtlaqa_score_rules import BANDS, comment_prompt, score_prompt, rubric

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "MTL-AQA"


SOURCE_FPS = 25.0


DEFAULT_FPS = frame_budget.sampling_fps(8.0)

RELATIVE_PREFIX = Path("data/MTL-AQA")

JUDGES = 7
COUNTED = slice(2, 5)


POSITIONS = {0: "free", 1: "pike", 2: "tuck"}
ROTATIONS = {0: "backward", 1: "forward", 2: "inward", 3: "reverse"}

TASK_LINE = "Score this dive."


COMMENT_TASK_LINE = "Comment on the execution of this dive."


def load_commentary(path: Path, source_name: str) -> dict:
    if not path.is_file():
        sys.exit(f"cleaned commentary not found: {path}; supply --commentary-file "
                 "or use --no-comment for score-only preprocessing")
    comments, seen, statuses = {}, set(), Counter()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for reference in row["sample_keys"]:
                if reference["source"] != source_name:
                    continue
                key = tuple(int(v) for v in reference["key"])
                if len(key) != 2 or key in seen:
                    raise ValueError(f"{path}:{line_no}: invalid or duplicate key {key}")
                seen.add(key)
                status = row["status"]
                if status not in {"usable", "needs_review", "empty"}:
                    raise ValueError(f"{path}:{line_no}: unknown status {status!r}")
                statuses[status] += 1
                if status != "usable":
                    continue
                text = row["cleaned"]
                if (row["usable_as_gt"] is not True or row.get("review_reason")
                        or not isinstance(text, str) or not text.strip()):
                    raise ValueError(f"{path}:{line_no}: invalid usable commentary")
                comments[key] = {"text": text, "index": row["index"]}
    if not comments:
        sys.exit(f"no usable commentary for source {source_name!r} in {path}")
    print(f"commentary: {len(comments)} usable for {source_name}; {dict(statuses)}")
    return comments


AWARD_REPAIRS = {(4, 100, 0): 7.5, (4, 49, 3): 4.0}

def load_pickle(path: Path):
    if not path.is_file():
        sys.exit(f"annotation not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def clip_name(key) -> str:
    video, dive = key
    return f"{int(video):02d}_{int(dive):02d}"


def parse_award(value, key, index: int, where: str) -> float:
    repair = AWARD_REPAIRS.get((key[0], key[1], index))
    valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
             and 0.0 <= value <= 10.0 and abs(value * 2 - round(value * 2)) <= 1e-9)
    if not valid:
        if repair is None:
            sys.exit(f"{where}: award cell {index + 1} is not on the 0-10 half-point "
                     f"scale: {value!r}")
        return repair
    return float(value)


def load_awards(root: Path) -> dict:
    try:
        import openpyxl
    except ImportError:
        sys.exit("openpyxl is required to read the per-judge awards: pip install openpyxl")

    annotation_dir = root / "MTL-AQA_dataset_release" / "Raw_Annotations"
    sheets = sorted(annotation_dir.glob("[0-9][0-9].xlsx"))
    if not sheets:
        sys.exit(f"no raw annotation workbooks under {annotation_dir}")

    awards, repaired = {}, 0
    for path in sheets:
        video = int(path.stem)
        rows = openpyxl.load_workbook(path, data_only=True).worksheets[0]
        for index, row in enumerate(rows.iter_rows(values_only=True)):
            if index == 0 or row[0] is None:


                continue
            key = (video, int(row[0]))
            where = f"{path.name} row {index + 1}"
            cells = list(row[12:12 + JUDGES])
            repaired += sum(1 for j in range(JUDGES) if (video, key[1], j) in AWARD_REPAIRS)
            awards[key] = [parse_award(c, key, j, where) for j, c in enumerate(cells)]
    print(f"read {len(awards)} judged dives from {annotation_dir}"
          + (f" ({repaired} malformed award cell(s) repaired)" if repaired else ""))
    return awards

def rule_score(difficulty: float, awards) -> float:
    if len(awards) != JUDGES:
        sys.exit(f"the strike rule needs {JUDGES} awards, got {len(awards)}")
    return round(difficulty * sum(sorted(awards)[COUNTED]), 2)


def band(award: float) -> str:
    for floor, name in BANDS:
        if award >= floor:
            return name
    sys.exit(f"award {award} is below the scale")


def halves(count: float) -> str:
    whole, half = divmod(round(count * 2), 2)
    if not half:
        return str(whole)
    return f"{whole} 1/2" if whole else "1/2"


def dive_phrase(annotation: dict) -> str:
    rotation = ROTATIONS.get(annotation["rotation_type"])
    position = POSITIONS.get(annotation["position"])
    if rotation is None or position is None:
        sys.exit(f"unknown rotation / position code in {annotation}")
    somersaults = annotation["ss_no"] / 2
    twists = annotation["tw_no"] / 2

    parts = ["armstand"] if annotation["armstand"] else []
    parts.append(rotation)
    if somersaults:
        parts.append(f"{halves(somersaults)} somersault"
                     + ("s" if somersaults != 1 else ""))
    else:

        parts.append("dive")
    if twists:
        parts.append(f"with {halves(twists)} twist" + ("s" if twists != 1 else ""))
    return f"{' '.join(parts)}, {position} position"


def dive_requirements(dive_name: str) -> str:
    try:
        return DIVE_REQUIREMENTS[dive_name]
    except KeyError:
        raise ValueError(
            f"no execution description for {dive_name!r}; "
            "add an entry to data/preprocess/mtlaqa_dive_descriptions.py"
        ) from None


def load_samples(root: Path, split_dir: Path, video_path_style: str):
    annotations = load_pickle(split_dir / "final_annotations_dict.pkl")
    awards = load_awards(root)
    video_dir = root / "videos"

    samples, notes = [], Counter()
    for key in sorted(annotations):
        annotation = annotations[key]
        judged = awards.get(tuple(int(v) for v in key))
        if judged is None:
            notes["no_award_row"] += 1
            continue
        clip = video_dir / f"{clip_name(key)}.mp4"
        if not clip.is_file():
            notes["missing_clip"] += 1
            continue
        official = round(float(annotation["final_score"]), 2)
        ranked = sorted(judged)
        samples.append({
            "key": tuple(int(v) for v in key),
            "source_collection": split_dir.name,
            "clip": clip_name(key),
            "dive": dive_phrase(annotation),
            "difficulty": float(annotation["difficulty"]),
            "awards": ranked,


            "award": ranked[JUDGES // 2],
            "final_score": official,
            "rule_score": rule_score(float(annotation["difficulty"]), judged),
            "video": str(clip) if video_path_style == "abs"
                     else str(RELATIVE_PREFIX / "videos" / clip.name),


            "abs_video": str(clip),
        })
    if not samples:
        sys.exit(f"no usable clips built from {split_dir}")
    consistent = sum(1 for s in samples if s["rule_score"] == s["final_score"])
    print(f"joined {len(samples)}/{len(annotations)} clips with awards and a clip under "
          f"{video_dir}")
    print(f"  the strike-and-sum rule reproduces final_score for {consistent}/"
          f"{len(samples)} clips ({len(samples) - consistent} contradict the release)")
    if notes:
        print(f"  notes {dict(sorted(notes.items()))}")
    return samples

def build_question(sample: dict, show_difficulty: bool, task_line: str) -> str:
    lines = [f"Dive: {sample['dive']}"]
    if show_difficulty:
        lines.append(f"Degree of difficulty: {sample['difficulty']:.1f}")
    lines.append("\n" + dive_requirements(sample["dive"]))
    return "<video>\n" + "\n".join(lines) + f"\n\n{task_line}"


def build_comment_record(sample: dict, comment: dict, fps: float | None,
                         show_difficulty: bool) -> dict:
    system, provenance = comment_prompt(sample["key"])
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": build_question(
                sample, show_difficulty, COMMENT_TASK_LINE)},
            {"role": "assistant", "content": comment["text"]},
        ],
        "videos": [sample["video"]],
        "chat_template_kwargs": {
            "fps": frame_budget.resolve_fps(sample["duration"], fps)},
        "clip": sample["clip"],
        "commentary_index": comment["index"],
        "commentary_status": "usable",
        **provenance,
    }


def build_record(sample: dict, fps: float | None, show_difficulty: bool,
                 target: str) -> dict:
    if sample.get("source_collection") != "MTL-AQA_split_0_data":
        raise ValueError("Event rules are audited for MTL-AQA_split_0_data only; audit other source collections before exporting scores")
    question = build_question(sample, show_difficulty, TASK_LINE)
    system, provenance = score_prompt(sample["key"], target)
    if target == "panel":
        answer = json.dumps({"awards": sample["awards"],
                             "final_score": sample["final_score"]},
                            separators=(",", ":"))
    else:
        answer = json.dumps({"execution_award": round(float(sample["award"]), 1)},
                            ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "videos": [sample["video"]],
        "chat_template_kwargs": {
            "fps": frame_budget.resolve_fps(sample["duration"], fps)},


        "clip": sample["clip"],
        "difficulty": sample["difficulty"],
        "gold_award": sample["award"],
        "awards": sample["awards"],
        "final_score": sample["final_score"],
        **provenance,
    }


def merge_records(score_rows: list, comment_rows: list) -> list:
    scores = {r["clip"]: r for r in score_rows}
    comments = {r["clip"]: r for r in comment_rows}
    if len(scores) != len(score_rows) or len(comments) != len(comment_rows):
        raise ValueError("Duplicate MTL-AQA clip within an annotation task")
    merged = []
    for clip in sorted(scores.keys() | comments.keys()):
        score, comment = scores.get(clip), comments.get(clip)
        base = score if score is not None else comment
        row = {**base, "messages": [dict(m) for m in base["messages"]]}
        answer, schema, tasks = {}, [], []
        if score is not None:
            answer.update(json.loads(score["messages"][-1]["content"]))
            schema.append('"execution_award": <number from 0.0 to 10.0 in increments of 0.5>')
            tasks.append("Give one overall execution award from 0.0 to 10.0 in increments of 0.5.")
        if comment is not None:
            if score is not None and any(score[k] != comment[k]
                                         for k in ("videos", "chat_template_kwargs")):
                raise ValueError(f"{clip}: paired annotations have different media or sampling")
            answer["comment"] = comment["messages"][-1]["content"]
            row.update({k: comment[k] for k in ("commentary_index", "commentary_status")})
            schema.append('"comment": "<broadcast commentary>"')
            tasks.append("Give a brief English commentary in one or a few sentences, speaking "
                         "naturally as during a sports broadcast. Use the judging criteria below "
                         "to discuss visible faults and merits, with a tone reflecting the performance.")
        role = "one execution judge" if score is not None else "a diving commentator"
        row["messages"][0]["content"] = "\n\n".join((
            f"You are {role} at a {row['apparatus']} diving competition.",
            " ".join(tasks), rubric(row).strip(),
            "Return only this JSON object and nothing else: {" + ", ".join(schema) + "}"))
        question = row["messages"][1]["content"].rsplit("\n\n", 1)[0]
        row["messages"][1]["content"] = question + "\n\nAssess the execution of this dive."
        row["messages"][-1]["content"] = json.dumps(answer, ensure_ascii=False)
        row["sample_id"] = "mtlaqa:" + clip
        row["answer_fields"] = list(answer)
        merged.append(row)
    return merged


def verify_rows(rows, samples, show_difficulty: bool, target: str):
    if len(rows) != len(samples):
        sys.exit(f"{len(rows)} rows against {len(samples)} clips")
    contradicting = []
    for row, sample in zip(rows, samples):
        text = row["messages"][-1]["content"]
        if target == "panel":
            answer = json.loads(text)
            awards, stated = answer["awards"], answer["final_score"]
        else:
            answer = json.loads(text)
            if set(answer) != {"execution_award"} or answer["execution_award"] != sample["award"]:
                sys.exit(f"{sample['clip']}: award did not survive the round trip")
            awards, stated = sample["awards"], sample["final_score"]
        if awards != sample["awards"]:
            sys.exit(f"{sample['clip']}: awards did not survive the round trip")
        if show_difficulty:
            line = f"Degree of difficulty: {sample['difficulty']:.1f}"
            if line not in row["messages"][1]["content"]:
                sys.exit(f"{sample['clip']}: {line!r} missing from the prompt")
        if rule_score(sample["difficulty"], awards) != stated:
            contradicting.append(sample["clip"])
        if Path(row["videos"][0]).stem != sample["clip"]:
            sys.exit(f"{sample['clip']}: row points at {row['videos'][0]}")
        if dive_requirements(sample["dive"]) not in row["messages"][1]["content"]:
            sys.exit(f"{sample['clip']}: expected execution missing from the prompt")
    return contradicting


def verify_comment_rows(rows, samples, comments) -> None:
    if len(rows) != len(samples):
        sys.exit(f"{len(rows)} commentary rows against {len(samples)} clips")
    for row, sample in zip(rows, samples):
        system = row["messages"][0]["content"]
        for marker in ("Use this judging scale:", "Execution criteria and deductions:",
                       "Overall assessment:", f"{row['apparatus']} diving competition"):
            if marker not in system:
                sys.exit(f"{sample['clip']}: {marker!r} missing from the commentary prompt")
        if row["messages"][-1]["content"] != comments[sample["key"]]["text"]:
            sys.exit(f"{sample['clip']}: commentary text did not survive the round trip")
        if dive_requirements(sample["dive"]) not in row["messages"][1]["content"]:
            sys.exit(f"{sample['clip']}: expected execution missing from the prompt")


def summarise(name: str, samples) -> None:
    scores = sorted(s["final_score"] for s in samples)
    difficulties = sorted(s["difficulty"] for s in samples)
    bands = Counter(band(s["award"]) for s in samples)
    awards = Counter(s["award"] for s in samples)


    gaps = [abs(round(s["difficulty"] * 3 * s["award"], 2) - s["final_score"])
            for s in samples]
    exact = sum(1 for g in gaps if g == 0)
    print(f"  {name}: {len(samples)} clips from "
          f"{len({s['key'][0] for s in samples})} competition videos")
    print(f"    final_score: min {scores[0]:g} / mean {sum(scores) / len(scores):.2f} / "
          f"max {scores[-1]:g}, {len(set(scores))} distinct values")
    print(f"    difficulty: min {difficulties[0]:g} / max {difficulties[-1]:g}; "
          f"R-l2 denominator max - min = {scores[-1] - scores[0]:g}")
    print(f"    gold award: {len(awards)} distinct values, "
          + ", ".join(f"{n} {bands[n] / len(samples):.1%}" for _, n in BANDS if bands[n]))
    print(f"    difficulty x 3 x award rebuilds final_score exactly for {exact}/"
          f"{len(samples)} clips, mean gap {sum(gaps) / len(gaps):.2f} points "
          f"(R-l2 {sum(gaps) / len(gaps) / (scores[-1] - scores[0]):.4f})")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="MTL-AQA root (default: %(default)s)")
    parser.add_argument("--split-dir", default="Ready_2_Use/MTL-AQA_split_0_data",
                        help="directory under MTL-AQA_dataset_release/ holding "
                             "final_annotations_dict.pkl and the split lists "
                             "(default: %(default)s)")
    parser.add_argument("--video-path-style", choices=["abs", "relative"], default="abs",
                        help=f"absolute paths (default) or {RELATIVE_PREFIX}/... relative")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "sports",
                        help="where to write the metadata (default: %(default)s)")
    parser.add_argument("--prefix", default="mtlaqa",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--fps", type=float, default=None,
                        help=f"pin one sampling fps for every row instead of "
                             f"data/preprocess/frame_budget.py's duration ladder; source is "
                             f"{SOURCE_FPS:g} fps (default: the ladder)")
    parser.add_argument("--jobs", type=int, default=32,
                        help="concurrent ffprobe calls for the clip durations "
                             "(default: %(default)s)")
    parser.add_argument("--target", choices=["award", "panel"], default="award",
                        help="what the assistant answers: one execution award (default) "
                             "or the seven-award JSON of the whole panel, which is the "
                             "only form that reproduces the official final score exactly")
    parser.add_argument("--inconsistent", choices=["drop-train", "drop", "keep"],
                        default=None,
                        help="what to do with the clips whose released final_score "
                             "contradicts the awards: drop them from train only, from "
                             "both splits, or keep them (default: keep with --target "
                             "award, where the clash is metadata-only, drop-train with "
                             "--target panel, where it sits in the answer)")
    parser.add_argument("--no-difficulty", action="store_true",
                        help="leave the degree of difficulty out of the prompt; the "
                             "final score is then not derivable from what the model sees")
    parser.add_argument("--commentary-file", type=Path, default=None,
                        help="cleaned commentary JSONL with sample_keys and quality flags "
                             "(default: <dataset-root>/commentary_cleaned/cleaning_pairs.jsonl)")
    parser.add_argument("--commentary-source", default=None,
                        help="sample_keys source to match; default: --split-dir relative "
                             "to Ready_2_Use (set explicitly for a custom split directory)")
    parser.add_argument("--no-comment", action="store_true",
                        help="export only the score/panel task without loading commentary")
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    args = parser.parse_args(argv)
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.inconsistent is None:
        args.inconsistent = "drop-train" if args.target == "panel" else "keep"
    return args


def write_rows(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.no_comment:
        commentary = args.commentary_file or (args.dataset_root / 'commentary_cleaned' / 'cleaning_pairs.jsonl')
        if not commentary.is_file():
            clean_commentary(['--source', str(args.dataset_root / 'MTL-AQA_dataset_release' / 'Ready_2_Use'),
                              '--output', str(args.dataset_root / 'commentary_cleaned')])
    split_dir = args.dataset_root / "MTL-AQA_dataset_release" / args.split_dir
    comments = None
    if not args.no_comment:
        source_name = args.commentary_source
        if source_name is None:
            ready = args.dataset_root / "MTL-AQA_dataset_release" / "Ready_2_Use"
            try:
                source_name = split_dir.resolve().relative_to(ready.resolve()).as_posix()
            except ValueError:
                sys.exit("custom split directory requires --commentary-source")
        comments = load_commentary(
            args.commentary_file or args.dataset_root / "commentary_cleaned/cleaning_pairs.jsonl",
            source_name)
    samples = load_samples(args.dataset_root, split_dir, args.video_path_style)
    by_key = {s["key"]: s for s in samples}
    durations = frame_budget.probe_durations([s["abs_video"] for s in samples],
                                            workers=args.jobs)
    for sample in samples:
        sample["duration"] = durations[sample["abs_video"]]
    print(f"target: {args.target}; sampling "
          + (f"pinned at {args.fps:g} fps" if args.fps else
             "per clip, data/preprocess/frame_budget.py's duration ladder")
          + f" (source {SOURCE_FPS:g} fps)")
    if args.no_difficulty:
        print("difficulty is withheld from the prompt")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test"):
        listed = load_pickle(split_dir / f"{name}_split_0.pkl")
        subset, dropped = [], Counter()
        for key in listed:
            sample = by_key.get(tuple(int(v) for v in key))
            if sample is None:
                dropped["no_sample"] += 1
                continue
            if sample["rule_score"] != sample["final_score"] and (
                    args.inconsistent == "drop"
                    or (args.inconsistent == "drop-train" and name == "train")):
                dropped["contradicts_the_rule"] += 1
                continue
            subset.append(sample)
        subset.sort(key=lambda s: s["key"])
        print(f"  {name}: {len(listed)} clips listed in {name}_split_0.pkl"
              + (f", dropped {dict(sorted(dropped.items()))}" if dropped else ""))
        if subset:
            summarise(name, subset)

        rows = [build_record(s, args.fps, not args.no_difficulty, args.target)
                for s in subset]
        print(f"    sampling: {frame_budget.fps_histogram(rows)}")
        contradicting = verify_rows(rows, subset, not args.no_difficulty, args.target)
        print(f"    the rule rebuilt from the awards matches final_score for "
              f"{len(rows) - len(contradicting)}/{len(rows)} rows")
        if contradicting:
            print(f"    kept with a contradicting release score, score them on "
                  f"final_score: {', '.join(contradicting)}")

        comment_rows = []
        if comments is not None:

            comment_samples = sorted(
                (by_key[key] for entry in listed
                 if (key := tuple(int(v) for v in entry)) in by_key and key in comments),
                key=lambda s: s["key"])
            comment_rows = [build_comment_record(
                s, comments[s["key"]], args.fps, not args.no_difficulty)
                for s in comment_samples]
            verify_comment_rows(comment_rows, comment_samples, comments)
            print(f"    commentary: {len(comment_rows)}/{len(listed)} clips; "
                  f"excluded {len(listed) - len(comment_rows)} without usable text/video")
        if args.target == "award":
            rows = merge_records(rows, comment_rows)
            out_path = args.out_dir / f"{args.prefix}_{name}.jsonl"
        else:
            out_path = args.out_dir / f"{args.prefix}_panel_{name}.jsonl"
            if comments is not None:
                write_rows(args.out_dir / f"{args.prefix}_comment_{name}.jsonl", comment_rows)
        if not rows:
            sys.exit(f"{name} split has no usable annotations")
        write_rows(out_path, rows)
        print(f"    -> {out_path}: {len(rows)} rows")
    if args.target == "award":
        for task in ("score", "comment"):
            for split in ("train", "test"):
                (args.out_dir / f"{args.prefix}_{task}_{split}.jsonl").unlink(missing_ok=True)
    return 0


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == 'scalar':
            return np._core.multiarray.scalar
        if module == 'numpy' and name == 'dtype':
            return np.dtype
        raise ValueError(f'Unexpected pickle class: {module}.{name}')


def read_pickle(path):
    with path.open('rb') as stream:
        return RestrictedUnpickler(stream, encoding='latin1').load()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))


def write_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(value, protocol=4))


def canonical_key(key):
    return tuple(int(v) for v in key)


def source_text(tokens):
    return ' '.join(t for t in tokens if t not in ('<sos>', '<eos>'))


def clean_commentary(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source', type=Path, default=workspace() / 'datasets' / 'MTL-AQA' / 'MTL-AQA_dataset_release' / 'Ready_2_Use')
    parser.add_argument('--annotations', type=Path,
                        default=workspace() / 'data' / 'annotations' / 'mtlaqa_commentary.json')
    parser.add_argument('--output', type=Path, default=workspace() / 'datasets' / 'MTL-AQA' / 'commentary_cleaned')
    args = parser.parse_args(argv)
    source, output, ann = args.source.resolve(), args.output.resolve(), args.annotations.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Export must be separate from the source tree')
    bundle = json.loads(ann.read_text())
    inventory = bundle['sample_keys']
    checksums = bundle['source_checksums']
    expected = {r['index'] for r in inventory}
    assert expected == set(range(len(inventory)))
    edits = {row[0]: row for row in bundle['edits']}
    assert set(edits) == expected, 'Missing or unexpected manual edits'


    originals = {}
    actual_files = {str(p.relative_to(source)) for p in source.rglob('final_captions_dict.pkl')}
    assert actual_files == set(checksums), 'Source file inventory changed'
    for relative, metadata in checksums.items():
        path = source / relative
        assert sha256(path) == metadata['sha256'], f'Source changed: {path}'
        raw = read_pickle(path)
        assert len(raw) == metadata['count']
        originals[str(Path(relative).parent)] = raw

    pairs, by_source = [], {}
    for record in inventory:
        if 'original' not in record:
            reference = record['sample_keys'][0]
            tokens = originals[reference['source']][canonical_key(reference['key'])]
            record = dict(record, original=source_text(tokens))
        row = edits[record['index']]
        cleaned = row[1]
        assert isinstance(cleaned, str) and cleaned == cleaned.strip()
        assert not re.search(r'[\u2013\u2014]|<(?:sos|eos)>|\[(?:Applause|Music)\]', cleaned)
        reason = row[2] if len(row) == 3 else ''
        if not cleaned:
            reason = reason or '删除背景、分数或无法辨认的转录片段后，无可用动作质量反馈。'
        status = 'empty' if not cleaned else ('needs_review' if reason else 'usable')
        pair = dict(record, cleaned=cleaned, status=status,
                    usable_as_gt=status == 'usable', review_reason=reason)
        pairs.append(pair)
        for reference in record['sample_keys']:
            mapping = by_source.setdefault(reference['source'], {})
            key = canonical_key(reference['key'])
            assert key not in mapping
            mapping[key] = pair

    for relative, raw in originals.items():
        mapping = by_source[relative]
        assert {canonical_key(k) for k in raw} == set(mapping)
        for key, tokens in raw.items():
            assert source_text(tokens) == mapping[canonical_key(key)]['original'], key

    per_source = {}
    for relative, raw in originals.items():
        src_dir, dst_dir = source / relative, output / 'Ready_2_Use' / relative
        mapping = by_source[relative]
        full, usable, rows = {}, {}, []
        for key in raw:
            pair = mapping[canonical_key(key)]
            tokens = ['<sos>', *pair['cleaned'].split(), '<eos>']
            full[key] = tokens
            if pair['usable_as_gt']:
                usable[key] = tokens
            rows.append(dict(key=list(canonical_key(key)), index=pair['index'],
                             original=pair['original'], cleaned=pair['cleaned'],
                             status=pair['status'], usable_as_gt=pair['usable_as_gt'],
                             review_reason=pair['review_reason']))
        write_pickle(dst_dir / 'final_captions_dict.pkl', full)
        write_pickle(dst_dir / 'final_captions_usable.pkl', usable)
        write_jsonl(dst_dir / 'captions.jsonl', rows)
        write_jsonl(dst_dir / 'captions_usable.jsonl', [r for r in rows if r['usable_as_gt']])
        assert read_pickle(dst_dir / 'final_captions_dict.pkl') == full
        assert read_pickle(dst_dir / 'final_captions_usable.pkl') == usable

        vocab = json.loads((src_dir / 'vocab.json').read_text())
        old_ids = dict(vocab['word_to_ix'])
        next_id = max(int(v) for v in old_ids.values()) + 1
        for tokens in full.values():
            for token in tokens:
                if token not in vocab['word_to_ix']:
                    vocab['word_to_ix'][token] = next_id
                    vocab['ix_to_word'][str(next_id)] = token
                    next_id += 1
        assert all(vocab['word_to_ix'][w] == idx for w, idx in old_ids.items())
        for tokens in full.values():
            for token in tokens:
                assert vocab['ix_to_word'][str(vocab['word_to_ix'][token])] == token
        write_json(dst_dir / 'vocab.json', vocab)

        split_stats = {}
        for path in sorted(src_dir.glob('*split*.pkl')):
            split = read_pickle(path)
            keys = [canonical_key(k) for k in split]
            assert len(keys) == len(set(keys)) and set(keys) <= set(mapping)
            selected = [k for k in split if mapping[canonical_key(k)]['usable_as_gt']]
            (dst_dir / path.name).write_bytes(path.read_bytes())
            write_pickle(dst_dir / (path.stem + '_usable.pkl'), selected)
            split_stats[path.name] = dict(original=len(split), usable=len(selected),
                                         excluded=len(split) - len(selected))
        if (src_dir / 'final_annotations_dict.pkl').exists():
            (dst_dir / 'final_annotations_dict.pkl').write_bytes(
                (src_dir / 'final_annotations_dict.pkl').read_bytes())
        per_source[relative] = dict(total=len(full), usable=len(usable),
                                   statuses=dict(Counter(r['status'] for r in rows)),
                                   added_vocab_tokens=len(vocab['word_to_ix']) - len(old_ids),
                                   splits=split_stats)

    write_jsonl(output / 'cleaning_pairs.jsonl', pairs)
    write_jsonl(output / 'captions_usable.jsonl', [r for r in pairs if r['usable_as_gt']])
    write_jsonl(output / 'needs_review.jsonl', [r for r in pairs if r['status'] == 'needs_review'])
    write_jsonl(output / 'empty_captions.jsonl', [r for r in pairs if r['status'] == 'empty'])
    summary = dict(method='Assistant-authored individual edits; local export only; no API calls',
                   total_unique=len(pairs), statuses=dict(Counter(r['status'] for r in pairs)),
                   source_files=len(originals), source_occurrences=sum(len(v) for v in originals.values()),
                   original_words=sum(len(r['original'].split()) for r in pairs),
                   cleaned_words=sum(len(r['cleaned'].split()) for r in pairs),
                   source_checksums_verified=True, key_coverage_verified=True,
                   pickle_roundtrip_verified=True, vocabulary_coverage_verified=True,
                   per_source=per_source)

    for relative, metadata in checksums.items():
        assert sha256(source / relative) == metadata['sha256']
    write_json(output / 'summary.json', summary)
    files = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (output / 'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(output)}\n' for p in files))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
