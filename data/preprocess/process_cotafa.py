#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

from . import frame_budget
from . import loss_scale

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "CoT-AFA"
STEP_INSTRUCTIONS = workspace() / "data" / "annotations" / "cotafa_step_instructions.json"


DEFAULT_FPS = frame_budget.sampling_fps(8.0)

TASKS = ("cot", "aqa")


N_VIDEOS, N_CATEGORIES = 3392, 141
N_STANDARD, N_NON_STANDARD = 2242, 1150


STANDARD, NON_STANDARD = "standard", "non-standard"
LABEL_TO_STANDARD = {0: True, 1: False, 2: False}


FITNESS_FAMILIES = {"徒手": "bodyweight", "器械": "equipment"}
FITNESS, MARTIAL_ARTS = "fitness", "martial arts"


MODE_DESTINATION = {FITNESS: ("fitness", "fitness"), MARTIAL_ARTS: ("sports", "martial")}


DESCRIPTOR = {FITNESS: "a {} exercise", MARTIAL_ARTS: "a {} technique"}


WORKOUT_TYPE_EN = {

    "单杠": "pull-up bar", "双杠": "parallel bars", "杠铃": "barbell", "哑铃": "dumbbell",
    "划船机": "rowing machine", "拉背机": "lat pulldown machine", "拉力器": "cable machine",
    "史密斯机": "Smith machine", "俯卧撑": "push-up", "健身操": "aerobics",
    "普拉提": "Pilates", "拉伸": "stretching", "热身": "warm-up", "瑜伽": "yoga",

    "长拳": "Changquan (long fist)", "南拳": "Nanquan (southern fist)",
    "少林拳": "Shaolin quan", "太极拳": "Tai chi quan", "翻子拳": "Fanziquan",
    "刀术": "broadsword (dao)", "南刀": "southern broadsword", "剑术": "straight sword (jian)",
    "太极剑": "Tai chi sword", "棍术": "staff (gun)", "南棍": "southern staff",
    "双截棍": "nunchaku", "枪术": "spear (qiang)", "九节鞭": "nine-section whip",
    "流星锤": "meteor hammer",
}


VIEW_EN = {"正面": "front", "背面": "back", "侧面": "side",
           "侧面（左）": "left side", "侧面(左)": "left side",
           "侧面（右）": "right side", "侧面(右)": "right side"}


PAREN = re.compile(r"[（(][^）)]*[）)]")


VERDICT_RE = re.compile(
    r"^The\s+.+?\s+action in this video is\s+(?:classified as\s+)?"
    r"['\u2018\u2019\"]?(?P<negated>not\s+)?(?P<verdict>good|bad)\b",
    re.IGNORECASE | re.DOTALL)


COT_INTRO = (
    "You assess whether the {action} ({descriptor}) in the given video was performed "
    "in accordance with the following action standard."
)

AQA_INTRO = (
    "You assess whether the {action} ({descriptor}) in the given video was performed "
    "in accordance with the following action standard."
)

COT_ANSWER_FORMAT = (
    "Open with one sentence naming the action and stating whether the execution is good "
    "or bad, then explain the judgement in a few sentences: what the performer does, "
    "which reference steps the execution does or does not follow, and how to correct any "
    "error. Respond in a single paragraph."
)

AQA_ANSWER_FORMAT = (
    'Return only this JSON object and nothing else: '
    f'{{"form_assessment": "{STANDARD}" or "{NON_STANDARD}"}}.'
)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_categories(root: Path) -> list:
    raw = read_json(STEP_INSTRUCTIONS)
    if len(raw) != N_CATEGORIES:
        sys.exit(f"expected {N_CATEGORIES} categories, got {len(raw)}")
    categories = []
    for name, steps in raw.items():
        if len(steps) != 6:
            sys.exit(f"category {name!r}: expected 6 instruction lines, got {len(steps)}")
        categories.append({"name": name.strip(),
                           "steps": [step.strip() for step in steps]})
    return categories


def check_categories(table, categories: list) -> None:
    problems = []
    for index, rows in sorted(table.items()):
        expected = categories[index]["name"]
        for column, label in (("category_cn", "具体类"), ("family", "大类"),
                              ("modality", "小类")):
            values = {row[column] for row in rows if row[column]}
            if len(values) > 1:
                problems.append(f"{index:02d}: {label} is ambiguous, {sorted(values)}")
            if column != "category_cn" and not values:
                problems.append(f"{index:02d}: {label} is blank for every row")
        named = {PAREN.sub("", row["category_cn"]).strip()
                 for row in rows if row["category_cn"]}
        off = [name for name in named if name.casefold() != expected.casefold()]
        if off:
            problems.append(f"{index:02d}: 具体类 {off} vs json key {expected!r}")
    if problems:
        sys.exit("GYM88.xlsx does not line up with the category list:\n  "
                 + "\n  ".join(problems))


def cell(value) -> str:
    return "" if value is None else str(value).strip()


def load_table(root: Path) -> dict:
    try:
        import openpyxl
    except ImportError:
        sys.exit("openpyxl is required to read GYM88.xlsx: pip install openpyxl")

    path = root / "dataset" / "GYM88.xlsx"
    if not path.exists():
        sys.exit(f"annotation sheet not found: {path}")
    table, seen = defaultdict(list), set()
    sheet = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    for index, row in enumerate(sheet.iter_rows(values_only=True)):
        if index == 0:
            continue
        video_id = cell(row[0])
        if not video_id:
            continue
        if video_id in seen:
            sys.exit(f"{path.name} row {index + 1}: duplicate video id {video_id}")
        seen.add(video_id)
        label = row[5]
        if label not in LABEL_TO_STANDARD:
            sys.exit(f"{path.name} row {index + 1}: 好坏 is {label!r}, expected "
                     f"one of {sorted(LABEL_TO_STANDARD)}")
        table[int(video_id.split("_")[0])].append({
            "video_id": video_id,
            "family": cell(row[1]),
            "modality": cell(row[2]),
            "category_cn": cell(row[3]),
            "label": label,
            "view": cell(row[6]),
        })
    if len(seen) != N_VIDEOS:
        sys.exit(f"{path.name}: expected {N_VIDEOS} videos, read {len(seen)}")
    if sorted(table) != list(range(N_CATEGORIES)):
        sys.exit(f"{path.name}: category prefixes are not 0..{N_CATEGORIES - 1}")
    return table


def load_captions(root: Path) -> tuple:
    comment_dir = root / "dataset" / "comment"
    if not comment_dir.is_dir():
        sys.exit(f"comment directory not found: {comment_dir}")
    comments = {}
    for path in sorted(comment_dir.glob("*_with_label.txt")):
        text = path.read_text(encoding="utf-8", errors="strict").strip()
        if text:
            comments[path.name[:-len("_with_label.txt")]] = text

    coco_path = root / "test_caption_coco_format_coco_format.json"
    coco = {}
    for annotation in read_json(coco_path)["annotations"]:
        video_id, text = annotation["image_id"], annotation["caption"].strip()
        if not text:
            sys.exit(f"{coco_path.name}: empty caption for {video_id}")
        coco[video_id] = text
    return comments, coco


def parse_verdict(text: str, video_id: str) -> str:
    match = VERDICT_RE.match(text)
    if not match:
        sys.exit(f"{video_id}: caption does not open with a verdict: {text[:120]!r}")
    verdict = match.group("verdict").lower()
    if match.group("negated"):
        verdict = "bad" if verdict == "good" else "good"
    return verdict


def build_units(root: Path, table: dict, categories: list, comments: dict,
                coco: dict) -> list:
    video_dir = root / "AQA_data" / "videos"
    clips = {path.stem: path for path in video_dir.iterdir() if path.is_file()}
    unknown = set(coco) - {row["video_id"] for rows in table.values() for row in rows}
    if unknown:
        sys.exit(f"coco captions for {len(unknown)} ids absent from GYM88.xlsx: "
                 f"{sorted(unknown)[:5]}")

    units = []
    for index, rows in sorted(table.items()):
        category = categories[index]
        family = next(row["family"] for row in rows if row["family"])
        modality = next(row["modality"] for row in rows if row["modality"])
        mode = FITNESS if family in FITNESS_FAMILIES else MARTIAL_ARTS
        workout_type = modality if mode == FITNESS else family
        if workout_type not in WORKOUT_TYPE_EN:
            sys.exit(f"{index:02d}: no English name for workout type {workout_type!r}")
        for row in rows:
            video_id = row["video_id"]
            if video_id not in clips:
                sys.exit(f"{video_id}: no clip under {video_dir}")
            if row["view"] and row["view"] not in VIEW_EN:
                sys.exit(f"{video_id}: unrecognised 角度 {row['view']!r}")
            caption = coco.get(video_id) or comments.get(video_id)
            units.append({
                "video_id": video_id,
                "video": clips[video_id],
                "category_index": index,
                "category": category["name"],
                "steps": category["steps"],
                "workout_mode": mode,
                "workout_type": WORKOUT_TYPE_EN[workout_type],
                "view": VIEW_EN.get(row["view"]),
                "is_standard": LABEL_TO_STANDARD[row["label"]],
                "caption": caption,
                "caption_source": "coco" if video_id in coco else
                                  ("comment" if caption else None),
                "verdict": parse_verdict(caption, video_id) if caption else None,
                "is_test": video_id in coco,
            })
    standard = sum(1 for unit in units if unit["is_standard"])
    if (standard, len(units) - standard) != (N_STANDARD, N_NON_STANDARD):
        sys.exit(f"好坏 gives {standard} standard / {len(units) - standard} "
                 f"non-standard, README says {N_STANDARD} / {N_NON_STANDARD}")
    return units


def build_system(unit: dict, intro: str, answer_format: str) -> str:
    descriptor = DESCRIPTOR[unit["workout_mode"]].format(unit["workout_type"])
    sections = [
        intro.format(action=unit["category"].lower(), descriptor=descriptor),
        "\n".join(unit["steps"]),
        answer_format,
    ]
    return "\n\n".join(sections)


def build_question(unit: dict) -> str:
    return "<video>"


def wrap(unit: dict, fps: float | None, system: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": build_question(unit)},
            {"role": "assistant", "content": answer},
        ],
        "videos": [str(unit["video"])],
        "chat_template_kwargs": {
            "fps": frame_budget.resolve_fps(unit["duration"], fps)},
        "video_id": unit["video_id"],
        "category_index": unit["category_index"],
        "category": unit["category"],
        "workout_mode": unit["workout_mode"],
        "workout_type": unit["workout_type"],
        "view": unit["view"],
        "gold_standard": unit["is_standard"],
    }


def build_cot_record(unit: dict, fps: float) -> dict:
    record = wrap(unit, fps, build_system(unit, COT_INTRO, COT_ANSWER_FORMAT),
                  unit["caption"])
    record["caption_source"] = unit["caption_source"]
    record["gold_verdict"] = unit["verdict"]

    record["verdict_matches_label"] = (unit["verdict"] == "good") == unit["is_standard"]
    return record


def build_aqa_record(unit: dict, fps: float) -> dict:
    return wrap(unit, fps, build_system(unit, AQA_INTRO, AQA_ANSWER_FORMAT),
                json.dumps({"form_assessment": (
                    STANDARD if unit["is_standard"] else NON_STANDARD)},
                    ensure_ascii=False))


def report(task: str, name: str, units: list) -> set:
    categories = {unit["category_index"] for unit in units}
    modes = Counter(unit["workout_mode"] for unit in units)
    labels = Counter(unit["is_standard"] for unit in units)
    majority = max(labels.values()) / len(units)
    print(f"  {task} {name}: {len(units)} rows over {len(categories)} categories, "
          f"{dict(sorted(modes.items()))}")
    print(f"    {STANDARD} {labels[True]} / {NON_STANDARD} {labels[False]}, "
          f"majority baseline {majority:.3f}")
    if task == "cot":
        sources = Counter(unit["caption_source"] for unit in units)
        verdicts = Counter(unit["verdict"] for unit in units)
        conflicts = sum(1 for unit in units
                        if (unit["verdict"] == "good") != unit["is_standard"])
        words = sorted(len(unit["caption"].split()) for unit in units)
        print(f"    captions {dict(sorted(sources.items()))}, verdict "
              f"{dict(sorted(verdicts.items()))}, {conflicts} disagree with 好坏 "
              f"({conflicts / len(units):.1%})")
        print(f"    caption words: p50 {words[len(words) // 2]}, "
              f"p99 {words[int(len(words) * 0.99)]}, max {words[-1]}")
    return categories


def write_rows(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"    -> {path}")


def emit(task: str, units: list, out_dir: Path, prefix: str, fps: float | None) -> None:
    build = build_cot_record if task == "cot" else build_aqa_record
    if task == "cot":
        units = [unit for unit in units if unit["caption"]]
    cell_of = lambda unit: (unit["workout_mode"], unit["category_index"])
    class_of = lambda unit: unit["is_standard"]
    seen = {}
    for name in ("train", "test"):
        part = sorted((unit for unit in units if unit["is_test"] == (name == "test")),
                      key=lambda unit: unit["video_id"])
        if not part:
            sys.exit(f"{task} {name} split is empty")
        seen[name] = report(task, name, part)
        weights = {}
        if task == "aqa" and name == "train":
            fitness = [unit for unit in part if unit["workout_mode"] == FITNESS]
            weights = loss_scale.balanced_weights(fitness, cell_of, class_of)
            print(f"    loss_scale ({FITNESS}): "
                  + loss_scale.summarise(fitness, cell_of, class_of, weights))
        rows = [loss_scale.attach(build(unit, fps),
                                 weights.get((cell_of(unit), class_of(unit)), 1.0))
                for unit in part]
        print(f"    sampling: {frame_budget.fps_histogram(rows)}")
        for mode, (folder, tag) in MODE_DESTINATION.items():
            subset = [row for row in rows if row["workout_mode"] == mode]
            if not subset:
                sys.exit(f"{task} {name} split has no {mode} rows")
            destination = out_dir if out_dir.name == folder else out_dir.parent / folder
            destination.mkdir(parents=True, exist_ok=True)
            write_rows(destination / f"{prefix}_{tag}_{task}_{name}.jsonl", subset)
    unseen = seen["test"] - seen["train"]
    if unseen:

        print(f"    note: {len(unseen)} of {len(seen['test'])} test categories are "
              f"absent from {task} train")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="CoT-AFA root (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "fitness",
                        help="where to write the metadata (default: %(default)s)")
    parser.add_argument("--prefix", default="cotafa",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--tasks", default=",".join(TASKS),
                        help=f"comma-separated subset of {'/'.join(TASKS)} to emit "
                             "(default: %(default)s)")
    parser.add_argument("--fps", type=float, default=None,
                        help="pin one sampling fps for every row instead of "
                             "data/preprocess/frame_budget.py's duration ladder (default: "
                             "the ladder)")
    parser.add_argument("--jobs", type=int, default=32,
                        help="concurrent ffprobe calls for the clip durations "
                             "(default: %(default)s)")
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    args = parser.parse_args(argv)
    args.tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = [task for task in args.tasks if task not in TASKS]
    if unknown:
        parser.error(f"unknown task(s) {unknown}, pick from {list(TASKS)}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    categories = load_categories(args.dataset_root)
    table = load_table(args.dataset_root)
    check_categories(table, categories)
    comments, coco = load_captions(args.dataset_root)
    units = build_units(args.dataset_root, table, categories, comments, coco)
    captioned = [unit for unit in units if unit["caption"]]
    print(f"loaded {len(units)} videos over {len(categories)} categories from "
          f"{args.dataset_root}")
    print(f"captions: {len(comments)} non-empty comment files + {len(coco)} coco test "
          f"references = {len(captioned)} videos "
          f"({len(set(comments) & set(coco))} in both, coco wins)")
    print(f"split: the released {len(coco)}-video test set, "
          f"{len(units) - len(coco)} train")

    durations = frame_budget.probe_durations([str(u["video"]) for u in units],
                                             workers=args.jobs)
    for unit in units:
        unit["duration"] = durations[str(unit["video"])]
    lengths = sorted(u["duration"] for u in units)
    print(f"clip seconds: min {lengths[0]:.1f} median {lengths[len(lengths) // 2]:.1f} "
          f"max {lengths[-1]:.1f}; sampling "
          + (f"pinned at {args.fps:g} fps" if args.fps else
             "per clip, data/preprocess/frame_budget.py's duration ladder"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        emit(task, units, args.out_dir, args.prefix, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
