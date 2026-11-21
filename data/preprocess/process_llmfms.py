#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "LLM-FMS"

SCORES = (1, 2, 3)


SYSTEM_INTRO = (
    "You assess the quality of the Functional Movement Screen (FMS) movement "
    "\"{movement}\" from a single keyframe. Judge every item below, then give the "
    "overall score."
)

ANSWER_FORMAT = (
    "Return only this JSON object and nothing else:\n"
    "{template}"
)


M08_ANSWER_FIX = {
    "一个手掌长度及以内": "Within the length of a palm",
    "一个手掌长度 ~ 1.5个手掌长度(包括)": "One palm length ~ 1.5 palm length",
    "＞ 1.5个手掌长度": "≥ 1.5 palm length",
}


STANDARD, NON_STANDARD = "standard", "non-standard"

DEEP_SQUAT_ITEMS = {
    "r01": ("The angle between the trunk (the line from the midpoint of the hips to the "
            "midpoint of the shoulders) and the tibia (the line from the knee to the "
            "ankle).",
            {"Parallel": f"{STANDARD}, 0-5 degrees",
             "Not parallel": f"{NON_STANDARD}, over 5 degrees"}),
    "r02": ("The height of the hip joint relative to the knee joint along the vertical "
            "axis.",
            {"Higher": f"{NON_STANDARD}, the hip stays above the knee",
             "Equal": f"{STANDARD}, level with the knee",
             "Lower": f"{STANDARD}, below the knee"}),
    "r03": ("The position of the wrist joint relative to the knee joint along the "
            "horizontal axis.",
            {"The Left": f"{NON_STANDARD}, the wrist is left of the knee",
             "Equal": f"{STANDARD}, level with the knee",
             "The right": f"{STANDARD}, right of the knee"}),
}

HURDLE_STEP_ITEMS = {
    "r01": ("The angle between the line connecting the left and right wrist joints and "
            "the horizontal axis.",
            {"Parallel": f"{STANDARD}, 0-5 degrees",
             "Not parallel": f"{NON_STANDARD}, over 5 degrees"}),
    "r02": ("The angle between the support leg (the line from its hip joint to its ankle "
            "joint) and the horizontal plane.",
            {"Vertical": f"{STANDARD}, 85-90 degrees",
             "Not vertical": f"{NON_STANDARD}, under 85 degrees"}),
    "r03": ("The angle between the shank of the stepping leg (the line from its knee "
            "joint to its ankle joint) and the horizontal plane.",
            {"Vertical": f"{STANDARD}, 85-90 degrees",
             "Not vertical": f"{NON_STANDARD}, under 85 degrees"}),
    "r04": ("The height of the stepping ankle joint relative to the knee joint of the "
            "support leg along the vertical axis.",
            {"Higher": f"{STANDARD}, above the support knee",
             "Equal": f"{NON_STANDARD}, level with the support knee",
             "Lower": f"{NON_STANDARD}, below the support knee"}),
}

INLINE_LUNGE_ITEMS = {
    "r01": ("The angle between the trunk (the line from the midpoint of the hips to the "
            "midpoint of the shoulders) and the vertical axis.",
            {"Parallel": f"{STANDARD}, 0-5 degrees",
             "Not parallel": f"{NON_STANDARD}, over 5 degrees"}),
    "r02": ("The distance between the knee joint of the back leg and the ankle joint of "
            "the front leg.",
            {"Near": f"{STANDARD}, 0-2 cm",
             "Far": f"{NON_STANDARD}, over 2 cm"}),
    "r03": ("The height of the back-leg knee joint relative to the front-leg ankle joint "
            "along the vertical axis.",
            {"Near": f"{STANDARD}, within 2 cm of the front ankle",
             "Far": f"{NON_STANDARD}, more than 2 cm above the front ankle"}),
}

SHOULDER_ITEMS = {
    "r01": ("The distance between the left and right wrist joints, one hand reaching over "
            "the shoulder and the other up from the lower back.",
            {"Within the length of a palm": "one palm length or less",
             "One palm length ~ 1.5 palm length": "over one palm, up to 1.5 palms",
             "≥ 1.5 palm length": "over 1.5 palm lengths"}),
}

ASLR_ITEMS = {
    "r01": ("Where the raised ankle joint falls along the ground leg, read off as a "
            "landmark on that leg.",
            {"Thigh midpoint and above (including thigh midpoint)":
                "at or past the midpoint of the ground-leg thigh, toward the head",
             "Knee point (included) ~ thigh midpoint (not included)":
                "between the ground-leg knee and the thigh midpoint",
             "Knee point and below (excluding knee point)":
                "below the ground-leg knee, toward the ankle"}),
}

PUSH_UP_ITEMS = {
    "r01": ("The position of the wrist joint relative to the nose tip and the shoulder "
            "joint along the horizontal axis.",
            {"Above the nose (excluding)":
                f"{STANDARD}, at or past the nose tip, toward the top of the head",
             "Shoulder joint (inclusive) ~ Nose (inclusive)":
                f"{NON_STANDARD}, between the shoulder joint and the nose tip",
             "Below shoulder joint (not included)":
                f"{NON_STANDARD}, past the shoulder joint, toward the feet"}),
    "r02": ("The angle between the trunk (the line from the midpoint of the hips to the "
            "midpoint of the shoulders) and the lower limb (the line from the hip joint "
            "to the ankle joint).",
            {"Parallel": f"{STANDARD}, 0-5 degrees",
             "Not parallel": f"{NON_STANDARD}, over 5 degrees"}),
    "r03": ("The angle between the thigh (the line from the hip joint to the ankle joint) "
            "and the horizontal plane.",
            {"Standard (> 10 degrees)": f"{STANDARD}, over 10 degrees",
             "Non-standard (≤10 degrees)": f"{NON_STANDARD}, 10 degrees or less"}),
}


def rotary_items(side: str) -> dict:
    return {
        "r01": (f"The distance between the elbow and the knee on {side} of the body.",
                {"Near": f"{STANDARD}, 0-2 cm",
                 "Far": f"{NON_STANDARD}, over 2 cm"}),
    }


RUBRIC = {
    "m01": {
        "items": DEEP_SQUAT_ITEMS,
        "scores": (3, 2, 1),
        "rating": ("3 = every item above is standard",
                   "2 = the squat is completed but at least one item is non-standard",
                   "1 = the squat cannot be completed on the floor"),
    },
    "m02": {
        "items": DEEP_SQUAT_ITEMS,
        "scores": (2, 1),
        "rating": ("2 = the squat is completed with the board under the heels",
                   "1 = the squat cannot be completed even with the board"),
    },
    "m03": {
        "items": HURDLE_STEP_ITEMS,
        "scores": (3, 2, 1),
        "rating": ("3 = every item above is standard",
                   "2 = item 4 is standard, but at least one of items 1-3 is not",
                   "1 = item 4 is non-standard"),
    },
    "m05": {
        "items": INLINE_LUNGE_ITEMS,
        "scores": (3, 2, 1),
        "rating": ("3 = every item above is standard",
                   "2 = item 3 is standard, but at least one of items 1-2 is not",
                   "1 = item 3 is non-standard"),
    },
    "m07": {
        "items": SHOULDER_ITEMS,
        "scores": (3, 2, 1),
        "rating": ("3 = the wrists are one palm length apart or less",
                   "2 = over one palm length apart, up to 1.5 palm lengths",
                   "1 = over 1.5 palm lengths apart"),
    },
    "m09": {
        "items": ASLR_ITEMS,
        "scores": (3, 2, 1),
        "rating": ("3 = the raised ankle falls at or past the thigh midpoint",
                   "2 = it falls between the knee and the thigh midpoint",
                   "1 = it falls below the knee"),
    },
    "m11": {
        "items": PUSH_UP_ITEMS,
        "scores": (3, 2, 1),
        "rating": ("3 = the push-up is performed correctly",
                   "2 = the push-up is completed with compensation",
                   "1 = the push-up cannot be completed"),
    },
    "m12": {
        "items": rotary_items("the same side"),
        "scores": (3, 2),
        "rating": ("3 = the elbow-knee spacing is standard",
                   "2 = the elbow-knee spacing is non-standard"),
    },
    "m14": {
        "items": rotary_items("opposite sides"),
        "scores": (2, 1),
        "rating": ("2 = the elbow-knee spacing is standard",
                   "1 = the elbow-knee spacing is non-standard"),
    },
}

for _left, _right in (("m03", "m04"), ("m05", "m06"), ("m07", "m08"), ("m09", "m10"),
                      ("m12", "m13"), ("m14", "m15")):
    RUBRIC[_right] = RUBRIC[_left]


def answer_template(spec: dict, scores: tuple) -> str:
    fields = []
    for rule_key in sorted(spec["rules"]):
        rule = spec["rules"][rule_key]
        name = json.dumps(rule["ruleName"], ensure_ascii=False)
        options = json.dumps("<" + " | ".join(rule["ruleOptions"]) + ">",
                             ensure_ascii=False)
        fields.append(f"{name}: {options}")
    fields.append('"Score": <' + " | ".join(map(str, scores)) + ">")
    return "{" + ", ".join(fields) + "}"


def check_rubric(rules: dict) -> None:
    problems = []
    for movement in sorted(rules):
        if movement not in RUBRIC:
            problems.append(f"{movement}: no rubric")
            continue
        items = RUBRIC[movement]["items"]
        expected = rules[movement]["rules"]
        if sorted(items) != sorted(expected):
            problems.append(f"{movement}: rules {sorted(expected)} vs rubric "
                            f"{sorted(items)}")
            continue
        for rule_key, (_, options) in items.items():
            if sorted(options) != sorted(expected[rule_key]["ruleOptions"]):
                problems.append(f"{movement}.{rule_key}: options "
                                f"{expected[rule_key]['ruleOptions']} vs "
                                f"{sorted(options)}")
    if problems:
        sys.exit("rubric does not match FMSRules.json:\n  " + "\n  ".join(problems))


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_rules(root: Path) -> dict:
    rules = read_json(root / "FMSRules.json")
    options = rules["m08"]["rules"]["r01"]["ruleOptions"]
    absent = [value for value in M08_ANSWER_FIX.values() if value not in options]
    if absent:
        sys.exit(f"M08_ANSWER_FIX targets absent from m08 ruleOptions: {absent}")
    return rules


def rubric_score(movement: str, answers: dict, source_score: int) -> int:
    if movement in ("m02", "m11") or (movement == "m01" and source_score == 1):
        return source_score
    if movement in ("m07", "m08"):


        return {
            "Within the length of a palm": 3,
            "One palm length ~ 1.5 palm length": 2,
            "≥ 1.5 palm length": 1,
        }[answers["r01"]]
    if movement in ("m09", "m10"):
        return {
            "Thigh midpoint and above (including thigh midpoint)": 3,
            "Knee point (included) ~ thigh midpoint (not included)": 2,
            "Knee point and below (excluding knee point)": 1,
        }[answers["r01"]]
    standard = {
        key: options[answers[key]].startswith("standard,")
        for key, (_, options) in RUBRIC[movement]["items"].items()
    }
    if movement == "m01":
        return 3 if all(standard.values()) else 2
    if movement in ("m03", "m04", "m05", "m06"):
        required = "r04" if movement in ("m03", "m04") else "r03"
        if not standard[required]:
            return 1
        return 3 if all(standard.values()) else 2
    if movement in ("m12", "m13"):
        return 3 if standard["r01"] else 2
    if movement in ("m14", "m15"):
        return 2 if standard["r01"] else 1
    raise ValueError(f"No score mapping for {movement}")


def load_units(root: Path, rules: dict) -> list:
    units, problems = [], Counter()
    for movement in sorted(rules):
        label_dir = root / "FMS-FineGrainedLabelInfo" / movement
        image_dir = root / "FMS-Keyframes" / movement
        spec = rules[movement]
        for label_path in sorted(label_dir.glob("*.json")):
            label = read_json(label_path)
            image = image_dir / label["picName"]
            if not image.exists():
                problems[f"{movement}_image_missing"] += 1
                continue
            answers = {}
            for rule_key in sorted(spec["rules"]):
                if rule_key not in label["ruleResults"]:
                    problems[f"{movement}_rule_missing"] += 1
                    continue
                answer = label["ruleResults"][rule_key]["ruleAnswer"]
                answer = M08_ANSWER_FIX.get(answer, answer)
                if answer not in spec["rules"][rule_key]["ruleOptions"]:
                    problems[f"{movement}_answer_off_option"] += 1
                    continue
                answers[rule_key] = answer
            score = label["score"]
            if len(answers) != len(spec["rules"]) or score not in SCORES:
                problems[f"{movement}_incomplete"] += 1
                continue
            if score not in RUBRIC[movement]["scores"]:


                problems[f"{movement}_score_outside_rubric_range"] += 1
                continue


            fields = image.stem.split("_")
            units.append({
                "image": image,
                "movement": movement,
                "subject": fields[1],
                "episode": int(fields[4][1:]),
                "answers": answers,
                "score": rubric_score(movement, answers, score),
                "source_score": score,
            })
    if problems:
        sys.exit(f"LLM-FMS: unusable pairs {dict(sorted(problems.items()))}")
    return units


def build_system(unit: dict, spec: dict) -> str:
    rubric = RUBRIC[unit["movement"]]
    rule_keys = sorted(spec["rules"])
    lines = []
    for index, rule_key in enumerate(rule_keys, start=1):
        definition, options = rubric["items"][rule_key]
        lines.append(f"{index}. {spec['rules'][rule_key]['ruleName']} -- {definition}")
        for option in spec["rules"][rule_key]["ruleOptions"]:
            lines.append(f"   - {option}: {options[option]}")
    lines.append("Overall score:")
    lines += [f"   {rule}" for rule in rubric["rating"]]
    sections = [
        SYSTEM_INTRO.format(movement=spec["actionName"]),
        "\n".join(lines),
        ANSWER_FORMAT.format(template=answer_template(spec, rubric["scores"])),
    ]
    return "\n\n".join(sections)


def build_question(unit: dict, spec: dict) -> str:
    return "<image>"


def build_answer(unit: dict, spec: dict) -> str:
    answer = {
        spec["rules"][rule_key]["ruleName"]: unit["answers"][rule_key]
        for rule_key in sorted(spec["rules"])
    }
    answer["Score"] = unit["score"]
    return json.dumps(answer, ensure_ascii=False)


def build_record(unit: dict, spec: dict) -> dict:
    rule_keys = sorted(spec["rules"])
    return {
        "messages": [
            {"role": "system", "content": build_system(unit, spec)},
            {"role": "user", "content": build_question(unit, spec)},
            {"role": "assistant", "content": build_answer(unit, spec)},
        ],
        "images": [str(unit["image"])],
        "movement": unit["movement"],
        "movement_name": spec["actionName"],
        "view": spec["view"],
        "subject": unit["subject"],
        "episode": unit["episode"],
        "rules": rule_keys,
        "rule_options": {key: spec["rules"][key]["ruleOptions"] for key in rule_keys},
        "gold_rules": {key: unit["answers"][key] for key in rule_keys},
        "gold_score": unit["score"],
        **({"source_score": unit["source_score"]}
           if unit.get("source_score", unit["score"]) != unit["score"] else {}),
        "score_range": list(RUBRIC[unit["movement"]]["scores"]),
    }


def split_subjects(units: list, every: int) -> tuple:
    subjects = sorted({unit["subject"] for unit in units})
    test = set(subjects[every - 1::every])
    train = [unit for unit in units if unit["subject"] not in test]
    held = [unit for unit in units if unit["subject"] in test]
    print(f"subjects: {len(subjects)} total, {len(test)} held out "
          f"({','.join(sorted(test))})")
    return train, held


def report(name: str, units: list) -> None:
    per_movement = defaultdict(Counter)
    for unit in units:
        per_movement[unit["movement"]][unit["score"]] += 1
    print(f"  {name}: {len(units)} rows over {len(per_movement)} movements")
    corrected = sum(unit.get("source_score", unit["score"]) != unit["score"]
                    for unit in units)
    print(f"    scores corrected to rubric: {corrected}")
    majorities = []
    for movement in sorted(per_movement):
        counts = per_movement[movement]
        total = sum(counts.values())
        majority = max(counts.values()) / total
        majorities.append(majority)
        spread = "/".join(str(counts.get(score, 0)) for score in SCORES)
        print(f"    {movement}: {total} rows, score 1/2/3 = {spread}, "
              f"majority {majority:.3f}")
    pooled = Counter(unit["score"] for unit in units)
    print(f"    majority baseline: unweighted mean over movements "
          f"{sum(majorities) / len(majorities):.3f}, pooled "
          f"{max(pooled.values()) / len(units):.3f}")


def write(path: Path, units: list, rules: dict) -> None:
    units = sorted(units, key=lambda unit: (unit["movement"], str(unit["image"])))
    with path.open("w", encoding="utf-8") as handle:
        for unit in units:
            record = build_record(unit, rules[unit["movement"]])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"    -> {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="LLM-FMS root (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "rehabilitation",
                        help="where to write the metadata (default: %(default)s)")
    parser.add_argument("--prefix", default="llmfms",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--split", choices=["subject", "all"], default="subject",
                        help="'subject' holds out every --test-every subject into "
                             "<prefix>_test.jsonl; 'all' writes every row to "
                             "<prefix>_all.jsonl, the paper's own zero-shot evaluation "
                             "surface (default: %(default)s)")
    parser.add_argument("--test-every", type=int, default=5,
                        help="with --split subject, hold out every N-th subject "
                             "(default: %(default)s)")
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.test_every < 2:
        sys.exit("--test-every must be at least 2")

    rules = load_rules(args.dataset_root)
    check_rubric(rules)
    units = load_units(args.dataset_root, rules)
    print(f"loaded {len(units)} keyframes over {len(rules)} movements from "
          f"{args.dataset_root}")
    if len(units) != 1812:
        print(f"warning: the release holds 1812 keyframes, loaded {len(units)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "all":
        report("all", units)
        write(args.out_dir / f"{args.prefix}_all.jsonl", units, rules)
        return 0

    train, test = split_subjects(units, args.test_every)
    for name, part in (("train", train), ("test", test)):
        if not part:
            sys.exit(f"{name} split is empty")
        report(name, part)
        write(args.out_dir / f"{args.prefix}_{name}.jsonl", part, rules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
