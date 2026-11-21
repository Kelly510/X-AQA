#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from dataset_catalog import DOMAIN_FOLDERS, dataset_for_artifact
import parsing
from data_io import read_references, reference_sha, add_dataset_arguments, load_reference_set
from metrics import BERT_MODEL, fisher_z_mean, regression_metrics, spearman_rho, text_metrics

SUITE = {
    "aqa7_score": ("AQA-7", "ID", 100, None, "S"),
    "mtlaqa": ("MTL-AQA", "ID", 179, None, "S,T"),
    "rhythmicgymnastics": ("RG", "ID", 100, None, "S"),
    "finefs_goe": ("FineFS", "ID", 100, None, "S"),
    "cricketvision_score": ("CricketVision", "ID", 100, None, "S"),
    "llmfms": ("LLM-FMS", "ID", 361, None, "S"),
    "lasana_score": ("LASANA", "ID", 195, None, "S"),
    "multisensebadminton_qa": ("MultiSenseBadminton", "ID", 167, ("Beginner", "Intermediate", "Expert"), "C"),
    "egoexo4d_sports": ("Ego-Exo4D", "ID", 166, ("Novice", "Early Expert", "Intermediate Expert", "Late Expert"), "C"),
    "cotafa_martial_aqa": ("CoT-AFA", "ID", 167, ("standard", "non-standard"), "C"),
    "cotafa_fitness_aqa": ("CoT-AFA", "ID", 125, ("standard", "non-standard"), "C"),
    "egoexo_fitness_kp": ("EgoExo-Fitness", "ID", 125, ("yes", "no"), "C"),
    "fitnessaqa_kp": ("Fitness-AQA", "ID", 125, ("yes", "no"), "C"),
    "flag3d_kp": ("FLAG3D", "ID", 125, ("yes", "no"), "C"),
    "rehab246": ("REHAB24-6", "ID", 200, ("correct", "incorrect"), "C"),
    "execheck_correctness": ("ExeCheck", "ID", 100, ("correct", "incorrect"), "C"),
    "cprcoach": ("CPR-Coach", "ID", 125, ("YES", "NO"), "C"),
    "egoexo4d_skill": ("Ego-Exo4D", "ID", 125, ("Novice", "Early Expert", "Intermediate Expert", "Late Expert"), "C"),
    "lasana_error": ("LASANA", "ID", 125, ("Yes", "No"), "C"),
    "captaincook4d": ("CaptainCook4D", "ID", 125, None, "C"),
    "expertaf_sports": ("ExpertAF", "ID", 166, None, "T"),
    "cotafa_martial_cot": ("CoT-AFA", "ID", 167, None, "T"),
    "cotafa_fitness_cot": ("CoT-AFA", "ID", 167, None, "T"),
    "egoexo_fitness_comment": ("EgoExo-Fitness", "ID", 167, None, "T"),
    "qevd_fitcoach": ("QEVD", "ID", 166, None, "T"),
    "jigsaws": ("JIGSAWS", "OOD", 339, None, "S"),
    "aixsuture_grs": ("AIxSuture", "ID", 46, None, "S"),
    "kimore_cts": ("KIMORE", "OOD", 348, None, "S"),
    "fitness7": ("Fitness-7", "OOD", 348, ("good", "bad"), "C"),
    "upperlimb_completion": ("UpperLimb", "ID", 80, ("complete", "incomplete"), "C"),
    "t3set_comment": ("T3Set", "OOD", 700, None, "T"),
}


FORMAT_METRICS = {"S": ("srcc",), "C": ("maf1", "kappa"), "T": ("bertscore_f1",)}
METRIC_DIRECTIONS = {
    "srcc": "higher",
    "maf1": "higher",
    "kappa": "higher",
    "bertscore_f1": "higher",
}
FORMAT_NAME = {"S": "score", "C": "category", "T": "text"}


EXPECTED_ROWS = sum(spec[2] for spec in SUITE.values())


def version_tasks(names, reference_dir):
    names = list(names)
    if reference_dir is None:
        return names
    manifest = Path(reference_dir).resolve().parent / "benchmark_sample.json"
    if not manifest.is_file():
        return names
    kept = json.loads(manifest.read_text())["tasks"]
    dropped = [n for n in names if n not in kept]
    if dropped:
        print(f'not in {manifest.parent.name}, skipped: {",".join(dropped)}', file=sys.stderr)
    return [n for n in names if n in kept]


@lru_cache(maxsize=None)
def _manifest_counts(manifest):
    tasks = json.loads(Path(manifest).read_text())["tasks"]
    return {name: info["selected"] for name, info in tasks.items()}


@lru_cache(maxsize=None)
def _variant_counts(manifest):
    tasks = json.loads(Path(manifest).read_text())["tasks"]
    return {name: info["rows"] for name, info in tasks.items()}


def expected_rows(name, reference_path):
    if isinstance(reference_path, Path):
        parents = reference_path.resolve().parents
        if len(parents) > 2:
            manifest = parents[2] / "benchmark_sample.json"
            if manifest.is_file():
                counts = _manifest_counts(str(manifest))
                if name in counts:
                    return counts[name]
        for parent in parents[:3]:
            variant = parent / "test_manifest.json"
            if variant.is_file():
                counts = _variant_counts(str(variant))
                if name in counts:
                    return counts[name]
                break
    return SUITE[name][2]


GROUP_FIELDS = (
    "exercise",
    "exercise_name",
    "movement",
    "camera",
    "view",
    "task",
    "error_category",
    "stroke_type",
    "technique",
)
ERRORS = (
    "Preparation Error",
    "Measurement Error",
    "Order Error",
    "Timing Error",
    "Technique Error",
    "Temperature Error",
    "Missing Step",
    "Other",
)
ASPECTS = ("Depth perception", "Efficiency", "Bimanual dexterity", "Tissue handling")
JIGSAWS_ELEMENTS = (
    "Respect for tissue",
    "Suture/needle handling",
    "Time and motion",
    "Flow of operation",
    "Overall performance",
    "Quality of final product",
)


AIXSUTURE_ELEMENTS = (
    "Respect for tissue",
    "Time and motion",
    "Instrument handling",
    "Suture technique",
    "Flow of operation and forward planning",
    "Knowledge of specific procedure",
    "Quality of final product",
    "Overall performance",
)


KIMORE_ITEMS = (
    "primary_objective_achieved",
    "movement_amplitude_complete",
    "repetitions_constant",
    "head_posture",
    "right_arm_posture",
    "left_arm_posture",
    "trunk_posture",
    "pelvis_posture",
    "right_leg_posture",
    "left_leg_posture",
)


def read(path, opts=None):
    return read_references(path, getattr(opts, "media_root", None))


def identity(row):
    return row.get("sample_id") or row.get("id")


def join_predictions(refs, predictions):
    ids, keys = {}, defaultdict(list)
    for i, ref in enumerate(refs):
        uid = identity(ref)
        if uid:
            if uid in ids:
                raise ValueError(f"Duplicate reference ID: {uid}")
            ids[uid] = i
        keys[parsing._benchmark_key(ref)].append(i)
    matched = {}
    for pred in predictions:
        if "response" not in pred or not isinstance(pred["response"], (str, list)):
            raise ValueError("Each prediction must have a string response (or a string list)")
        if isinstance(pred["response"], list) and any(not isinstance(x, str) for x in pred["response"]):
            raise ValueError("response list must contain strings")
        uid = identity(pred)
        key = parsing._benchmark_key(pred)
        if uid:
            if uid not in ids:
                raise ValueError(f"Unknown prediction ID: {uid}")
            i = ids[uid]
            if key[0] and key != parsing._benchmark_key(refs[i]):
                raise ValueError(f"Input mismatch for prediction ID: {uid}")
        else:
            candidates = keys.get(key, [])
            label = parsing.gold_text(pred).strip()
            if len(candidates) > 1 and label:
                candidates = [i for i in candidates if parsing.gold_text(refs[i]).strip() == label]
            if len(candidates) != 1:
                raise ValueError(f"Unmatched/ambiguous prediction: {key[0]}; preserve sample_id/id")
            i = candidates[0]
        label = parsing.gold_text(pred).strip()
        if label and label != parsing.gold_text(refs[i]).strip():
            raise ValueError(f"Conflicting reference answer: {identity(refs[i]) or key[0]}")
        if i in matched:
            raise ValueError(f"Duplicate prediction: {identity(refs[i]) or key[0]}")
        matched[i] = pred["response"]

    joined = [{**r, "response": matched.get(i, ""), "labels": parsing.gold_text(r)} for i, r in enumerate(refs)]
    return joined, {
        "expected_rows": len(refs),
        "predicted_rows": len(matched),
        "missing_rows": len(refs) - len(matched),
        "prediction_coverage": len(matched) / len(refs) if refs else 0.0,
    }


def answer(row):
    return parsing.pred_text(row)


def obj(text, declared=None):
    return parsing.canonical_keys(parsing.json_object(text), declared)


def label(text, classes):
    return text if isinstance(text, str) and text in classes else None


def number(text, low, high, step=None):

    if isinstance(text, str):
        try:
            text = float(text.strip())
        except ValueError:
            return None
    if not isinstance(text, (int, float)) or isinstance(text, bool):
        return None
    try:
        value = float(text)
    except OverflowError:
        return None
    if not math.isfinite(value) or not low <= value <= high:
        return None
    if step and (
        not math.isfinite(value / step) or not math.isclose(value / step, round(value / step), rel_tol=0, abs_tol=1e-6)
    ):
        return None
    return value


def classify(golds, preds, classes):
    if any(g not in classes for g in golds):
        raise ValueError("Reference contains an unknown class")
    n = len(golds)
    details = {}
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(golds, preds))
        fp = sum(g != c and p == c for g, p in zip(golds, preds))
        fn = sum(g == c and p != c for g, p in zip(golds, preds))
        support = golds.count(c)
        details[str(c)] = {
            "support": support,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / support if support else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    observed = [d for d in details.values() if d["support"]]
    acc = sum(g == p for g, p in zip(golds, preds)) / n if n else None
    chance = sum(golds.count(c) * preds.count(c) for c in classes) / n**2 if n else None
    return {
        "n": n,
        "n_valid": sum(p in classes for p in preds),
        "balanced_accuracy": sum(d["recall"] for d in observed) / len(observed) if observed else None,
        "macro_f1": sum(d["f1"] for d in details.values()) / len(classes),
        "macro_f1_gold_classes": sum(d["f1"] for d in observed) / len(observed) if observed else None,
        "kappa": (acc - chance) / (1 - chance) if chance is not None and chance < 1 else None,
        "per_class": details,
        "confusion": {
            str(c): {str(p): sum(g == c and v == p for g, v in zip(golds, preds)) for p in (*classes, None)}
            for c in classes
        },
    }


def grouped(rows, score, fields=GROUP_FIELDS):
    out = {}
    for field in fields:
        if not any(field in r for r in rows):
            continue
        groups = defaultdict(list)
        for r in rows:
            groups[str(r.get(field, "unknown"))].append(r)
        out[field] = {k: score(v) for k, v in sorted(groups.items())}
    return out


def numeric(golds, preds):
    out = regression_metrics(golds, preds)
    out["rmse"] = math.sqrt(out["mse"]) if out["mse"] is not None else None
    pairs = [(g, p) for g, p in zip(golds, preds) if p is not None]
    out["ccc_population"] = None
    if pairs:
        gs, ps = zip(*pairs)
        gm, pm = sum(gs) / len(gs), sum(ps) / len(ps)
        cov = sum((g - gm) * (p - pm) for g, p in pairs) / len(pairs)
        den = sum((g - gm) ** 2 + (p - pm) ** 2 for g, p in pairs) / len(pairs) + (gm - pm) ** 2
        out["ccc_population"] = 2 * cov / den if den else None
    return out


def rank_agreement(rhos):
    usable = [v for v in rhos.values() if v is not None]
    return {
        "spearman_group_fisher_z": fisher_z_mean(list(rhos.values())),
        "spearman_group_mean": sum(usable) / len(usable) if usable else None,
        "groups_at_unit_rho": sum(abs(v) >= 1 for v in usable),
        "groups_total": len(rhos),
        "groups_used": len(usable),
        "groups_undefined": sorted(k for k, v in rhos.items() if v is None),
        "per_group_spearman": rhos,
    }


SRCC_GROUP_KEY = {"aqa7_score": "sport"}


def jigsaws_task(row):
    videos = row.get("videos") or []
    name = Path(videos[0]).name if videos else ""
    for needle, label in (("Knot_Tying", "KT"), ("Needle_Passing", "NP"), ("Suturing", "S")):
        if needle in name:
            return label
    rid = str(row.get("original_recording_id") or "")
    for needle, label in (("Knot_Tying", "KT"), ("Needle_Passing", "NP"), ("Suturing", "S")):
        if needle in rid:
            return label
    return "unknown"


PROJECT_GROUP_KEY = {
    "aqa7_score": lambda row: row.get("sport"),
    "mtlaqa": lambda row: row.get("apparatus") or "unknown",
    "rhythmicgymnastics": lambda row: row.get("apparatus") or "unknown",
    "lasana_score": lambda row: (row.get("stratum") or "unknown").split("=", 1)[-1],
    "jigsaws": jigsaws_task,
    "kimore_cts": lambda row: row.get("exercise") or "unknown",
}


def project_srcc_block(name, rows, golds, preds):
    key = PROJECT_GROUP_KEY.get(name)
    groups = defaultdict(lambda: ([], []))
    for row, gold, pred in zip(rows, golds, preds):
        if pred is None:
            continue
        groups[key(row) if key else "__all__"][0].append(gold)
        groups[key(row) if key else "__all__"][1].append(pred)
    return {
        "group_key": name if key else "source",
        "per_project": {
            str(k): {"srcc": primary_srcc(gs, ps), "valid_rows": len(gs)} for k, (gs, ps) in sorted(groups.items())
        },
    }


ORDINAL_LEVELS = {
    "egoexo4d_sports": ("Novice", "Early Expert", "Intermediate Expert", "Late Expert"),
    "egoexo4d_skill": ("Novice", "Early Expert", "Intermediate Expert", "Late Expert"),
    "multisensebadminton_qa": ("Beginner", "Intermediate", "Expert"),
}


def primary_srcc(golds, preds):
    pairs = [(g, p) for g, p in zip(golds, preds) if p is not None and math.isfinite(p)]
    if not pairs:
        return None
    rho = spearman_rho([g for g, _ in pairs], [p for _, p in pairs])
    return 0.0 if rho is None else rho


def within_group_srcc(name, rows, golds, preds):
    key = SRCC_GROUP_KEY[name]
    pairs = defaultdict(lambda: ([], []))
    for row, gold, pred in zip(rows, golds, preds):
        group = pairs[row.get(key)]
        if pred is not None:
            group[0].append(gold)
            group[1].append(pred)
    if None in pairs:
        raise ValueError(f"{name}: row without a {key!r} group")
    rhos = {group: primary_srcc(gs, ps) for group, (gs, ps) in sorted(pairs.items())}
    out = rank_agreement(rhos)
    out["group_key"] = key
    out["group_valid_rows"] = {group: len(gs) for group, (gs, _) in sorted(pairs.items())}
    out["srcc"] = None if out["groups_undefined"] else out["spearman_group_fisher_z"]
    return out


def caption(preds, golds, opts):
    if opts.skip_text:
        return {"status": "skipped_by_request", "n": len(golds)}
    if not golds:
        return {"status": "no_references", "n": 0}
    return text_metrics(
        preds,
        golds,
        bert_model=opts.bert_model,
        bert_batch_size=opts.bert_batch_size,
    )


def binary_score(name, rows, classes):
    return classify(
        [category(name, parsing.gold_text(r), classes) for r in rows],
        [category(name, answer(r), classes) for r in rows],
        classes,
    )


UPPERLIMB_EXERCISES = ("Lifting an Object", "Extending the Elbow", "Lifting the Wrist", "Opening the Hand")


ACTION_GROUP_KEY = {
    "fitness7": "exercise_name",
    "upperlimb_completion": "exercise",
    "t3set_comment": "technique",
    "jigsaws": "original_recording_id",
}
ACTION_TIERS = {
    "fitness7": {
        "Overhead dumbbell triceps extension": "O1",
        "Bent-over dumbbell row": "O1",
        "Alternating dumbbell bicep curl": "O1",
        "Bodyweight squat": "O1",
        "Standard push-up": "O1",
        "Standing dumbbell shoulder press": "O1v",
        "Standing dumbbell side bend": "O2",
    },
    "jigsaws": {"Suturing": "O1", "Knot Tying": "O1", "Needle Passing": "O2"},
    "t3set_comment": dict.fromkeys(("push", "topspin", "short", "attack", "flick", "pendulum", "block"), "O2"),
    "upperlimb_completion": dict.fromkeys(UPPERLIMB_EXERCISES, "O2"),
}


def action_group(name, row):
    value = row.get(ACTION_GROUP_KEY[name])
    if name == "jigsaws":
        return str(value or "").rsplit("_", 1)[0].replace("_", " ")
    return value


def action_breakdown(name, rows, opts):
    if name not in ACTION_TIERS:
        return None
    by_action, by_tier = defaultdict(list), defaultdict(list)
    for row in rows:
        action = action_group(name, row)
        tier = ACTION_TIERS[name].get(action)
        if tier is None:
            raise ValueError(f"{name}: action outside the overlap map: {action!r}")
        by_action[action].append(row)
        by_tier[tier].append(row)

    def block(subset):
        primary, details = format_metrics(name, subset, opts)
        return {"n_rows": len(subset), "valid_rows": details["n_valid_rows"], "primary": primary}

    def within_action(tier):
        actions = [a for a, t in ACTION_TIERS[name].items() if t == tier and a in by_action]
        rhos = [block(by_action[a])["primary"]["srcc"] for a in sorted(actions)]
        return {"actions": len(rhos), "srcc_within_action": None if None in rhos else fisher_z_mean(rhos)}

    tiers = {tier: block(subset) for tier, subset in sorted(by_tier.items())}
    if SUITE[name][4] == "S":
        for tier in tiers:
            tiers[tier].update(within_action(tier))
    return {
        "group_key": "trial id" if name == "jigsaws" else ACTION_GROUP_KEY[name],
        "per_action": {
            action: {"tier": ACTION_TIERS[name][action], **block(subset)}
            for action, subset in sorted(by_action.items())
        },
        "per_tier": tiers,
    }


def upperlimb(rows):
    states = ("complete", "incomplete")
    parsed = []
    for r in rows:
        if r.get("target_definition") != "upperlimb_completion_v3":
            raise ValueError("UpperLimb requires regenerated completion v3 references")
        exercise, state = r["exercise"], r["completion"]
        if exercise not in UPPERLIMB_EXERCISES or state not in states:
            raise ValueError("Unknown UpperLimb reference class")
        parsed.append({**r, "_gold": state, "_pred": category("upperlimb_completion", answer(r), states)})

    def score(rs):
        out = classify([r["_gold"] for r in rs], [r["_pred"] for r in rs], states)
        for metric in ("precision", "recall", "f1"):
            out["weighted_" + metric] = (
                (sum(d["support"] * (d[metric] or 0) for d in out["per_class"].values()) / len(rs)) if rs else None
            )
        return out

    pooled = score(parsed)
    return {"pooled": pooled, "per_group": grouped(parsed, score, ("exercise",)), "n_valid_rows": pooled["n_valid"]}


def llmfms(rows):
    def parse(r):
        rubric = parsing._llmfms_rubric(parsing.system_text(r))
        if rubric is None:
            raise ValueError("Unrecognized LLM-FMS rubric")
        _, items, allowed = rubric
        verdicts, value = parsing._llmfms_answer(answer(r), items)
        valid = value in allowed and all(v is not None for v in verdicts.values())
        return value if valid else None, verdicts, items

    per = {}
    for movement in sorted({r["movement"] for r in rows}):
        rs = [r for r in rows if r["movement"] == movement]
        per[movement] = classify([r["gold_score"] for r in rs], [parse(r)[0] for r in rs], tuple(rs[0]["score_range"]))
    metrics = ("balanced_accuracy", "macro_f1", "macro_f1_gold_classes", "kappa")
    avg = {}
    for m in metrics:
        values = [b[m] for b in per.values() if b[m] is not None]
        avg[m] = sum(values) / len(values) if values else None
        avg[m + "_n_movements"] = len(values)
    item_pairs = defaultdict(list)
    for r in rows:
        _, verdicts, items = parse(r)
        golds, _ = parsing._llmfms_answer(parsing.gold_text(r), items)
        for key, gold in golds.items():
            item_pairs[(r["movement"], key)].append((gold, verdicts.get(key)))
    valid_rows = 0
    for r in rows:
        value, verdicts, _ = parse(r)
        valid_rows += value is not None and all(v is not None for v in verdicts.values())
    item_scores = [
        classify([g for g, _ in pairs], [p for _, p in pairs], tuple(dict.fromkeys(g for g, _ in pairs)))[
            "balanced_accuracy"
        ]
        for pairs in item_pairs.values()
    ]
    return {
        "per_movement": per,
        "movement_avg": avg,
        "item_balanced_accuracy": sum(item_scores) / len(item_scores) if item_scores else None,
        "n_valid_rows": valid_rows,
    }


def badminton(rows, opts):
    classes = ("Beginner", "Intermediate", "Expert")

    def metrics(golds, preds):

        if any(g not in classes for g in golds):
            raise ValueError("Reference contains an unknown skill level")
        recalls, f1s = [], []
        for c in classes:
            tp = sum(g == c and p == c for g, p in zip(golds, preds))
            fp = sum(g != c and p == c for g, p in zip(golds, preds))
            fn = sum(g == c and p != c for g, p in zip(golds, preds))
            support = golds.count(c)
            if support:
                recalls.append(tp / support)
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        return {
            "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
            "macro_f1": sum(f1s) / len(classes),
        }

    if any(
        r.get("target_definition") != "original_skill_level_v1" or r["skill_level"] != r.get("original_skill_level")
        for r in rows
    ):
        raise ValueError("MultiSenseBadminton requires reprocessed original skill-level references")

    def parse_level(text):
        return category("multisensebadminton_qa", text, classes)

    levels = {id(r): parse_level(answer(r)) for r in rows}

    def score(rs):
        return {"skill_level": metrics([r["skill_level"] for r in rs], [levels[id(r)] for r in rs])}

    sources = defaultdict(list)
    for r in rows:
        sources[r["source_id"]].append(r)
    source_rows = []
    for rs in sources.values():
        if len({r["skill_level"] for r in rs}) != 1:
            raise ValueError("Conflicting MultiSenseBadminton source-level reference labels")
        source_levels = [levels[id(r)] for r in rs]
        valid_levels = [x for x in source_levels if x is not None]
        counts = Counter(valid_levels)
        best = max(counts.values()) if counts else 0
        winners = [c for c in classes if counts[c] == best] if counts else []

        source_rows.append(
            {**rs[0], "_level": winners[0] if len(winners) == 1 and all(x is not None for x in source_levels) else None}
        )

    def source_score(rs):
        return {"skill_level": metrics([r["skill_level"] for r in rs], [r["_level"] for r in rs])}

    return {
        "source_level": source_score(source_rows),
        "source_per_stroke": grouped(source_rows, source_score, ("stroke_type",)),
        "window": score(rows),
        "n_valid_rows": sum(v is not None for v in levels.values()),
    }


def lasana_scores(rows):
    def parse(text):
        value = obj(text)
        if value is None or set(value) != set(ASPECTS):
            return {}
        return {aspect: number(value.get(aspect), 1, 5, 0.5) for aspect in ASPECTS}

    def score(rs):
        golds = [parse(parsing.gold_text(r)) for r in rs]
        if any(any(g.get(a) is None for a in ASPECTS) for g in golds):
            raise ValueError("Invalid LASANA reference scores")
        preds = [parse(answer(r)) for r in rs]
        totals = [sum(p[a] for a in ASPECTS) if all(p.get(a) is not None for a in ASPECTS) else None for p in preds]
        return {
            "raw_total": numeric([sum(g[a] for a in ASPECTS) for g in golds], totals),
            "aspects": {a: numeric([g[a] for g in golds], [p.get(a) for p in preds]) for a in ASPECTS},
        }

    per_task = grouped(rows, score, ("task",))
    return {
        "pooled": score(rows),
        "per_task": per_task,
        "rank_agreement": rank_agreement({k: b["raw_total"]["spearman"] for k, b in per_task.get("task", {}).items()}),
        "n_valid_rows": sum(all(parse(answer(r)).get(a) is not None for a in ASPECTS) for r in rows),
    }


def lasana_paper_errors(rows):
    selections = {
        "peg_transfer": ("object_dropped_within_fov",),
        "circle_cutting": ("gauze_detached",),
        "balloon_resection": ("balloon_opened", "balloon_damaged"),
        "suture_and_knot": ("needle_dropped",),
        "suture_knot": ("needle_dropped",),
    }
    groups = defaultdict(dict)
    for r in rows:
        groups[(r["task"], r["recording_id"])][r["error_category"]] = r
    out = defaultdict(list)
    for (task, _), categories in groups.items():
        wanted = selections.get(task, ("needle_dropped",) if "suture" in task else ())
        if not wanted or any(c not in categories for c in wanted):
            continue
        rs = [categories[c] for c in wanted]

        def present(text):
            return category("lasana_error", text, ("Yes", "No"))

        gold = any(present(parsing.gold_text(r)) == "Yes" for r in rs)
        ps = [present(answer(r)) for r in rs]
        pred = True if "Yes" in ps else (False if all(p == "No" for p in ps) else None)
        out[task].append((gold, pred))
    return {t: classify([g for g, _ in pairs], [p for _, p in pairs], (False, True)) for t, pairs in out.items()}


def cooking_object(text):
    value = obj(text, ERRORS)
    return (
        value
        if value is not None and all(k in ERRORS and isinstance(v, str) and v.strip() for k, v in value.items())
        else None
    )


def cooking(rows, opts):
    gs = [cooking_object(parsing.gold_text(r)) for r in rows]
    ps = [cooking_object(answer(r)) for r in rows]
    if any(g is None for g in gs):
        raise ValueError("Invalid CaptainCook4D reference")
    categories = {
        c: classify([c in g for g in gs], [c in p if p is not None else None for p in ps], (False, True))
        for c in ERRORS
    }
    return {
        "error_detection": classify(
            [bool(g) for g in gs], [bool(p) if p is not None else None for p in ps], (False, True)
        ),
        "per_category": categories,
        "category_macro_f1_positive": sum(v["per_class"]["True"]["f1"] for v in categories.values()) / len(ERRORS),
        "exact_category_set_balanced_accuracy": classify(
            [tuple(sorted(g)) for g in gs],
            [tuple(sorted(p)) if p is not None else None for p in ps],
            tuple(dict.fromkeys(tuple(sorted(g)) for g in gs)),
        )["balanced_accuracy"],
        "n_valid_rows": sum(p is not None for p in ps),
    }


def mtlaqa_object(text, row=None):
    reference = obj(parsing.gold_text(row)) if row is not None else {"execution_award": 0}
    data = obj(text, tuple(reference) if reference else None)
    if (
        not reference
        or not set(reference) <= {"execution_award", "comment"}
        or data is None
        or set(data) != set(reference)
    ):
        return None
    if "execution_award" in data and number(data["execution_award"], 0, 10, 0.5) is None:
        return None
    if "comment" in data and (not isinstance(data["comment"], str) or not data["comment"].strip()):
        return None
    return data


def scalar(name, text, row=None):
    if name in ORDINAL_LEVELS:
        level = category(name, text, ORDINAL_LEVELS[name])
        return None if level is None else float(ORDINAL_LEVELS[name].index(level) + 1)
    if name == "mtlaqa":
        data = mtlaqa_object(text, row)
        return data.get("execution_award") if data is not None else None
    data = obj(text)
    if data is None:
        return None
    if name == "rhythmicgymnastics":
        data = parsing.canonical_keys(data, ("execution_score", "penalty"))
        if set(data) != {"execution_score", "penalty"}:
            return None
        if number(data["penalty"], 0, math.inf, 0.05) is None:
            return None
        return number(data["execution_score"], 0, 10, 0.1)
    if name == "llmfms":
        rubric = parsing._llmfms_rubric(parsing.system_text(row or {}))
        if rubric is None:
            return None
        _, items, allowed = rubric
        verdicts, value = parsing._llmfms_answer(text, items)
        return float(value) if value in allowed and all(v is not None for v in verdicts.values()) else None
    if name == "aqa7_score":
        field = {
            "diving": "execution_award",
            "gym_vault": "execution_score",
            "ski_big_air": "overall_score",
            "snowboard_big_air": "overall_score",
            "sync_diving_3m": "dive_total",
            "sync_diving_10m": "dive_total",
        }.get((row or {}).get("sport"))
        if field is None:
            return None
        bounds = {
            "execution_award": (0, 10, 0.5),
            "execution_score": (0, 10, 0.1),
            "overall_score": (0, 50, 1),
            "dive_total": (0, math.inf, 0.01),
        }
        data = parsing.canonical_keys(data, (field,))
        if set(data) != {field} or (field == "overall_score" and type(data[field]) is not int):
            return None
        return number(data[field], *bounds[field])
    single = {"finefs_goe": ("goe_grade", -5, 5, 0.5)}
    if name in single:
        field, low, high, step = single[name]
        data = parsing.canonical_keys(data, (field,))
        return number(data[field], low, high, step) if set(data) == {field} else None
    multi = {
        "cricketvision_score": (("head", "shoulder", "hands", "hips", "feet"), 1, 10, 1),
        "lasana_score": (ASPECTS, 1, 5, 0.5),
        "jigsaws": (JIGSAWS_ELEMENTS, 1, 5, 1),
        "aixsuture_grs": (AIXSUTURE_ELEMENTS, 1, 5, None),
        "kimore_cts": (KIMORE_ITEMS, 1, 5, 1),
    }
    if name not in multi:
        return None
    fields, low, high, step = multi[name]
    data = parsing.canonical_keys(data, fields)
    if set(data) != set(fields):
        return None
    if name not in ("lasana_score", "aixsuture_grs") and any(type(data[k]) is not int for k in fields):
        return None
    values = [number(data[k], low, high, step) for k in fields]
    if any(value is None for value in values):
        return None
    return sum(values)


def category(name, text, classes):
    text = str(text).strip()
    if name == "captaincook4d":
        data = cooking_object(text)
        return None if data is None else ("error" if data else "no_error")
    if name == "multisensebadminton_qa":
        data = obj(text, ("skill_level", "feedback"))
        if data is None or set(data) != {"skill_level", "feedback"} or not isinstance(data["feedback"], str):
            return None
        return label(data["skill_level"], classes)
    fields = {
        "egoexo4d_sports": "proficiency_level",
        "egoexo4d_skill": "proficiency_level",
        "cotafa_martial_aqa": "form_assessment",
        "cotafa_fitness_aqa": "form_assessment",
        "egoexo_fitness_kp": "technical_point_satisfied",
        "fitnessaqa_kp": "technical_point_satisfied",
        "flag3d_kp": "technical_point_satisfied",
        "rehab246": "rehabilitation_form",
        "execheck_correctness": "exercise_form",
        "cprcoach": "error_present",
        "lasana_error": "error_present",
        "fitness7": "exercise_form",
        "upperlimb_completion": "completion",
    }
    field = fields.get(name)
    data = obj(text, (field,) if field else None)
    if field is None or data is None or set(data) != {field}:
        return None
    return label(data.get(field), classes)


def class_set(name, rows):
    declared = SUITE[name][3]
    if declared:
        return declared
    if name == "captaincook4d":
        return ("error", "no_error")
    if name == "multisensebadminton_qa":
        return ("Beginner", "Intermediate", "Expert")
    return tuple(sorted({parsing.gold_text(r).strip() for r in rows}))


def scored_classes(name, rows):
    return class_set(name, rows)


def scored_category(name, text, row, classes):
    return category(name, text, classes)


def mtlaqa_metrics(rows, opts):
    golds = [mtlaqa_object(parsing.gold_text(r), r) for r in rows]
    if any(g is None for g in golds):
        raise ValueError("MTL-AQA: invalid reference answer")
    preds = [mtlaqa_object(answer(r), r) for r in rows]
    chosen = {"S": [], "T": []}
    for i, (row, gold) in enumerate(zip(rows, golds)):
        available = [f for f, k in [("S", "execution_award"), ("T", "comment")] if k in gold]
        selected = row.get("evaluation_formats", available)
        if (
            not isinstance(selected, list)
            or not selected
            or any(f not in available for f in selected)
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("MTL-AQA: invalid reference evaluation_formats")
        for fmt in selected:
            chosen[fmt].append(i)
    scores, texts = chosen["S"], chosen["T"]
    regression = numeric(
        [golds[i]["execution_award"] for i in scores],
        [preds[i]["execution_award"] if preds[i] else None for i in scores],
    )
    feedback = caption(
        [preds[i]["comment"] if preds[i] else "" for i in texts], [golds[i]["comment"] for i in texts], opts
    )
    primary = {"srcc": regression["spearman"], **{k: feedback.get(k) for k in FORMAT_METRICS["T"]}}
    project = project_srcc_block(
        "mtlaqa",
        [rows[i] for i in scores],
        [golds[i]["execution_award"] for i in scores],
        [preds[i]["execution_award"] if preds[i] else None for i in scores],
    )
    return primary, {
        "regression": regression,
        "feedback": feedback,
        "component_counts": {f: len(ids) for f, ids in chosen.items()},
        "n_valid_rows": sum(p is not None for p in preds),
        "project_srcc": project,
    }


def format_metrics(name, rows, opts):
    fmt = SUITE[name][4]
    if name == "mtlaqa":
        return mtlaqa_metrics(rows, opts)
    if fmt == "C":
        classes = scored_classes(name, rows)
        golds = [scored_category(name, parsing.gold_text(r), r, classes) for r in rows]
        preds = [scored_category(name, answer(r), r, classes) for r in rows]
        if any(g is None for g in golds):
            raise ValueError(f"{name}: reference answer outside the declared classes")
        block = classify(golds, preds, classes)
        return (
            {"maf1": block["macro_f1_gold_classes"], "kappa": block["kappa"]},
            {
                "classes": list(classes),
                "classification": block,
                "per_group": grouped(
                    rows,
                    lambda rs: classify(
                        [scored_category(name, parsing.gold_text(r), r, classes) for r in rs],
                        [scored_category(name, answer(r), r, classes) for r in rs],
                        classes,
                    ),
                ),
                "n_valid_rows": block["n_valid"],
            },
        )
    if fmt == "S":
        if name in ("aixsuture_grs", "kimore_cts"):

            field = "grs" if name == "aixsuture_grs" else "cts"
            golds = [r.get(field) for r in rows]
            preds = [scalar(name, answer(r), r) for r in rows]
            if any(g is None for g in golds):
                raise ValueError(f"{name}: row is missing its {field} target")
            block = numeric(golds, preds)
            return {"srcc": primary_srcc(golds, preds)}, {
                "regression": block,
                "n_valid_rows": sum(p is not None for p in preds),
                "project_srcc": project_srcc_block(name, rows, golds, preds),
            }
        golds = [scalar(name, parsing.gold_text(r), r) for r in rows]
        preds = [scalar(name, answer(r), r) for r in rows]
        if any(g is None for g in golds):
            raise ValueError(f"{name}: reference score could not be parsed")
        block = numeric(golds, preds)
        details = {
            "regression": block,
            "n_valid_rows": sum(p is not None for p in preds),
            "project_srcc": project_srcc_block(name, rows, golds, preds),
        }
        if name in SRCC_GROUP_KEY:
            details["srcc_within_group"] = within_group_srcc(name, rows, golds, preds)
            return {"srcc": details["srcc_within_group"]["srcc"]}, details
        return {"srcc": primary_srcc(golds, preds)}, details
    feedback = caption([answer(r) for r in rows], [parsing.gold_text(r) for r in rows], opts)
    return (
        {key: feedback.get(key) for key in FORMAT_METRICS["T"]},
        {"feedback": feedback, "n_valid_rows": sum(bool(answer(r)) for r in rows)},
    )


def score_dataset_details(name, rows, opts):
    if name == "upperlimb_completion":
        return upperlimb(rows)
    classes = SUITE[name][3]
    if classes:
        block = binary_score(name, rows, classes)
        out = {
            "pooled": block,
            "per_group": grouped(rows, lambda rs: binary_score(name, rs, classes)),
            "n_valid_rows": block["n_valid"],
        }
        if name == "lasana_error":
            out["paper_selected_errors"] = lasana_paper_errors(rows)
        return out
    if name == "llmfms":
        return llmfms(rows)
    if name == "multisensebadminton_qa":
        return badminton(rows, opts)
    if name == "lasana_score":
        return lasana_scores(rows)
    if name == "captaincook4d":
        return cooking(rows, opts)
    if name == "t3set_comment":
        return {
            "feedback": caption([answer(r) for r in rows], [parsing.gold_text(r) for r in rows], opts),
            "n_valid_rows": sum(bool(answer(r)) for r in rows),
            "n_rounds": len({(r["player_id"], r["technique"], r["round_id"]) for r in rows}),
        }
    raise ValueError(f"No scorer: {name}")


METRIC_SCHEMA_VERSION = 20


def source_weight(block, fmt):
    comp = block.get("metrics", {}).get("component_counts", {})
    return comp.get(fmt) or block.get("coverage", {}).get("expected_rows", 0)


def project_fisher(blocks):
    pairs = []
    for block in blocks:
        per = ((block.get("metrics") or {}).get("project_srcc") or {}).get("per_project") or {}
        for value in per.values():
            rho, n = value.get("srcc"), value.get("valid_rows", 0)
            if rho is not None and n > 0:
                pairs.append((rho, n))
    if not pairs:
        return None
    limit = 1 - 1e-12
    total = sum(n for _, n in pairs)
    z = sum(n * math.atanh(max(-limit, min(limit, rho))) for rho, n in pairs)
    return math.tanh(z / total)


def format_summary(report, model_label="X-AQA (ours)"):
    summary, lines = {}, []
    for part in ("ID", "OOD"):
        blocks = report["groups"].get(part, {})
        per_form = {}
        for fmt, metrics in FORMAT_METRICS.items():
            tasks = {n: b for n, b in blocks.items() if n in SUITE and fmt in SUITE[n][4].split(",")}
            if not tasks:
                continue
            if fmt == "S":
                means = {"srcc": project_fisher(tasks.values())}
            else:
                means = {}
                for metric in metrics:
                    pairs = [(b["metrics"]["primary"].get(metric), source_weight(b, fmt)) for b in tasks.values()]
                    total = sum(w for _, w in pairs)
                    means[metric] = (
                        sum(v * w for v, w in pairs) / total if total and all(v is not None for v, _ in pairs) else None
                    )
            per_form[fmt] = {
                "form": FORMAT_NAME[fmt],
                "sources": len(tasks),
                "instances": sum(
                    b["metrics"].get("component_counts", {}).get(fmt, b["coverage"]["expected_rows"])
                    for b in tasks.values()
                ),
                "mean": means,
                "complete": all(v is not None for v in means.values()),
                "per_source": {
                    n: {k: b["metrics"]["primary"].get(k) for k in metrics} for n, b in sorted(tasks.items())
                },
            }
            shown = ", ".join(f"{k}={v:.4f}" if v is not None else f"{k}=--" for k, v in means.items())
            lines.append(
                f"{part:3s} {fmt} ({FORMAT_NAME[fmt]:8s}) {len(tasks):2d} sources, "
                f"{per_form[fmt]['instances']:5d} instances: {shown}"
            )
        summary[part] = per_form
    cells = [
        f"{summary[part][fmt]['mean'][metric] * 100:.1f}" if summary[part][fmt]["mean"][metric] is not None else "--"
        for part in ("ID", "OOD")
        for fmt in "SCT"
        if summary[part].get(fmt)
        for metric in FORMAT_METRICS[fmt]
    ]
    complete = all(f["complete"] for part in summary.values() for f in part.values())
    summary["metric_directions"] = METRIC_DIRECTIONS.copy()
    summary["table"] = "\n".join(lines)
    summary["latex_row"] = (model_label + " & " + " & ".join(cells) + r" \\") if complete else None
    summary["note"] = (
        "Sample-size-weighted cell aggregation: S is a valid-row-weighted "
        "Fisher-z mean over separately ranked projects; C/T are weighted means "
        "over sources. Each source is scored under its own protocol and rows of "
        "different sources are never pooled. S is SRCC, C macro-F1 over the gold "
        "classes and unweighted kappa, T BERTScore F1; all are "
        "higher-is-better. Raw values; LaTeX values are multiplied by 100. An "
        "incomplete cell emits no LaTeX row."
    )
    return summary


def domain_summary(report):
    summary = {part: {} for part in ("ID", "OOD")}
    selected = {name for group in report["groups"].values() for name in group}
    selected.update(report.get("missing_tasks", []))
    by_cell = defaultdict(list)
    for name in SUITE:
        if name not in selected:
            continue
        source = dataset_for_artifact(name)
        if source is None or source.benchmark_type != SUITE[name][1]:
            raise ValueError(f"Unknown or inconsistent domain assignment: {name}")
        for form in SUITE[name][4].split(","):
            by_cell[(SUITE[name][1], source.domain, form)].append(name)
    for part in summary:
        blocks = report["groups"].get(part, {})
        for domain in DOMAIN_FOLDERS:
            cells = {}
            for form, metrics in FORMAT_METRICS.items():
                names = by_cell.get((part, domain, form))
                if not names:
                    continue
                per_source = {
                    name: {
                        metric: blocks.get(name, {}).get("metrics", {}).get("primary", {}).get(metric)
                        for metric in metrics
                    }
                    for name in names
                }
                weights = {name: source_weight(blocks.get(name, {}), form) for name in names}
                if form == "S":
                    means = {"srcc": project_fisher(blocks.get(n, {}) for n in names)}
                else:
                    means = {}
                    for metric in metrics:
                        pairs = [(per_source[name][metric], weights[name]) for name in names]
                        total = sum(w for _, w in pairs)
                        means[metric] = (
                            sum(v * w for v, w in pairs) / total
                            if total and all(v is not None for v, _ in pairs)
                            else None
                        )
                cells[form] = {
                    "form": FORMAT_NAME[form],
                    "sources": len(names),
                    "instances": sum(
                        blocks.get(name, {})
                        .get("metrics", {})
                        .get("component_counts", {})
                        .get(form, blocks.get(name, {}).get("coverage", {}).get("expected_rows", SUITE[name][2]))
                        for name in names
                    ),
                    "mean": means,
                    "complete": all(v is not None for v in means.values()),
                    "per_source": per_source,
                }
            if cells:
                summary[part][domain] = cells
    return summary


def equal_group_mean(blocks, metrics, expected_groups):
    blocks = list(blocks)
    return {
        key: (
            sum(b[key] for b in blocks) / expected_groups
            if len(blocks) == expected_groups and all(b.get(key) is not None for b in blocks)
            else None
        )
        for key in metrics
    }


def score_dataset(name, rows, opts):
    primary, out = format_metrics(name, rows, opts)
    out["primary"] = primary
    try:
        out["source_protocol"] = score_dataset_details(name, rows, opts)
        out["source_protocol"].pop("n_valid_rows", None)
    except (ValueError, KeyError):
        out["source_protocol"] = None
    if getattr(opts, "action_breakdown", False):
        overlap = action_breakdown(name, rows, opts)
        if overlap is not None:
            out["action_overlap"] = overlap
    return out


def evaluate(name, prediction_path, reference_path, opts):
    refs = read(reference_path, opts)
    if not refs:
        raise ValueError(f"Empty reference: {reference_path}")
    rows, coverage = join_predictions(refs, read(prediction_path, opts))
    result = score_dataset(name, rows, opts)
    valid = result.pop("n_valid_rows")
    expected = expected_rows(name, reference_path)
    coverage.update(
        valid_rows=valid,
        invalid_or_missing_rows=len(rows) - valid,
        parse_failure_rows=coverage["predicted_rows"] - valid,
        parse_failure_rate=(
            (coverage["predicted_rows"] - valid) / coverage["predicted_rows"] if coverage["predicted_rows"] else None
        ),
        valid_rate=valid / len(rows),
        table_expected_rows=expected,
        table_count_matches=len(rows) == expected,
    )
    return {
        "dataset": SUITE[name][0],
        "type": SUITE[name][1],
        "supervision_format": SUITE[name][4],
        "prediction_file": str(prediction_path),
        "reference_file": str(reference_path),
        "prediction_sha256": hashlib.sha256(Path(prediction_path).read_bytes()).hexdigest(),
        "reference_sha256": reference_sha(reference_path),
        "coverage": coverage,
        "benchmark_complete": not coverage["missing_rows"] and coverage["table_count_matches"],
        "text_metrics_complete": (
            all(result["primary"].get(k) is not None for k in FORMAT_METRICS["T"]) if "T" in SUITE[name][4] else True
        ),
        "metrics": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "predictions", nargs="*", type=Path, help="JSONL files, named <prefix>.jsonl or <prefix>_test.jsonl"
    )
    parser.add_argument("--prediction-dir", type=Path, help="Load the selected suite from this directory")
    add_dataset_arguments(parser)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(SUITE), help=f"Default: all {len(SUITE)} tasks with --prediction-dir"
    )
    parser.add_argument("--type", choices=("all", "ID", "OOD"), default="all")
    parser.add_argument(
        "--format", choices=("all", "S", "C", "T"), default="all", help="Restrict to one supervision form"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "evaluate/results/metrics.json")
    parser.add_argument("--list", action="store_true", help="List task prefixes and reference paths")
    parser.add_argument("--model-label", default="X-AQA (ours)", help="Model label in both exported LaTeX rows")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit successfully for missing/invalid predictions; still report coverage",
    )
    parser.add_argument(
        "--action-breakdown",
        action="store_true",
        help="Add per-action and O1/O2 action-overlap primary metrics for the hand-mapped OOD sources (diagnostic; omitted by default)",
    )
    opts = parser.parse_args(argv)
    opts.skip_text = False
    opts.bert_model = BERT_MODEL
    opts.bert_batch_size = 16
    selected = [
        n
        for n in (opts.datasets or SUITE)
        if (opts.type == "all" or SUITE[n][1] == opts.type)
        and (opts.format == "all" or opts.format in SUITE[n][4].split(","))
    ]
    selected = version_tasks(selected, opts.reference_dir)
    if opts.list:
        refs, _ = load_reference_set(opts, selected, SUITE)
        for n in selected:
            print(n, SUITE[n][1], SUITE[n][4], SUITE[n][2], refs[n], sep="\t")
        return 0
    if bool(opts.predictions) == bool(opts.prediction_dir):
        parser.error("Pass prediction files OR --prediction-dir")
    paths = {}
    if opts.prediction_dir:
        for name in selected:
            candidates = [opts.prediction_dir / (name + suffix) for suffix in (".jsonl", "_test.jsonl")]
            existing = [p for p in candidates if p.is_file()]
            if len(existing) > 1:
                raise ValueError(f"Ambiguous result files for {name}")
            if existing:
                paths[name] = existing[0]
    else:
        for path in opts.predictions:
            name = path.stem.removesuffix("_test")
            if name not in SUITE:
                parser.error(f"Unknown prediction filename: {path.name}")
            if name in paths:
                parser.error(f"Duplicate task: {name}")
            if name in selected:
                paths[name] = path
        if opts.datasets is None:
            selected = [n for n in selected if n in paths]
    if not selected:
        parser.error("No datasets selected")
    refs, dataset_identity = load_reference_set(opts, list(paths), SUITE)
    protected = list(paths.values()) + [r for r in refs.values() if isinstance(r, Path)]
    if opts.out.resolve() in {p.resolve() for p in protected}:
        parser.error("--out must not overwrite a prediction or reference file")
    report = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "dataset": dataset_identity,
        "groups": {"ID": {}, "OOD": {}},
        "missing_tasks": [n for n in selected if n not in paths],
        "notes": [
            "Metrics are raw fractions, not percentages. No heterogeneous ID/OOD scalar average.",
            "Five primary metrics: S SRCC, C macro-F1/kappa, T BERTScore F1; all higher-is-better.",
            "A form cell is the sample-size-weighted mean over its sources; rows of different sources are never pooled.",
            "Type follows the supplied table; it does not verify training-data exclusion.",
            "Classification failures count as misses; numeric metrics are valid-only.",
        ],
        "text_backend": {
            "name": "eval.metrics.text_metrics",
            "skipped": opts.skip_text,
            "bert_model": opts.bert_model,
            "bert_batch_size": opts.bert_batch_size,
        },
    }
    for name, path in paths.items():
        print(f"Evaluating {name}", file=sys.stderr)
        reference = refs[name]
        block = evaluate(name, path, reference, opts)
        report["groups"][SUITE[name][1]][name] = block
    blocks = [b for group in report["groups"].values() for b in group.values()]
    report["complete"] = not report["missing_tasks"] and all(
        b["benchmark_complete"] and b["text_metrics_complete"] for b in blocks
    )
    report["format_summary"] = format_summary(report, opts.model_label)
    report["domain_summary"] = domain_summary(report)
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    opts.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(report["format_summary"]["table"])
    print(f'Report: {opts.out}; complete={report["complete"]}')
    return 0 if report["complete"] or opts.allow_partial else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
