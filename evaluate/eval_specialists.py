#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "data"))

import evaluate as ev
from metrics import fisher_z_mean

TASKS = {
    "mtlaqa": {
        "display": "MTL-AQA",
        "domain": "Sports",
        "references": ["data/exports/original/sports/mtlaqa_test.jsonl"],
        "metric": "SRCC / R-L2 / CCC",
    },
    "fitnessaqa_kp": {
        "display": "Fitness-AQA",
        "domain": "Fitness",
        "references": ["data/exports/original/fitness/fitnessaqa_kp_test.jsonl"],
        "metric": "per-error F1",
    },
    "kimore_cts": {
        "display": "KIMORE",
        "domain": "Rehab",
        "references": ["data/exports/original/rehabilitation/kimore_cts_test.jsonl"],
        "metric": "per-exercise SRCC",
    },
    "lasana_error": {
        "display": "LASANA (error)",
        "domain": "Skill",
        "references": ["data/exports/original/skills/lasana_error_test.jsonl"],
        "metric": "per-error accuracy",
    },

}

DEFAULT_TASKS = list(TASKS)


COMPARABLE = {
    "mtlaqa": True,
    "fitnessaqa_kp": True,
    "kimore_cts": True,
    "lasana_error": True,
}


FITNESS_GROUP_ORDER = ("BS_KIE", "BS_KFE", "OHP_ELBOW", "OHP_KNEE")
FITNESS_DIAGNOSTIC = {
    "BS_SS": "paper number is Waseda Squat, not Fitness-AQA",
    "BR_LUMBAR": "local export has no BarbellRow",
    "BR_TORSO": "local export has no BarbellRow",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


SPECIALIST_REFS = ROOT / "benchmark" / "specialist_refs"


def load_references(task: str) -> list[dict]:
    rows: list[dict] = []
    paths = []
    fallback = SPECIALIST_REFS / f"{task}_test.jsonl"
    for rel in TASKS[task]["references"]:
        path = ROOT / rel
        if not path.is_file():
            path = fallback
        paths.append(path)
    for path in dict.fromkeys(paths):
        if not path.is_file():
            raise FileNotFoundError(f"{task}: missing reference {path}")
        rows.extend(ev.read(path))
    return rows


def prediction_path(pred_dir: Path, task: str) -> Path | None:
    for name in (f"{task}.jsonl", f"{task}_test.jsonl"):
        path = pred_dir / name
        if path.is_file():
            return path
    return None


def _row_id(row: dict):
    return row.get("sample_id") or row.get("id")


def _row_key(row: dict):
    media = row.get("videos") or row.get("images") or []
    if isinstance(media, str):
        media = [media]
    return (tuple(media), ev.parsing.user_text(row))


def join_rows(refs: list[dict], preds: list[dict]) -> tuple[list[dict], dict]:

    if preds and all(
        isinstance(p.get("_benchmark_index"), int) and 0 <= p["_benchmark_index"] < len(refs) for p in preds
    ):

        def media(row):
            value = row.get("videos") or row.get("images") or []
            return value if isinstance(value, list) else [value]

        if all(media(refs[p["_benchmark_index"]]) == media(p) for p in preds):
            joined = [{**r, "response": "", "labels": ev.parsing.gold_text(r)} for r in refs]
            for p in preds:
                joined[p["_benchmark_index"]]["response"] = p.get("response", "")
            return joined, {
                "expected_rows": len(refs),
                "predicted_rows": len(preds),
                "missing_rows": 0,
                "prediction_coverage": 1.0,
                "join": "benchmark_index",
            }

    have_ids = all(_row_id(r) for r in refs) and all(_row_id(p) for p in preds)
    if have_ids:
        try:
            return ev.join_predictions(refs, preds)
        except ValueError:

            pass

    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, ref in enumerate(refs):
        buckets[_row_key(ref)].append(i)

    joined = [{**r, "response": "", "labels": ev.parsing.gold_text(r)} for r in refs]
    matched = 0
    for pred in preds:
        if "response" not in pred:
            raise ValueError("prediction row has no string response")
        candidates = buckets.get(_row_key(pred), [])
        if len(candidates) != 1:
            continue
        i = candidates[0]
        if joined[i]["response"]:
            continue
        joined[i]["response"] = pred["response"]
        matched += 1

    if matched < len(refs) and len(refs) == len(preds):

        ok = True
        for ref, pred in zip(refs, preds):
            ru, pu = ev.parsing.user_text(ref), ev.parsing.user_text(pred)
            if ru and pu and ru != pu:
                ok = False
                break
        if ok:
            joined = [
                {**ref, "response": pred["response"], "labels": ev.parsing.gold_text(ref)}
                for ref, pred in zip(refs, preds)
            ]
            matched = len(refs)
            kind = "index"
        else:
            kind = "media+prompt"
    else:
        kind = "media+prompt"

    coverage = {
        "expected_rows": len(refs),
        "predicted_rows": matched,
        "missing_rows": len(refs) - matched,
        "prediction_coverage": matched / len(refs) if refs else 0.0,
        "join": kind,
    }
    return joined, coverage


def accuracy(golds, preds) -> float | None:
    if not golds:
        return None
    return sum(1 for g, p in zip(golds, preds) if g is not None and p == g) / len(golds)


def mean(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def error_f1(block: dict) -> float | None:
    entry = block.get("per_class", {}).get("no")
    if not entry or not entry.get("support"):
        return None
    return entry["f1"]


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def score_mtlaqa(rows, opts):
    def award(text):
        data = ev.obj(text)
        if data is None:
            return None
        data = ev.parsing.canonical_keys(data, ("execution_award", "comment"))
        return ev.number(data.get("execution_award"), 0, 10, 0.5)

    golds, preds = [], []
    for row in rows:
        gold = row.get("final_score")
        if gold is None:
            raise ValueError("MTL-AQA: row is missing final_score")
        difficulty = float(row.get("difficulty") or 0.0)
        value = award(ev.answer(row))
        pred = difficulty * 3.0 * value if value is not None and difficulty else None
        golds.append(float(gold))
        preds.append(pred)
    block = ev.numeric(golds, preds)
    return {
        "srcc": block["spearman"],
        "r_l2_x100": block["relative_l2_x100"],
        "r_l2": block["relative_l2"],
        "mae": block["mae"],
        "n": block["n"],
        "n_valid": block["n_valid"],
        "coverage": block["coverage"],
        "target": "final_score",
    }, (
        "Prediction = difficulty x 3 x execution_award; gold = released final_score. "
        "R-L2 is squared relative error x100, matching the specialist tables."
    )


def _fitness_group(row) -> str | None:
    user = ev.parsing.user_text(row) or ""
    exercise = ""
    point = ""
    for line in user.splitlines():
        if line.startswith("Exercise:"):
            exercise = line.split(":", 1)[1].strip().lower()
        elif line.startswith("Technical point:"):
            point = line.split(":", 1)[1].strip().lower()
    if "back squat" in exercise:
        if "full depth" in point:
            return "BS_SS"
        if "tracking outward" in point or "cave inward" in point:
            return "BS_KIE"
        if "shins close to vertical" in point or "traveling far forward" in point:
            return "BS_KFE"
    if "overhead press" in exercise:
        if "elbows" in point:
            return "OHP_ELBOW"
        if "knees" in point:
            return "OHP_KNEE"
    return None


def score_fitnessaqa(rows, opts):
    classes = ("yes", "no")
    golds = [ev.category("fitnessaqa_kp", ev.parsing.gold_text(r), classes) for r in rows]
    preds = [ev.category("fitnessaqa_kp", ev.answer(r), classes) for r in rows]
    groups = defaultdict(list)
    for row, gold, pred in zip(rows, golds, preds):
        key = _fitness_group(row)
        if key:
            groups[key].append((gold, pred))

    def block_for(key):
        pairs = groups.get(key, [])
        if not pairs:
            return {"f1_error": None, "accuracy": None, "n": 0}
        block = ev.classify([g for g, _ in pairs], [p for _, p in pairs], classes)
        return {
            "f1_error": error_f1(block),
            "accuracy": accuracy([g for g, _ in pairs], [p for _, p in pairs]),
            "n": block["n"],
            "n_valid": block["n_valid"],
        }

    per_group = {key: block_for(key) for key in FITNESS_GROUP_ORDER}
    diagnostic = {key: block_for(key) for key in FITNESS_DIAGNOSTIC}
    available = [v["f1_error"] for v in per_group.values() if v["f1_error"] is not None]
    return {
        "per_group": per_group,
        "diagnostic": diagnostic,
        "diagnostic_note": FITNESS_DIAGNOSTIC,
        "macro_f1_error": mean(available),
        "n": len(rows),
        "n_valid": sum(p is not None for p in preds),
        "coverage": sum(p is not None for p in preds) / len(rows) if rows else 0.0,
    }, (
        "Direct comparison uses the four points present in both the local export and "
        "the paper: BS-KIE, BS-KFE, OHP-Elbow, OHP-Knees. BS-SS is diagnostic only "
        "(the paper's SS number is Waseda Squat); the local export has no BarbellRow."
    )


def score_kimore(rows, opts):
    golds = [r.get("cts") for r in rows]
    if any(g is None for g in golds):
        raise ValueError("KIMORE: row is missing cts")
    preds = [ev.scalar("kimore_cts", ev.answer(r), r) for r in rows]
    per_exercise = defaultdict(lambda: ([], []))
    for row, gold, pred in zip(rows, golds, preds):
        key = f"E{row.get('exercise_id')}"
        per_exercise[key][0].append(gold)
        per_exercise[key][1].append(pred)
    per = {key: ev.primary_srcc(gs, ps) for key, (gs, ps) in sorted(per_exercise.items())}
    pooled = ev.primary_srcc(golds, preds)
    return {
        "srcc_pooled": pooled,
        "per_exercise": per,
        "srcc_fisher_z": fisher_z_mean(list(per.values())),
        "srcc_arithmetic_mean": mean([v for v in per.values() if v is not None]),
        "n": len(rows),
        "n_valid": sum(p is not None for p in preds),
        "coverage": sum(p is not None for p in preds) / len(rows) if rows else 0.0,
    }, "X-AQA is OOD zero-shot; the specialist is 5-fold trained. SRCC is scale-free."


LASANA_PAPER_GROUPS = {
    "peg_transfer": ("object_dropped_within_fov",),
    "circle_cutting": ("gauze_detached",),
    "balloon_resection": ("balloon_opened", "balloon_damaged"),
    "suture_and_knot": ("needle_dropped",),
}


def score_lasana_error(rows, opts):
    classes = ("Yes", "No")
    by_recording = defaultdict(dict)
    for row in rows:
        by_recording[(row.get("task"), row.get("recording_id"))][row.get("error_category")] = row

    per_group = {}
    for task, wanted in LASANA_PAPER_GROUPS.items():
        pairs = []
        for (t, recording), cats in sorted(by_recording.items()):
            if t != task or any(c not in cats for c in wanted):
                continue
            gold = any(ev.category("lasana_error", ev.parsing.gold_text(cats[c]), classes) == "Yes" for c in wanted)
            preds = [ev.category("lasana_error", ev.answer(cats[c]), classes) for c in wanted]
            pred = True if "Yes" in preds else (False if all(p == "No" for p in preds) else None)
            pairs.append((gold, pred))
        block = ev.classify([g for g, _ in pairs], [p for _, p in pairs], (False, True))
        per_group[task] = {
            "error": " OR ".join(wanted),
            "accuracy": accuracy([g for g, _ in pairs], [p for _, p in pairs]),
            "balanced_accuracy": block["balanced_accuracy"],
            "n": block["n"],
            "n_valid": block["n_valid"],
        }

    n_recordings = sum(v["n"] for v in per_group.values())
    n_valid = sum(v["n_valid"] for v in per_group.values())
    return {
        "comparable": True,
        "per_group": per_group,
        "macro_accuracy": mean([v["accuracy"] for v in per_group.values()]),
        "macro_balanced_accuracy": mean([v["balanced_accuracy"] for v in per_group.values()]),
        "n": len(rows),
        "n_recordings": n_recordings,
        "n_valid": n_valid,
        "coverage": n_valid / n_recordings if n_recordings else 0.0,
    }, (
        "Four representative groups: object_dropped_within_fov, gauze_detached, "
        "balloon_opened OR balloon_damaged, needle_dropped. Paper test macro: "
        "accuracy 0.8638, balanced accuracy 0.8535."
    )


SCORERS = {
    "mtlaqa": score_mtlaqa,
    "fitnessaqa_kp": score_fitnessaqa,
    "kimore_cts": score_kimore,
    "lasana_error": score_lasana_error,
}


def markdown_report(results: dict) -> str:
    lines = [
        "| Dataset | Direct | Coverage | X-AQA metrics |",
        "| --- | :---: | ---: | --- |",
    ]
    for task, block in results.items():
        if block.get("status") != "ok":
            lines.append(f"| {block['display']} | — | — | {block.get('status')} |")
            continue
        cov = block["coverage"]
        expected = cov.get("expected_rows", 0)
        ratio = cov.get("predicted_rows", 0) / expected if expected else 0
        metrics = block["metrics"]
        parts = []
        for key, value in metrics.items():
            if key == "coverage" or isinstance(value, dict) or value is None:
                continue
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
        per = metrics.get("per_group") or metrics.get("per_task") or metrics.get("per_exercise")
        if isinstance(per, dict):
            for k, v in per.items():
                if isinstance(v, dict):
                    if v.get("f1_error") is not None:
                        parts.append(f"{k}_f1={fmt(v['f1_error'])}")
                    elif v.get("accuracy") is not None:
                        parts.append(f"{k}_acc={fmt(v['accuracy'])}")
                elif isinstance(v, float):
                    parts.append(f"{k}={fmt(v)}")
        cls = metrics.get("classification")
        if isinstance(cls, dict):
            parts.append(f"maf1={fmt(cls.get('macro_f1'))}")
            parts.append(f"acc={fmt(cls.get('accuracy'))}")
            parts.append(f"kappa={fmt(cls.get('kappa'))}")
        direct = "✅" if block.get("comparable") else "❌"
        lines.append(
            f"| {block['display']} | {direct} | {cov.get('predicted_rows', 0)}/{expected} "
            f"({ratio:.0%}) | {'; '.join(parts)} |"
        )
    return "\n".join(lines) + "\n"


AIXSUTURE_THRESHOLDS = (15.5, 23.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--pred-dir", type=Path, required=True, help="Directory containing <task>.jsonl or <task>_test.jsonl"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=DEFAULT_TASKS, choices=DEFAULT_TASKS, help="Subset of tasks (default: all)"
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "results" / "specialist_metrics.json", help="JSON report path"
    )
    parser.add_argument("--md", type=Path, help="Optional Markdown summary path")
    args = parser.parse_args()

    opts = SimpleNamespace(skip_text=True, bert_model=None, bert_batch_size=64)
    results = {}
    for task in args.tasks:
        cfg = TASKS[task]
        refs = load_references(task)
        pred_path = prediction_path(args.pred_dir, task)
        if pred_path is None:
            results[task] = {
                "display": cfg["display"],
                "domain": cfg["domain"],
                "status": "no_predictions",
                "expected_rows": len(refs),
                "prediction_file": None,
            }
            continue
        preds = load_jsonl(pred_path)
        joined, coverage = join_rows(refs, preds)
        metrics, note = SCORERS[task](joined, opts)
        results[task] = {
            "display": cfg["display"],
            "domain": cfg["domain"],
            "status": "ok",
            "comparable": COMPARABLE[task],
            "prediction_file": str(pred_path),
            "reference_files": cfg["references"],
            "coverage": coverage,
            "metrics": metrics,
            "note": note,
        }
        flat = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items() if not isinstance(v, dict)}
        print(
            f"{cfg['display']:16s} {coverage.get('predicted_rows', 0):5d}/"
            f"{coverage.get('expected_rows', 0):<5d} {json.dumps(flat)[:180]}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}")
    if args.md:
        args.md.write_text(markdown_report(results))
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
