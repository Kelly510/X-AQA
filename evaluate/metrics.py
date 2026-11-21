#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import warnings

BERT_MODEL = str(Path(__file__).resolve().parents[1] / "models" / "roberta-large")

def bertscore_f1(preds: list[str], golds: list[str], model: str, batch_size: int):
    from bert_score import score as bert_score
    from bert_score.utils import model2layers

    _patch_bert_score_empty()
    num_layers = model2layers.get(model, 17)
    _, _, f1 = bert_score(
        preds,
        golds,
        model_type=model,
        num_layers=num_layers,
        lang="en",
        batch_size=batch_size,
        rescale_with_baseline=False,
        verbose=False,
    )
    return f1.mean().item()


def _patch_bert_score_empty() -> None:
    from bert_score import utils

    if getattr(utils.sent_encode, "_uniaqa_patched", False):
        return
    original = utils.sent_encode

    def sent_encode(tokenizer, sent):
        if not (sent or "").strip():
            return tokenizer.encode("", add_special_tokens=True)
        return original(tokenizer, sent)

    sent_encode._uniaqa_patched = True
    utils.sent_encode = sent_encode


def text_metrics(
    preds: list[str],
    golds: list[str],
    *,
    bert_model: str | None = None,
    bert_batch_size: int = 64,
) -> dict:
    if len(preds) != len(golds):
        raise ValueError("Text prediction/reference length mismatch")
    if not preds:
        return {"n": 0}
    out = {
        "n": len(preds),
        "empty_pred_rate": sum(1 for p in preds if not p.strip()) / len(preds),
    }
    if bert_model:
        out["bertscore_f1"] = bertscore_f1(preds, golds, bert_model, bert_batch_size)
    return out


def balanced_accuracy(golds: list, preds: list) -> dict:
    if len(golds) != len(preds):
        raise ValueError("Classification gold/prediction lengths differ")
    if not golds:
        return {"n": 0, "balanced_accuracy": None}
    classes = list(dict.fromkeys(golds))
    recalls = [sum(g == c and p == c for g, p in zip(golds, preds)) / golds.count(c) for c in classes]
    return {"n": len(golds), "balanced_accuracy": sum(recalls) / len(recalls)}


def regression_metrics(golds: list[float], preds: list[float | None], *, span=None) -> dict:
    if len(golds) != len(preds):
        raise ValueError("Regression gold/prediction lengths differ")
    if any(not math.isfinite(g) for g in golds):
        raise ValueError("Non-finite reference score")
    if span is None:
        span = max(golds) - min(golds) if golds else 0.0
    pairs = [(g, p) for g, p in zip(golds, preds) if p is not None and math.isfinite(p)]
    out = {
        "n": len(golds),
        "n_valid": len(pairs),
        "coverage": len(pairs) / len(golds) if golds else 0.0,
        "score_range": span,
        "spearman": None,
        "mae": None,
        "mse": None,
        "relative_l2": None,
        "relative_l2_x100": None,
        "population": "valid_predictions_only",
    }
    if pairs:
        gs, ps = zip(*pairs)
        mse = sum((g - p) ** 2 for g, p in pairs) / len(pairs)
        rl2 = mse / span**2 if span > 0 else None
        out.update(
            spearman=spearman_rho(gs, ps),
            mse=mse,
            mae=sum(abs(g - p) for g, p in pairs) / len(pairs),
            relative_l2=rl2,
            relative_l2_x100=100 * rl2 if rl2 is not None else None,
        )
    return out


def binary_prf(golds: list[bool], preds: list[bool], *, positive: bool) -> dict:
    if not golds:
        return {"n": 0}
    tp = sum(1 for g, p in zip(golds, preds) if p == positive and g == positive)
    fp = sum(1 for g, p in zip(golds, preds) if p == positive and g != positive)
    fn = sum(1 for g, p in zip(golds, preds) if p != positive and g == positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(golds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy(golds, preds)["balanced_accuracy"],
        "positive_rate_gold": sum(1 for g in golds if g == positive) / len(golds),
        "positive_rate_pred": sum(1 for p in preds if p == positive) / len(preds),
    }


def average_precision(golds: list[bool], scores: list[float], *, positive: bool) -> float:
    if not golds:
        return 0.0
    order = sorted(range(len(golds)), key=lambda i: -scores[i])
    total_positive = sum(1 for g in golds if g == positive)
    if not total_positive:
        return 0.0
    ap = tp = seen = 0.0
    index = 0
    while index < len(order):
        stop = index
        while stop < len(order) and scores[order[stop]] == scores[order[index]]:
            stop += 1
        recall_before = tp / total_positive
        for position in order[index:stop]:
            tp += golds[position] == positive
        seen = stop
        precision = tp / seen
        ap += (tp / total_positive - recall_before) * precision
        index = stop
    return ap


def score_metrics(golds: list[float], preds: list[float], *, groups: list | None = None, margin: float = 1.5) -> dict:
    if len(golds) < 2:
        return {"n": len(golds)}
    from scipy.stats import spearmanr

    rho = spearmanr(golds, preds).statistic
    groups = groups or [None] * len(golds)
    pairs = hits = 0
    for i in range(len(golds)):
        for j in range(len(golds)):
            if groups[i] == groups[j] and golds[i] > golds[j] + margin:
                pairs += 1
                hits += preds[i] > preds[j]
    return {
        "n": len(golds),
        "spearman": float(rho) if rho == rho else 0.0,
        "mae": sum(abs(g - p) for g, p in zip(golds, preds)) / len(golds),
        "pairwise_rank_acc": hits / pairs if pairs else 0.0,
        "n_pairs": pairs,
    }


def spearman_rho(golds: list[float], preds: list[float]):
    if len(golds) < 2:
        return None
    from scipy.stats import spearmanr

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")
        rho = spearmanr(golds, preds).statistic
    return float(rho) if rho == rho else None


def fisher_z_mean(rhos: list[float]):
    usable = [r for r in rhos if r is not None]
    if not usable:
        return None
    limit = 1 - 1e-12
    zs = [math.atanh(max(-limit, min(limit, r))) for r in usable]
    return math.tanh(sum(zs) / len(zs))


def relative_l2(golds: list[float], preds: list[float], *, span: float) -> float:
    if not golds or span <= 0:
        return 0.0
    return sum(abs(g - p) for g, p in zip(golds, preds)) / len(golds) / span


def macro_f1(golds: list, preds: list, *, classes=None):
    if not golds:
        return None
    labels = list(classes) if classes is not None else sorted(set(golds), key=str)
    if not labels:
        return None
    total = 0.0
    for label in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        total += 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return total / len(labels)


def cohen_kappa(golds: list, preds: list, *, classes=None):
    if not golds:
        return None
    labels = set(classes) if classes is not None else set(golds) | set(preds)
    labels |= set(golds) | set(preds)
    n = len(golds)
    observed = sum(1 for g, p in zip(golds, preds) if g == p) / n
    expected = sum(
        (sum(1 for g in golds if g == label) / n) * (sum(1 for p in preds if p == label) / n) for label in labels
    )
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)
