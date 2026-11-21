#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict

MAX_LOSS_SCALE = 20.0


def balanced_weights(rows, cell_of, class_of) -> dict:
    counts = defaultdict(Counter)
    for row in rows:
        counts[cell_of(row)][class_of(row)] += 1
    weights = {}
    for cell, classes in counts.items():
        target = sum(classes.values()) / len(classes)
        for label, count in classes.items():
            weights[(cell, label)] = target / count
    return weights


def attach(record: dict, weight: float) -> dict:
    weight = min(weight, MAX_LOSS_SCALE)
    if abs(weight - 1.0) > 1e-9:
        record["messages"][-1]["loss_scale"] = round(weight, 6)
    else:
        record["messages"][-1].pop("loss_scale", None)
    return record


def summarise(rows, cell_of, class_of, weights: dict) -> str:
    natural = Counter(class_of(row) for row in rows)
    weighted = Counter()
    for row in rows:
        weighted[class_of(row)] += min(weights[(cell_of(row), class_of(row))], MAX_LOSS_SCALE)
    cells = len({cell_of(row) for row in rows})
    parts = ", ".join(f"{label}: {natural[label]} rows -> {weighted[label]:.0f} weighted"
                      for label in sorted(natural, key=str))
    return (f"{cells} cells, {parts}; total mass {sum(weighted.values()):.0f} "
            f"of {len(rows)} rows")
