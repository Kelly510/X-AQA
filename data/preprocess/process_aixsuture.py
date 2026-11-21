#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocess.paths import workspace
from data.preprocess import frame_budget

DATASET = workspace() / "datasets" / "AIxSuture"
SPLITS_FILE = workspace() / "data" / "annotations" / "aixsuture_splits.json"
WINDOW_SEC = 32.0
OUT = REPO_ROOT / "data" / "exports" / "original" / "skills"


ELEMENTS = (
    ("Respect for tissue", {
        1: "Often used unnecessary force on tissue or caused damage by inappropriate use of instruments",
        3: "Careful handling of tissue but occasionally caused inadvertent damage",
        5: "Consistently handled tissues appropriately with minimal damage"}),
    ("Time and motion", {
        1: "Many unnecessary moves",
        3: "Efficient time and motion, but some unnecessary moves",
        5: "Economy of movement and maximum efficiency"}),
    ("Instrument handling", {
        1: "Repeatedly makes tentative or awkward moves with instruments",
        3: "Competent use of instruments, although occasionally appeared stiff or awkward",
        5: "Fluid moves with instruments and no awkwardness"}),
    ("Suture technique", {
        1: "Clumsy and insecure, inadequately tying knots and inability to maintain tension",
        3: "Careful and slow with most knots placed properly with proper tension",
        5: "Excellent control of the suture, with proper placement of knots and correct tension"}),
    ("Flow of operation and forward planning", {
        1: "Frequently stopped operation or needed to discuss next move",
        3: "Demonstrated ability for forward planning with steady progression of operative procedure",
        5: "Obviously planned course of operation with effortless flow from one move to the next"}),
    ("Knowledge of specific procedure", {
        1: "Deficient knowledge; needed specific instruction at most operative steps",
        3: "Knew all important aspects of the operation",
        5: "Demonstrated familiarity with all aspects of the operation"}),
    ("Quality of final product", {1: "Very poor", 3: "Competent", 5: "Exceptional"}),
    ("Overall performance", {1: "Very poor", 3: "Competent", 5: "Exceptional"}),
)

COLUMNS = {
    "Respect for tissue": "OSATS_RESPECT",
    "Time and motion": "OSATS_MOTION",
    "Instrument handling": "OSATS_INSTRUMENT",
    "Suture technique": "OSATS_SUTURE",
    "Flow of operation and forward planning": "OSATS_FLOW",
    "Knowledge of specific procedure": "OSATS_KNOWLEDGE",
    "Quality of final product": "OSATS_FINAL_QUALITY",
    "Overall performance": "OSATS_PERFORMANCE",
}

SYSTEM = """You are a surgical skills examiner. Evaluate the open suturing performance in the video.
The subject performs single interrupted sutures on an ethylene-vinyl acetate bench model with a
needle holder, forceps and scissors; judge only the performance visible in the video.

Score the performance on the eight dimensions of the Objective Structured Assessment of Technical
Skills (OSATS) global rating score. Apply each dimension to the suturing task and use a five-point
Likert scale. The rubric gives anchor descriptions for the lowest anchor (score value 1), the middle
anchor (score value 3) and the highest anchor (score value 5); use score values 2 or 4 when the
performance falls between adjacent anchors. The numbers in the rubric are score values.

%s

Return only this JSON object and nothing else, with one numeric score for each dimension:
%s""" % (
    "\n\n".join(
        name + ":\n" + "\n".join(f"- Anchor {k}: {v}" for k, v in anchors.items())
        for name, anchors in ELEMENTS
    ),
    "{" + ", ".join(f'"{name}": <integer from 1 to 5>' for name, _ in ELEMENTS) + "}",
)

USER = "Assess the suturing performance in the following video.\n\n<video>"


def durations(paths: list[Path]) -> dict[str, float]:
    cache = REPO_ROOT / "outputs" / "analysis" / "aixsuture.durations.json"
    return frame_budget.probe_durations(paths, workers=32, cache_path=cache)


def item_medians(rows) -> dict[str, int]:
    out = {}
    for name, column in COLUMNS.items():
        values = sorted(float(v) for v in rows[column].tolist() if v == v)
        out[name] = int(values[len(values) // 2]) if len(values) % 2 else int(values[len(values)//2 - 1])
    return out


def raw_ratings(rows) -> dict:
    out = {}
    for _, row in rows.iterrows():
        rater = str(row["INVESTIGATOR"])
        out[rater] = {name: int(row[column]) for name, column in COLUMNS.items()
                      if row[column] == row[column]}
        out[rater]["grs"] = int(row["GLOBA_RATING_SCORE"])
    return dict(sorted(out.items()))


def build(split: str, args) -> list[dict]:
    import pandas as pd

    frame = pd.read_excel(DATASET / "OSATS.xlsx")
    splits = json.loads(SPLITS_FILE.read_text())
    wanted = set(splits["videos"][split])
    frame = frame[frame["VIDEO"].isin(wanted)]
    videos = {vid: DATASET / "videos" / f"{vid}.mp4" for vid in wanted}
    lengths = durations(list(videos.values()))

    out = []
    for video_id, group in frame.groupby("VIDEO"):
        path = videos[video_id]
        if not path.is_file():
            continue
        items = item_medians(group)
        grs = sum(items.values())
        raw = raw_ratings(group)
        grs_mean = round(sum(r["grs"] for r in raw.values()) / len(raw), 3)
        first = group.iloc[0]
        duration = lengths[str(path)]
        for index, (start, end) in enumerate(frame_budget.windows(duration, WINDOW_SEC)):
            out.append({
                "sample_id": f"aixsuture_{video_id}_w{index:02d}",
                "task": "aixsuture_grs",
                "split": split,
                "student_id": str(first["STUDENT"]),
                "group": str(first["GROUP"]),
                "time": str(first["TIME"]),
                "skill": ("novice" if grs < 16 else "intermediate" if grs < 24 else "proficient"),
                "grs": grs,
                "grs_mean": grs_mean,
                "item_scores": items,
                "raw_ratings": raw,
                "score_target": "median_of_3_raters",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER},
                    {"role": "assistant", "content": json.dumps(items, ensure_ascii=False)},
                ],
                "videos": [str(path)],
                "chat_template_kwargs": {
                    "video_start": round(start, 2),
                    "video_end": round(end, 2),
                    "fps": frame_budget.sampling_fps(end - start),
                },
            })
    return sorted(out, key=lambda r: r["sample_id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in ("train", "validation", "test"):
        rows = build(split, args)
        path = args.out / f"aixsuture_grs_{split}.jsonl"
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        total += len(rows)
        print(f"{path.name}: {len(rows)} rows", flush=True)
    print(f"total {total} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
