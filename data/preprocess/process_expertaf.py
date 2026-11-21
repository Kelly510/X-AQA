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
from collections import Counter
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

from . import frame_budget
from .feedback_style import feedback_person_phrase

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERTAF_ROOT = workspace() / "datasets" / "ExpertAF"
DEFAULT_EGOEXO_ROOT = workspace() / "datasets" / "Ego-Exo4D"


SOURCE_FPS = 30.0


DOWNSCALE_PARTS = ("downscaled", "448")

ACTIVITY_RE = re.compile(r"person doing ([^.]+?)\.\s*Give feedback")


EMITTED_ACTIVITY_RE = re.compile(r"^Activity: (.+)$", re.M)

SYSTEM_PROMPT = (
    "You analyze videos of skilled activities and give one concise, actionable "
    "coaching suggestion in a single English sentence. Write the feedback "
    "{person}."
)


USER_TEMPLATE = (
    "<video>\nActivity: {activity}\nGive feedback on the execution."
)


def read_json(path: Path):
    if not path.is_file():
        sys.exit(f"not found: {path}\n"
                 "download it from the links in https://github.com/thechargedneutron/"
                 "ExpertAF (README -> Dataset and path)")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def activity_of(row: dict) -> str:
    turns = row.get("conversations") or []
    if turns:
        match = ACTIVITY_RE.search(turns[0].get("value") or "")
        if match:
            return match.group(1).strip()
    return ""


def clip_videos(egoexo_root: Path, take: dict, views: str):
    streams = take.get("frame_aligned_videos") or {}
    best = take.get("best_exo")

    def resolve(relative: str) -> Path:
        raw = Path(f"{take['root_dir']}/{relative}")
        return egoexo_root / raw.parent / Path(*DOWNSCALE_PARTS) / raw.name

    ego, exos = None, []
    for camera in sorted(streams):
        if camera in ("collage", "best_exo"):
            continue
        if camera.startswith("aria"):

            relative = (streams[camera].get("rgb") or {}).get("relative_path")
            if relative and (path := resolve(relative)).is_file():
                ego = ego or path
            continue
        relative = (streams[camera].get("0") or {}).get("relative_path")
        if relative and (path := resolve(relative)).is_file():
            exos.append(path)

    best_exo = next((p for p in exos if p.stem == best), exos[0] if exos else None)
    chosen = []
    if views in ("all", "ego") and ego is not None:
        chosen.append(("ego", ego))
    if views in ("all", "best-exo") and best_exo is not None:
        chosen.append(("exo", best_exo))
    return chosen


def build_record(row: dict, kind: str, video: Path, fps: float | None) -> dict:
    start_frame, end_frame = json.loads(row["video1_range"])
    activity = activity_of(row)


    duration = (end_frame - start_frame) / SOURCE_FPS
    answer = row["text"].strip()
    system = SYSTEM_PROMPT.format(person=feedback_person_phrase(answer))
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",
             "content": USER_TEMPLATE.format(activity=activity or "a skilled activity")},
            {"role": "assistant", "content": answer},
        ],
        "videos": [str(video)],
        "chat_template_kwargs": {"video_start": round(start_frame / SOURCE_FPS, 2),
                                 "video_end": round(end_frame / SOURCE_FPS, 2),
                                 "fps": frame_budget.resolve_fps(duration, fps)},


        "view": kind,
    }


def convert(rows: list[dict], takes: dict, egoexo_root: Path, fps: float,
            dedup: bool, views: str):
    records, stats, seen = [], Counter(), set()
    for row in rows:
        if dedup:
            key = (row["take"], row["expert_id"], row["commentary_idx"])
            if key in seen:
                stats["duplicate_commentary"] += 1
                continue
            seen.add(key)
        if not (row.get("text") or "").strip():
            stats["empty_target"] += 1
            continue
        take = takes.get(row["take"])
        if take is None:
            stats["take_missing_from_takes_json"] += 1
            continue
        chosen = clip_videos(egoexo_root, take, views)
        if not chosen:
            stats["no_video_on_disk"] += 1
            continue
        if len(chosen) < (2 if views == "all" else 1):
            stats["incomplete_view_pair"] += 1
        if not activity_of(row):
            stats["activity_not_in_prompt"] += 1
        for kind, video in chosen:
            records.append(build_record(row, kind, video, fps))
    return records, stats


def summarise(name: str, rows: list[dict], records: list[dict]) -> None:
    takes = {r["take"] for r in rows}
    words = sorted(len(r["messages"][2]["content"].split()) for r in records)
    spans = sorted(round(r["chat_template_kwargs"]["video_end"]
                         - r["chat_template_kwargs"]["video_start"], 2)
                   for r in records)
    activities = Counter(EMITTED_ACTIVITY_RE.search(r["messages"][1]["content"]).group(1)
                         if EMITTED_ACTIVITY_RE.search(r["messages"][1]["content"])
                         else "?"
                         for r in records)
    print(f"  {name}: {len(records)} samples from {len(rows)} source rows, "
          f"{len(takes)} takes")
    print(f"    views: {dict(Counter(r['view'] for r in records))}")
    print(f"    activities: {dict(activities.most_common())}")
    print(f"    clip seconds: median {spans[len(spans) // 2]:.2f}, "
          f"min {spans[0]:.2f}, max {spans[-1]:.2f}")
    print(f"    sampling: {frame_budget.fps_histogram(records)}")
    print(f"    target words: median {words[len(words) // 2]}, "
          f"p90 {words[int(len(words) * 0.9)]}, max {words[-1]}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--expertaf-root", type=Path, default=DEFAULT_EXPERTAF_ROOT,
                        help="directory holding expertaf_{train,val}_N5.json "
                             "(default: %(default)s)")
    parser.add_argument("--egoexo-root", type=Path, default=DEFAULT_EGOEXO_ROOT,
                        help="Ego-Exo4D root, for takes.json and the videos "
                             "(default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original",
                        help="output root; creates sports/ and skills/ subdirectories (default: %(default)s)")
    parser.add_argument("--prefix", default="expertaf",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--fps", type=float, default=None,
                        help=f"pin one sampling fps for every row instead of "
                             f"data/preprocess/frame_budget.py's duration ladder; source is "
                             f"{SOURCE_FPS:g} fps, so --fps 30 reproduces ExpertAF's "
                             f"120 frames exactly (default: the ladder)")
    parser.add_argument("--views", choices=["all", "best-exo", "ego"], default="best-exo",
                        help="the best exocentric camera only (default), ego only, or "
                             "both with one sample each")
    parser.add_argument("--dedup-test", action="store_true",
                        help="also collapse the test split to one row per commentary; "
                             "off by default because ExpertAF averages over all 1272 "
                             "rows and collapsing them reweights the metrics")
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    return parser.parse_args(argv)


SCENARIO_DOMAINS = {
    "Basketball": "sports", "Soccer": "sports", "Rock Climbing": "sports",
    "Cooking": "skill", "Music": "skill", "Dance": "skill",
    "Bike Repair": "skill", "Health": "skill",
}


def record_domain(record: dict, takes: dict) -> str:
    names = [part for part in Path(record["videos"][0]).parts if part in takes]
    if len(names) != 1:
        raise ValueError(f"Cannot resolve take from {record['videos']}")
    return SCENARIO_DOMAINS[takes[names[0]]["parent_task_name"]]


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    takes = {t["take_name"]: t for t in
             read_json(args.egoexo_root / "takes.json")}
    print(f"clip source: video1_range frames at {SOURCE_FPS:g} fps, decoded at "
          + (f"{args.fps:g} fps (pinned)" if args.fps else
             "data/preprocess/frame_budget.py's duration ladder"))
    print(f"views: {args.views} (one sample per view; ExpertAF instead concatenates "
          f"ego+exo features into a single input)")
    print("dropped: video2/video2_range (expert demonstration), pose branch, "
          "original_text")

    take_ids = {}
    for split, source, dedup in (("train", "expertaf_train_N5.json", True),
                                 ("test", "expertaf_val_N5.json", args.dedup_test)):
        rows = read_json(args.expertaf_root / source)
        records, stats = convert(rows, takes, args.egoexo_root, args.fps, dedup,
                                 args.views)
        if not records:
            sys.exit(f"no usable samples for split {split}")
        for domain in ("sports", "skill"):
            selected = [r for r in records if record_domain(r, takes) == domain]
            if not selected:
                continue
            domain_dir = args.out_dir / ('skills' if domain == 'skill' else domain)
            domain_dir.mkdir(parents=True, exist_ok=True)
            out_path = domain_dir / f"{args.prefix}_{domain}_{split}.jsonl"
            with out_path.open("w", encoding="utf-8") as handle:
                for record in selected:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{out_path}: {len(selected)} records")
        print(f"{split} (from {source}, dedup={dedup})")
        for reason, count in sorted(stats.items()):
            print(f"    {reason}: {count}")
        summarise(split, rows, records)
        take_ids[split] = {r["take"] for r in rows}

    overlap = take_ids["train"] & take_ids["test"]
    assert not overlap, f"{len(overlap)} takes appear in both splits"
    print(f"take-disjoint: {len(take_ids['train'])} train / {len(take_ids['test'])} test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
