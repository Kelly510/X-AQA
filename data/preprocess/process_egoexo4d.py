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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "Ego-Exo4D"


PROFICIENCY_CLASSES = ("Novice", "Early Expert", "Intermediate Expert", "Late Expert")


DEFAULT_EXCLUDED_SCENARIOS = ("Bike Repair", "Health")
SCENARIO_DOMAINS = {
    "Basketball": "sports", "Soccer": "sports", "Rock Climbing": "sports",
    "Cooking": "skill", "Music": "skill", "Dance": "skill",
}


DEFAULT_WINDOW_SEC = 32.0
DEFAULT_FPS = frame_budget.sampling_fps(DEFAULT_WINDOW_SEC)


DOWNSCALE_PARTS = ("downscaled", "448")


EGO_RE = re.compile(r"(^|/)aria")

PROFICIENCY_LINE = "Proficiency levels: " + ", ".join(PROFICIENCY_CLASSES)

SYSTEM_PROMPT = (
    "You assess proficiency in short videos of skilled activities.\n\n"
    f"{PROFICIENCY_LINE}\n\n"
    "Return only this "
    'JSON object and nothing else: {"proficiency_level": "<one of the listed levels>"}. '
    "Do not output Markdown, code fences, or extra keys."
)


def read_json(path: Path):
    if not path.is_file():
        sys.exit(f"annotation not found: {path}\n"
                 "download it with --parts annotations --benchmarks "
                 "proficiency_demonstrator")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def view_kind(relative_path: str) -> str:
    return "ego" if EGO_RE.search(relative_path) else "exo"


def downscaled(dataset_root: Path, relative_path: str) -> Path:
    raw = Path(relative_path)
    return dataset_root / raw.parent / Path(*DOWNSCALE_PARTS) / raw.name


def camera_paths(take: dict, entry: dict) -> dict:
    paths = {}
    streams = take.get("frame_aligned_videos") or {}
    exo_index = 0
    for camera in sorted(streams):
        if camera in ("collage", "best_exo"):
            continue
        substreams = streams[camera] or {}
        if camera.startswith("aria"):
            relative = (substreams.get("rgb") or {}).get("relative_path")
            if relative:
                paths["ego"] = f"{take['root_dir']}/{relative}"
            continue
        relative = (substreams.get("0") or {}).get("relative_path")
        if relative:
            exo_index += 1
            paths[f"exo{exo_index}"] = f"{take['root_dir']}/{relative}"
    if len(paths) >= len(entry.get("video_paths") or {}):
        return paths
    return dict(entry["video_paths"])


def clips(dataset_root: Path, take: dict, entry: dict, views: str, max_exo: int = 0):
    resolved = []
    for name, relative in sorted((camera_paths(take, entry)).items()):
        kind = view_kind(relative)
        if views == "all-exo" and kind == "ego":
            continue
        path = downscaled(dataset_root, relative)
        if path.is_file():
            resolved.append((name, kind, path))
    if views in ("best-exo", "ego+best-exo"):


        best = take.get("best_exo")
        chosen = [c for c in resolved if c[1] == "exo" and c[2].stem == best]
        if not chosen:
            chosen = [c for c in resolved if c[1] == "exo"][:1]
        if views == "ego+best-exo":

            chosen = [c for c in resolved if c[1] == "ego"][:1] + chosen
        return chosen
    if max_exo:


        exo_kept = 0
        capped = []
        for name, kind, path in resolved:
            if kind == "exo":
                if exo_kept >= max_exo:
                    continue
                exo_kept += 1
            capped.append((name, kind, path))
        return capped
    return resolved


def windows(duration: float, window_sec: float, max_windows: int = 0):
    if duration <= window_sec + 1e-6:
        return [(0.0, max(duration, 0.0))]
    spans = []
    start = 0.0
    while duration - start >= window_sec - 1e-6:
        spans.append((start, start + window_sec))
        start += window_sec
    if duration - start > 1e-6:
        spans.append((duration - window_sec, duration))
    if max_windows and len(spans) > max_windows:
        if max_windows == 1:
            return [spans[len(spans) // 2]]
        step = (len(spans) - 1) / (max_windows - 1)
        spans = [spans[round(i * step)] for i in range(max_windows)]
    return spans


def build_records(sample: dict, fps: float | None, window_sec: float, max_windows: int):
    context = sample["task_name"] or sample["scenario"]
    lines = []
    if context:
        lines.append(f"Activity: {context}")


    question = "<video>\n" + "\n".join(lines)
    records = []
    for start, end in windows(sample["duration"], window_sec, max_windows):
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": json.dumps(
                    {"proficiency_level": sample["proficiency"]}, ensure_ascii=False)},
            ],
            "videos": [sample["video"]],
            "chat_template_kwargs": {"video_start": round(start, 2),
                                     "video_end": round(end, 2),
                                     "fps": frame_budget.resolve_fps(end - start, fps)},

            "view": sample["view_kind"],
        })
    return records


def collect(dataset_root: Path, views: str, excluded, max_exo: int = 0):
    takes = {t["take_uid"]: t for t in read_json(dataset_root / "takes.json")}
    official = read_json(dataset_root / "annotations" / "splits.json")["take_uid_to_split"]

    labelled = {}
    for name in ("train", "val"):
        path = dataset_root / "annotations" / f"proficiency_demonstrator_{name}.json"
        for entry in read_json(path)["annotations"]:
            if entry.get("proficiency_score") in PROFICIENCY_CLASSES:
                labelled[entry["take_uid"]] = entry


    split_name = {"train": "train", "val": "test"}
    samples = defaultdict(list)
    stats = Counter()
    for take_uid in sorted(labelled):
        entry = labelled[take_uid]
        take = takes.get(take_uid)
        if take is None:
            stats["take_missing_from_takes_json"] += 1
            continue
        split = split_name.get(official.get(take_uid))
        if split is None:
            stats[f"official_split:{official.get(take_uid)}"] += 1
            continue
        scenario = take.get("parent_task_name") or ""
        if scenario in excluded:
            stats[f"excluded_scenario:{scenario}"] += 1
            continue
        chosen = clips(dataset_root, take, entry, views, max_exo)
        if not chosen:
            stats["no_video_on_disk"] += 1
            continue
        for name, kind, video in chosen:
            samples[split].append({
                "take_uid": take_uid,
                "take_name": take["take_name"],
                "participant_uid": take.get("participant_uid") or "",
                "scenario": scenario,
                "task_name": take.get("task_name") or entry.get("task_name") or "",
                "duration": float(take.get("duration_sec") or 0.0),
                "view": name,
                "view_kind": kind,
                "proficiency": entry["proficiency_score"],
                "video": str(video),
            })
    return samples, stats


def summarise(name: str, samples, records) -> None:
    takes = {s["take_uid"] for s in samples}
    per_take = Counter()
    for s in samples:
        per_take[s["take_uid"]] = s["proficiency"]
    take_levels = Counter(per_take.values())
    lengths = sorted(r["chat_template_kwargs"]["video_end"]
                     - r["chat_template_kwargs"]["video_start"] for r in records)
    windows_per_clip = len(records) / max(len(samples), 1)
    print(f"  {name}: {len(records)} samples from {len(takes)} takes "
          f"x {len(samples) / max(len(takes), 1):.1f} cameras "
          f"x {windows_per_clip:.1f} windows")
    print(f"    cameras: {dict(Counter(s['view'] for s in samples).most_common())}")
    print(f"    viewpoints: {dict(Counter(s['view_kind'] for s in samples))}")
    print(f"    window seconds: median {lengths[len(lengths) // 2]:.1f}, "
          f"min {lengths[0]:.1f}, max {lengths[-1]:.1f}")
    print(f"    sampling: {frame_budget.fps_histogram(records)}")
    print(f"    labels (takes) {dict(sorted(take_levels.items(), key=lambda kv: PROFICIENCY_CLASSES.index(kv[0])))}")
    majority = max(take_levels.values()) / max(sum(take_levels.values()), 1)
    print(f"    majority-class prior (take level): {majority:.4f}")
    print(f"    scenarios (takes) "
          f"{dict(Counter(s['scenario'] for s in samples if s['view'] == 'ego').most_common() or Counter(s['scenario'] for s in samples).most_common())}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="Ego-Exo4D root (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports" / "original",
                        help="output root; creates sports/ and skills/ subdirectories (default: %(default)s)")
    parser.add_argument("--prefix", default="egoexo4d",
                        help="output filename prefix (default: %(default)s)")
    parser.add_argument("--suffix", default="",
                        help="appended to the output filenames (default: none)")
    parser.add_argument("--views", choices=["all", "all-exo", "best-exo",
                                            "ego+best-exo"], default="best-exo",
                        help="one canonical exocentric camera per take -- takes.json's "
                             "best_exo (default), all cameras including the egocentric "
                             "Aria RGB stream, every exocentric camera, or ego plus "
                             "best_exo")
    parser.add_argument("--exclude-scenarios", default=",".join(DEFAULT_EXCLUDED_SCENARIOS),
                        help="comma-separated scenarios to drop (default: %(default)s); "
                             "pass an empty string to keep everything")
    parser.add_argument("--fps", type=float, default=None,
                        help="pin one sampling fps for every row instead of "
                             "data/preprocess/frame_budget.py's duration ladder (default: "
                             "the ladder, which gives 4 fps for a 32 s window)")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC,
                        help=f"window length; the ladder samples a window of "
                             f"{DEFAULT_WINDOW_SEC:g}s at {DEFAULT_FPS:g} fps, which "
                             f"is the full 128-frame budget (default: %(default)s)")
    parser.add_argument("--max-exo-views", type=int, default=0,
                        help="keep at most this many exocentric cameras per take, in "
                             "sorted camera-name order, 0 = all (default). Only bites "
                             "under --views all / all-exo; Table 10's Exos column is "
                             "M = 4 while takes here carry 4, 5 or 6")
    parser.add_argument("--max-windows", type=int, default=0,
                        help="keep at most this many evenly spread windows per camera, "
                             "0 = all (default: %(default)s)")
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    excluded = {s.strip() for s in args.exclude_scenarios.split(",") if s.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("task: demonstrator proficiency estimation only (no commentary samples)")
    print(f"excluded scenarios: {sorted(excluded) or 'none'}")
    print(f"views: {args.views}"
          + (f", at most {args.max_exo_views} exo cameras per take"
             if args.max_exo_views else ""))
    window_fps = frame_budget.resolve_fps(args.window_sec, args.fps)
    print(f"windows: {args.window_sec:g}s = "
          f"{frame_budget.frames_for(args.window_sec, window_fps)} frames at "
          f"{window_fps:g} fps, non-overlapping, last window anchored at the end"
          + (f", at most {args.max_windows} per camera" if args.max_windows else ""))

    samples, stats = collect(args.dataset_root, args.views, excluded, args.max_exo_views)
    if not samples:
        sys.exit("no usable samples")
    for reason, count in sorted(stats.items()):
        print(f"    skipped {count} ({reason})")

    take_ids, participants = {}, {}
    for split in ("train", "test"):
        group = samples.get(split) or []
        if not group:
            sys.exit(f"no usable samples for split {split}")
        group.sort(key=lambda s: (s["take_name"], s["view"]))
        records = [r for s in group
                   for r in build_records(s, args.fps, args.window_sec,
                                          args.max_windows)]
        for domain in ("sports", "skill"):
            domain_samples = [s for s in group if SCENARIO_DOMAINS[s["scenario"]] == domain]
            domain_records = [r for sample in domain_samples
                              for r in build_records(sample, args.fps, args.window_sec,
                                                     args.max_windows)]
            domain_dir = args.out_dir / ('skills' if domain == 'skill' else domain)
            domain_dir.mkdir(parents=True, exist_ok=True)
            out_path = domain_dir / f"{args.prefix}_{domain}_{split}{args.suffix}.jsonl"
            with out_path.open("w", encoding="utf-8") as handle:
                for record in domain_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{out_path}: {len(domain_records)} records")
        summarise(split, group, records)
        take_ids[split] = {s["take_uid"] for s in group}
        participants[split] = {s["participant_uid"] for s in group if s["participant_uid"]}

    overlap = take_ids["train"] & take_ids["test"]
    assert not overlap, f"{len(overlap)} takes appear in both splits"
    shared = participants["train"] & participants["test"]


    print(f"participants: {len(participants['train'])} train / {len(participants['test'])} "
          f"test, shared {len(shared)}")
    if shared:
        print(f"  WARNING: {len(shared)} participants appear in both splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
