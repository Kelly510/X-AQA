#!/usr/bin/env python3
from __future__ import annotations

from build_data_protocol import main as protocol_main
from build_specialist_refs import main as spec_main


from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_catalog import DOMAIN_FOLDERS, dataset_for_file

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT
DEFAULT_SOURCE = REPO_ROOT / "data" / "exports" / "original"


FORMAT = {
    "aqa7_score": "S", "mtlaqa": "S,T", "rhythmicgymnastics": "S", "finefs_goe": "S",
    "cricketvision_score": "S", "llmfms": "S", "lasana_score": "S", "jigsaws": "S",
    "multisensebadminton_qa": "C", "egoexo4d_sports": "C", "cotafa_martial_aqa": "C",
    "cotafa_fitness_aqa": "C", "egoexo_fitness_kp": "C", "fitnessaqa_kp": "C",
    "flag3d_kp": "C", "execheck_correctness": "C", "rehab246": "C",
    "captaincook4d": "C", "cprcoach": "C", "egoexo4d_skill": "C", "lasana_error": "C",
    "fitness7": "C", "upperlimb_completion": "C", "aixsuture_grs": "S",
    "kimore_cts": "S",
    "expertaf_sports": "T", "cotafa_martial_cot": "T",
    "cotafa_fitness_cot": "T", "egoexo_fitness_comment": "T", "qevd_fitcoach": "T",
    "t3set_comment": "T",
}

OOD_TASKS = ("t3set_comment", "fitness7", "jigsaws", "kimore_cts")


TRAINING_ONLY = ("keraal",)

STRATA_FIELDS = ("exercise", "exercise_name", "movement", "task", "error_category",
                 "stroke_type", "technique", "category", "error_name", "scenario")
DOMAIN_OF_FOLDER = {v: k for k, v in DOMAIN_FOLDERS.items()}


def read_rows(path: Path) -> list:
    return [json.loads(line) for line in path.open() if line.strip()]


def media_of(row: dict) -> str:
    media = row.get("videos") or row.get("images") or [""]
    return media[0]


def row_identity(row: dict, task: str, split: str, index: int) -> str:
    return str(row.get("sample_id") or row.get("id") or f"{task}#{split}#{index}")


def sort_key(identity: str) -> str:
    return hashlib.sha1(identity.encode()).hexdigest()


def stratum_of(row: dict) -> str:
    for field in STRATA_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            return f"{field}={value}"
    return "all"


def target_of(row: dict, task: str | None = None) -> str:
    text = row["messages"][-1]["content"]
    if task is None:
        return text
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return text
    if task == "captaincook4d":
        return "error" if value else "no_error"
    if task == "multisensebadminton_qa":
        return str(value.get("skill_level", ""))
    if isinstance(value, dict) and len(value) == 1:
        return str(next(iter(value.values())))
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def allocate(sizes: dict, budget: int) -> dict:
    out = {key: 0 for key in sizes}
    left = min(budget, sum(sizes.values()))
    while left > 0:
        hungry = [k for k in sizes if out[k] < sizes[k]]
        share = max(1, left // len(hungry))
        for key in sorted(hungry, key=lambda k: (sizes[k], k)):
            if left <= 0:
                break
            take = min(share, sizes[key] - out[key], left)
            out[key] += take
            left -= take
    return out


def interleave_by_media(rows: list) -> list:
    by_media = defaultdict(list)
    for row in rows:
        by_media[media_of(row)].append(row)
    for group in by_media.values():
        group.sort(key=lambda r: r["_sort"])
    order = sorted(by_media, key=lambda m: by_media[m][0]["_sort"])
    out = []
    while order:
        for medium in list(order):
            group = by_media[medium]
            out.append(group.pop(0))
            if not group:
                order.remove(medium)
    return out


def pick(rows: list, quota: int, fmt: str, task: str | None = None) -> list:
    if quota >= len(rows):
        return list(rows)
    cells = defaultdict(list)
    for row in rows:
        key = (row["_stratum"], target_of(row, task)) if fmt == "C" else (row["_stratum"], "")
        cells[key].append(row)
    shares = allocate({k: len(v) for k, v in cells.items()}, quota)
    chosen = []
    for key, take in shares.items():
        chosen += interleave_by_media(cells[key])[:take]
    return sorted(chosen, key=lambda r: r["_sort"])


def collect(source: Path) -> dict:
    files = defaultdict(dict)
    for folder, domain in DOMAIN_OF_FOLDER.items():
        for path in sorted((source / folder).glob("*.jsonl")):
            if dataset_for_file(path.name) is None:
                continue
            split = path.stem.rsplit("_", 1)[-1]
            if split not in ("train", "test", "val", "validation"):
                continue
            task = path.stem.rsplit("_", 1)[0]
            files[task][split] = (path, folder, domain)

    found = {}
    for task, splits in sorted(files.items()):
        if task in TRAINING_ONLY:
            continue
        part = "OOD" if task in OOD_TASKS else "ID"
        wanted = sorted(splits) if part == "OOD" else [s for s in splits if s == "test"]
        if not wanted:
            continue
        if task not in FORMAT:
            raise ValueError(f"No supervision form declared for benchmark task {task}")
        path, folder, domain = splits[wanted[0]]
        entry = found.setdefault(task, {
            "dataset": dataset_for_file(path.name).name, "domain": domain,
            "folder": folder, "format": FORMAT[task], "part": part, "rows": []})
        for split in wanted:
            path = splits[split][0]
            for index, row in enumerate(read_rows(path)):
                row["_source_split"] = split
                row["_identity"] = row_identity(row, task, split, index)
                row["_sort"] = sort_key(row["_identity"])
                row["_stratum"] = stratum_of(row)
                entry["rows"].append(row)
    return found


def portable_path(text: str) -> str:
    value = str(text).replace('\\', '/')
    if value.startswith('datasets/'):
        return value
    marker = '/datasets/'
    if value.startswith('/') and marker in value:
        return 'datasets/' + value.split(marker, 1)[1]
    return value


def portable_value(value):
    if isinstance(value, str):
        return portable_path(value)
    if isinstance(value, list):
        return [portable_value(v) for v in value]
    if isinstance(value, dict):
        return {k: portable_value(v) for k, v in value.items()}
    return value


def relative_media(path: str) -> str:
    return portable_path(Path(path).resolve())


def parquet_row(task: str, part: str, row: dict) -> dict:
    kwargs = row.get("chat_template_kwargs") or {}
    duration = row.get("duration") or row.get("duration_seconds")
    if duration is None and {"video_start", "video_end"} <= set(kwargs):
        duration = round(kwargs["video_end"] - kwargs["video_start"], 3)
    recording = (row.get("recording_id") or row.get("video_id") or row.get("record_id")
                 or row.get("round_id") or Path(media_of(row)).stem)
    metadata = {k: v for k, v in row.items()
                if k not in ("messages", "videos", "images", "chat_template_kwargs")
                and not k.startswith("_")}
    metadata.update(source_split=row["_source_split"], stratum=row["_stratum"],
                    supervision_format=FORMAT[task], task=task, benchmark_part=part)
    if "_evaluation_formats" in row:
        metadata["evaluation_formats"] = sorted(row["_evaluation_formats"])
    metadata = portable_value(metadata)
    video = images = None
    if row.get("videos"):
        video = {"path": relative_media(row["videos"][0])}
    if row.get("images"):
        images = {"path": relative_media(row["images"][0])}
    return {
        "task": task, "split": part, "sample_id": row["_identity"],
        "messages": [{"role": m["role"], "content": m["content"]}
                     for m in row["messages"] if m["role"] != "assistant"],


        "target": target_of(row),
        "sampling": json.dumps(kwargs, ensure_ascii=False, sort_keys=True),
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True,
                               default=str),
        "original_recording_id": str(recording),
        "original_window_duration": float(duration) if duration is not None else None,
        "video": video, "image": images,
    }


def write_parquet(path: Path, rows: list) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    media = pa.struct([("path", pa.string())])
    schema = pa.schema([
        ("task", pa.string()), ("split", pa.string()), ("sample_id", pa.string()),
        ("messages", pa.list_(pa.struct([("role", pa.string()),
                                         ("content", pa.string())]))),
        ("target", pa.string()), ("sampling", pa.string()), ("metadata", pa.string()),
        ("original_recording_id", pa.string()),
        ("original_window_duration", pa.float64()),
        ("video", media), ("image", media),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def dedup_stratum(rows: list, keep: int = 1) -> list:
    groups = defaultdict(list)
    for row in rows:
        groups[(media_of(row), row["_stratum"])].append(row)
    chosen = []
    for group in groups.values():
        group.sort(key=lambda r: r["_sort"])
        chosen += group[:keep]
    return sorted(chosen, key=lambda r: r["_sort"])


def sample_stratum(found: dict, cap: int = 300) -> dict:
    keep = 1
    selected = {}
    for task, entry in sorted(found.items()):
        for fmt in entry["format"].split(","):
            if task == "mtlaqa":
                field = {"S": "execution_award", "T": "comment"}.get(fmt)
                rows = [r for r in entry["rows"] if field in r["answer_fields"]]
            else:
                rows = entry["rows"]
            if entry["part"] == "OOD":
                dedup = chosen = sorted(rows, key=lambda r: r["_sort"])
            else:
                dedup = dedup_stratum(rows, keep)
                chosen = (pick(dedup, min(cap, len(dedup)), fmt, task)
                          if entry["part"] == "ID" else dedup)
            bucket = selected.setdefault(task, {})
            for row in chosen:
                item = bucket.setdefault(row["_identity"], dict(row))
                if task == "mtlaqa":
                    item.setdefault("_evaluation_formats", set()).add(fmt)
            print(f"  {entry['part']:3s} {entry['domain']:14s} / {fmt}  {task:24s}"
                  f"  rows {len(rows):6d} -> dedup {len(dedup):5d} -> {len(chosen):4d}"
                  f"  strata {len({r['_stratum'] for r in chosen}):3d}"
                  f"  media {len({media_of(r) for r in chosen}):4d}")
    return {task: (sorted(rows.values(), key=lambda r: r["_sort"])
                   if task == "mtlaqa" else list(rows.values()))
            for task, rows in selected.items()}


def build_core() -> int:
    version = "v3"
    out_root = REPO_ROOT / "benchmark"
    cap = 300
    out_dir = out_root / version / "data"
    found = collect(DEFAULT_SOURCE)
    print(f"sources: {len(found)} tasks "
          f"({sum(e['part'] == 'ID' for e in found.values())} ID / "
          f"{sum(e['part'] == 'OOD' for e in found.values())} OOD)")
    print(f"{version}: stratum sampling, cap {cap}, full OOD")
    selected = sample_stratum(found, cap)

    manifest = {"version": version, "scheme": "stratum", "cap": cap,
                "selection": "stratified, sha1-ordered, no RNG",
                "parts": {}, "tasks": {}}
    totals = defaultdict(Counter)
    for task, entry in sorted(found.items()):
        rows = selected[task]
        out = out_dir / entry["part"] / f"{task}.parquet"
        write_parquet(out, [parquet_row(task, entry["part"], r) for r in rows])
        component_counts = {
            fmt: sum(fmt in r.get("_evaluation_formats", entry["format"].split(","))
                     for r in rows) for fmt in entry["format"].split(",")}
        totals[entry["part"]].update(component_counts)
        totals[entry["part"]]["total"] += len(rows)
        manifest["tasks"][task] = {
            "dataset": entry["dataset"], "domain": entry["domain"],
            "format": entry["format"], "part": entry["part"],
            "pool": len(entry["rows"]), "selected": len(rows),
            "file": (str(out.relative_to(REPO_ROOT))
                     if out.is_relative_to(REPO_ROOT) else str(out)),
            "source_splits": dict(Counter(r["_source_split"] for r in rows)),
            "strata": len({r["_stratum"] for r in rows}),
            "media": len({media_of(r) for r in rows}),
            "classes": (dict(Counter(target_of(r, task) for r in rows))
                        if entry["format"] == "C" and
                        len({target_of(r, task) for r in rows}) <= 12 else None),
        }
        if task == "mtlaqa":
            manifest["tasks"][task]["component_counts"] = component_counts
    for part, counts in totals.items():
        manifest["parts"][part] = {**{f: counts[f] for f in "SCT"},
                                   "total": counts["total"]}
    if "mtlaqa" in found:
        for old_task in ("mtlaqa_score", "mtlaqa_comment"):
            (out_dir / "ID" / f"{old_task}.parquet").unlink(missing_ok=True)
    path = out_root / version / "benchmark_sample.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("\nby format: " + json.dumps(manifest["parts"], ensure_ascii=False))
    print(f"manifest: {path}")
    return 0


def main() -> int:
    protocol_main()
    build_core()
    spec_main([])
    print('X-AQA-Bench build complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
