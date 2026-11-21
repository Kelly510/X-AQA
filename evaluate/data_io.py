import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "v3" / "data"


def default_workspace():
    return Path(__file__).resolve().parents[1]


def add_dataset_arguments(parser):
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Local benchmark reference directory (default: %(default)s)",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        default=default_workspace(),
        help="Root for the workspace-relative media paths stored in the references (default: %(default)s)",
    )


def resolve_data_file(root, task):
    root = Path(root)
    name = task.removesuffix("_test.jsonl")
    for split in ("ID", "OOD"):
        path = root / split / (name + ".parquet")
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing reference parquet for {name} under {root}")


def media_path(value, root=None):
    path = Path(value)
    if root is None or path.is_absolute():
        return str(path)
    if ".." in path.parts:
        raise ValueError("Unsafe media path")
    return str(Path(root).resolve() / path)


def reference_row(item, root=None):
    row = json.loads(item["metadata"])
    row.update(sample_id=item["sample_id"], messages=item["messages"], labels=item["target"])
    row["chat_template_kwargs"] = json.loads(item["sampling"])
    row["original_recording_id"] = item["original_recording_id"]
    row["original_window_duration"] = item["original_window_duration"]
    for field, feature in [("videos", "video"), ("images", "image")]:
        value = item[feature]["path"] if item[feature] else None
        row[field] = [media_path(value, root)] if value else []
    return row


def read_references(path, media_root=None):
    path = Path(path)
    if path.suffix != ".parquet":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    root = Path(media_root) if media_root else default_workspace()
    return [reference_row(item, root) for item in pq.read_table(path).to_pylist()]


def reference_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference_set(args, names, suite):
    root = Path(args.reference_dir)
    refs = {n: resolve_data_file(root, n + "_test.jsonl") for n in names}
    return refs, {"source": "local", "reference_dir": str(root.resolve())}


def dataset_cli(args, identity):
    result = ["--reference-dir", identity["reference_dir"]]
    if getattr(args, "media_root", None):
        result += ["--media-root", str(args.media_root)]
    return result
