#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


FRAME_CAP = 128
FRAME_FLOOR = 4
FRAME_FACTOR = 2


FPS_LADDER = ((8.0, 16.0), (16.0, 8.0), (32.0, 4.0))
TAIL_DECIMALS = 4


DURATION_CACHE = REPO_ROOT / "outputs" / "analysis" / "duration_cache.json"

_EPS = 1e-6


def sampling_fps(duration: float) -> float:
    if duration <= 0:
        return FPS_LADDER[0][1]
    for bound, fps in FPS_LADDER:
        if duration <= bound + _EPS:
            return fps
    scale = 10 ** TAIL_DECIMALS
    return math.ceil(FRAME_CAP / duration * scale) / scale


def resolve_fps(duration: float, override: float | None) -> float:
    return float(override) if override else sampling_fps(duration)


def frames_for(duration: float, fps: float | None = None) -> int:
    if duration <= 0:
        return FRAME_FLOOR
    want = duration * (sampling_fps(duration) if fps is None else fps)
    want = min(max(want, FRAME_FLOOR), FRAME_CAP)
    return max(FRAME_FACTOR, int(want // FRAME_FACTOR) * FRAME_FACTOR)


def fps_histogram(records) -> str:
    rungs = {fps for _, fps in FPS_LADDER}
    counts, dynamic = Counter(), []
    images = 0
    for record in records:
        fps = (record.get("chat_template_kwargs") or {}).get("fps")
        if not fps:
            images += 1
        elif float(fps) in rungs:
            counts[float(fps)] += 1
        else:
            dynamic.append(float(fps))
    parts = [f"{fps:g} fps x {n}" for fps, n in sorted(counts.items(), reverse=True)]
    if dynamic:
        parts.append(f"dynamic {min(dynamic):.2f}-{max(dynamic):.2f} fps "
                     f"x {len(dynamic)}")
    if images:
        parts.append(f"no fps (image rows) x {images}")
    return ", ".join(parts) or "no rows"


def windows(duration: float, window_sec: float):
    if duration <= window_sec + _EPS:
        return [(0.0, max(duration, 0.0))]
    spans, start = [], 0.0
    while duration - start >= window_sec - _EPS:
        spans.append((start, start + window_sec))
        start += window_sec
    if duration - start > _EPS:
        spans.append((duration - window_sec, duration))
    return spans


def probe_duration(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def probe_durations(paths, workers: int = 32, cache_path: Path | None = None,
                    verbose: bool = True) -> dict[str, float]:
    cache_path = cache_path or DURATION_CACHE
    cache: dict[str, float] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, ValueError):


            cache = {}
    wanted = [str(p) for p in dict.fromkeys(paths)]
    todo = [p for p in wanted if p not in cache]
    if todo:
        if verbose:
            print(f"probing {len(todo)} durations "
                  f"({len(wanted) - len(todo)} cached)", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for path, duration in zip(todo, pool.map(probe_duration, todo)):
                cache[path] = duration
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        tmp = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(cache))
        tmp.replace(cache_path)
    return {p: cache[p] for p in wanted}
