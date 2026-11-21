#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import copy
import json
import math
import mimetypes
import subprocess
from functools import lru_cache
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import evaluate as benchmark
from data_io import add_dataset_arguments, load_reference_set, media_path, dataset_cli

MAX_TOKENS = 2048
TEMPERATURE = 0.0
SEED = 42
MIN_FRAMES = 4
MAX_FRAMES = 128
MAX_PIXELS = 458752


def image_url(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


@lru_cache(maxsize=1024)
def exact_video_frames(path):

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-threads",
            "1",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    if result.stderr.strip():
        raise ValueError(f"FFprobe reported video decoding errors: {path}")
    count = int(json.loads(result.stdout)["streams"][0]["nb_read_frames"])
    if count <= 0:
        raise ValueError(f"Video has no decodable frames: {path}")
    return count


def video_url(path, kwargs, args):
    import cv2

    cap = cv2.VideoCapture(str(path), cv2.CAP_ANY, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        fps = float(kwargs.get("source_fps") or cap.get(cv2.CAP_PROP_FPS))
        total = exact_video_frames(str(path))
        if not math.isfinite(fps) or fps <= 0 or total <= 0:
            raise ValueError(f"Cannot read video metadata: {path}")
        start = float(kwargs.get("video_start", 0))
        end = min(float(kwargs.get("video_end", total / fps)), total / fps)
        if not 0 <= start < end:
            raise ValueError(f"Invalid video window: {start}, {end}")
        first, stop = math.ceil(start * fps), min(total, math.ceil(end * fps))
        available = stop - first
        if available < 1:
            raise ValueError("Video window contains no frames")
        target_fps = float(kwargs.get("fps", 4))
        count = min(
            available,
            MAX_FRAMES,
            max(MIN_FRAMES, round(float(kwargs.get("sampling_window_duration", end - start)) * target_fps)),
        )

        pin_budget = float(os.environ.get("XAQA_PIN_FRAME_BUDGET") or 0)
        frame_pixels = min(MAX_PIXELS, pin_budget / count) if pin_budget else MAX_PIXELS
        if args.frame_scale != 1.0:

            count = min(available, max(2, round(count * args.frame_scale)))
        indices = [first] if count == 1 else [first + round(i * (available - 1) / (count - 1)) for i in range(count)]
        frames = []

        current = 0
        for index in indices:
            if index - current > max(64, round(fps * 2)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                current = index
            while current < index:
                if not cap.grab():
                    raise ValueError(f"Cannot decode frame {current}: {path}")
                current += 1
            ok, frame = cap.read()
            current = index + 1
            if not ok:
                raise ValueError(f"Cannot decode frame {index}: {path}")
            h, w = frame.shape[:2]
            if h * w > frame_pixels:
                scale = math.sqrt(frame_pixels / (h * w))
                frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError("JPEG encoding failed")
            frames.append(base64.b64encode(encoded).decode())
        metadata = {
            "num_frames": len(frames),
            "fps": fps,
            "frames_indices": [i - first for i in indices],
            "total_num_frames": available,
            "duration": available / fps,
            "do_sample_frames": False,
        }
        return "data:video/jpeg;base64," + ",".join(frames), metadata
    finally:
        cap.release()


def request_body(row, args):
    kwargs = row.get("chat_template_kwargs") or {}
    images, videos = iter(row.get("images", [])), iter(row.get("videos", []))
    used = {"image": 0, "video": 0}
    metadata = None
    messages = []

    text_only = getattr(args, "frame_scale", 1.0) == 0
    for message in row["messages"]:
        if message["role"] == "assistant":
            continue
        if message["role"] not in ("system", "user") or not isinstance(message["content"], str):
            raise ValueError("Expected benchmark system/user messages with string content")

        text = re.sub(r"</?(?:image|video)>", "", message["content"]) if text_only else message["content"]
        parts = []
        for part in re.split(r"(<image>|<video>)", text):
            if part in ("<image>", "<video>"):
                kind = part[1:-1]
                if text_only:
                    continue
                try:
                    path = media_path(next(images if kind == "image" else videos))
                except StopIteration:
                    raise ValueError(f"Missing media for {part}") from None
                if kind == "video":
                    if used["video"]:
                        raise ValueError("Per-request metadata currently supports one video")
                    url, metadata = video_url(path, kwargs, args)
                else:
                    url = image_url(path)
                parts.append({"type": kind + "_url", kind + "_url": {"url": url}})
                used[kind] += 1
            elif part:
                parts.append({"type": "text", "text": part})
        content = parts if any(p["type"] != "text" for p in parts) else text
        messages.append({"role": message["role"], "content": content})
    if not text_only and used != {"image": len(row.get("images", [])), "video": len(row.get("videos", []))}:
        raise ValueError("Media and placeholder counts differ")
    if os.environ.get("XAQA_MERGE_SYSTEM") and messages and messages[0]["role"] == "system":

        system_text = messages.pop(0)["content"]
        user = next(m for m in messages if m["role"] == "user")
        if isinstance(user["content"], str):
            user["content"] = system_text + "\n\n" + user["content"]
        else:
            user["content"] = [{"type": "text", "text": system_text + "\n\n"}] + user["content"]
    body = {
        "model": args.model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": "think" in args.model.lower()},
    }
    if metadata is not None:
        body["media_io_kwargs"] = {"video": metadata}
    return body


def complete(body, args):
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(args.api_key_env)
    if key:
        headers["Authorization"] = "Bearer " + key
    for attempt in range(args.retries + 1):
        try:
            request = Request(args.base_url.rstrip("/") + "/chat/completions", data=payload, headers=headers)
            with urlopen(request, timeout=args.timeout) as response:
                data = json.load(response)
            choice = data["choices"][0]
            content = choice["message"].get("content")
            if not isinstance(content, str):
                raise ValueError("Service did not return string message.content")
            return {
                "response": content,
                "finish_reason": choice.get("finish_reason"),
                "service_model": data.get("model"),
                "usage": data.get("usage"),
            }
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == args.retries:

                raise RuntimeError(f"vLLM HTTP {exc.code}; check server logs") from None
        except (URLError, TimeoutError):
            if attempt == args.retries:
                raise
        time.sleep(min(2**attempt, 10))


def read_resume(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):

        tail = raw.rfind(b"\n") + 1
        try:
            json.loads(raw[tail:])
        except (ValueError, UnicodeDecodeError):
            raw = raw[:tail]
        else:
            raw += b"\n"
        path.write_bytes(raw)
    return benchmark.read(path)


def summarize(report, output):
    lines = ["# Benchmark results", "", f'Complete: {report["complete"]}', ""]
    summary = report["format_summary"]
    for part in ("ID", "OOD"):
        cells = summary.get(part) or {}
        if not cells:
            continue
        lines += [
            f"## {part}",
            "",
            "| Form | Sources | Instances | Metrics | Complete |",
            "| --- | ---: | ---: | --- | --- |",
        ]
        for form, entry in cells.items():
            shown = " / ".join(f"{k}={v:.4f}" if v is not None else f"{k}=--" for k, v in entry["mean"].items())
            lines.append(
                f'| {form} ({entry["form"]}) | {entry["sources"]} | '
                f'{entry["instances"]} | {shown} | {entry["complete"]} |'
            )
        lines.append("")
    if summary.get("latex_row"):
        lines += ["```latex", summary["latex_row"], "```", ""]
    for part, domains in report.get("domain_summary", {}).items():
        if not domains:
            continue
        lines += [
            f"## {part} by domain",
            "",
            "| Domain | Form | Sources | Instances | Metrics |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for domain, cells in domains.items():
            for form, entry in cells.items():
                shown = " / ".join(f"{k}={v:.4f}" if v is not None else f"{k}=--" for k, v in entry["mean"].items())
                lines.append(f'| {domain} | {form} | {entry["sources"]} | ' f'{entry["instances"]} | {shown} |')
        lines.append("")
    lines += [
        "Definitions, scales and paper comparability: README.md.",
        "Full coverage and diagnostics: metrics.json.",
        "",
    ]
    output.write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", required=True, help="Served model name sent in every request")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument(
        "--replica-urls",
        nargs="+",
        help="Additional equivalent model replicas; preserve model, sampling and generation settings",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output-dir", required=True, type=Path)
    add_dataset_arguments(parser)
    parser.add_argument("--datasets", nargs="+", choices=tuple(benchmark.SUITE))
    parser.add_argument("--type", choices=("all", "ID", "OOD"), default="all")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=2)

    parser.add_argument("--frame-scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.timeout <= 0 or args.retries < 0 or args.frame_scale < 0:
        parser.error("Invalid request limits")
    names = [n for n in (args.datasets or benchmark.SUITE) if args.type == "all" or benchmark.SUITE[n][1] == args.type]
    names = benchmark.version_tasks(names, args.reference_dir)
    if not names:
        parser.error("No datasets selected")
    refs, dataset_identity = load_reference_set(args, names, benchmark.SUITE)
    sources = {n: benchmark.read(p, args) for n, p in refs.items()}
    for n, rows in sources.items():
        if len(rows) != benchmark.expected_rows(n, refs[n]):
            raise ValueError(f"{n}: reference count does not match benchmark")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import fcntl

    with (args.output_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for name, rows in sources.items():
            previous = read_resume(args.output_dir / (name + ".jsonl"))
            benchmark.join_predictions(rows, previous)
            indices = {r["_benchmark_index"] for r in previous}
            if len(indices) != len(previous) or any(i not in range(len(rows)) for i in indices):
                raise ValueError("Invalid resume indices")
            for r in previous:
                i = r["_benchmark_index"]
                if {k: r[k] for k in rows[i]} != rows[i]:
                    raise ValueError("Resume index/reference mismatch")
            pending.extend((name, i, r) for i, r in enumerate(rows) if i not in indices)
        endpoints = list(dict.fromkeys([args.base_url, *(args.replica_urls or [])]))
        if args.concurrency < len(endpoints):
            raise ValueError("Concurrency must be at least the number of replicas")

        def infer(item, endpoint):
            name, index, row = item
            request_args = copy(args)
            request_args.base_url = endpoint
            try:
                result = complete(request_body(row, request_args), request_args)
                return item, {**result, "service_url": endpoint}, None
            except Exception as exc:
                return item, None, f"{type(exc).__name__}: {exc}"

        failures = processed = 0
        started = time.monotonic()

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            todo = iter(pending)
            futures = {}
            for slot in range(min(args.concurrency, len(pending))):
                endpoint = endpoints[slot % len(endpoints)]
                futures[pool.submit(infer, next(todo), endpoint)] = endpoint
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    endpoint = futures.pop(future)
                    (name, index, row), result, error = future.result()
                    if error:
                        failures += 1
                        with (args.output_dir / "failures.jsonl").open("a") as f:
                            f.write(
                                json.dumps({"task": name, "index": index, "endpoint": endpoint, "error": error}) + "\n"
                            )
                    else:
                        with (args.output_dir / (name + ".jsonl")).open("a") as f:
                            f.write(json.dumps({**row, **result, "_benchmark_index": index}, ensure_ascii=False) + "\n")
                    processed += 1
                    following = next(todo, None)
                    if following is not None:
                        futures[pool.submit(infer, following, endpoint)] = endpoint
                    if processed % 4 == 0 or processed == len(pending):
                        rate = processed * 60 / max(time.monotonic() - started, 0.001)
                        print(
                            f"Processed {processed}/{len(pending)} new requests; failures={failures}; rate={rate:.1f}/min",
                            flush=True,
                        )
        cli = [
            "--prediction-dir",
            str(args.output_dir),
            *dataset_cli(args, dataset_identity),
            "--media-root",
            str(args.media_root),
            "--datasets",
            *names,
            "--out",
            str(args.output_dir / "metrics.json"),
            "--model-label",
            args.model,
        ]
        code = benchmark.main(cli)
        summarize(json.loads((args.output_dir / "metrics.json").read_text()), args.output_dir / "summary.md")
        return code


if __name__ == "__main__":
    raise SystemExit(main())
