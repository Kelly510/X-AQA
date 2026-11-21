from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import tempfile

VERSION = 2
DEFAULT_BLUR_SIGMA = 0.04
VIDEO_SUBDIR = 'processed/videos_masked'


def probe(path: Path) -> dict:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-threads', '2', '-select_streams', 'v:0', '-count_frames',
         '-show_entries', 'stream=width,height,nb_read_frames,duration,avg_frame_rate',
         '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)['streams'][0]
    return dict(width=int(stream['width']), height=int(stream['height']),
                frames=int(stream['nb_read_frames']), duration=float(stream['duration']),
                fps=stream['avg_frame_rate'])


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def receipt_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + '.mask.json')


def require_masked(path: Path, dataset: str) -> None:
    try:
        receipt = json.loads(receipt_path(path).read_text())
        if (receipt['policy']['version'] == VERSION
                and receipt['policy']['dataset'] == dataset
                and receipt['output'] == fingerprint(path)):
            return
    except (OSError, ValueError, KeyError):
        pass
    raise ValueError(f'Missing/stale scoreboard mask: {path}; run '
                     f'python data/preprocess/process_finefs.py or process_rhythmicgymnastics.py')


def load_manifest(path: Path, source: Path, dataset: str) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get('source') != fingerprint(source) or manifest.get('dataset') != dataset:
        raise ValueError(f'Manifest source/dataset mismatch: {path}')
    status = manifest.get('status')
    if status not in ('detected', 'approved', 'clear'):
        raise ValueError(f'Scoreboard mask needs review: {path}')
    masks = manifest.get('masks')
    if not isinstance(masks, list) or (status != 'clear' and not masks):
        raise ValueError(f'Missing masks: {path}')
    if status == 'clear' and masks:
        raise ValueError('A reviewed clear video must have no masks')
    if status in ('approved', 'clear') and not manifest.get('review_note', '').strip():
        raise ValueError('Manual review requires a nonempty review_note')
    for mask in masks:
        box = mask['box']
        if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
            raise ValueError(f'Invalid normalized box: {box}')
        if not (math.isfinite(mask['start']) and math.isfinite(mask['end'])
                and 0 <= mask['start'] < mask['end']):
            raise ValueError('Invalid mask interval')
        if status == 'detected' and not mask.get('reliable'):
            raise ValueError('Unreliable automatic mask requires review')
    return manifest


def mask_filters(masks: list[dict], width: int, height: int,
                 blur_sigma: float = DEFAULT_BLUR_SIGMA) -> list[str]:
    if not math.isfinite(blur_sigma) or not 0 < blur_sigma <= 1:
        raise ValueError('blur_sigma must be finite and in (0, 1]')
    sigma = min(1024, max(1, height * blur_sigma))
    filters = ['[0:v:0]setpts=PTS-STARTPTS[mask0]']
    for index, mask in enumerate(masks):
        x1, y1, x2, y2 = mask['box']

        left, top = 2*math.floor(x1*width/2), 2*math.floor(y1*height/2)
        right, bottom = min(width, 2*math.ceil(x2*width/2)), min(height, 2*math.ceil(y2*height/2))
        enable = f"between(t,{mask['start']:.9f},{mask['end'] + .1:.9f})"
        filters.extend([
            f'[mask{index}]split=2[base{index}][region{index}]',
            f'[region{index}]crop=w={right-left}:h={bottom-top}:x={left}:y={top},'
            f'gblur=sigma={sigma:.6f}:steps=3:planes=15[blur{index}]',
            f'[base{index}][blur{index}]overlay=x={left}:y={top}:eof_action=pass:'
            f"repeatlast=0:enable='{enable}'[mask{index+1}]",
        ])
    return filters


def _same_geometry(old: dict, new: dict) -> bool:
    keys = ('version', 'dataset', 'long_side', 'fill', 'blur_sigma', 'blur_steps')
    if any(old.get(k) != new.get(k) for k in keys):
        return False
    shape = lambda masks: [(m.get('box'), m.get('start'), m.get('end'), m.get('reliable'))
                           for m in masks]
    return shape(old.get('masks') or []) == shape(new.get('masks') or [])


def encode(source: Path, target: Path, dataset: str, *, manifest_path: Path,
           long_side: int | None = None, overwrite: bool = False,
           blur_sigma: float = DEFAULT_BLUR_SIGMA) -> dict:
    if source.resolve() == target.resolve():
        raise ValueError('Source and masked output must be different files')
    if long_side is not None and (long_side < 2 or long_side % 2):
        raise ValueError('long_side must be a positive even number >= 2')
    mask_filters([], 2, 2, blur_sigma)
    manifest = load_manifest(manifest_path, source, dataset)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    policy = dict(version=VERSION, dataset=dataset, manifest_sha256=digest,
                  masks=manifest['masks'], long_side=long_side,
                  fill='gaussian', blur_sigma=blur_sigma, blur_steps=3)
    source_stat = fingerprint(source)
    try:
        old = json.loads(receipt_path(target).read_text())
        if (not overwrite and old['source'] == source_stat
                and (old['policy'] == policy or _same_geometry(old['policy'], policy))):
            require_masked(target, dataset)
            return dict(source=str(source), status='cached')
    except (OSError, ValueError, KeyError):
        pass
    info = probe(source)
    filters = mask_filters(manifest['masks'], info['width'], info['height'], blur_sigma)
    final_filter = 'null'
    if long_side:
        final_filter = (f'scale={long_side}:-2' if info['width'] >= info['height']
                        else f'scale=-2:{long_side}')
    filters.append(f"[mask{len(manifest['masks'])}]{final_filter}[out]")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.stem + '.', suffix='.mp4', dir=target.parent)
    os.close(fd)
    tmp = Path(name)
    receipt_tmp = tmp.with_suffix('.json')
    try:
        if not manifest['masks'] and long_side is None:

            shutil.copyfile(source, tmp)
        else:
            command = ['ffmpeg', '-y', '-v', 'error', '-threads', '2', '-i', str(source),
                       '-filter_complex_threads', '1', '-filter_complex', ';'.join(filters),
                       '-map', '[out]', '-fps_mode', 'passthrough',
                       '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                       '-threads', '2', '-pix_fmt', 'yuv420p', '-an', str(tmp)]
            subprocess.run(command, check=True, capture_output=True, text=True)
        output_info = probe(tmp)
        if (output_info['frames'] != info['frames'] or output_info['fps'] != info['fps']
                or abs(output_info['duration'] - info['duration']) > 0.05):
            raise ValueError(f'Video timing changed: {source}: {info} -> {output_info}')
        if fingerprint(source) != source_stat:
            raise ValueError(f'Source changed during encoding: {source}')
        tmp.replace(target)
        receipt = dict(policy=policy, source=source_stat, output=fingerprint(target),
                       input_video=info, output_video=output_info, filters=filters)
        receipt_tmp.write_text(json.dumps(receipt, indent=2) + '\n')
        receipt_tmp.replace(receipt_path(target))
    finally:
        tmp.unlink(missing_ok=True)
        receipt_tmp.unlink(missing_ok=True)
    return dict(source=str(source), status='encoded', frames=info['frames'])


def parse_box(text: str) -> list[float]:
    try:
        box = [float(v) for v in text.split(',')]
    except ValueError as exc:
        raise ValueError('box must be x1,y1,x2,y2') from exc
    if (len(box) != 4 or not all(math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
        raise ValueError(f'invalid normalized box: {text}')
    return box


def load_regions(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError('regions JSON must map video stem -> box or list of masks')
    if 'regions' in data:
        return data['regions'], data
    return data, {}


def masks_for(key: str, regions: dict, default_box: list[float] | None,
              start: float, end: float) -> list[dict] | None:
    value = regions.get(key) if regions else None
    if value is None:
        if default_box is None:
            return None
        return [{'box': default_box, 'start': start, 'end': end}]
    items = value if isinstance(value, list) else [value]
    masks = []
    for item in items:
        if isinstance(item, list):
            box, s, e = item[0], item[1] if len(item) > 1 else start, item[2] if len(item) > 2 else end
        elif isinstance(item, dict):
            box = item['box']
            s = item.get('start', start)
            e = item.get('end', end)
        else:
            raise ValueError(f'{key}: region must be a box or an object')
        if isinstance(box, str):
            box = parse_box(box)
        box = [float(v) for v in box]
        if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
            raise ValueError(f'{key}: invalid box {box}')
        s, e = float(s), float(e)
        if not (math.isfinite(s) and math.isfinite(e) and 0 <= s < e):
            raise ValueError(f'{key}: invalid interval {s}..{e}')
        masks.append({'box': box, 'start': s, 'end': e, 'reliable': True})
    return masks


def mask_regions(dataset: str, dataset_root: Path, regions_path: Path, *,
                 jobs: int = 8, limit: int | None = None,
                 allow_missing: bool = False, overwrite: bool = False) -> None:
    regions, meta = load_regions(regions_path)
    root = dataset_root
    source_dir = root / (meta.get('source_subdir')
                         or ('video' if dataset == 'finefs' else 'videos'))
    out = root / VIDEO_SUBDIR
    manifests = out / 'manifests'
    manifests.mkdir(parents=True, exist_ok=True)
    sources = sorted(source_dir.glob('*.mp4'))
    if limit:
        sources = sources[:limit]
    if not sources:
        raise SystemExit(f'no source videos under {source_dir}')
    if not allow_missing:
        missing = [s.stem for s in sources if s.stem not in regions]
        if missing:
            raise SystemExit(f'{len(missing)} source videos are absent from {regions_path} '
                             f'(first: {missing[:3]})')
    policy = meta.get('policy', {})
    blur_sigma = policy.get('blur_sigma', DEFAULT_BLUR_SIGMA)
    long_side = policy.get('long_side', 800 if dataset == 'finefs' else None)

    def work(source: Path):
        duration = probe(source)['duration']
        masks = masks_for(source.stem, regions, None, 0.0, duration)
        if masks is None:
            return source.name, 'skip (no region)'
        manifest = {'dataset': dataset, 'source': fingerprint(source),
                    'status': 'clear' if not masks else 'approved',
                    'review_note': 'replay from published receipt', 'masks': masks}
        manifest_path = manifests / (source.stem + '.json')
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        result = encode(source, out / source.name, dataset, manifest_path=manifest_path,
                        long_side=long_side, overwrite=overwrite, blur_sigma=blur_sigma)
        return source.name, result['status']

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, (name, status) in enumerate(pool.map(work, sources), 1):
            print(f'{i}/{len(sources)} {name} {status}', flush=True)
