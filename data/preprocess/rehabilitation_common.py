from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace
import random
import subprocess

from . import frame_budget
from . import loss_scale

REPO_ROOT = Path(__file__).resolve().parents[2]


VIDEO_MAX_PIXELS = 448 * 32 * 32


def arguments(dataset: str, description: str):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--dataset-root', type=Path, default=workspace() / 'datasets' / dataset)
    p.add_argument('--output-dir', type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "rehabilitation")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fps', type=float)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--balance', action='store_true',
                   help='uniform-interface flag; these rehabilitation datasets have no class '
                        'imbalance to upsample, so it is a no-op')
    return p


def videos(root: Path):
    return sorted(p.resolve() for p in root.rglob('*') if p.suffix.lower() in {'.mp4', '.mov'} and p.is_file())


def sha256(path: Path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def attach_hashes(samples, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sample, digest in zip(samples, pool.map(sha256, (Path(s['video']) for s in samples))):
            sample['content_hash'] = 'sha256:' + digest


def subject_split(samples, test_count, seed):
    subjects = sorted({s['subject'] for s in samples})
    if not 0 < test_count < len(subjects):
        raise ValueError('Invalid test participant count')
    random.Random(seed).shuffle(subjects)
    heldout = set(subjects[:test_count])
    for s in samples:
        s['split'] = 'test' if s['subject'] in heldout else 'train'


def stratified_split(samples, seed):
    by_hash = defaultdict(list)
    for s in samples:
        by_hash[s['content_hash']].append(s)
    strata = defaultdict(list)
    for digest, group in sorted(by_hash.items()):
        keys = {(s['exercise'], s['label']) for s in group}
        if len(keys) != 1:
            raise ValueError(f'Conflicting labels for identical video: {digest}')
        strata[next(iter(keys))].append(group)
    rng = random.Random(seed)
    for key, groups in sorted(strata.items()):
        if len(groups) < 2:
            raise ValueError(f'Cannot split stratum {key}')
        rng.shuffle(groups)
        target = max(1, round(sum(map(len, groups)) * .2))
        ntest = 0
        for i, group in enumerate(groups):
            split = 'test' if ntest < target and i < len(groups) - 1 else 'train'
            if split == 'test':
                ntest += len(group)
            for s in group:
                s['split'] = split


def probe(path):
    p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=index,codec_name,width,height,duration,avg_frame_rate,nb_frames:format=duration',
                        '-of', 'json', str(path)], capture_output=True, text=True, timeout=60)
    if p.returncode:
        raise ValueError(f'ffprobe failed: {path}: {p.stderr[:200]}')
    info = json.loads(p.stdout)
    if not info.get('streams'):
        raise ValueError(f'No video stream: {path}')
    stream = info['streams'][0]
    duration = float(stream.get('duration', info.get('format', {}).get('duration', 0)))
    if not math.isfinite(duration) or duration <= 0 or min(stream['width'], stream['height']) <= 0:
        raise ValueError(f'Invalid media dimensions/duration: {path}')
    return {'duration_seconds': duration, 'media': stream}


def attach_media(samples, workers):
    print(f'Probing {len(samples)} videos', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for s, info in zip(samples, pool.map(probe, (s['video'] for s in samples))):
            s.update(info)


def order_record(row):
    leading = ('messages', 'videos', 'chat_template_kwargs', 'dataset', 'id',
               'split', 'exercise', 'exercise_code', 'subject', 'label', 'score')
    return {**{k: row[k] for k in leading if k in row},
            **{k: v for k, v in row.items() if k not in leading}}


def record(sample, dataset, system, rubric_source, rubric_version, fps):
    return order_record({
        **{k: v for k, v in sample.items() if k != 'video'},
        'dataset': dataset,
        'messages': [{'role': 'system', 'content': system},
                     {'role': 'user', 'content': '<video>'},
                     {'role': 'assistant', 'content': sample.get('target', sample['label'])}],
        'videos': [sample['video']],
        'chat_template_kwargs': {'fps': frame_budget.resolve_fps(sample['duration_seconds'], fps),
                                 'min_frames': 4, 'max_frames': 128,
                                 'max_pixels': VIDEO_MAX_PIXELS},
        'rubric_source': rubric_source, 'rubric_version': rubric_version,
        'label_granularity': 'whole_video', 'audio_in_model_input': False,
    })


def validate(rows, subject_disjoint=True, test_only=False):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty export or duplicate IDs')
    groups = defaultdict(set)
    coverage = defaultdict(set)
    for r in rows:
        if r['split'] not in {'train', 'test'}:
            raise ValueError('Invalid split')
        if not Path(r['videos'][0]).is_file():
            raise ValueError('Missing video')
        if [m['role'] for m in r['messages']] != ['system', 'user', 'assistant'] or r['messages'][1]['content'] != '<video>':
            raise ValueError('Invalid conversation')
        if r['messages'][-1]['content'] != r.get('target', r['label']):
            raise ValueError('Target mismatch')
        if not math.isfinite(r['chat_template_kwargs']['fps']) or r['chat_template_kwargs']['fps'] <= 0:
            raise ValueError('Invalid fps')
        groups[('hash', r['content_hash'])].add(r['split'])
        if subject_disjoint:
            groups[('subject', r['subject'])].add(r['split'])
        coverage[r['split']].add((r['exercise'], r.get('class_label', r['label'])))
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError('Content/participant leakage across splits')
    if test_only:
        if any(r['split'] != 'test' for r in rows):
            raise ValueError('Test-only export contains training rows')
        return
    if coverage['train'] != coverage['test']:
        raise ValueError(f'Task/class coverage differs: {coverage}')


def export(samples, args, dataset, prefix, prompt, rubric_source, rubric_version, audit, subject_disjoint=True, test_only=False, loss_weighted=False):
    if args.workers < 1 or (args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0)):
        raise ValueError('Workers and fps must be positive')
    attach_media(samples, args.workers)
    usable = []
    for sample in samples:
        frames = sample['media'].get('nb_frames')
        if frames is not None and frames != 'N/A' and int(frames) < 2:
            audit.setdefault('excluded', []).append({'video': sample['source_path'], 'reason': 'fewer_than_two_video_frames'})
        else:
            usable.append(sample)
    rows = [record(s, dataset, prompt(s), rubric_source, rubric_version, args.fps)
            for s in sorted(usable, key=lambda s: s['id'])]
    validate(rows, subject_disjoint, test_only=test_only)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in (('test',) if test_only else ('train', 'test')):
        subset = [r for r in rows if r['split'] == split]
        if loss_weighted and split == 'train':
            cell_of, class_of = (lambda r: r['exercise']), (lambda r: r['label'])
            weights = loss_scale.balanced_weights(subset, cell_of, class_of)
            for row in subset:
                loss_scale.attach(row, weights[(cell_of(row), class_of(row))])
            audit['loss_scale'] = loss_scale.summarise(subset, cell_of, class_of, weights)
        path = args.output_dir / f'{prefix}_{split}.jsonl'
        tmp = path.with_suffix('.jsonl.tmp')
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in subset))
        tmp.replace(path)
    if test_only:
        (args.output_dir / f'{prefix}_train.jsonl').unlink(missing_ok=True)
    audit.update(dataset=dataset, seed=args.seed, subject_disjoint=subject_disjoint,
                 rubric_version=rubric_version, rubric_source=rubric_source,
                 rows=len(rows), split_counts=dict(Counter(r['split'] for r in rows)),
                 task_counts=dict(Counter(f"{r['split']}/{r['exercise']}/{r.get('class_label', r['label'])}" for r in rows)),
                 participants={split: sorted({r.get('subject') for r in rows if r['split'] == split and r.get('subject') is not None}) for split in ('train', 'test')},
                 duplicate_groups=[ids for ids in _duplicates(rows).values() if len(ids) > 1],
                 fps_histogram=frame_budget.fps_histogram(rows))
    print(json.dumps({**audit, 'excluded': {'count': len(audit.get('excluded', [])),
                                         'examples': audit.get('excluded', [])[:3]}},
                     indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return rows


def _duplicates(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['content_hash']].append(r['id'])
    return groups
