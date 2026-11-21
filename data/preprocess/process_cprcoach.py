#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace
import re
import random
import subprocess
import uuid

from . import frame_budget
from .cprcoach_descriptions import (DESCRIPTION_BASIS, DESCRIPTION_VERSION,
                                    ERROR_DESCRIPTIONS, PROVISIONAL_DESCRIPTION_IDS)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = workspace() / 'datasets/CPR-Coach'
SOURCE = 'https://github.com/Shunli-Wang/CPR-Coach'
SOURCE_FPS = 25
LABELS = ('Correct', 'Overlap Hands', 'Clenching Hands', 'Single Hand',
          'Bending Arms', 'Tilting Arms', 'Jump Pressing', 'Squatting', 'Standing',
          'Wrong Position', 'Insufficient Pressing', 'Slow Frequency',
          'Excessive Pressing', 'Random Position Pressing')
LISTS = {'train': 'trainlist.txt', 'test': 'testlist.txt',
         'composite_test': 'all_errors_testlist.txt'}


def read_list(root, name):
    result = {}
    for number, line in enumerate((root / 'ann' / name).read_text().splitlines(), 1):
        if not line.strip():
            continue
        relative, count, *labels = line.split()
        parts = Path(relative).parts
        ids = tuple(map(int, labels))
        if (len(parts) != 4 or '..' in parts or Path(relative).is_absolute()
                or not re.fullmatch(r'ch[0-3]', parts[-1])
                or not re.fullmatch(r'r\d+', parts[-2]) or int(count) <= 0
                or not ids or len(set(ids)) != len(ids)
                or any(i not in range(14) for i in ids)
                or (0 in ids and len(ids) != 1) or relative in result):
            raise ValueError(f'Invalid/duplicate annotation: {name}:{number}: {line}')
        result[relative] = {'source_relative_path': relative, 'num_frames': int(count),
                            'label_ids': sorted(ids), 'recording_id': '/'.join(parts[:-1]),
                            'view': parts[-1]}
    if not result:
        raise ValueError(f'Empty annotation: {name}')
    return result


def load_samples(root):
    names = (root / 'ann/ActionList.txt').read_text().splitlines()[:14]
    if tuple(names) != LABELS:
        raise ValueError('ActionList.txt label order differs from expected mapping')
    splits = {split: read_list(root, name) for split, name in LISTS.items()}
    double = read_list(root, 'double_errors_testlist.txt')
    triquad = read_list(root, 'triquad_errors_testlist.txt')
    if set(double) & set(triquad) or double | triquad != splits['composite_test']:
        raise ValueError('Composite list must equal the disjoint double/triquad union')
    if any(len(s['label_ids']) != 2 for s in double.values()) or any(
            len(s['label_ids']) not in (3, 4) for s in triquad.values()):
        raise ValueError('Composite error cardinality disagrees with subset')
    seen_recordings, samples = set(), []
    for split, entries in splits.items():
        groups = {}
        for s in entries.values():
            if split != 'composite_test' and len(s['label_ids']) != 1:
                raise ValueError('Single-class split contains composite labels')
            groups.setdefault(s['recording_id'], []).append(s)
            samples.append({**s, 'split': 'train' if split == 'train' else 'test',
                            'test_subset': 'single' if split == 'test' else
                            ('double' if len(s['label_ids']) == 2 else 'triquad')
                            if split == 'composite_test' else None,
                            'annotation_file': LISTS[split]})
        if seen_recordings & groups.keys():
            raise ValueError('Recording leakage between official splits')
        seen_recordings.update(groups)
        for rid, views in groups.items():
            if {s['view'] for s in views} != {'ch0', 'ch1', 'ch2', 'ch3'} or len({
                    tuple(s['label_ids']) for s in views}) != 1:
                raise ValueError(f'Missing views or inconsistent labels: {rid}')
    return samples


def video_path(root, sample):
    return root / 'processed/videos' / (sample['source_relative_path'] + '.mp4')


def probe_video(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=codec_name,width,height,nb_frames,avg_frame_rate,duration',
                             '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)['streams'][0]


def validate_media(info, sample):
    if (info['codec_name'] != 'h264' or info['avg_frame_rate'] != '25/1'
            or int(info['nb_frames']) != sample['num_frames']
            or not math.isclose(float(info['duration']), sample['num_frames'] / SOURCE_FPS, abs_tol=1e-5)
            or max(info['width'], info['height']) != 800):
        raise ValueError(f"Invalid converted media: {sample['source_relative_path']}: {info}")


def build_record(root, sample, error_id, fps=None):
    if error_id not in range(1, len(LABELS)):
        raise ValueError('Expected an error ID from 1 to 13')
    duration = sample['num_frames'] / SOURCE_FPS
    present = error_id in sample['label_ids']
    instruction = (
        'Assess whether the specified error is present in the chest-compression action.\n\n'
        'Return only this JSON object and nothing else: '
        '{"error_present": "YES" or "NO"}.'
    )
    return {
        'messages': [
            {'role': 'system', 'content': instruction},
            {'role': 'user', 'content': f'<video>\nError: {LABELS[error_id]}\n'
             f'Description: {ERROR_DESCRIPTIONS[error_id]}'},
            {'role': 'assistant', 'content': json.dumps(
                {'error_present': 'YES' if present else 'NO'}, ensure_ascii=False)},
        ],
        'videos': [str(video_path(root, sample))],
        'chat_template_kwargs': {'fps': frame_budget.resolve_fps(duration, fps),
                                 'use_audio_in_video': False},
        'sample_id': 'cprcoach_' + sample['source_relative_path'].replace('/', '_')
                     + f'_error_{error_id:02d}',
        **sample, 'duration': duration, 'source_fps': SOURCE_FPS,
        'error_id': error_id, 'error_name': LABELS[error_id],
        'error_description': ERROR_DESCRIPTIONS[error_id],
        'description_version': DESCRIPTION_VERSION,
        'description_basis': DESCRIPTION_BASIS,
        'description_provisional': error_id in PROVISIONAL_DESCRIPTION_IDS,
        'error_present': present, 'answer_group': 'error' if present else 'no_error',
        'label_granularity': 'whole_video', 'source_url': SOURCE,
    }


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def build_rows(root, samples, seed=42, fps=None, view='ch0'):
    rng = random.Random(seed)
    groups = {}
    for sample in samples:
        groups.setdefault(sample['recording_id'], []).append(sample)
    rows = []
    for rid in sorted(groups):
        views = sorted(groups[rid], key=lambda s: s['view'])
        chosen = next((s for s in views if s['view'] == view), None)
        if chosen is None:
            raise ValueError(f'Recording without {view}: {rid}')
        present = sorted(i for i in chosen['label_ids'] if i != 0)
        absent = [i for i in range(1, len(LABELS)) if i not in present]
        count = max(1, len(present))
        if len(absent) < count:
            raise ValueError(f"Not enough distinct absent errors: {chosen['recording_id']}")
        for error_id in present + sorted(rng.sample(absent, count)):
            row = build_record(root, chosen, error_id, fps)
            row['view_selection_unit'] = 'recording'
            row['sample_polarity'] = 'negative' if row['error_present'] else 'positive'
            row['error_selection_seed'] = seed
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root', '--dataset-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out-dir', type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "skills")
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--fps', type=float)
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for absent-error selection (default: 42)')
    parser.add_argument('--view', default='ch0', choices=['ch0', 'ch1', 'ch2', 'ch3'],
                        help='the one channel every recording is exported from; the four '
                             'are simultaneous views of the same sequence (default: %(default)s)')
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    args = parser.parse_args(argv)
    encode_videos(['--dataset-root', str(args.root)])
    if args.workers < 1 or (args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0)):
        parser.error('workers/fps must be positive and finite')
    root = args.root.resolve()
    rows = build_rows(root, load_samples(root), args.seed, args.fps, args.view)
    if {r['view'] for r in rows} != {args.view}:
        raise ValueError('Exported rows mix camera channels')
    samples = list({r['source_relative_path']: r for r in rows}.values())
    print(f'Verifying {len(samples)} converted RGB videos', flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for sample, info in zip(samples, pool.map(probe_video, [video_path(root, s) for s in samples])):
            validate_media(info, sample)
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in ('train', 'test'):
        target = args.out_dir / f'cprcoach_{split}.jsonl'
        tmp = target.with_suffix('.jsonl.tmp')
        with tmp.open('w') as handle:
            for row in rows:
                if row['split'] == split:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        tmp.replace(target)

    obsolete = ['error', 'no_error'] + [f'{split}_{group}'
                for split in ('train', 'test') for group in ('error', 'no_error')]
    for suffix in obsolete:
        (args.out_dir / f'cprcoach_{suffix}.jsonl').unlink(missing_ok=True)
    print(f"Answer groups: {dict(Counter(r['answer_group'] for r in rows))}")
    print(f"Split counts: {dict(Counter(r['split'] for r in rows))}")
    print(f"Sampling: {frame_budget.fps_histogram(rows)}")


def encode_videos(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root', '--dataset-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--jobs', type=int, default=32)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    root = args.root.resolve()
    samples = load_samples(root)
    print(f'Encoding/verifying {len(samples)} RGB videos with {args.jobs} workers', flush=True)
    counts = Counter()
    with ThreadPoolExecutor(args.jobs) as pool:
        for index, status in enumerate(pool.map(lambda s: encode_rgb(root, s), samples), 1):
            counts[status] += 1
            if index % 100 == 0 or index == len(samples):
                print(f'{index}/{len(samples)}: {dict(counts)}', flush=True)
    atomic_json(root / 'processed/encoding_manifest.json', {
        'recipe': 'rgb_jpg_25fps_h264_crf20_veryfast_800px_v1',
        'source_fps': SOURCE_FPS, 'counts': dict(counts),
        'videos': [{**s, 'output': str(video_path(root, s))} for s in samples],
    })


def validate_frames(root, sample):
    source = root / sample['source_relative_path']
    actual = {p.name for p in source.iterdir() if p.suffix.lower() == '.jpg'}
    expected = {f'{i:08d}.jpg' for i in range(1, sample['num_frames'] + 1)}
    if actual != expected:
        raise ValueError(f'RGB frame sequence mismatch: {source}: '
                         f'{len(actual)} files / {sample["num_frames"]} annotated')


def encode_rgb(root, sample):
    validate_frames(root, sample)
    target = video_path(root, sample)
    if target.is_file():
        validate_media(probe_video(target), sample)
        return 'reused'
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f'.{uuid.uuid4().hex}.tmp.mp4')
    cmd = ['ffmpeg', '-y', '-nostdin', '-v', 'error', '-threads', '1',
           '-framerate', str(SOURCE_FPS), '-start_number', '1',
           '-i', str(root / sample['source_relative_path'] / '%08d.jpg'),
           '-frames:v', str(sample['num_frames']), '-an',
           '-vf', "scale=w='if(gte(iw,ih),800,-2)':h='if(gte(iw,ih),-2,800)',setsar=1",
           '-filter_threads', '1', '-c:v', 'libx264', '-threads', '1',
           '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(tmp)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        validate_media(probe_video(tmp), sample)
        check = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror',
                                '-threads', '1', '-i', str(tmp), '-map', '0:v:0',
                                '-f', 'null', '-', '-progress', 'pipe:1'],
                               capture_output=True, text=True, check=True)
        frames = [int(line.split('=', 1)[1]) for line in check.stdout.splitlines()
                  if line.startswith('frame=')]
        if not frames or frames[-1] != sample['num_frames']:
            raise ValueError(f'Decoded frame count mismatch: {target}')
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return 'encoded'


if __name__ == '__main__':
    main()