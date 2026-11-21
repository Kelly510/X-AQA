#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import subprocess

from .rehabilitation_common import (arguments, attach_hashes, attach_media, order_record,
                                   record, sha256, validate)
from . import frame_budget

SOURCE = 'https://arxiv.org/html/2412.10573v1#S4.T1'
VERSION = 'execheck_rgb_correctness_v1'

ERRORS = {
    'arm_circle': 'Raised shoulders or bent arms.',
    'forward_lunge': 'Forward trunk lean, knee beyond toes, or the trailing knee striking the floor.',
    'high_knee_raise': 'Trunk wobble, insufficient knee lift, or an abrupt foot drop.',
    'hip_abduction': 'Trunk tilt away from the moving leg.',
    'leg_extension': 'Forward trunk lean or incomplete leg straightening.',
    'shoulder_abduction': 'Uneven arm elevation, compensatory head tilt, or raised shoulders.',
    'shoulder_external_rotation': 'Elbow moving away from the torso.',
    'shoulder_flexion': 'Uneven or insufficient arm elevation.',
    'side_step_squat': 'Trunk rotation or forward trunk lean.',
    'squat': 'Knees collapsing inward or loss of an upright, straight back.',
}


def prompt(sample):
    return (f"Assess the form of this rehabilitation exercise: {sample['exercise']}.\n"
            f"The dataset's incorrect-form criteria are: {ERRORS[sample['exercise_code']]}\n"
            'Judge the visible repetition against these criteria. Incorrect means at least one '
            'listed form error; correct means the intended form is maintained. '
            'Return only this JSON object and nothing else: '
            '{"exercise_form": "correct" or "incorrect"}.')


def load_recordings(root):
    root = root.resolve()
    result = []
    with (root / 'processing_scripts/RepSeg.csv').open(newline='') as f:
        for row in csv.reader(f):
            if not row or not row[0].startswith('file://'):
                continue
            original = Path(row[0][7:])
            subject = original.parent.name
            exercise, label = original.stem.rsplit('_', 1)
            bounds = list(map(int, row[1:]))
            if (subject not in {f'p{i}' for i in range(7)} or exercise not in ERRORS
                    or label not in {'correct', 'incorrect'} or original.suffix != '.mkv'):
                raise ValueError(f'Unknown annotation: {row}')
            if len(bounds) != 6 or bounds[0] < 0 or any(a >= b for a, b in zip(bounds, bounds[1:])):
                raise ValueError(f'Invalid zero-based repetition boundaries: {row}')
            source = root / 'raw_data' / subject / original.name
            skeleton = source.with_name(source.stem + '_sdk.json')
            if not source.is_file() or not skeleton.is_file():
                raise ValueError(f'Missing raw recording/skeleton: {source}')
            result.append(dict(id=f'execheck_{subject}_{original.stem}', subject=subject,
                               exercise=exercise.replace('_', ' '), exercise_code=exercise,
                               label=label, boundaries=bounds,
                               source_path=str(source.relative_to(root)),
                               skeleton_path=str(skeleton.relative_to(root)),
                               video=str(root / 'processed/videos' / subject / (original.stem + '.mp4')),
                               split='test' if subject == 'p6' else 'train'))
    keys = {(r['subject'], r['exercise_code'], r['label']) for r in result}
    expected = {(f'p{i}', ex, label) for i in range(7) for ex in ERRORS for label in ('correct', 'incorrect')}
    if keys != expected or len(result) != 140:
        raise ValueError('Expected exactly 140 unique subject/exercise/label recordings')
    if {root / r['source_path'] for r in result} != set((root / 'raw_data').rglob('*.mkv')):
        raise ValueError('Raw MKV / annotation coverage mismatch')
    return sorted(result, key=lambda r: r['id'])


def effective_boundaries(bounds, nframes):
    if bounds[-1] > nframes + 1 or bounds[-2] >= nframes:
        raise ValueError('Annotation exceeds RGB frame count by more than the allowed final frame')
    return [*bounds[:-1], min(bounds[-1], nframes)]


def build_rows(samples, fps=None):
    rows = []
    for sample in samples:
        nframes = int(sample['media']['nb_frames'])
        bounds = effective_boundaries(sample['boundaries'], nframes)
        for repetition, (start, end) in enumerate(zip(bounds, bounds[1:]), 1):
            segment = {k: v for k, v in sample.items() if k != 'boundaries'}
            segment.update(id=f"{sample['id']}_R{repetition}", recording_id=sample['id'],
                           repetition=repetition, start_frame=start, end_frame_exclusive=end,
                           annotated_end_frame_exclusive=sample['boundaries'][repetition],
                           end_frame_clamped=sample['boundaries'][repetition] != end,
                           duration_seconds=(end - start) / 30,
                           recording_duration_seconds=sample['duration_seconds'],
                           label_source='recording filename; processing_scripts/label.txt',
                           segmentation_source='processing_scripts/RepSeg.csv',
                           boundary_convention='zero-based [start, end)',
                           source_fps=30, split_protocol='paper_subjects_1_to_6_train_7_test_no_mirror')
            segment['target'] = json.dumps(
                {'exercise_form': segment['label']}, ensure_ascii=False)
            row = record(segment, 'ExeCheck', prompt(segment), SOURCE, VERSION, fps)
            row['label_granularity'] = 'repetition (inherits recording correctness)'
            row['chat_template_kwargs'].update(video_start=start / 30, video_end=end / 30)
            rows.append(order_record(row))
    return rows


def main():
    args = arguments('ExeCheck', '').parse_args()
    encode_videos(['--dataset-root', str(args.dataset_root)])
    if args.workers < 1 or (args.fps is not None and (not 0 < args.fps < float('inf'))):
        raise ValueError('Workers and fps must be positive and finite')
    samples = load_recordings(args.dataset_root)
    manifest_path = args.dataset_root / 'processed/encoding_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    encoded = {r['source_path']: r for r in manifest['recordings']}
    for sample in samples:
        info = encoded[sample['source_path']]
        if info['color_title'] != 'COLOR' or info['output_frames'] != info['source_frames']:
            raise ValueError(f"Invalid RGB conversion: {sample['id']}")
        sample['source_color_stream_index'] = info['color_stream_index']
        sample['source_sha256'] = info['source_sha256']
    attach_media(samples, args.workers)
    attach_hashes(samples, args.workers)
    for sample in samples:
        info = encoded[sample['source_path']]
        if (sample['content_hash'] != info['output_sha256']
                or int(sample['media']['nb_frames']) != info['source_frames']
                or sample['media']['avg_frame_rate'] != '30/1'):
            raise ValueError(f"Converted media differs from manifest: {sample['id']}")
    rows = build_rows(samples, args.fps)
    validate(rows)
    if len(rows) != 700 or Counter(r['split'] for r in rows) != {'train': 600, 'test': 100}:
        raise ValueError('Unexpected split counts')
    for split in ('train', 'test'):
        coverage = Counter((r['exercise_code'], r['label']) for r in rows if r['split'] == split)
        if set(coverage.values()) != {30 if split == 'train' else 5}:
            raise ValueError('Unbalanced exercise/label coverage')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'test'):
        path = args.output_dir / f'execheck_correctness_{split}.jsonl'
        tmp = path.with_suffix('.jsonl.tmp')
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows if r['split'] == split))
        tmp.replace(path)
    print(f'Exported {len(rows)} repetitions: 600 train / 100 test.')
    print(frame_budget.fps_histogram(rows))
    print(f"Clamped final boundaries: {sum(r['end_frame_clamped'] for r in rows)}")


def encode_videos(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset-root', type=Path, default=Path('datasets/ExeCheck'))
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    root = args.dataset_root.resolve()
    samples = load_recordings(root)
    manifest_path = root / 'processed/encoding_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'recordings': []}
    cached = {r['source_path']: r for r in previous['recordings']}
    records, counts = [], {}
    print(f'Converting {len(samples)} COLOR recordings -> {root / "processed/videos"}', flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(encode, root, s, cached.get(s['source_path']), args.overwrite) for s in samples]
        for future in as_completed(futures):
            info, status = future.result()
            records.append(info)
            counts[status] = counts.get(status, 0) + 1

            cached[info['source_path']] = info
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = manifest_path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps({'recipe': RECIPE, 'recordings': sorted(cached.values(), key=lambda r: r['source_path'])}, indent=2) + '\n')
            tmp.replace(manifest_path)
            if len(records) % 10 == 0 or len(records) == len(samples):
                print(f'{len(records)}/{len(samples)}: {counts}', flush=True)
    print(f'All {len(records)} outputs passed full decode and source/output frame-count checks.', flush=True)


RECIPE = 'color_h264_crf20_veryfast_800px_30fps_frame_order_v1'


def probe(path, source=False):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v',
               '-count_packets' if source else '-count_frames', '-show_streams', '-of', 'json', str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise ValueError(f'Cannot probe {path}: {result.stderr[-1000:]}')
    streams = json.loads(result.stdout)['streams']
    if source:
        streams = [s for s in streams if s.get('tags', {}).get('title') == 'COLOR']
    if len(streams) != 1:
        raise ValueError(f'Expected exactly one {"COLOR" if source else "output"} stream: {path}')
    stream = streams[0]
    if Fraction(stream['avg_frame_rate']) != 30:
        raise ValueError(f'Expected 30 fps: {path}')
    stream['frames'] = int(stream['nb_read_packets' if source else 'nb_read_frames'])
    return stream


def encode(root, sample, previous, overwrite):
    source, target = root / sample['source_path'], Path(sample['video'])
    stream = probe(source, source=True)
    effective_boundaries(sample['boundaries'], stream['frames'])
    sidecar = Path(str(source) + '.sha256')
    fingerprint = dict(source_path=sample['source_path'], source_size=source.stat().st_size,
                       source_mtime_ns=source.stat().st_mtime_ns,
                       source_sha256=sidecar.read_text().split()[0],
                       source_sha256_origin='local acquisition sidecar (not publisher checksum)',
                       color_stream_index=stream['index'], color_title=stream['tags']['title'],
                       source_frames=stream['frames'], recipe=RECIPE)
    if target.is_file() and not overwrite and previous and all(previous.get(k) == v for k, v in fingerprint.items()):
        out = probe(target)
        if (out['frames'] == stream['frames'] and out['codec_name'] == 'h264'
                and max(out['width'], out['height']) == 800
                and 'sha256:' + sha256(target) == previous['output_sha256']):
            return previous, 'cached'
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp.mp4')
    scale = 'scale=800:-2' if stream['width'] >= stream['height'] else 'scale=-2:800'
    try:
        cmd = ['ffmpeg', '-hide_banner', '-v', 'error', '-xerror', '-y', '-threads', '2',
               '-i', str(source), '-map', f"0:{stream['index']}",
               '-vf', scale + ',setsar=1,setpts=N/(30*TB)', '-filter_threads', '1',
               '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough',
               '-an', '-sn', '-dn', '-map_metadata', '-1', '-movflags', '+faststart', str(tmp)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError(f'Encoding failed for {source}: {result.stderr[-2000:]}')
        out = probe(tmp)
        if out['frames'] != stream['frames'] or out['codec_name'] != 'h264':
            raise ValueError(f'Frame count/codec mismatch for {target}')
        digest = 'sha256:' + sha256(tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return {**fingerprint, 'output_path': str(target.relative_to(root)),
            'output_frames': out['frames'], 'width': out['width'], 'height': out['height'],
            'output_sha256': digest}, 'encoded'


if __name__ == '__main__':
    main()
