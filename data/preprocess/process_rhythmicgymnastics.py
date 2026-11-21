#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
import json
import math
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

from . import frame_budget
from .scoreboard_mask import VIDEO_SUBDIR, mask_regions, require_masked
from .rhythmicgymnastics_rules import (
    APPARATUS, RULE_VERSION, event_metadata, system_prompt, user_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = workspace() / "datasets" / "RhythmicGymnastics"
HEADER = ["Video_Name", "Difficulty_Score", "Execution_Score", "Total_Score", "Penalty", "Video_length"]
KEYS = ["difficulty_score", "execution_score", "total_score", "penalty"]


def read_split(root: Path, split: str, video_dir: str = 'videos') -> list[dict]:
    path = root / f"{split}.txt"
    lines = path.read_text().splitlines()
    if not lines or lines[0].split() != HEADER:
        raise ValueError(f"Unexpected header: {path}")
    samples = []
    seen = set()
    for line_number, line in enumerate(lines[1:], 2):
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"{path}:{line_number}: expected six fields")
        clip = parts[0]
        apparatus = clip.split('_')[0]
        if apparatus not in APPARATUS or clip in seen or '/' in clip:
            raise ValueError(f"Invalid/duplicate clip: {clip}")
        seen.add(clip)
        scores = {k: Decimal(v) for k, v in zip(KEYS, parts[1:5])}
        if any(not v.is_finite() or v < 0 for v in scores.values()):
            raise ValueError(f"Invalid score: {clip}")
        if scores['execution_score'] > 10:
            raise ValueError(f"Execution score above 10: {clip}")
        delta = scores['total_score'] - (scores['difficulty_score'] + scores['execution_score'] - scores['penalty'])
        if abs(delta) > Decimal('0.01'):
            raise ValueError(f"Unresolved score discrepancy: {clip}: {delta}")
        length = float(parts[5])
        if not math.isfinite(length) or length <= 0:
            raise ValueError(f"Invalid length: {clip}")
        video = (root / video_dir / f'{clip}.mp4').resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        samples.append(dict(clip=clip, apparatus=apparatus, split=split,
                            scores=scores, score_delta=delta, video=str(video),
                            annotated_length=length, annotation_line=line_number,
                            raw_annotation=line, event_metadata=event_metadata(clip, line)))
    expected = 200 if split == 'train' else 50
    if Counter(s['apparatus'] for s in samples) != Counter({a: expected for a in APPARATUS}):
        raise ValueError(f"{split} does not match the official four-by-{expected} split")
    return samples


def duration(video: str) -> float:
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=duration', '-of', 'json', video],
                            check=True, capture_output=True, text=True, timeout=30)
    value = float(json.loads(result.stdout)['streams'][0]['duration'])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid video duration: {video}")
    return value


def judge_scores(scores: dict) -> dict:
    return {
        'execution_score': scores['execution_score'].quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
        'penalty': (scores['penalty'] / Decimal('0.05')).quantize(
            Decimal('1'), rounding=ROUND_HALF_UP) * Decimal('0.05'),
    }


def answer(scores: dict) -> str:
    rounded = judge_scores(scores)
    return (f'{{"execution_score": {rounded["execution_score"]:.1f}, '
            f'"penalty": {rounded["penalty"]:.2f}}}')


def build_record(sample: dict, fps: float | None) -> dict:
    return {
        'messages': [
            {'role': 'system', 'content': system_prompt(sample['apparatus'], sample['event_metadata']['rule_profile'])},
            {'role': 'user', 'content': user_prompt(sample['apparatus'])},
            {'role': 'assistant', 'content': answer(sample['scores'])},
        ],
        'videos': [sample['video']],
        'chat_template_kwargs': {'fps': frame_budget.resolve_fps(sample['duration'], fps)},
        'dataset': 'RhythmicGymnastics', 'clip': sample['clip'],
        'apparatus': sample['apparatus'], 'split': sample['split'], 'target': 'execution_penalty',
        'duration_seconds': sample['duration'],
        'annotated_video_length': sample['annotated_length'],
        'scores': {k: float(v) for k, v in judge_scores(sample['scores']).items()},
        'original_scores': {k: float(v) for k, v in sample['scores'].items()},
        'target_rounding': {'execution_score': '0.1', 'penalty': '0.05', 'mode': 'ROUND_HALF_UP'},
        'score_formula_delta': float(sample['score_delta']),
        'annotation_file': sample['split'] + '.txt',
        'annotation_line': sample['annotation_line'],
        'raw_annotation': sample['raw_annotation'],
        'rubric_version': RULE_VERSION,
        **sample['event_metadata'],
        'label_granularity': 'whole_routine',
        'audio_in_model_input': False,
    }


def verify(rows: list[dict], samples: list[dict]) -> None:
    if len(rows) != len(samples):
        raise ValueError('Row count changed')
    for row, sample in zip(rows, samples):
        if row != build_record(sample, row['chat_template_kwargs']['fps']):
            raise ValueError(f"Record changed on serialization: {sample['clip']}")
        if row['messages'][1]['content'].count('<video>') != len(row['videos']):
            raise ValueError('Video placeholder mismatch')
        result = json.loads(row['messages'][-1]['content'], parse_float=Decimal)
        expected = judge_scores(sample['scores'])
        if result != expected:
            raise ValueError(f"Target differs from rounded source: {sample['clip']}")

        if row['messages'][:2] != [
            {'role': 'system', 'content': system_prompt(sample['apparatus'], sample['event_metadata']['rule_profile'])},
            {'role': 'user', 'content': user_prompt(sample['apparatus'])},
        ]:
            raise ValueError('Unexpected model input')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--video-dir', default=VIDEO_SUBDIR, help='Scoreboard-masked videos under dataset root')
    parser.add_argument('--out-dir', type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "sports")
    parser.add_argument('--fps', type=float, help='Override the shared 128-frame budget; also configure the loader frame cap.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    args = parser.parse_args()
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error('--fps must be finite and positive')
    if args.workers < 1:
        parser.error('--workers must be positive')
    mask_regions('rg', args.dataset_root,
                 REPO_ROOT / 'data' / 'annotations' / 'rg_scoreboard.json')
    root = args.dataset_root.resolve()
    splits = {name: read_split(root, name, args.video_dir) for name in ['train', 'test']}
    if {s['clip'] for s in splits['train']} & {s['clip'] for s in splits['test']}:
        raise ValueError('Train/test overlap')
    all_samples = splits['train'] + splits['test']
    for sample in all_samples:
        require_masked(Path(sample['video']), 'rg')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        durations = list(pool.map(duration, [s['video'] for s in all_samples]))
    for sample, seconds in zip(all_samples, durations):
        sample['duration'] = seconds
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = 'rhythmicgymnastics'
    for split, samples in splits.items():
        rows = [build_record(s, args.fps) for s in samples]
        verify(rows, samples)
        path = args.out_dir / f'{prefix}_{split}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        verify([json.loads(line) for line in path.read_text().splitlines()], samples)
        print(f'{split}: {len(rows)} rows -> {path}')
        print(f'  {frame_budget.fps_histogram(rows)}')
    discrepancies = sum(bool(s['score_delta']) for s in all_samples)
    print(f'Validated all targets, paths, schemas and splits; {discrepancies} 0.01-point discrepancies preserved.')
    if args.fps:
        print(f'Requested up to {math.ceil(max(durations)*args.fps)} frames; the loader cap must be raised if above 128.')


if __name__ == '__main__':
    main()
