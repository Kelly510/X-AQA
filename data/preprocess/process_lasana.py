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
from statistics import median

from . import frame_budget
from . import loss_scale

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = workspace() / 'datasets' / 'LASANA'
ERROR_SAMPLING_SEED = 42
ASPECTS = ('depth_perception', 'efficiency', 'bimanual_dexterity', 'tissue_handling')
ASPECT_NAMES = ('Depth perception', 'Efficiency', 'Bimanual dexterity', 'Tissue handling')
SCORE_OUTPUT_EXAMPLE = "{" + ", ".join(
    f'"{name}": <number from 1.0 to 5.0 in increments of 0.5>' for name in ASPECT_NAMES) + "}"
ASPECT_DEFINITIONS = {
    'Depth perception': 'accuracy of instrument targeting, grasping and placement.',
    'Efficiency': 'purposeful progress, avoiding pauses and unnecessary adjustments.',
    'Bimanual dexterity': 'coordination and mutual assistance of both instruments.',
    'Tissue handling': 'controlled force and preservation of the task material.',
}

TASKS = {
    'peg_transfer': ('Peg transfer', {
        'object_dropped_within_fov': 'A dropped object remains visible and retrievable.',
        'object_dropped_outside_of_fov': 'An object is dropped out of view or cannot be retrieved.',
    }),
    'circle_cutting': ('Circle cutting', {
        'cutting_imprecise': 'The cut deviates from the marked circle.',
        'gauze_detached': 'The gauze detaches from at least one retaining clip.',
    }),
    'balloon_resection': ('Balloon resection', {
        'cutting_imprecise': 'The cut deviates from the marked line.',
        'cutting_incomplete': 'Part of the marked line remains uncut.',
        'balloon_opened': 'The inner balloon opens and loses over half its water.',
        'balloon_damaged': 'The inner balloon is damaged and loses less than half its water.',
        'balloon_perforated': 'The inner balloon has a puncture with minor water leakage.',
    }),
    'suture_and_knot': ('Suture and knot', {
        'needle_dropped': 'The needle is dropped.',
        'suture_imprecise': 'Needle placement misses at least one marked dot.',
        'fewer_than_three_throws': 'Fewer than three knot throws are completed successfully.',
        'slit_not_closed': 'The slit remains open.',
        'knot_comes_apart': 'The knot unravels.',
        'drain_detached': 'The drain detaches from its mounting.',
    }),
}

TASK_ANCHORS = {
    'peg_transfer': {
        'Depth perception': ('Missed grasps', 'Slow accuracy', 'Fast precise transfers'),
        'Efficiency': ('Hesitation', 'Minor realignment', 'Uninterrupted progress'),
        'Bimanual dexterity': ('Isolated movements', 'Stationary handoffs', 'Anticipatory coordination'),
        'Tissue handling': ('Squeezing', 'Some pressure', 'Secure pressure-free placement'),
    },
    'circle_cutting': {
        'Depth perception': ('Air cuts', 'Mostly successful cuts', 'Consistent gauze contact'),
        'Efficiency': ('Repeated unproductive cuts', 'Usually progresses', 'Every cut progresses'),
        'Bimanual dexterity': ('Interference', 'Delayed assistance', 'Timely supportive traction'),
        'Tissue handling': ('Tearing', 'Minor deviations', 'Intact fibers, precise circle'),
    },
    'balloon_resection': {
        'Depth perception': ('Repeated probing', 'Cautious successful entry', 'Stable cutting plane'),
        'Efficiency': ('Loses opening', 'Fluid assisted cutting', 'Continuous cutting'),
        'Bimanual dexterity': ('Interference', 'Usually supports', 'Continuous unobstructed support'),
        'Tissue handling': ('Tearing, water loss', 'Gentle separation', 'Precise cuts, intact inner balloon'),
    },
    'suture_and_knot': {
        'Depth perception': ('Missed targets', 'Slow accurate placement', 'Precise stitching, loops'),
        'Efficiency': ('Repeated reorientation', 'Minor adjustments', 'Direct stitching, handoff'),
        'Bimanual dexterity': ('Unassisted movements', 'Sequential assistance', 'Coordinated needle, thread control'),
        'Tissue handling': ('Deformation, tearing', 'Limited tension', 'Undamaged material, controlled tightening'),
    },
}

def collect_samples(root):
    samples = {'train': [], 'test': []}
    seen = set()
    for task, (_, error_defs) in TASKS.items():
        for split in ('train', 'validation', 'test'):
            metadata = root / task / split / 'metadata.jsonl'
            rows = [json.loads(line) for line in metadata.read_text(encoding='utf-8').splitlines() if line.strip()]
            if not rows:
                raise ValueError(f'Empty metadata: {metadata}')
            for row in rows:
                relative = Path(row['left_file_name'])
                if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != 'left' or relative.suffix != '.mkv':
                    raise ValueError(f'Unexpected left video path: {relative}')
                key = (task, relative.stem)
                if key in seen:
                    raise ValueError(f'Duplicate or cross-split recording: {key}')
                seen.add(key)
                if type(row['grs']) not in (float, int) or not math.isfinite(row['grs']):
                    raise ValueError(f'Invalid GRS: {key}')
                if any(type(row.get(tag)) is not bool for tag in error_defs):
                    raise ValueError(f'Missing/non-binary error label: {key}')
                aspect_scores, raw_ratings = raw_aspect_target(row)
                if split != 'validation':
                    samples[split].append({
                        'task': task, 'recording_id': relative.stem, 'split': split,
                        'video': str((metadata.parent / relative).resolve()),
                        'aspect_scores': aspect_scores, 'raw_ratings': raw_ratings,
                        'grs': row['grs'], 'errors': [tag for tag in error_defs if row[tag]],
                    })
    return samples

def raw_aspect_target(row):
    raters = sorted({match.group(1) for key in row
                     if (match := re.fullmatch(r'(rater\d+)_(?:' + '|'.join(ASPECTS) + ')', key))})
    ratings, valid = {}, []
    for rater in raters:
        values = {aspect: row.get(f'{rater}_{aspect}') for aspect in ASPECTS}
        ratings[rater] = values
        if all(v is None for v in values.values()):
            continue
        if any(type(v) not in (int, float) or not math.isfinite(v)
               or not 1 <= v <= 5 or v != int(v) for v in values.values()):
            raise ValueError(f'Incomplete or invalid raw aspect ratings: {rater}')
        valid.append(values)
    scores = ({aspect: float(median(r[aspect] for r in valid)) for aspect in ASPECTS}
              if len(valid) >= 2 else None)
    return scores, ratings


def task_scoring_prompt(task):
    lines = ['Scoring criteria:']
    for dimension, anchors in TASK_ANCHORS[task].items():
        levels = '; '.join(f'{score}: {description}' for score, description in zip((1, 3, 5), anchors))
        lines.append(f'- {dimension}: {ASPECT_DEFINITIONS[dimension]}\n  {levels}.')
    return '\n'.join(lines)


def build_record(sample, duration, supervision, fps=None, *, error_category=None):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid video duration: {sample['video']}")
    task = sample['task']
    name, error_defs = TASKS[task]
    if supervision == 'score':
        if sample['aspect_scores'] is None:
            raise ValueError('Insufficient raw ratings for score supervision')
        system = (
            'Assess the four skill dimensions over the complete laparoscopic training video.\n\n'
            'Assess each dimension independently from 1.0 to 5.0 in increments of 0.5 using the task-specific anchors below. '
            'Use intermediate values for performance between adjacent anchors.\n\n'
            + task_scoring_prompt(task) + '\n\n'
            'Output format:\nReturn only JSON, following this example:\n'
            + SCORE_OUTPUT_EXAMPLE
        )
        request = 'Assess the four skill dimensions.'
        answer = json.dumps(
            {name: sample["aspect_scores"][aspect]
             for name, aspect in zip(ASPECT_NAMES, ASPECTS)},
            ensure_ascii=False)
    elif supervision == 'error':
        if error_category not in error_defs:
            raise ValueError(f'Invalid error category for {task}: {error_category}')
        system = (
            'Determine whether the specified error occurs in the laparoscopic training video.\n\n'
            'Return only this JSON object and nothing else: '
            '{"error_present": "Yes" or "No"}.'
        )
        request = (f'Error: {error_category}\nDefinition: {error_defs[error_category]}\n'
                   'Does this error occur?')
        present = error_category in sample['errors']
        answer = json.dumps(
            {'error_present': 'Yes' if present else 'No'},
            ensure_ascii=False)
    else:
        raise ValueError(f'Unknown supervision task: {supervision}')
    return {
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': f'<video>\nTask: {name}\n{request}'},
            {'role': 'assistant', 'content': answer},
        ],
        'aspect_scores': sample['aspect_scores'], 'raw_ratings': sample['raw_ratings'],
        'official_normalized_grs': sample['grs'],
        'score_target': 'median_available_raters_raw_min2' if sample['aspect_scores'] is not None else None,
        'supervision': 'aspects_only' if supervision == 'score' else 'errors_only',
        'task': task, 'recording_id': sample['recording_id'], 'split': sample['split'],
        'errors': sample['errors'],
        'videos': [sample['video']],
        'chat_template_kwargs': {'fps': frame_budget.resolve_fps(duration, fps), 'use_audio_in_video': False},
        'sample_id': f"lasana_{task}_{sample['recording_id']}_{supervision}"
                     + (f'_{error_category}' if supervision == 'error' else ''),
        **({'error_category': error_category, 'error_present': present}
           if supervision == 'error' else {}),
    }


def select_error_questions(rows, seed=ERROR_SAMPLING_SEED):
    selected = []
    for row in rows:
        if row['split'] == 'test':
            selected.append(row | {'error_sampling': 'all_task_categories'})
        elif row['split'] == 'train':
            selected.append(row | {'error_sampling': 'all_positive_all_negative'})
        else:
            raise ValueError(f"Unexpected error split: {row['split']}")
    return sorted(selected, key=lambda r: (r['split'], r['sample_id']))


def main(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root', '--dataset-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out-dir', type=Path, default=REPO_ROOT / 'data' / 'exports' / 'original' / 'skills')
    parser.add_argument('--fps', type=float)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--balance', action='store_true',
                        help='duplicate the minority error_present class up to the majority '
                             'count per (task, error_category); opt-in experiment, the pipeline '
                             'uses loss_scale weights instead; '
                             'default: no resampling')
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0)):
        parser.error('workers and fps must be positive and finite')
    samples = collect_samples(args.root.resolve())
    paths = sorted({s['video'] for records in samples.values() for s in records})
    existing = [p for p in paths if Path(p).is_file()]
    print(f'Probing {len(existing)} left-camera videos; validation excluded', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        durations = dict(zip(existing, pool.map(frame_budget.probe_duration, existing)))
    outputs = {'score': [], 'error': []}
    for split, records in samples.items():
        counts = Counter()
        for sample in records:
            duration = durations.get(sample['video'], 0)
            if sample['video'] not in durations:
                counts['missing_video'] += 1
            elif not math.isfinite(duration) or duration <= 0:
                counts['invalid_video_duration'] += 1
            else:
                outputs['error'].extend(build_record(sample, duration, 'error', args.fps,
                                                     error_category=tag)
                                        for tag in TASKS[sample['task']][1])
                counts['error'] += 1
                if sample['aspect_scores'] is not None:
                    outputs['score'].append(build_record(sample, duration, 'score', args.fps))
                    counts['score'] += 1
                else:
                    counts['insufficient_ratings'] += 1
        if not counts['error']:
            raise ValueError(f'No usable samples: {split}')
        print(f'{split}: {dict(counts)}', flush=True)
    outputs['error'] = select_error_questions(outputs['error'])
    print(f"Error answers: {dict(Counter((r['split'], r['messages'][-1]['content']) for r in outputs['error']))}", flush=True)
    ids = [r['sample_id'] for rows in outputs.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task sample IDs')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def dump(path, selected):
        tmp = path.with_suffix('.jsonl.tmp')
        with tmp.open('w', encoding='utf-8') as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        tmp.replace(path)
        print(f'{path.name}: {len(selected)}', flush=True)

    def upsample_error(rows, seed):


        groups = {}
        for r in rows:
            groups.setdefault((r['task'], r['error_category']),
                              {True: [], False: []})[r['error_present']].append(r)
        out = []
        for key, classes in sorted(groups.items()):
            ordered = {k: sorted(v, key=lambda r: r['sample_id']) for k, v in classes.items()}
            target = max(len(ordered[True]), len(ordered[False]))
            for present in (True, False):
                items = ordered[present]
                if not items:
                    continue
                rng = random.Random(f'{seed}:{key[0]}:{key[1]}:{present}')
                grown = list(items)
                while len(grown) < target:
                    need = target - len(grown)
                    grown.extend(items if need >= len(items) else rng.sample(items, need))
                out.extend(grown)
        return sorted(out, key=lambda r: (r['split'], r['sample_id']))

    for supervision, rows in outputs.items():
        for split in ('train', 'test'):
            selected = [row for row in rows if row['split'] == split]
            if supervision == 'error' and split == 'train':
                if getattr(args, 'balance', False):
                    selected = upsample_error(selected, ERROR_SAMPLING_SEED)
                else:


                    cell_of = lambda r: (r['task'], r['error_category'])
                    class_of = lambda r: r['error_present']
                    weights = loss_scale.balanced_weights(selected, cell_of, class_of)
                    for row in selected:
                        loss_scale.attach(row, weights[(cell_of(row), class_of(row))])
                    print('    loss_scale: ' + loss_scale.summarise(
                        selected, cell_of, class_of, weights), flush=True)
            dump(args.out_dir / f'lasana_{supervision}_{split}.jsonl', selected)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
