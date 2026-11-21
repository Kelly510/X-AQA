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
from functools import lru_cache
import json
import math
import random
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

from . import frame_budget
from . import loss_scale

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = workspace() / 'datasets' / 'CaptainCook4D'
ERROR_CATEGORIES = (
    'Preparation Error', 'Measurement Error', 'Order Error', 'Timing Error',
    'Technique Error', 'Temperature Error', 'Missing Step', 'Other',
)
SYSTEM = (
    'Assess the execution of the specified cooking step in the video. '
    'The input video shows a segment of a single cooking step.\n\n'
    'Assessment criteria:\n'
    'Use the intended step as the reference for ingredients, quantities, utensils, '
    'preparation, technique, timing and temperature. Compare the visible actions '
    'and outcomes with these requirements to identify deviations. '
    'Alternative step orders that satisfy the required dependencies are valid. '
    'Judge order and omissions only when supported by the available context; '
    'absence from this clip alone does not establish that a step was skipped.\n\n'
    'Error categories:\n'
    'Classify each identified error using the definitions below. '
    'A step may contain multiple errors.\n'
    '- Preparation Error: Use of soiled or wrong ingredients, or different tools.\n'
    '- Measurement Error: Use of incorrectly measured amounts of ingredients.\n'
    '- Order Error: Execution of steps in a sequence that violates the required dependencies.\n'
    '- Timing Error: Execution lasting less or more time than prescribed.\n'
    '- Technique Error: Incorrect execution of the required action leading to an unintended outcome.\n'
    '- Temperature Error: Microwave or stove power or heat settings above or below those prescribed.\n'
    '- Missing Step: Omission of a required step.\n'
    '- Other: An identified error outside the categories above.\n\n'
    'Error descriptions:\n'
    'Describe each error briefly and concretely, identifying the action, ingredient, '
    'utensil, quantity, setting or outcome that differs from the intended step. '
    'State the deviation directly; include the expected alternative when needed to '
    'make the error clear. Avoid generic assessments, speculative causes, repeated '
    'category names and additional coaching.\n\n'
    'Output format:\n'
    'Return only a valid JSON object. Each key must be an error category name copied '
    'exactly from the list above, and each value must be a nonempty description string. '
    'Include only categories with identified errors. Return {} when there are no errors.'
)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def description_merges():
    path = Path(__file__).resolve().parents[1] / 'annotations' / 'captaincook4d_description_merges.json'
    rows = read_json(path)['merges']
    return {(r['recording_id'], r['step_index'], r['category']): r for r in rows}


def merge_descriptions(recording_id, index, tag, descriptions):
    if len(descriptions) == 1:
        return descriptions[0]
    entry = description_merges().get((recording_id, index, tag))
    if entry is None or entry['original_descriptions'] != descriptions:
        raise ValueError(f'Unreviewed repeated-category descriptions: {recording_id}/{index}/{tag}')
    return entry['merged_description']


def validate_splits(splits, annotations, protocol):
    seen, groups = set(), set()
    for split in ('train', 'val', 'test'):
        ids = splits[split]
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f'Empty, duplicate or overlapping recording split: {split}')
        seen.update(ids)
        if protocol in ('person', 'environment'):
            field = 'person_id' if protocol == 'person' else 'environment'
            current = {annotations[r][field] for r in ids}
            if groups & current:
                raise ValueError(f'Overlapping {field}: {split}')
            groups.update(current)
    if seen != set(annotations):
        raise ValueError('Split does not cover the annotation recordings exactly')


def aligned_steps(recording, errors):
    steps, detailed = recording['steps'], errors['step_annotations']
    if len(steps) != len(detailed):
        raise ValueError('Step annotation lengths disagree')
    for index, (step, detail) in enumerate(zip(steps, detailed)):
        for key in ('step_id', 'start_time', 'end_time', 'description'):
            if step[key] != detail[key]:
                raise ValueError(f'Step annotations disagree: {key}')
        if type(step['has_errors']) is not bool or step['has_errors'] != bool(detail.get('errors')):
            raise ValueError('Binary and detailed error annotations disagree')
        yield index, step, detail.get('errors', [])


def interval_reason(start, end, duration):
    if not all(math.isfinite(x) for x in (start, end)) or start < 0 or end <= start:
        return 'invalid_or_missing_interval'
    if not math.isfinite(duration) or duration <= 0:
        return 'invalid_video_duration'
    if end > duration:
        return 'interval_exceeds_video'
    return None


def build_record(video, recording, index, step, errors, fps=None):
    start, end = step['start_time'], step['end_time']
    instruction = 'Assess this step and return the error descriptions by category.'
    if step['has_errors'] != bool(errors):
        raise ValueError('Binary and detailed error annotations disagree')
    grouped = {}
    for error in errors:
        tag, description = error['tag'], error['description']
        if tag not in ERROR_CATEGORIES or not isinstance(description, str) or not description.strip():
            raise ValueError(f'Invalid error annotation: {error}')
        grouped.setdefault(tag, []).append(description)
    target = json.dumps({tag: merge_descriptions(recording['recording_id'], index, tag, descriptions)
                         for tag, descriptions in grouped.items()},
                        ensure_ascii=False)
    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': f"<video>\nRecipe: {recording['activity_name']}\n"
             f"Intended step: {step['description']}\n{instruction}"},
            {'role': 'assistant', 'content': target},
        ],
        'videos': [str(video)],
        'chat_template_kwargs': {'video_start': start, 'video_end': end,
                                'fps': frame_budget.resolve_fps(end - start, fps),
                                'use_audio_in_video': False},
        'sample_id': f"captaincook4d_{recording['recording_id']}_{index:03d}",
    }


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root', '--dataset-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out-dir', type=Path, default=REPO_ROOT / "data" / "exports" / "original" / "skills")
    parser.add_argument('--fps', type=float)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--balance', action='store_true',
                        help='duplicate the minority class (error/normal) up to the majority '
                             'count; opt-in experiment without loss weights; default uses '
                             'loss_scale to balance no-error and individual error categories')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error('--fps must be finite and positive')
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.root.resolve()
    ann = root / 'annotations'
    recordings = read_json(ann / 'annotation_json/complete_step_annotations.json')
    error_list = read_json(ann / 'annotation_json/error_annotations.json')
    details = {r['recording_id']: r for r in error_list}
    if len(details) != len(error_list) or set(details) != set(recordings):
        raise ValueError('Duplicate or inconsistent annotation recording IDs')
    splits = read_json(ann / 'data_splits/person_data_split_combined.json')
    validate_splits(splits, recordings, 'person')

    aligned = {rid: list(aligned_steps(rec, details[rid])) for rid, rec in recordings.items()}
    selected_ids = splits['train'] + splits['test']
    videos = {r: root / 'gopro_360p' / f'{r}_360p.mp4' for r in selected_ids}
    existing = sorted(p for p in videos.values() if p.is_file())
    print(f'Probing {len(existing)} videos; official val excluded', flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        durations = dict(zip(existing, pool.map(frame_budget.probe_duration, existing)))
    outputs = {}
    for split in ('train', 'test'):
        rows, counts = [], Counter()
        for rid in splits[split]:
            rec, video = recordings[rid], videos[rid]
            for index, step, errors in aligned[rid]:
                reason = ('missing_video' if video not in durations else interval_reason(
                    step['start_time'], step['end_time'], durations[video]))
                if reason:
                    counts[reason] += 1
                    continue
                rows.append(build_record(video, rec, index, step, errors, args.fps))
                counts['error' if step['has_errors'] else 'normal'] += 1
        if not rows:
            raise ValueError(f'No usable samples: {split}')
        outputs[split] = rows
        print(f'{split}: {len(rows)} samples; {dict(counts)}', flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.out_dir, outputs, args.balance, args.seed)
    return 0


def with_balanced_loss_scales(rows):
    if not rows:
        return []
    labels = []
    for row in rows:
        target = json.loads(row['messages'][-1]['content'])
        if not isinstance(target, dict) or set(target) - set(ERROR_CATEGORIES):
            raise ValueError(f'Invalid CaptainCook4D target: {target}')
        labels.append(tuple(sorted(target)) or ('no_error',))
    counts = Counter(label for group in labels for label in group)
    budget = len(rows) / len(counts)
    contribution = {label: budget / count for label, count in counts.items()}
    weighted = []
    for row, group in zip(rows, labels):
        record = {**row, 'messages': [dict(message) for message in row['messages']]}
        record['messages'][-1].pop('loss_scale', None)
        weighted.append(loss_scale.attach(record, math.fsum(contribution[label] for label in group)))
    print('    loss_scale class counts: ' + json.dumps(dict(sorted(counts.items()))), flush=True)
    print(f'    loss_scale: {len(counts)} classes, {budget:.6f} contribution budget per class; '
          f'row weights sum to {len(rows)} before clipping; '
          f'exported sum {sum(r["messages"][-1].get("loss_scale", 1.) for r in weighted):.6f}, '
          f'cap {loss_scale.MAX_LOSS_SCALE:g}', flush=True)
    return weighted


def write_outputs(out_dir, outputs, upsample=False, seed=42):


    normal, errors = [], []
    for row in outputs['train']:
        (errors if json.loads(row['messages'][-1]['content']) else normal).append(row)
    normal.sort(key=lambda row: row['sample_id'])
    errors.sort(key=lambda row: row['sample_id'])
    if upsample and errors and normal:
        minority, majority = (errors, normal) if len(errors) <= len(normal) else (normal, errors)
        rng = random.Random(seed)
        grown = list(minority)
        while len(grown) < len(majority):
            need = len(majority) - len(grown)
            grown.extend(minority if need >= len(minority) else rng.sample(minority, need))
        train = sorted(majority + grown, key=lambda row: row['sample_id'])
    else:
        train = sorted(errors + normal, key=lambda row: row['sample_id'])
    if not upsample:
        train = with_balanced_loss_scales(train)
    write_jsonl(out_dir / 'captaincook4d_train.jsonl', train)
    write_jsonl(out_dir / 'captaincook4d_test.jsonl', outputs['test'])
    for suffix in ('errors', 'no_errors'):
        (out_dir / f'captaincook4d_train_{suffix}.jsonl').unlink(missing_ok=True)
    print(f'exported train: {len(train)} (errors {len(errors)}, normal {len(normal)}, '
          f'{"minority upsampled" if upsample else "natural rows with class loss weights"})', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
