import argparse
from collections import Counter
import os
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_catalog import DATASETS, dataset_for_file, discover_data_files

ROOT = Path(__file__).resolve().parents[1]


ANNOTATION_ROOT = 'data/exports/original'
TRAIN_ROOTS = ('data/exports/original',)

TRAINING_ONLY_PREFIXES = {'keraal'}

TABLE_COUNTS = (
    (803, 303), (1059, 353), (800, 200), (7469, 2096), (1663, 225),
    (1980, 550), (419, 143), (6439, 1755), (3655, 1272), (1599, 350),
    (9669, 1706), (37476, 9417), (6068, 2574), (258675, 1993),
    (2340, 670), (278, 70), (540, 491), (727, 345),
    (600, 100), (1451, 361), (0, 348), (411, 80),
    (281, 58), (2820, 1397), (648, 4118), (3725, 1035), (4427, 958),
    (2201, 460),
)


def validate_training(specs, root=ROOT):
    seen = set()
    if not specs:
        raise ValueError('Empty training mixture')
    allowed = [(root / r).resolve() for r in TRAIN_ROOTS]
    for spec in specs:
        filename, sep, cap = spec.rpartition('#')
        path = Path(filename if sep else spec).resolve()
        dataset = dataset_for_file(path.name)
        if not dataset or not path.name.endswith('_train.jsonl'):
            raise ValueError(f'Only catalogued train exports are allowed: {spec}')
        if dataset.benchmark_type == 'OOD':
            raise ValueError(f'OOD training source is forbidden: {dataset.name}')
        if not any(path.is_relative_to(a) for a in allowed) or not path.is_file():
            raise ValueError('Not an existing training export under '
                             f'{"/".join(TRAIN_ROOTS)}: {path}')
        if path in seen:
            raise ValueError(f'Duplicate training source: {path}')
        seen.add(path)
        if sep and (not cap.isdigit() or int(cap) <= 0):
            raise ValueError(f'Invalid row cap: {spec}')


def build(root=ROOT):
    files = discover_data_files(root / ANNOTATION_ROOT)
    entries, media = [], {'training': set(), 'benchmark': set()}
    hashes = {'training': set(), 'benchmark': set()}
    for dataset, expected in zip(DATASETS, TABLE_COUNTS, strict=True):
        entry = {**dataset.metadata(), 'table_train': expected[0], 'table_test': expected[1],
                 'files': [], 'train': 0, 'test': 0, 'validation': 0}
        for path in files:
            if dataset_for_file(path.name) != dataset:
                continue
            suffix = path.stem.rsplit('_', 1)[-1]
            split = 'validation' if suffix in ('val', 'validation') else suffix
            if split not in ('train', 'test', 'validation'):
                raise ValueError(f'Unclassified data export: {path}')
            selected = ('training' if split == 'train' and dataset.benchmark_type != 'OOD'
                        else 'benchmark' if split == 'test' and dataset.benchmark_type in ('ID', 'OOD')
                        else None)
            count = 0
            with path.open('rb') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    count += 1
                    if selected:
                        for field in ('videos', 'images'):
                            media[selected].update(os.path.abspath(p) for p in row.get(field, []))
                        if row.get('content_hash'):
                            hashes[selected].add(row['content_hash'])
            entry[split] += count
            entry['files'].append({'path': path.relative_to(root).as_posix(), 'split': split,
                                   'rows': count, 'use': selected})
        entry['training_rows'] = entry['train'] if dataset.benchmark_type != 'OOD' else 0
        entry['benchmark_rows'] = entry['test'] if dataset.benchmark_type in ('ID', 'OOD') else 0
        entry['counts_match_table'] = (entry['train'], entry['test']) == expected
        entries.append(entry)
    overlap = sorted(media['training'] & media['benchmark'])
    hash_overlap = sorted(hashes['training'] & hashes['benchmark'])
    totals = Counter()
    for e in entries:
        for key in ('train', 'test', 'validation', 'training_rows', 'benchmark_rows'):
            totals[key] += e[key]
        if e['benchmark_type'] in ('ID', 'OOD'):
            totals[e['benchmark_type'] + '_available_test'] += e['test']


    sample = root / 'benchmark/v3/benchmark_sample.json'
    sampled = (json.loads(sample.read_text())['parts'] if sample.is_file() else None)
    manifest = {'protocol': 'X-AQA-Bench v3',
                'training_rule': 'ID and N/A train only; never OOD train or any test',
                'benchmark_rule': 'ID test partitions (sampled) plus every partition of the four OOD sources',
                'count_unit': 'JSONL rows (tasks, views, windows or feedback may share media)',
                'totals': dict(totals), 'benchmark_sampled': sampled, 'datasets': entries,
                'leakage_audit': {'shared_media_paths': overlap, 'shared_available_content_hashes': hash_overlap,
                                 'scope': 'Exact paths and existing content_hash fields only; not a global subject or near-duplicate audit'}}
    if overlap or hash_overlap:
        raise ValueError(f'Training/benchmark media overlap: {overlap[:5]}, hashes: {hash_overlap[:5]}')
    return manifest


def write(manifest, root=ROOT):
    folder = root / 'benchmark'
    folder.mkdir(parents=True, exist_ok=True)
    lists = folder / 'manifests'
    lists.mkdir(parents=True, exist_ok=True)
    selections = {'train': [], 'test_all': [], 'benchmark_ID': [], 'benchmark_OOD': [],
                  'excluded_ood_train': []}
    for e in manifest['datasets']:
        for f in e['files']:
            path = f['path']
            if f['use'] == 'training':
                selections['train'].append(path)
            if f['split'] == 'test' and e['prefix'] not in TRAINING_ONLY_PREFIXES:
                selections['test_all'].append(path)
            if f['use'] == 'benchmark' and e['prefix'] not in TRAINING_ONLY_PREFIXES:
                selections['benchmark_' + e['benchmark_type']].append(path)
            if f['split'] == 'train' and e['benchmark_type'] == 'OOD':
                selections['excluded_ood_train'].append(path)
    for name, paths in selections.items():
        (lists / (name + '.txt')).write_text('\n'.join(paths) + '\n')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--validate-training', nargs='+')
    args = parser.parse_args()
    if args.validate_training:
        validate_training(args.validate_training)
        print(f'Training policy OK: {len(args.validate_training)} sources; no OOD/test exports')
    else:
        manifest = build()
        write(manifest)
        mismatches = [e['name'] for e in manifest['datasets'] if not e['counts_match_table']]
        print(json.dumps({'totals': manifest['totals'],
                          'benchmark_sampled': manifest['benchmark_sampled'],
                          'count_mismatches': mismatches}, ensure_ascii=False))


if __name__ == '__main__':
    main()
