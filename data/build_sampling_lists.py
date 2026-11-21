#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data'))
from dataset_catalog import dataset_for_file

DOMAINS = ('sports', 'fitness', 'skills', 'rehabilitation')
ARM = {'skills': 'skill'}


def rows_of(path: Path) -> int:
    with path.open('rb') as fh:
        return sum(1 for line in fh if line.strip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source', type=Path, default=ROOT / 'data' / 'exports' / 'original')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'data' / 'configs' / 'cap2k')
    args = parser.parse_args(argv)

    by_domain = {d: [] for d in DOMAINS}
    skipped = []
    for domain in DOMAINS:
        for path in sorted((args.source / domain).glob('*_train.jsonl')):
            dataset = dataset_for_file(path.name)
            if dataset is None:
                skipped.append((path.name, 'not catalogued'))
                continue
            if dataset.benchmark_type == 'OOD':
                skipped.append((path.name, 'OOD'))
                continue
            rows = rows_of(path)
            try:
                stored = str(path.relative_to(ROOT))
            except ValueError:
                stored = str(path)
            by_domain[domain].append({'path': stored, 'rows': rows,
                                      'cap': min(rows, 2000)})

    args.out.mkdir(parents=True, exist_ok=True)
    full = [e for d in DOMAINS for e in by_domain[d]]
    total = sum(e['cap'] for e in full)
    (args.out / 'full.txt').write_text(''.join(f"{e['path']}#{e['cap']}\n" for e in full))
    manifest = {'rule': 'per-source #N = min(rows, 2000), one epoch',
                'data_root': str(args.source.relative_to(ROOT)) if args.source.is_relative_to(ROOT)
                else str(args.source), 'total_draws': total, 'sources': len(full), 'runs': {}}
    print(f"full: {len(full)} sources, {total:,} draws")
    manifest['runs']['full'] = {'sources': len(full), 'draws': total}
    for domain in DOMAINS:
        arm = ARM.get(domain, domain)
        members = by_domain[domain]
        draws = sum(e['cap'] for e in members)
        (args.out / f'{arm}_plain.txt').write_text(
            ''.join(f"{e['path']}#{e['cap']}\n" for e in members))
        manifest['runs'][f'{arm}_plain'] = {'sources': len(members), 'draws': draws,
                                            'per_source': members}
        print(f"{arm}_plain: {len(members)} sources, {draws:,} draws "
              f"({draws // 32} steps)")
    for name, why in skipped:
        print(f'  skipped {name} ({why})')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
