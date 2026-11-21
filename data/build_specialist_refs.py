#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'data' / 'exports' / 'original'
DEFAULT_OUT = ROOT / 'benchmark' / 'specialist_refs'
TASKS = {
    'mtlaqa': ('sports', ['mtlaqa_test.jsonl']),
    'fitnessaqa_kp': ('fitness', ['fitnessaqa_kp_test.jsonl']),
    'kimore_cts': ('rehabilitation', ['kimore_cts_test.jsonl']),
    'lasana_error': ('skills', ['lasana_error_test.jsonl']),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for task, (domain, files) in TASKS.items():
        paths = [args.source / domain / name for name in files]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f'{task}: missing {missing}')
        target = args.out / f'{task}_test.jsonl'
        rows = 0
        with target.open('wb') as out:
            for path in paths:
                out.write(path.read_bytes())
                rows += sum(1 for line in path.open('rb') if line.strip())
        manifest[task] = {'rows': rows}
        print(f'{task:16s} {rows:6d} rows <- {", ".join(p.name for p in paths)}')
    (args.out / 'test_manifest.json').write_text(
        json.dumps({'tasks': manifest}, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
