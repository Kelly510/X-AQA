#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from build_data_protocol import main as protocol_main
from build_sampling_lists import main as lists_main

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    protocol_main()
    lists_main([])
    print('X-AQA-40K build complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
