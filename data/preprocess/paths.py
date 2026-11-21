from __future__ import annotations

from pathlib import Path


def workspace() -> Path:
    return Path(__file__).resolve().parents[2]
