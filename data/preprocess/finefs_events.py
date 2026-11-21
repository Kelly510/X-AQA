import hashlib
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def manifest():
    path = Path(__file__).resolve().parents[1] / "annotations" / "finefs_events.json"
    return json.loads(path.read_text())


def context(program_id, competition):
    record = manifest().get(str(program_id))
    if record is None or record["raw_competition"] != competition:
        raise ValueError(f"program {program_id}: competition does not match audited release")
    return {key: record[key] for key in ("event", "season", "segment")}


def verify_annotation(path):
    record = manifest().get(path.stem)
    if record is None or hashlib.sha256(path.read_bytes()).hexdigest() != record["annotation_sha256"]:
        raise ValueError(f"{path}: annotation differs from audited release; re-audit event identity")


EVENT_NAMES = {
    "4cc": "Four Continents Championships", "wc": "World Championships",
    "ec": "European Championships", "og": "Olympic Winter Games",
    "wtt": "World Team Trophy", "gpf": "Grand Prix Final",
    "sa": "Skate America", "sc": "Skate Canada", "idf": "Internationaux de France",
    "rc": "Rostelecom Cup", "coc": "Cup of China", "nhk": "NHK Trophy",
}


def describe(record):
    family, year, gender = record["event"].split("_")
    discipline = "Men's singles" if gender == "men" else "Women's singles"
    segment = "Short program" if record["segment"] == "sp" else "Free skating"
    return f"{year} {EVENT_NAMES[family]}; {discipline}; {segment}; season {record['season']}"
