from dataclasses import asdict, dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Dataset:
    domain: str
    name: str
    prefix: str
    annotation: str
    benchmark_type: str

    def metadata(self) -> dict:
        return asdict(self)


DATASETS = (
    Dataset('Sports', 'AQA-7', 'aqa7', 'S', 'ID'),
    Dataset('Sports', 'MTL-AQA', 'mtlaqa', 'S,T', 'ID'),
    Dataset('Sports', 'RG', 'rhythmicgymnastics', 'S', 'ID'),
    Dataset('Sports', 'FineFS', 'finefs', 'S', 'ID'),
    Dataset('Sports', 'MultiSenseBadminton', 'multisensebadminton', 'S,C,T', 'ID'),
    Dataset('Sports', 'Ego-Exo4D', 'egoexo4d_sports', 'C,T', 'ID'),
    Dataset('Sports', 'T3Set', 't3set', 'C,T', 'OOD'),
    Dataset('Sports', 'CricketVision', 'cricketvision', 'S', 'ID'),
    Dataset('Sports', 'ExpertAF', 'expertaf_sports', 'T', 'ID'),
    Dataset('Sports', 'CoT-AFA', 'cotafa_martial', 'C,T', 'ID'),
    Dataset('Fitness', 'Fitness-AQA', 'fitnessaqa', 'C', 'ID'),
    Dataset('Fitness', 'FLAG3D', 'flag3d', 'C', 'ID'),
    Dataset('Fitness', 'EgoExo-Fitness', 'egoexo_fitness', 'S,C,T', 'ID'),
    Dataset('Fitness', 'QEVD', 'qevd', 'T', 'ID'),
    Dataset('Fitness', 'CoT-AFA', 'cotafa_fitness', 'C,T', 'ID'),
    Dataset('Fitness', 'Fitness-7', 'fitness7', 'C', 'OOD'),
    Dataset('Rehabilitation', 'KERAAL', 'keraal', 'C', 'ID'),
    Dataset('Rehabilitation', 'REHAB24-6', 'rehab246', 'C', 'ID'),
    Dataset('Rehabilitation', 'ExeCheck', 'execheck', 'C', 'ID'),
    Dataset('Rehabilitation', 'LLM-FMS', 'llmfms', 'S', 'ID'),
    Dataset('Rehabilitation', 'KIMORE', 'kimore', 'S', 'OOD'),
    Dataset('Rehabilitation', 'UpperLimb', 'upperlimb', 'C', 'ID'),
    Dataset('General Skills', 'JIGSAWS', 'jigsaws', 'S', 'OOD'),
    Dataset('General Skills', 'CaptainCook4D', 'captaincook4d', 'C,T', 'ID'),
    Dataset('General Skills', 'CPR-Coach', 'cprcoach', 'C', 'ID'),
    Dataset('General Skills', 'Ego-Exo4D', 'egoexo4d_skill', 'C,T', 'ID'),
    Dataset('General Skills', 'LASANA', 'lasana', 'S,C', 'ID'),
    Dataset('General Skills', 'AIxSuture', 'aixsuture', 'S', 'ID'),
)


def dataset_for_file(name: str) -> Dataset | None:
    path = Path(name)
    if path.suffix != ".jsonl":
        return None
    for dataset in DATASETS:
        prefix = dataset.prefix + "_"
        if path.stem.startswith(prefix):
            tail = path.stem[len(prefix):]
            if dataset.prefix == 'cprcoach' and tail in {'error', 'no_error'}:
                return dataset
            if re.search(r"(?:^|_)(?:train|test|val|validation|all)(?:_|$)", tail):
                return dataset
    return None


ORIGINAL_DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'exports' / 'original'
DOMAIN_FOLDERS = {'Sports': 'sports', 'Fitness': 'fitness',
                  'Rehabilitation': 'rehabilitation', 'General Skills': 'skills'}


def dataset_for_artifact(name: str) -> Dataset | None:
    stem = Path(name).stem
    return next((d for d in DATASETS if stem == d.prefix or stem.startswith(d.prefix + '_')), None)


def discover_data_files(data_dir: Path) -> list[Path]:
    root = data_dir / 'original' if (data_dir / 'original').is_dir() else data_dir
    folders = [root, *(root / domain for domain in DOMAIN_FOLDERS.values())]
    return sorted(p for folder in folders for p in folder.glob('*.jsonl')
                  if dataset_for_file(p.name) is not None)


def resolve_data_file(data_dir: Path, name: str) -> Path:
    direct = data_dir / name
    if direct.is_file():
        return direct
    matches = [p for p in discover_data_files(data_dir) if p.name == Path(name).name]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous dataset name: {name}')
    if matches:
        return matches[0]
    return direct
