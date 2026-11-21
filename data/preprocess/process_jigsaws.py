#!/usr/bin/env python3
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"


import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.preprocess.paths import workspace

from . import frame_budget


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = workspace() / "datasets" / "JIGSAWS"

TASKS = ("Suturing", "Knot_Tying", "Needle_Passing")


DEFAULT_WINDOW_SEC = 32.0
FPS = frame_budget.sampling_fps(DEFAULT_WINDOW_SEC)


ELEMENTS = (
    (
        "Respect for tissue",
        (
            "Frequently used unnecessary force on tissue",
            "Careful tissue handling but occasionally caused inadvertent damage",
            "Consistent appropriate tissue handling",
        ),
    ),
    (
        "Suture/needle handling",
        (
            "Awkward and unsure with repeated entanglement and poor knot tying",
            "Majority of knots placed correctly with appropriate tension",
            "Excellent suture control",
        ),
    ),
    (
        "Time and motion",
        (
            "Made unnecessary moves",
            "Efficient time/motion but some unnecessary moves",
            "Clear economy of movement and maximum efficiency",
        ),
    ),
    (
        "Flow of operation",
        (
            "Frequently interrupted flow to discuss the next move",
            "Demonstrated some forward planning and reasonable procedure progression",
            "Obviously planned course of operation with efficient transitions between moves",
        ),
    ),
    (
        "Overall performance",
        (
            "Very poor",
            "Competent",
            "Clearly Superior",
        ),
    ),
    (
        "Quality of final product",
        (
            "Very poor",
            "Competent",
            "Clearly Superior",
        ),
    ),
)


TASK_TEXT = {
    "Suturing": (
        "Using the robotic instruments, the subject picks up the needle, passes it "
        "through the artificial tissue from a marked entry point on one side of the "
        "incision to the corresponding exit point on the other side, extracts the "
        "needle, transfers it to the right hand, and repeats the pass."
    ),
    "Knot_Tying": (
        "Using the robotic instruments, the subject ties two sutures, one after the "
        "other, around a flexible tube fixed at both ends. Each completed knot should "
        "be taut and secure around the tube."
    ),
    "Needle_Passing": (
        "Using the robotic instruments, the subject picks up a threaded needle and "
        "passes it through the small metal hoops mounted on rubber stands above the "
        "bench-top model."
    ),
}


SYSTEM_TEMPLATE = """You are a surgical skills examiner. Evaluate the \
robot-assisted surgical training trial provided by the user. The video may be an \
excerpt of a longer trial; score the trial it comes from.

Score the performance using the six named dimensions of the modified OSATS global \
rating score. Apply each dimension to the task identified in the user message and \
judge only the performance visible in the video.

Use a five-point Likert scale. The rubric gives anchor descriptions for the lowest \
anchor (score value 1), the middle anchor (score value 3), and the highest anchor \
(score value 5). Use score values 2 or 4 when the performance falls between adjacent \
anchors. The numbers in the rubric are score values, not item numbering.

{rubric}

Return only JSON, following this example:
{output_example}"""


USER_TEMPLATE = """Assess the trial in the following video.

Task: {task_name}
Expected procedure: {task_description}

<video>"""


def output_example() -> str:
    return "{" + ", ".join(f'"{name}": <integer from 1 to 5>' for name, _ in ELEMENTS) + "}"


def system_prompt() -> str:
    rubric = []
    for name, anchors in ELEMENTS:
        rubric.append(
            f"{name}:\n"
            f"- Lowest anchor (score value 1): {anchors[0]}\n"
            f"- Middle anchor (score value 3): {anchors[1]}\n"
            f"- Highest anchor (score value 5): {anchors[2]}"
        )
    return SYSTEM_TEMPLATE.format(
        rubric="\n\n".join(rubric),
        output_example=output_example(),
    )


def user_prompt(task: str) -> str:
    return USER_TEMPLATE.format(
        task_name=task.replace("_", " "),
        task_description=TASK_TEXT[task],
    )


def answer(scores: list[int]) -> str:
    return json.dumps({name: score for (name, _), score in zip(ELEMENTS, scores)},
                      ensure_ascii=False)


def read_meta(root: Path, task: str) -> list[dict]:
    trials = []
    path = root / task / f"meta_file_{task}.txt"

    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue

        if len(parts) != 9:
            print(
                f"{path.name}: unexpected column count in {line!r}",
                file=sys.stderr,
            )
            continue

        name, level, grs = parts[0], parts[1], int(parts[2])
        scores = [int(value) for value in parts[3:9]]

        if sum(scores) != grs:
            print(
                f"{name}: GRS {grs} != sum {sum(scores)}, dropped",
                file=sys.stderr,
            )
            continue

        trials.append(
            {
                "trial": name,
                "task": task,
                "self_level": level,
                "grs": grs,
                "scores": scores,
                "subject": name.split("_")[-1][0],
                "trial_index": int(name.split("_")[-1][1:]),
            }
        )

    return trials


def folds(trials: list[dict], protocol: str) -> dict:
    key = "trial_index" if protocol == "loso" else "subject"
    grouped = defaultdict(list)

    for trial in trials:
        grouped[trial[key]].append(trial["trial"])

    return {
        fold: sorted(names)
        for fold, names in sorted(grouped.items())
    }


def official_folds(root: Path, setup: str) -> dict:
    base = (
        root
        / "Experimental_setup"
        / "Suturing"
        / "unBalanced"
        / "SkillDetection"
        / setup
    )

    if not base.is_dir():
        return {}

    out = {}
    for fold_dir in sorted(base.iterdir()):
        names = set()
        for test in fold_dir.glob("itr_*/Test.txt"):
            for line in test.read_text().splitlines():
                if line.strip():
                    names.add("_".join(line.split()[0].split("_")[:-2]))
        out[fold_dir.name] = sorted(names)

    return out


def check_official(root: Path, trials: list[dict]) -> bool:
    suturing = [trial for trial in trials if trial["task"] == "Suturing"]
    if not suturing:
        return True

    ok = True
    fold_specs = (
        ("loso", "SuperTrialOut", lambda fold: f"{fold}_Out"),
        ("louo", "UserOut", None),
    )

    for protocol, setup, label in fold_specs:
        released = official_folds(root, setup)
        if not released:
            print(f"no released {setup} folds to check against", file=sys.stderr)
            continue

        generated = folds(suturing, protocol)

        if label is not None:
            generated = {
                label(fold): names
                for fold, names in generated.items()
            }
        else:

            order = sorted(generated)
            generated = {
                f"{index}_Out": generated[subject]
                for index, subject in enumerate(order, start=1)
            }

        if generated == released:
            print(
                f"official check: Suturing {protocol.upper()} == {setup} "
                f"({len(released)} folds)"
            )
        else:
            ok = False
            print(f"official check FAILED for {setup}", file=sys.stderr)
            for fold in sorted(set(generated) | set(released)):
                if generated.get(fold) != released.get(fold):
                    print(
                        f"  {fold}: generated={generated.get(fold)} "
                        f"released={released.get(fold)}",
                        file=sys.stderr,
                    )

    return ok


def single_holdout(trials: list[dict], spec: str) -> set[str] | None:
    match = re.fullmatch(r"trial(\d+)", spec.strip())

    if match:
        index = int(match.group(1))
        names = {
            trial["trial"]
            for trial in trials
            if trial["trial_index"] == index
        }
    else:
        subjects = {
            letter.strip().upper()
            for letter in spec.split(",")
            if letter.strip()
        }
        names = {
            trial["trial"]
            for trial in trials
            if trial["subject"] in subjects
        }

    return names or None


def build_row(trial: dict, video: Path, start: float, end: float) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt(),
            },
            {
                "role": "user",
                "content": user_prompt(trial["task"]),
            },
            {
                "role": "assistant",
                "content": answer(trial["scores"]),
            },
        ],
        "videos": [str(video)],
        "grs": trial["grs"],
        "chat_template_kwargs": {
            "video_start": round(start, 2),
            "video_end": round(end, 2),
            "fps": frame_budget.sampling_fps(end - start),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        help=(
            "default: <dataset-root>/processed/trials, written by "
            "process_jigsaws.py"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "exports" / "original" / "skills",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help=(
            "comma separated; the default trains one model across all three "
            "tasks (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--prefix",
        default="jigsaws",
        help=(
            "file name prefix; must contain 'jigsaws' so evaluate/evaluate.py "
            "routes it (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--holdout",
        default="trial5",
        help=(
            "test side for the single split: 'trial<N>' holds out the N-th "
            "trial of every subject, or supply a comma-separated list of "
            "subject letters (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help=(
            "also write the five LOSO and eight LOUO fold pairs; each pair "
            "requires a separate fine-tuning run"
        ),
    )
    parser.add_argument(
        "--protocols",
        default="loso,louo",
        help="fold sets written by --cv (default: %(default)s)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=DEFAULT_WINDOW_SEC,
        help=(
            "window length in seconds; data/preprocess/frame_budget.py's ladder samples a "
            f"{DEFAULT_WINDOW_SEC:g} s window at {FPS:g} fps, which is the full "
            "128-frame budget (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--skip-official-check",
        action="store_true",
        help=(
            "do not compare the generated Suturing folds with the released "
            "SkillDetection splits"
        ),
    )
    parser.add_argument('--balance', action='store_true', help='uniform-interface flag; this dataset has no class imbalance to upsample, so it is a no-op')
    args = parser.parse_args(argv)
    encode_videos(['--dataset-root', str(args.dataset_root)])

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        print(f"unknown tasks: {unknown}", file=sys.stderr)
        return 1

    if "jigsaws" not in args.prefix:
        print(
            f"--prefix must contain 'jigsaws', got {args.prefix!r}",
            file=sys.stderr,
        )
        return 1

    protocols = [
        protocol.strip()
        for protocol in args.protocols.split(",")
        if protocol.strip()
    ]
    unknown_protocols = [
        protocol
        for protocol in protocols
        if protocol not in {"loso", "louo"}
    ]
    if unknown_protocols:
        print(f"unknown protocols: {unknown_protocols}", file=sys.stderr)
        return 1

    clips_dir = (
        args.clips_dir
        or args.dataset_root / "processed" / "trials"
    )
    trials = [
        trial
        for task in tasks
        for trial in read_meta(args.dataset_root, task)
    ]

    if not trials:
        print("no trials read", file=sys.stderr)
        return 1

    missing = [
        trial["trial"]
        for trial in trials
        if not (clips_dir / f"{trial['trial']}.mp4").is_file()
    ]
    if missing:
        print(
            f"{len(missing)} clips missing from {clips_dir}, run "
            f"process_jigsaws.py first: {missing[:5]}",
            file=sys.stderr,
        )
        return 1

    if (
        not args.skip_official_check
        and not check_official(args.dataset_root, trials)
    ):
        return 1

    by_name = {trial["trial"]: trial for trial in trials}
    assignments = {
        protocol: folds(trials, protocol)
        for protocol in protocols
    }

    holdout = single_holdout(trials, args.holdout)
    if holdout is None:
        print(
            f"cannot read --holdout {args.holdout!r}",
            file=sys.stderr,
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    durations = frame_budget.probe_durations(
        [str(clips_dir / f"{trial['trial']}.mp4") for trial in trials])
    lengths = sorted(durations.values())
    spans = {name: frame_budget.windows(duration, args.window_sec)
             for name, duration in durations.items()}
    per_trial = sorted(len(s) for s in spans.values())
    print(
        f"trial seconds: min {lengths[0]:.0f} median "
        f"{lengths[len(lengths) // 2]:.0f} max {lengths[-1]:.0f}"
    )
    print(
        f"windows: {args.window_sec:g}s at "
        f"{frame_budget.sampling_fps(args.window_sec):g} fps = "
        f"{frame_budget.frames_for(args.window_sec)} frames, last window anchored at "
        f"the end, trial scores copied onto every window; per trial min {per_trial[0]} "
        f"median {per_trial[len(per_trial) // 2]} max {per_trial[-1]}"
    )

    def write_pair(suffix: str, test_names: set[str]) -> tuple[int, int]:
        test = [by_name[name] for name in sorted(test_names)]
        train = [
            trial
            for trial in trials
            if trial["trial"] not in test_names
        ]

        counts = {}
        for split, trials_in_split in (("train", train), ("test", test)):
            path = args.out_dir / f"{args.prefix}{suffix}_{split}.jsonl"
            written = 0
            with path.open("w", encoding="utf-8") as handle:
                for trial in sorted(trials_in_split, key=lambda row: row["trial"]):
                    video = clips_dir / f"{trial['trial']}.mp4"
                    for start, end in spans[str(video)]:
                        row = build_row(trial, video, start, end)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
            counts[split] = (len(trials_in_split), written)

        return counts["train"], counts["test"]

    n_train, n_test = write_pair("", holdout)
    held = Counter(by_name[name]["task"] for name in holdout)
    held_text = "  ".join(
        f"{task}={count}"
        for task, count in sorted(held.items())
    )
    print(
        f"single split ({args.holdout}): train {n_train[0]} trials / {n_train[1]} rows  "
        f"test {n_test[0]} trials / {n_test[1]} rows ({held_text})"
    )

    if args.cv:
        for protocol in protocols:
            for fold, test_names in assignments[protocol].items():
                n_train, n_test = write_pair(
                    f"_{protocol}{fold}",
                    set(test_names),
                )
                per_task = Counter(
                    by_name[name]["task"]
                    for name in test_names
                )
                per_task_text = "  ".join(
                    f"{task}={count}"
                    for task, count in sorted(per_task.items())
                )
                print(
                    f"{protocol}{fold}: train {n_train[0]} trials / {n_train[1]} rows  "
                    f"test {n_test[0]} trials / {n_test[1]} rows ({per_task_text})"
                )

    counts = Counter(trial["task"] for trial in trials)
    count_text = "  ".join(
        f"{task}={count}"
        for task, count in sorted(counts.items())
    )
    print(f"-> {args.out_dir}/{args.prefix}_*.jsonl")
    print(f"trials: {count_text}  total={len(trials)}")

    grs = [trial["grs"] for trial in trials]
    print(
        f"GRS: min={min(grs)} max={max(grs)} "
        f"mean={sum(grs) / len(grs):.2f}"
    )

    return 0


def gesture_span(path: Path) -> tuple[int, int]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    return int(rows[0][0]), int(rows[-1][1])


def probe_frames(path: Path) -> int | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    value = out.stdout.strip().rstrip(",")
    return int(value) if value.isdigit() else None


def plan(root: Path, tasks) -> list[dict]:
    jobs = []
    for task in tasks:
        for transcription in sorted((root / task / "transcriptions").glob("*.txt")):
            trial = transcription.stem
            source = root / task / "video" / f"{trial}_capture1.avi"
            if not source.is_file():
                print(f"missing source: {source}", file=sys.stderr)
                continue
            start, end = gesture_span(transcription)
            jobs.append({"task": task, "trial": trial, "source": source,
                         "start": start, "end": end})
    return jobs


def encode(job: dict, out_dir: Path, overwrite: bool) -> tuple[str, str]:
    target = out_dir / f"{job['trial']}.mp4"
    if target.is_file() and not overwrite:
        return job["trial"], "skipped"
    trim = f"trim=start_frame={job['start'] - 1}:end_frame={job['end']},setpts=PTS-STARTPTS"
    out = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(job["source"]),
         "-vf", trim, "-an", "-c:v", "libx264", "-crf", "18",
         "-pix_fmt", "yuv420p", str(target)],
        capture_output=True, text=True)
    if out.returncode != 0:
        print(f"ffmpeg failed on {job['trial']}: {out.stderr.strip()}", file=sys.stderr)
        return job["trial"], "failed"
    return job["trial"], "written"


def encode_videos(argv=None) -> int:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path,
                        help="default: <dataset-root>/processed/trials")
    parser.add_argument("--tasks", default=",".join(TASKS),
                        help="comma separated (default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="probe every written clip and report length mismatches")
    args = parser.parse_args(argv)

    tasks = [t for t in args.tasks.split(",") if t]
    out_dir = args.out_dir or args.dataset_root / "processed" / "trials"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = plan(args.dataset_root, tasks)
    if not jobs:
        print("nothing to do", file=sys.stderr)
        return 1

    counts = {"written": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for _, status in pool.map(lambda j: encode(j, out_dir, args.overwrite), jobs):
            counts[status] += 1
    print(f"{len(jobs)} trials: " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    if args.verify:
        bad = 0
        for job in jobs:
            target = out_dir / f"{job['trial']}.mp4"
            want = job["end"] - job["start"] + 1
            got = probe_frames(target)
            if got is None or abs(got - want) > 1:
                print(f"{job['trial']}: expected {want} frames, probed {got}",
                      file=sys.stderr)
                bad += 1
        print(f"verify: {len(jobs) - bad}/{len(jobs)} clips match their annotated span")
        if bad:
            return 1
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
