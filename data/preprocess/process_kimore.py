#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocess.paths import workspace
from data.preprocess import frame_budget

DATASET = workspace() / "datasets" / "KIMORE"
OUT = REPO_ROOT / "data" / "exports" / "original" / "rehabilitation"


ITEMS = (
    ("primary_objective_achieved",
     "Is the primary objective of the exercise achieved (i.e., extension of the upper limbs, "
     "trunk rotation with upper limbs elevated to 90 degrees, squatting, etc.)?"),
    ("repetitions_constant", "Is the exercise repeated in a constant manner?"),
    ("movement_amplitude_complete", "Is the amplitude of the movement complete?"),
    ("head_posture", "Is the posture of the head correct?"),
    ("right_arm_posture", "Is the posture of the right arm correct?"),
    ("left_arm_posture", "Is the posture of the left arm correct?"),
    ("trunk_posture", "Is the posture of the trunk correct?"),
    ("pelvis_posture", "Is the posture of the pelvis correct?"),
    ("right_leg_posture", "Is the posture of the right leg correct?"),
    ("left_leg_posture", "Is the posture of the left leg correct?"),
)

EXERCISES = {
    1: ("Arm Raise", "The subject holds a bar with both hands and arms extended along the body, "
                     "slightly apart, and raises the arms above the head keeping the elbows "
                     "extended, to stretch the trunk muscles. The feet stay on the ground, "
                     "slightly apart, with the knees slightly flexed; the pelvis must not tilt."),
    2: ("Lateral Trunk Tilt", "The subject raises the arms above the head with the elbows "
                              "completely extended and a bar in both hands, then tilts the trunk "
                              "slowly left and then right, keeping it in the frontal plane. The "
                              "feet stay on the ground, the knees slightly flexed, and the trunk "
                              "must not bend forwards or backwards."),
    3: ("Trunk Rotation", "The subject holds the arms parallel at ninety degrees to the torso "
                          "with the elbows extended, then rotates the torso slowly to the left "
                          "and then to the right. The body stays aligned, the feet on the ground, "
                          "the knees slightly flexed."),
    4: ("Pelvis Rotation", "The subject stands still with the feet slightly apart and, without "
                           "moving the feet, makes a circular rotation with the pelvis, first "
                           "clockwise and then counter-clockwise."),
    5: ("Squat", "The subject holds the arms at ninety degrees to the trunk with the elbows "
                 "extended, flexes the knees up to 60-70 degrees and returns to the starting "
                 "position, keeping the body aligned in the sagittal plane."),
}

SYSTEM = """You are a physiotherapy examiner. Evaluate the rehabilitation exercise in the video.
The exercise is %s.

%s

Rate the performance on the ten items of the Exercise Accuracy Assessment Questionnaire
(EAAQ), using the scale 1 = Never, 2 = Rarely, 3 = Sometimes, 4 = Often, 5 = Always.
Consider the whole exercise, all repetitions, and judge only what is visible in the video.

%s

Return only this JSON object and nothing else, with one score for every item:
%s

The clinical Total Score is the sum of the ten scores."""

USER = "<video>"


def video_index() -> dict[tuple[str, str, str], Path]:
    out = {}
    for path in DATASET.glob("*/*/*/Es*/rgb/*"):
        if path.suffix.lower() not in (".mp4", ".mxf"):
            continue
        parts = path.parts[len(DATASET.parts):]
        group = f"{parts[0]}/{parts[1]}"
        out[(group, parts[2], parts[3])] = path
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    labels = list(csv.DictReader((DATASET / "labels" / "clinical_assessment.csv").open()))
    videos = video_index()
    paths = list(videos.values())
    lengths = frame_budget.probe_durations(
        paths, workers=32,
        cache_path=REPO_ROOT / "outputs" / "analysis" / "kimore.durations.json")

    rows, skipped = [], 0
    for label in labels:
        if not label["cTS"]:
            continue
        key = (label["group"], label["subject_id"], f"Es{label['exercise_id']}")
        path = videos.get(key)
        if path is None:
            skipped += 1
            continue
        name, description = EXERCISES[int(label["exercise_id"])]
        system = SYSTEM % (
            name, description,
            "\n".join(f"- {key_}: {question}" for key_, question in ITEMS),
            "{" + ", ".join(f'"{key_}": <integer from 1 to 5>' for key_, _ in ITEMS) + "}",
        )
        duration = lengths[str(path)]
        rows.append({
            "sample_id": f"kimore_{label['group'].replace('/', '_')}_{label['subject_id']}_Es{label['exercise_id']}",
            "task": "kimore_cts",
            "split": "test",
            "group": label["group"],
            "subject_id": label["subject_id"],
            "exercise_id": int(label["exercise_id"]),
            "exercise": name,
            "cts": float(label["cTS"]),
            "cpo": float(label["cPO"]) if label["cPO"] else None,
            "ccf": float(label["cCF"]) if label["cCF"] else None,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": USER},
                {"role": "assistant", "content": json.dumps({"cTS": round(float(label["cTS"]), 3)})},
            ],
            "videos": [str(path)],
            "chat_template_kwargs": {
                "video_start": 0.0,
                "video_end": round(duration, 2),
                "fps": frame_budget.sampling_fps(duration),
            },
        })

    rows.sort(key=lambda r: r["sample_id"])
    path = args.out / "kimore_cts_test.jsonl"
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(rows)} rows (skipped {skipped} without video)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
