#!/usr/bin/env python3
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"

import json
import re

from .rehabilitation_common import arguments, attach_hashes, export, videos

EXERCISES = {
    '01': ('Lifting an Object', 'Begin seated with the object held securely. Lift it steadily to the final raised position while maintaining the grip and control. A visible lowering phase should also be controlled. Failure to lift to the endpoint, losing the grip, or stopping early indicates incomplete execution. Do not require an unseen lowering phase when the recorded endpoint is the raised position.'),
    '02': ('Extending the Elbow', 'Begin with the elbow supported at the table and the forearm upright. Extend the elbow smoothly until the forearm lies along and contacts the table. Failure to achieve that endpoint or an early stop indicates incomplete execution.'),
    '03': ('Lifting the Wrist', 'Keep the forearm resting on an elevated support with the wrist free beyond the edge. Raise the wrist from its hanging position to the final lifted position in a controlled movement. Restricted movement that fails to reach the lifted endpoint, loss of control, or early stopping indicates incomplete execution.'),
    '04': ('Opening the Hand', 'Keep the forearm supported and the hand free beyond the edge. From a closed or relaxed hand, extend and spread the fingers until the hand is fully open. Fingers remaining partly closed or a stop before the open-hand endpoint indicates incomplete execution.'),
}


OUTPUT_VERSION = 'upperlimb_completion_v3'


def prompt(s):
    candidates = '\n'.join(f'- {name}: {criteria}' for name, criteria in EXERCISES.values())
    return ('Assess whether the upper-limb exercise in the video is completed.\n\n'
            'Exercise candidates and expected procedures:\n' + candidates + '\n\n'
            'Apply the procedure for the exercise shown in the video. Complete means the required '
            'movement reaches its endpoint successfully; incomplete means it visibly fails or stops short. '
            'Assess the recorded cycle without requiring repeated cycles or exact joint angles. '
            'Do not infer unseen failure from occlusion alone.\n\n'
            'Return only one JSON object and nothing else, without Markdown or code fences:\n'
            '{"completion": "complete" or "incomplete"}.')


def load_samples(root):
    samples = []
    for p in videos(root / 'Exercise'):
        rel = p.relative_to(root.resolve())
        if len(rel.parts) != 5:
            raise ValueError(f'Unexpected directory structure: {rel}')
        _, folder, label, split, _ = rel.parts
        m = re.fullmatch(r'(\d{2})_(0[1-4])_(0[01])_(\d{2})', p.stem)
        if not m or label not in {'Complete', 'Incomplete'} or split not in {'Train', 'Test'}:
            raise ValueError(f'Invalid UpperLimb metadata: {rel}')
        subject, code, status, repetition = m.groups()
        conflict = (status == '01') != (label == 'Complete')
        known_conflict = (subject == '04' and code == '02' and status == '00' and label == 'Complete'
                          and split == 'Train' and 1 <= int(repetition) <= 10)
        if folder != f'{int(code)}_{EXERCISES[code][0]}' or (conflict and not known_conflict):
            raise ValueError(f'Filename/directory label mismatch: {rel}')
        samples.append(dict(id=f'upperlimb_{p.stem}_{label.lower()}', video=str(p), source_path=rel.as_posix(),
                            subject=subject, exercise=EXERCISES[code][0], exercise_code=code,
                            label=label.lower(), original_status=status, repetition=repetition,
                            split=split.lower(), label_source='original directory',
                            filename_label_conflict=conflict,
                            label_resolution='known source filename typo; preserve released directory label' if conflict else 'filename and directory agree'))
    return samples


def main():
    args = arguments('UpperLimb', '').parse_args()
    samples = load_samples(args.dataset_root)
    for sample in samples:
        sample['completion'] = sample['label']
        sample['class_label'] = sample['completion']
        sample['target'] = json.dumps(
            {'completion': sample['completion']},
            ensure_ascii=False)
        sample['target_definition'] = OUTPUT_VERSION
    attach_hashes(samples, args.workers)
    export(samples, args, 'UpperLimb', 'upperlimb_completion', prompt,
           'https://pmc.ncbi.nlm.nih.gov/articles/PMC13195782/ Sections 2-3 and Table 2',
           OUTPUT_VERSION, {'split_protocol': 'released Train/Test folders; seed does not alter official split',
                                      'excluded': [], 'rubric_note': 'Exercise 1 follows the explicit completion paragraph (raised endpoint); a return is assessed only if visible.'})


if __name__ == '__main__':
    main()
