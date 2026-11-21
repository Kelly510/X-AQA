# Data preparation

X-AQA is built from publicly released source datasets. Download each dataset from its official release
into `datasets/<SOURCE>/` under the repository root, then run its converter. Each
`process_*.py` performs the required video encoding, scoreboard masking, and commentary
cleaning internally.

Extra annotations needed by the converters are provided in `data/annotations/`.

## Workflow

1. Download each source dataset into `datasets/<SOURCE>/`.
2. Run its converter, e.g. `python data/preprocess/process_aqa7.py`.
3. Build the training and test data:

```bash
python data/build_train.py      # X-AQA-40K
python data/build_benchmark.py  # X-AQA-Bench
```

## Source index

**Competitive sports**  
[AQA-7](#aqa-7) ·
[MTL-AQA](#mtl-aqa) ·
[Rhythmic Gymnastics](#rhythmic-gymnastics) ·
[FineFS](#finefs) ·
[MultiSenseBadminton](#multisensebadminton) ·
[Ego-Exo4D](#ego-exo4d) ·
[T3Set](#t3set) ·
[CricketVision](#cricketvision) ·
[ExpertAF](#expertaf) ·
[CoT-AFA](#cot-afa)

**Fitness exercises**  
[Fitness-AQA](#fitness-aqa) ·
[FLAG3D](#flag3d) ·
[EgoExo-Fitness](#egoexo-fitness) ·
[QEVD](#qevd) ·
[Fitness-7](#fitness-7)

**Rehabilitation**  
[KERAAL](#keraal) ·
[REHAB24-6](#rehab24-6) ·
[ExeCheck](#execheck) ·
[LLM-FMS](#llm-fms) ·
[UpperLimb](#upperlimb) ·
[KIMORE](#kimore)

**General skills**  
[JIGSAWS](#jigsaws) ·
[CaptainCook4D](#captaincook4d) ·
[CPR-Coach](#cpr-coach) ·
[LASANA](#lasana) ·
[AIxSuture](#aixsuture)

## Competitive sports

### AQA-7 — `datasets/AQA-7/`
Download: https://github.com/ParitoshParmar/AQA-7

```text
AQA-7/
  Actions/<sport>/*.avi
  Split_4/split_4_{train,test}_list.mat
  source_annotations/unlv/
```

```bash
python data/preprocess/process_aqa7.py
```

### MTL-AQA — `datasets/MTL-AQA/`
Download: https://github.com/ParitoshParmar/MTL-AQA

```text
MTL-AQA/
  MTL-AQA_dataset_release/Raw_Annotations/
  MTL-AQA_dataset_release/Ready_2_Use/
  videos/
```

`process_mtlaqa.py` first exports `commentary_cleaned/cleaning_pairs.jsonl` from
`data/annotations/mtlaqa_commentary.json`, then writes score and text rows.

```bash
python data/preprocess/process_mtlaqa.py
```

### Rhythmic Gymnastics — `datasets/RhythmicGymnastics/`
Download: dataset link from https://github.com/qinghuannn/ACTION-NET

```text
RhythmicGymnastics/
  train.txt
  test.txt
  videos/*.mp4
  processed/videos_masked/        # generated
```

```bash
python data/preprocess/process_rhythmicgymnastics.py
```

Scoreboard masking replays `data/annotations/rg_scoreboard.json`.

### FineFS — `datasets/FineFS/`
Download: https://github.com/yanliji/FineFS-dataset

```text
FineFS/
  annotation/*.json
  video/*.mp4
  processed/videos/               # generated (long side 800)
  processed/videos_masked/        # generated
```

```bash
python data/preprocess/process_finefs.py
```

Scoreboard masking replays `data/annotations/finefs_scoreboard.json`.

### MultiSenseBadminton — `datasets/MultiSenseBadminton/`
Download: DOI https://doi.org/10.1038/s41597-024-03144-z

```text
MultiSenseBadminton/
  Sub00/ ... Sub24/              # official videos and wearable recordings
  Documentations/Survey Data File.xlsx
```

The three-coach feedback merged into one English paragraph per stroke is provided in
`data/annotations/multisensebadminton_feedback.jsonl`.

```bash
python data/preprocess/process_multisensebadminton.py
```

### Ego-Exo4D — `datasets/Ego-Exo4D/`
Download: https://ego-exo4d-data.org/

```text
Ego-Exo4D/
  takes.json
  annotations/splits.json
  annotations/proficiency_demonstrator_*.json
  <raw take videos referenced by takes.json>
```

One command writes both sports and general-skills tasks.

```bash
python data/preprocess/process_egoexo4d.py
```

### T3Set — `datasets/T3Set/`
Download: https://zenodo.org/records/15516144

```text
T3Set/
  anonymous_*/<technique>/<round>/...   # official videos and coach annotations
```

The English coach-comment summaries derived from the official annotations are provided in
`data/annotations/t3set_merged_summaries.jsonl`.

```bash
python data/preprocess/process_t3set.py
```

### CricketVision — `datasets/CricketVision/`
Download: https://github.com/dvanderhaar/uj-aqa-cricketvision

```text
CricketVision/
  JSON/*.json
  Videos/*.mp4
```

```bash
python data/preprocess/process_cricketvision.py
```

### ExpertAF — `datasets/ExpertAF/` and `datasets/Ego-Exo4D/`
Download: https://github.com/thechargedneutron/ExpertAF

```text
ExpertAF/
  expertaf_val_N5.json
  <other released annotation JSON>
Ego-Exo4D/
  takes.json
  <raw take videos>
```

```bash
python data/preprocess/process_expertaf.py
```

### CoT-AFA — `datasets/CoT-AFA/`
Download: https://github.com/MICLAB-BUPT/EFA

```text
CoT-AFA/
  dataset/GYM88.xlsx
  dataset/comment/
  AQA_data/videos/
  test_caption_coco_format_coco_format.json
```

The English step instructions keyed by the 141 categories are provided in
`data/annotations/cotafa_step_instructions.json`.

One command writes both martial-arts and fitness tasks.

```bash
python data/preprocess/process_cotafa.py
```

## Fitness exercises

### Fitness-AQA — `datasets/FitnessAQA/`
Download: https://github.com/ParitoshParmar/Fitness-AQA

```text
FitnessAQA/
  OHP/Labeled_Dataset/
  Squat/Labeled_Dataset/
  Squat/processed/ss_frames/      # generated when missing
```

```bash
python data/preprocess/process_fitnessaqa.py
```

### FLAG3D — `datasets/FLAG3D/`
Download: https://andytang15.github.io/FLAG3D/

```text
FLAG3D/
  annotations/*I001.txt
  FLAG3D++/annotation/*.csv
  <raw videos>
  processed/clips/                # generated
```

```bash
python data/preprocess/process_flag3d.py
```

### EgoExo-Fitness — `datasets/EgoExo-Fitness/`
Download: https://github.com/iSEE-Laboratory/EgoExo-Fitness

```text
EgoExo-Fitness/
  raw_annotations/meta_records.json
  raw_annotations/interpretable_action_judgement.json
  <ego/exo frame directories>
  processed/clips/                # generated
```

```bash
python data/preprocess/process_egoexo_fitness.py
```

### QEVD — `datasets/QEVD/`
Download: [QEVD instructions](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/QSC-QEVD-FIT-COACH-Dataset-Download-Instructions.pdf)

```text
QEVD/
  <FIT-300K annotations>
  short_clips/*.mp4
  <FIT-COACH annotations and videos>
```

The first run caches clip durations under `outputs/analysis/`.

```bash
python data/preprocess/process_qevd.py
```

### Fitness-7 — `datasets/Fitness-7/`
Download: https://data.mendeley.com/datasets/kgbb3yn47p/3

```text
Fitness-7/
  abs/ back/ bicep_curl/ push_up/ shoulder/ squat/ tricep/
    good/<view>/*.mp4
    bad/<view>/*.mp4
```

```bash
python data/preprocess/process_fitness7.py
```

## Rehabilitation

### KERAAL — `datasets/KERAAL/`
Download: https://keraal.enstb.org/KeraalDataset.html

```text
KERAAL/
  <released group and annotation files>
  processed/videos/               # generated
```

KERAAL contributes training data only.

```bash
python data/preprocess/process_keraal.py
```

### REHAB24-6 — `datasets/REHAB24-6/`
Download: https://zenodo.org/records/13305826

```text
REHAB24-6/
  Segmentation.csv
  <released video directories>
  processed/videos/               # generated
```

```bash
python data/preprocess/process_rehab246.py
```

### ExeCheck — `datasets/ExeCheck/`
Download: https://www.cs.bu.edu/faculty/betke/ExeChecker/

```text
ExeCheck/
  processing_scripts/RepSeg.csv
  raw_data/<subject>/*.mkv
  processed/videos/<subject>/*.mp4   # generated
```

```bash
python data/preprocess/process_execheck.py
```

### LLM-FMS — `datasets/LLM-FMS/`
Download: https://doi.org/10.6084/m9.figshare.c.7601630.v1

```text
LLM-FMS/
  FMSRules.json
  FMS-FineGrainedLabelInfo/<movement>/
  FMS-Keyframes/<movement>/
```

```bash
python data/preprocess/process_llmfms.py
```

### UpperLimb — `datasets/UpperLimb/`
Download: https://data.mendeley.com/datasets/49h9dcwx5v/1

```text
UpperLimb/
  Exercise/<exercise>/<Complete|Incomplete>/<Train|Test>/*.mp4
```

```bash
python data/preprocess/process_upperlimb.py
```

### KIMORE — `datasets/KIMORE/`
Download: https://doi.org/10.1109/TNSRE.2019.2923060 (RGB by request)

```text
KIMORE/
  labels/clinical_assessment.csv
  <subject>/<exercise>/.../rgb/*.mp4
```

```bash
python data/preprocess/process_kimore.py
```

## General skills

### JIGSAWS — `datasets/JIGSAWS/`
Download: https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/

```text
JIGSAWS/
  <task>/meta_file_<task>.txt
  <task>/transcriptions/*.txt
  <task>/video/*.mp4
  processed/trials/               # generated
```

```bash
python data/preprocess/process_jigsaws.py
```

### CaptainCook4D — `datasets/CaptainCook4D/`
Download: https://captaincook4d.github.io/captain-cook/

```text
CaptainCook4D/
  annotations/
  gopro_360p/<recording>_360p.mp4
```

```bash
python data/preprocess/process_captaincook4d.py
```

### CPR-Coach — `datasets/CPR-Coach/`
Download: https://github.com/Shunli-Wang/CPR-Coach

```text
CPR-Coach/
  ann/ActionList.txt
  ann/trainlist.txt
  ann/testlist.txt
  ann/*.txt
  <released videos>
  processed/videos/               # generated
```

```bash
python data/preprocess/process_cprcoach.py
```

### LASANA — `datasets/LASANA/`
Download: https://huggingface.co/datasets/nct-tso/lasana

```text
LASANA/
  <task>/<split>/metadata.jsonl
  <task>/<split>/<videos>
```

```bash
python data/preprocess/process_lasana.py
```

### AIxSuture — `datasets/AIxSuture/`
Download: https://zenodo.org/records/7940583

```text
AIxSuture/
  OSATS.xlsx
  videos/*.mp4
```

The student-level 70/15/15 split is provided in `data/annotations/aixsuture_splits.json`.

```bash
python data/preprocess/process_aixsuture.py
```

## Build training and test data

After every source has been processed:

```bash
# X-AQA-40K: 29-file, 40,163-draw mixture
python data/build_train.py

# X-AQA-Bench: v3 and specialist references
python data/build_benchmark.py
```

Outputs:

```text
data/configs/cap2k/                       # generated training lists
data/exports/original/                    # generated QA
benchmark/v3/                             # 5,366 ID + 1,735 OOD
benchmark/specialist_refs/                # four full test partitions
```

Benchmark protocol: each ID source is deduplicated by (recording, stratum) and capped at 300
units; the four OOD sources (T3Set, Fitness-7, JIGSAWS, KIMORE) keep every released row.
KERAAL contributes training data only.

## Extra annotations

- `data/annotations/finefs_scoreboard.json`, `rg_scoreboard.json` — per-video scoreboard
  boxes and intervals.
- `data/annotations/mtlaqa_commentary.json` — edited MTL-AQA commentary, sample mapping and
  source checksums.
- `data/annotations/finefs_events.json`, `rhythmicgymnastics_events.json` — audited event
  identities used by the FineFS and rhythmic-gymnastics converters.
- `data/annotations/captaincook4d_description_merges.json` — merged step descriptions.
- `data/annotations/multisensebadminton_feedback.jsonl` — merged three-coach feedback.
- `data/annotations/t3set_merged_summaries.jsonl` — merged English coach comments.
- `data/annotations/cotafa_step_instructions.json` — English step instructions.
- `data/annotations/aixsuture_splits.json` — student-level train/val/test split.

## Licenses

Source datasets keep their original licenses. Code and annotations in this repository are
Apache-2.0; see `LICENSE`.
