# X-AQA

**Unifying Action Quality Assessment with Multimodal Large Language Models and Multi-Domain Learning**

[![arXiv](https://img.shields.io/badge/Paper-arXiv%3A2611.28548-B31B1B.svg)](https://arxiv.org/abs/2611.28548)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

X-AQA is a unified training and evaluation framework for general-purpose action quality
assessment. It converts heterogeneous AQA annotations into one question-answering format and
trains a single multimodal LLM across four domains: competitive sports, fitness exercises,
rehabilitation, and general skills.

- **X-AQA-40K** — 40,163 training samples over 19,614 media files from 22 public datasets.
- **X-AQA-Bench** — 5,366 in-domain and 1,735 out-of-domain test requests over 31 tasks.
- **X-AQA-9B** — the released model, initialized from Qwen3.5-9B and trained with LoRA.

This repository provides the data construction pipeline, the multi-domain SFT recipe, and the
X-AQA-Bench evaluation code. It does not redistribute source videos, annotations, generated QA,
or model weights.

## Contents

- [Environment](#environment)
- [Models](#models)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Metrics](#metrics)
- [Citation](#citation)
- [License](#license)

## Environment

```bash
pip install -r requirements.txt
```

Python 3.10+ is required. Training uses PyTorch, `ms-swift`, and 16 H20 GPUs; evaluation uses
vLLM. `ffmpeg` and `ffprobe` are required for video preparation.

## Models

Training starts from Qwen3.5-9B, and text metrics use RoBERTa-large:

```bash
huggingface-cli download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
huggingface-cli download FacebookAI/roberta-large --local-dir models/roberta-large
```

The trained X-AQA model is available at
[Hugging Face](https://huggingface.co/Kelly0510/X-AQA-9B):

```bash
huggingface-cli download Kelly0510/X-AQA-9B --local-dir models/X-AQA-9B
```

`models/X-AQA-9B` is the merged model and can be evaluated directly. LoRA adapters trained
locally are merged with ms-swift before evaluation when needed.

## Data

Follow **[data/README.md](data/README.md)** to download the 26 public source datasets into `datasets/` and
run their converters. Each `process_*.py` also performs the required video encoding,
scoreboard masking, or commentary cleaning.

Build the training and test data:

```bash
# X-AQA-40K: sampling lists
python data/build_train.py

# X-AQA-Bench: benchmark and specialist references
python data/build_benchmark.py
```

Generated outputs:

```text
data/configs/cap2k/                 # training lists
data/exports/original/              # generated QA
benchmark/v3/data/{ID,OOD}/         # 5,366 ID + 1,735 OOD
benchmark/specialist_refs/          # 4 specialist test partitions
```

## Training

All experiments use Qwen3.5-9B with LoRA rank 32, alpha 64, dropout 0.05 on all linear layers
and the visual merger, a frozen ViT, one epoch, bf16, maximum sequence length 32,768, adaptive
sampling up to 128 frames, seed 42, and global batch size 32.

```bash
MODEL=models/Qwen3.5-9B bash train/run_joint.sh
```

The released model corresponds to checkpoint 1255.

## Evaluation

Inference uses 128 frames, greedy decoding, `max_tokens=2048`, and four vLLM replicas.

```bash
# Released X-AQA-9B
NAME=X-AQA-9B MODEL_PATH="$PWD/models/X-AQA-9B" PREFIX=xaqa_9b_benchv3_ \
  bash evaluate/run_model_eval.sh

# Locally trained joint adapter
MODEL=models/Qwen3.5-9B TRAIN_DIR=/path/to/joint_run \
  bash evaluate/run_xaqa_main.sh

# Dataset-specific specialists
bash evaluate/run_specialist.sh mtlaqa

# Untuned base model
bash evaluate/run_base.sh
THINKING=1 bash evaluate/run_base.sh
```

Each run directory under `evaluate/results/` contains `metrics.json`, `summary.md`, and one
`<task>.jsonl` file with the full prompt, reference, and model response for every sample.

## Metrics

- **Score (S)** — SRCC inside each separately ranked project, combined by a sample-size-weighted
  Fisher-z mean.
- **Category (C)** — macro-F1 over reference classes and unweighted Cohen's κ.
- **Text (T)** — BERTScore F1 with RoBERTa-large.
- **Specialists** — native metrics on full test partitions: MTL-AQA SRCC / R-L2, Fitness-AQA
  per-error F1, KIMORE per-exercise SRCC, and LASANA accuracy / balanced accuracy.

## Citation

```bibtex
@misc{zheng2026xaqa,
  title={X-AQA: Unifying Action Quality Assessment with Multimodal Large Language Models and Multi-Domain Learning}, 
  author={Kaili Zheng and Wei Ding and Yuchen Ding and Kaiwen Wang and Xun Zhu and Chenyi Guo and Yudong Zhang and Ji Wu},
  year={2026},
  eprint={2611.28548},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2611.28548}, 
}
```

## License

Code and extra annotations are released under the [Apache License 2.0](LICENSE). Source
datasets remain under their original licenses; see [data/README.md](data/README.md).
