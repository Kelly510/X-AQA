#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export DATASET_LIST=$ROOT/data/configs/cap2k/full.txt
export RUN_NAME=qwen35_9b_lora_full
exec bash "$ROOT/train/sft_lora_full.sh"
