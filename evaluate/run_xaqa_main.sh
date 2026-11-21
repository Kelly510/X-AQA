#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON_BIN:-python3}
MODEL=${MODEL:-"$ROOT/models/Qwen3.5-9B"}
TRAIN_DIR=${TRAIN_DIR:-$ROOT/runs/qwen35_9b_lora_full}
CK_STEP=1255
MERGED=${MERGED:-$TRAIN_DIR.merged/checkpoint-${CK_STEP}-merged}
if [ ! -f "$MERGED/config.json" ]; then
  mkdir -p "$(dirname "$MERGED")"
  "$PY" -m swift.cli.merge_lora --model "$MODEL" \
    --adapters "$TRAIN_DIR/checkpoint-$CK_STEP" --output_dir "$MERGED"
fi
NAME="X-AQA" \
  MODEL_PATH="$MERGED" PREFIX="xaqa_benchv3_" REF="$ROOT/benchmark/v3/data" \
  bash "$ROOT/evaluate/run_model_eval.sh"
