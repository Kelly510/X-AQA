#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${MODEL:-"$ROOT/models/Qwen3.5-9B"}
if [ "${THINKING:-0}" = 1 ]; then
  NAME=Qwen3.5-9B-thinking
  PREFIX=qwen35_9b_base_thinking_benchv3_
else
  NAME=Qwen3.5-9B
  PREFIX=qwen35_9b_base_benchv3_
fi
NAME="$NAME" MODEL_PATH="$MODEL" PREFIX="$PREFIX" REF="$ROOT/benchmark/v3/data" \
  bash "$ROOT/evaluate/run_model_eval.sh"
