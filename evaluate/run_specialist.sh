#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON_BIN:-python3}
RESULTS=${XAQA_RESULTS:-$ROOT/evaluate/results/specialists}
MODEL=${MODEL:-"$ROOT/models/Qwen3.5-9B"}
TRAIN_DIR=${TRAIN_DIR:-$ROOT/runs/qwen35_9b_lora_full}
CK_STEP=1255
TASK=${1:?usage: bash evaluate/run_specialist.sh mtlaqa|fitnessaqa_kp|kimore_cts|lasana_error}
case "$TASK" in mtlaqa | fitnessaqa_kp | kimore_cts | lasana_error) ;;
*)
  echo "unknown specialist task: $TASK" >&2
  exit 1
  ;;
esac
MERGED=${MODEL_PATH:-$TRAIN_DIR.merged/checkpoint-${CK_STEP}-merged}
mkdir -p "$RESULTS"
if [ ! -f "$ROOT/benchmark/specialist_refs/test_manifest.json" ]; then
  "$PY" "$ROOT/data/build_specialist_refs.py"
fi
if [ ! -f "$MERGED/config.json" ]; then
  mkdir -p "$(dirname "$MERGED")"
  "$PY" -m swift.cli.merge_lora --model "$MODEL" \
    --adapters "$TRAIN_DIR/checkpoint-$CK_STEP" --output_dir "$MERGED"
fi
RUN="$RESULTS/${TASK}_$(date -u +%Y%m%dT%H%M%SZ)"
MODEL_PATH="$MERGED" MODEL_NAME="X-AQA" \
  XAQA_REFERENCE_DIR="$ROOT/benchmark/specialist_refs" XAQA_TASKS="$TASK" \
  XAQA_REPLICAS=8 XAQA_GPUS=${XAQA_GPUS:-0,1,2,3,4,5,6,7} XAQA_CONCURRENCY=32 \
  bash "$ROOT/evaluate/run_vllm_eval.sh" "$RUN"
"$PY" "$ROOT/evaluate/eval_specialists.py" --pred-dir "$RUN" --tasks "$TASK" \
  --out "$RUN/specialist_metrics.json"
