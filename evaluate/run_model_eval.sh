#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVAL=$ROOT/evaluate
RESULTS=${XAQA_RESULTS:-$ROOT/evaluate/results}
PY=${PYTHON_BIN:-python3}
NAME=${NAME:?Set NAME}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH}
PREFIX=${PREFIX:?Set PREFIX}
REF=${REF:-$ROOT/benchmark/v3/data}
[[ $REF = /* ]] || REF=$ROOT/$REF
[ -d "$REF" ] || {
  echo "missing reference tree: $REF (run data/build_benchmark.py)" >&2
  exit 1
}
RUN="$RESULTS/${PREFIX}$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"
MODEL_NAME="$NAME" MODEL_PATH="$MODEL_PATH" XAQA_REFERENCE_DIR="$REF" \
  bash "$EVAL/run_vllm_eval.sh" "$RUN"
printf 'metrics: %s\n' "$RUN/metrics.json"
