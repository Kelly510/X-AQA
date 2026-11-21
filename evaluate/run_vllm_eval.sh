#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_PATH=${MODEL_PATH:-"$ROOT/models/Qwen3.5-9B"}
RUN_DIR=${1:?Pass a run directory}
REPLICAS=${XAQA_REPLICAS:-4}
CONCURRENCY=${XAQA_CONCURRENCY:-16}
PORT_BASE=8019
MAX_MODEL_LEN=65536
GPU_MEM_UTIL=0.85
MAX_NUM_SEQS=4
MAX_BATCHED_TOKENS=8192

mkdir -p -- "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export NO_PROXY=localhost,127.0.0.1
export MODEL_NAME MODEL_PATH
export XAQA_REFERENCE_DIR=${XAQA_REFERENCE_DIR:-"$ROOT/benchmark/v3/data"}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

if [ -n "${XAQA_GPUS:-}" ]; then
  IFS=, read -ra GPUS <<<"$XAQA_GPUS"
else
  GPUS=()
  for ((i = 0; i < REPLICAS; i++)); do GPUS+=("$i"); done
fi
if [ "${#GPUS[@]}" -lt "$REPLICAS" ]; then
  echo "XAQA_GPUS has ${#GPUS[@]} devices for $REPLICAS replicas" >&2
  exit 1
fi

PORTS=()
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for ((i = 0; i < REPLICAS; i++)); do
  gpu=${GPUS[$i]}
  port=$((PORT_BASE + i))
  PORTS+=("$port")
  VLLM_PORT=$((19000 + 100 * gpu)) MASTER_PORT=$((20000 + 100 * gpu)) \
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 --port "$port" --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --limit-mm-per-prompt '{"image":1,"video":1}' \
    --trust-remote-code --enforce-eager \
    >"$RUN_DIR/server_gpu$gpu.log" 2>&1 &
  PIDS+=("$!")
done

deadline=$((SECONDS + 1800))
for port in "${PORTS[@]}"; do
  while ! curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; do
    for pid in "${PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "vLLM server $pid exited early; see $RUN_DIR/server_gpu*.log" >&2
        exit 1
      fi
    done
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "timed out waiting for vLLM on port $port" >&2
      exit 1
    fi
    sleep 5
  done
done

ARGS=(--reference-dir "$XAQA_REFERENCE_DIR" --model "$MODEL_NAME"
  --base-url "http://127.0.0.1:${PORTS[0]}/v1"
  --output-dir "$RUN_DIR" --concurrency "$CONCURRENCY"
  --frame-scale "${XAQA_FRAME_SCALE:-1}")
if [ "${#PORTS[@]}" -gt 1 ]; then
  ARGS+=(--replica-urls)
  for port in "${PORTS[@]:1}"; do ARGS+=("http://127.0.0.1:$port/v1"); done
fi
if [ -n "${XAQA_TASKS:-}" ]; then
  IFS=, read -ra TASKS <<<"$XAQA_TASKS"
  ARGS+=(--datasets "${TASKS[@]}")
fi
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -B -u "$SCRIPT_DIR/run_vllm.py" "${ARGS[@]}"
