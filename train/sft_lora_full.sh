#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL=${MODEL:-"$ROOT/models/Qwen3.5-9B"}
DATA_ROOT=${DATA_ROOT:-$ROOT/data/exports/original}
DATASET_LIST=${DATASET_LIST:-$ROOT/data/configs/cap2k/full.txt}
RUN_NAME=${RUN_NAME:-qwen35_9b_lora_full}
OUT_DIR=${OUT_DIR:-$ROOT/runs/$RUN_NAME}
LORA_RANK=32
LORA_ALPHA=64
MIN_LR_RATE=0.1
GLOBAL_BATCH_SIZE=32

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")

NODE_IPS=${NODE_IPS:-}
if [[ -n "$NODE_IPS" ]]; then
  read -r -a _NODES <<<"$NODE_IPS"
  export NNODES=${NNODES:-${#_NODES[@]}}
  export MASTER_ADDR=${MASTER_ADDR:-${_NODES[0]}}
  if [ -z "${NODE_RANK:-}" ]; then
    _local_ips=" $(hostname -I) "
    for i in "${!_NODES[@]}"; do
      if [[ "$_local_ips" == *" ${_NODES[$i]} "* ]]; then
        NODE_RANK=$i
        break
      fi
    done
    [ -n "${NODE_RANK:-}" ] || {
      echo "ERROR: host IPs ($_local_ips) not in NODE_IPS=$NODE_IPS" >&2
      exit 1
    }
  fi
  export NODE_RANK
else
  export NNODES=${NNODES:-1}
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  export NODE_RANK=${NODE_RANK:-0}
fi
export MASTER_PORT=${MASTER_PORT:-39550}
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
if [ $((GLOBAL_BATCH_SIZE % WORLD_SIZE)) -ne 0 ]; then
  echo "ERROR: global batch 32 is not divisible by world size $WORLD_SIZE" >&2
  exit 1
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / WORLD_SIZE))
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
echo "[dist] node $NODE_RANK/$NNODES  master=$MASTER_ADDR:$MASTER_PORT  gpus=$CUDA_VISIBLE_DEVICES"

export FPS_MAX_FRAMES=128
export FPS_MIN_FRAMES=4
export VIDEO_MAX_TOKEN_NUM=512
export FORCE_QWENVL_VIDEO_READER=decord
export MODEL_SEQ_LEN=30720
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ -f "$OUT_DIR/args.json" ]] || compgen -G "$OUT_DIR/checkpoint-*" >/dev/null; then
  echo "ERROR: existing training artifacts in $OUT_DIR; choose a new RUN_NAME or OUT_DIR" >&2
  exit 1
fi

mapfile -t DATASETS <"$DATASET_LIST"
[ "${#DATASETS[@]}" -gt 0 ] || {
  echo "Empty DATASET_LIST: $DATASET_LIST" >&2
  exit 1
}

"$PYTHON_BIN" "$ROOT/data/build_data_protocol.py" --validate-training "${DATASETS[@]}"
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node "$NPROC_PER_NODE" \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  -m swift.cli.sft \
  --model "$MODEL" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --dataset "${DATASETS[@]}" \
  --split_dataset_ratio 0 \
  --loss_scale default \
  --is_binary_loss_scale false \
  --max_length 32768 \
  --truncation_strategy delete \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner false \
  --aligner_lr 5e-5 \
  --vit_gradient_checkpointing true \
  --use_liger_kernel true \
  --gradient_checkpointing true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs "{\"min_lr_rate\": $MIN_LR_RATE}" \
  --warmup_ratio 0.03 \
  --dataloader_num_workers 8 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 10 \
  --eval_strategy no \
  --output_dir "$OUT_DIR" \
  --add_version false \
  --report_to tensorboard \
  --seed 42 \
  2>&1 | tee "$OUT_DIR/train.node$NODE_RANK.log"
