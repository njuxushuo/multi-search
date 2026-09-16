#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SFT_GPUS="${SFT_GPUS:-0,1,2,3}"
IFS=',' read -r -a gpu_list <<< "$SFT_GPUS"
if [[ "${#gpu_list[@]}" -ne 4 ]]; then
  echo "SFT_GPUS must contain exactly four GPU ids: $SFT_GPUS" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$SFT_GPUS"
export TOKENIZERS_PARALLELISM=false
export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_API_KEY is required}"

for gpu in "${gpu_list[@]}"; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$used" -gt 1024 ]]; then
    echo "GPU $gpu is not idle (memory.used=${used} MiB); refusing to start R3.0 Full SFT" >&2
    exit 2
  fi
done

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/sft_qwen35_4b_full_v3.json}"
TRAIN_FILE="data/processed/searchqa_repro_v3_0_0/sft_tokenized/train.parquet"
EVAL_FILE="data/processed/searchqa_repro_v3_0_0/sft_tokenized/eval.parquet"
[[ -f "$TRAIN_FILE" && -f "$EVAL_FILE" ]] || { echo "Missing validated R3.0 tokenized dataset" >&2; exit 2; }
mkdir -p outputs/v3/searchqa_repro_v3_0_0/sft_qwen35_4b_full

PYTHON_BIN="${SFT_PYTHON:-/data3/xs/conda/envs/llamafactory/bin/python}"
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi
STOP_ARGS=()
if [[ -n "${STOP_AFTER_STEP:-}" ]]; then
  STOP_ARGS=(--stop-after-step "$STOP_AFTER_STEP")
fi
EPOCH_GATE_ARGS=()
if [[ "${APPROVE_SECOND_EPOCH:-0}" == "1" ]]; then
  EPOCH_GATE_ARGS=(--approve-second-epoch)
fi
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node 4 \
  scripts/train_sft_v3.py "$TRAIN_CONFIG" "${RESUME_ARGS[@]}" "${STOP_ARGS[@]}" "${EPOCH_GATE_ARGS[@]}" \
  2>&1 | tee -a outputs/v3/searchqa_repro_v3_0_0/sft_qwen35_4b_full/run.log
