#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SFT_GPUS="${SFT_GPUS:-0,1,2,3}"
IFS=',' read -r -a sft_gpu_list <<< "$SFT_GPUS"
if [[ "${#sft_gpu_list[@]}" -ne 4 ]]; then
  echo "SFT_GPUS must contain exactly four GPU ids: $SFT_GPUS" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$SFT_GPUS"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#sft_gpu_list[@]}}"
export TOKENIZERS_PARALLELISM=false
export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_API_KEY is required}"
# The lowercase proxy inherited by the shell is stale; uppercase values are
# the working route used by SwanLab and ModelScope in this machine.
unset https_proxy http_proxy
for gpu in "${sft_gpu_list[@]}"; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$used" -gt 1024 ]]; then
    echo "GPU $gpu is not idle (memory.used=${used} MiB); refusing to start SFT" >&2
    exit 2
  fi
done

DATASET_DIR="data/processed/search_sft_qwen35_4b"
if [[ ! -f "$DATASET_DIR/train.jsonl" || ! -f "$DATASET_DIR/eval.jsonl" ]]; then
  echo "Missing validated teacher dataset under $DATASET_DIR" >&2
  exit 2
fi
TRAIN_COUNT=$(wc -l < "$DATASET_DIR/train.jsonl")
if [[ "$TRAIN_COUNT" -ne 15000 ]]; then
  echo "Expected exactly 15000 SFT traces, found $TRAIN_COUNT" >&2
  exit 2
fi
mkdir -p outputs/sft_qwen35_4b_lora_multiturn_masked

conda run --no-capture-output -n llamafactory \
  python -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
  scripts/run_llamafactory_ddp_with_swanlab.py configs/sft_qwen35_4b_lora.yaml \
  2>&1 | tee outputs/sft_qwen35_4b_lora_multiturn_masked/run.log
