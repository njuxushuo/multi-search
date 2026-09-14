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
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#gpu_list[@]}}"
export TOKENIZERS_PARALLELISM=false
export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_API_KEY is required}"
unset https_proxy http_proxy

for gpu in "${gpu_list[@]}"; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$used" -gt 1024 ]]; then
    echo "GPU $gpu is not idle (memory.used=${used} MiB); refusing to start full SFT" >&2
    exit 2
  fi
done

DATASET_DIR="data/processed/search_sft_qwen35_4b"
[[ -f "$DATASET_DIR/train.jsonl" && -f "$DATASET_DIR/eval.jsonl" ]] || { echo "Missing validated teacher dataset" >&2; exit 2; }
[[ "$(wc -l < "$DATASET_DIR/train.jsonl")" -eq 15000 ]] || { echo "Expected exactly 15000 train traces" >&2; exit 2; }
mkdir -p outputs/sft_qwen35_4b_full_multiturn_masked

LF_SRC="${LF_SRC:-/data0/xs/LLaMA-Factory/src}"
DS_SITE="${DS_SITE:-/data3/xs/conda/envs/llamafactory/lib/python3.11/site-packages}"
LF_PY="${LF_PY:-/data3/xs/conda/envs/llamafactory/bin/python}"
PYTHONPATH="$LF_SRC:$DS_SITE${PYTHONPATH:+:$PYTHONPATH}" \
  "$LF_PY" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
  scripts/run_llamafactory_ddp_full_swanlab.py configs/sft_qwen35_4b_full.yaml \
  2>&1 | tee -a outputs/sft_qwen35_4b_full_multiturn_masked/run.log
