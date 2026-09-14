#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="3"
export TOKENIZERS_PARALLELISM=false
export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_API_KEY is required}"
# The inherited lower-case proxy points to a dead endpoint; keep the working
# upper-case HTTPS_PROXY/HTTP_PROXY values for SwanLab uploads.
unset https_proxy http_proxy
mkdir -p outputs/sft_qwen35_2b_lora_pilot

conda run --no-capture-output -n llamafactory \
  python scripts/run_llamafactory_with_swanlab.py \
  2>&1 | tee outputs/sft_qwen35_2b_lora_pilot/run.log
