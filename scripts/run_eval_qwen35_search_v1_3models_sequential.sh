#!/usr/bin/env bash
set -euo pipefail

# Evaluate the three local 4B checkpoints sequentially under the frozen v1
# protocol. Each model uses GPUs 0,1,2,3 as four TP=1 data-parallel workers.
cd "$(dirname "$0")/.."

run_one() {
  local run_id="$1" model="$2" v0_results="$3"
  echo "[$(date -Is)] starting ${run_id}"
  env \
    EVAL_MODEL="$model" \
    EVAL_RUN_ID="$run_id" \
    EVAL_V0_RESULTS="$v0_results" \
    EVAL_COUNT="${EVAL_COUNT:-50000}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}" \
    EVAL_MANIFEST="${EVAL_MANIFEST:-data/processed/search_eval/searchqa_50k_seed42_hnb_first.jsonl}" \
    EVAL_RETRIEVER_URL="${EVAL_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}" \
    bash scripts/run_eval_qwen35_search_v1_4gpu.sh
  echo "[$(date -Is)] completed ${run_id}"
}

run_one \
  qwen35_4b_base_v1_neutral \
  /data3/xs/models/Qwen3.5-4B-Base \
  outputs/eval/qwen35_4b_base_50k_docnovelty.jsonl

run_one \
  qwen35_4b_lora_v1_neutral \
  /data0/xs/search/outputs/sft_qwen35_4b_lora_multiturn_masked_merged \
  outputs/eval/qwen35_4b_lora_50k_docnovelty.jsonl

run_one \
  qwen35_4b_full_v1_neutral \
  /data0/xs/search/outputs/sft_qwen35_4b_full_multiturn_masked \
  outputs/eval/qwen35_4b_full_50k_docnovelty.jsonl
