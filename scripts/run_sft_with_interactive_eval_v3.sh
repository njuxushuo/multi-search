#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export SFT_GPUS="${SFT_GPUS:-0,1,2,3}"
export EVAL_GPUS="$SFT_GPUS"
BASE_OUT="outputs/v3/searchqa_repro_v3_0_0/sft_qwen35_4b_full"
QUICK_MANIFEST="data/processed/search_eval/searchqa_v3_interactive_dev_quick_1k_seed43_train_holdout.jsonl"
FULL_MANIFEST="data/processed/search_eval/searchqa_v3_interactive_dev_5k_seed43_train_holdout.jsonl"
DEV_INPUT="data/processed/nq_hotpotqa_train/train.parquet"
PYTHON_BIN="${EVAL_PYTHON:-/data3/xs/conda/envs/search-r1/bin/python}"

resume="${RESUME_FROM_CHECKPOINT:-}"
for step in 500 1000 1500; do
  checkpoint="$BASE_OUT/checkpoint-$step"
  if [[ ! -d "$checkpoint" ]]; then
    RESUME_FROM_CHECKPOINT="$resume" STOP_AFTER_STEP="$step" \
      bash scripts/run_sft_qwen35_4b_full_v3.sh
  fi
  [[ -d "$checkpoint" ]] || { echo "Missing expected checkpoint: $checkpoint" >&2; exit 2; }
  EVAL_INPUT="$DEV_INPUT" EVAL_MANIFEST="$QUICK_MANIFEST" EVAL_COUNT=1000 \
    bash scripts/run_eval_qwen35_search_v3_4gpu.sh "$checkpoint" "sft_checkpoint_${step}_dev1k"
  "$PYTHON_BIN" scripts/log_eval_summary_swanlab_v3.py \
    "outputs/v3/searchqa_repro_v3_0_0/eval/sft_checkpoint_${step}_dev1k.summary.json" \
    --step "$step"
  resume="$checkpoint"
done

RESUME_FROM_CHECKPOINT="$resume" STOP_AFTER_STEP="" bash scripts/run_sft_qwen35_4b_full_v3.sh
FINAL_MODEL="$BASE_OUT/final"
EVAL_INPUT="$DEV_INPUT" EVAL_MANIFEST="$FULL_MANIFEST" EVAL_COUNT=5000 \
  bash scripts/run_eval_qwen35_search_v3_4gpu.sh "$FINAL_MODEL" "sft_final_dev5k"
"$PYTHON_BIN" scripts/log_eval_summary_swanlab_v3.py \
  "outputs/v3/searchqa_repro_v3_0_0/eval/sft_final_dev5k.summary.json" \
  --step 1875 --run-name "R3.0-interactive-dev-final"
