#!/usr/bin/env bash
set -euo pipefail

# Evaluate one model at a time with four TP=1 data-parallel workers on GPUs
# 0,1,2,3. Each worker appends and fsyncs one 32-example batch.
cd "$(dirname "$0")/.."
COUNT="${EVAL_COUNT:-50000}"
LABEL="${EVAL_LABEL:-50k_docnovelty}"
MANIFEST="${EVAL_MANIFEST:-data/processed/search_eval/searchqa_50k_seed42_hnb_first.jsonl}"
SOURCE_ORDER="hotpotqa,nq,bamboogle"
RETRIEVER_URL="${EVAL_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"

BASE="/data3/xs/models/Qwen3.5-4B-Base"
LORA="/data0/xs/search/outputs/sft_qwen35_4b_lora_multiturn_masked_merged"
FULL="/data0/xs/search/outputs/sft_qwen35_4b_full_multiturn_masked"
TEACHER="/data0/xs/search/models/Qwen3.5-27B"
START_AT="${EVAL_START_AT:-base}"
RUN_TEACHER="${EVAL_RUN_TEACHER:-0}"

for path in "$BASE" "$LORA" "$FULL"; do
  [[ -d "$path" ]] || { echo "Missing model: $path" >&2; exit 2; }
done
if [[ "$RUN_TEACHER" == "1" ]]; then
  [[ -d "$TEACHER" ]] || { echo "Missing model: $TEACHER" >&2; exit 2; }
fi
for gpu in 0 1 2 3; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "$used" -le 1024 ]] || { echo "GPU $gpu is busy (${used} MiB)" >&2; exit 2; }
done

conda run --no-capture-output -n search-r1 python scripts/evaluate_qwen35_search.py \
  --prepare-manifest-only \
  --input data/processed/nq_hotpotqa_train/test.parquet \
  --manifest "$MANIFEST" --count "$COUNT" --seed 42 \
  --source-order "$SOURCE_ORDER"

if ! curl -fsS -X POST "$RETRIEVER_URL" -H 'Content-Type: application/json' \
  -d '{"queries":["test"],"topk":1,"return_scores":true}' >/dev/null 2>&1; then
  echo "BM25 retrieval service is unavailable: $RETRIEVER_URL" >&2
  exit 2
fi

mkdir -p outputs/eval
run_model() {
  local run_id="$1" model="$2"
  local output="outputs/eval/${run_id}_${LABEL}.jsonl" status=0
  local pids=() shards=()
  echo "[$(date -Is)] starting $run_id with four data-parallel workers"
  for gpu in 0 1 2 3; do
    shard="outputs/eval/${run_id}_${LABEL}.shard${gpu}.jsonl"
    shards+=("$shard")
    env EVAL_GPU="$gpu" EVAL_TP_SIZE=1 EVAL_COUNT="$COUNT" \
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}" EVAL_MANIFEST="$MANIFEST" \
      EVAL_RETRIEVER_URL="$RETRIEVER_URL" EVAL_RUN_ID="${run_id}_shard${gpu}" \
      EVAL_MODEL="$model" EVAL_OUTPUT="$shard" \
      EVAL_SHARD_INDEX="$gpu" EVAL_NUM_SHARDS=4 \
      bash scripts/run_eval_qwen35_search.sh &
    pids+=("$!")
  done
  cleanup_workers() {
    for pid in "${pids[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
    wait "${pids[@]}" 2>/dev/null || true
  }
  trap cleanup_workers INT TERM
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  trap - INT TERM
  [[ "$status" -eq 0 ]] || return "$status"
  conda run --no-capture-output -n search-r1 python scripts/merge_eval_shards.py \
    --manifest "$MANIFEST" --output "$output" "${shards[@]}"
  echo "[$(date -Is)] completed $run_id"
}

case "$START_AT" in
  base)
    run_model qwen35_4b_base "$BASE"
    run_model qwen35_4b_lora "$LORA"
    run_model qwen35_4b_full "$FULL"
    [[ "$RUN_TEACHER" == "1" ]] && run_model qwen35_27b_teacher "$TEACHER"
    ;;
  lora)
    run_model qwen35_4b_lora "$LORA"
    run_model qwen35_4b_full "$FULL"
    [[ "$RUN_TEACHER" == "1" ]] && run_model qwen35_27b_teacher "$TEACHER"
    ;;
  full)
    run_model qwen35_4b_full "$FULL"
    [[ "$RUN_TEACHER" == "1" ]] && run_model qwen35_27b_teacher "$TEACHER"
    ;;
  teacher)
    run_model qwen35_27b_teacher "$TEACHER"
    ;;
  *)
    echo "Invalid EVAL_START_AT=$START_AT (expected base, lora, full, or teacher)" >&2
    exit 2
    ;;
esac
