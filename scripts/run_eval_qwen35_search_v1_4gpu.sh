#!/usr/bin/env bash
set -euo pipefail

# Run exactly one model under v1 with four TP=1 data-parallel workers.
# Existing v0 results are read-only inputs; every v1 artifact is isolated under
# outputs/eval/v1/ so a future v2 runner can use outputs/eval/v2/ independently.
cd "$(dirname "$0")/.."
MODEL_PATH="${EVAL_MODEL:?EVAL_MODEL is required}"
RUN_ID="${EVAL_RUN_ID:?EVAL_RUN_ID is required}"
V0_RESULTS="${EVAL_V0_RESULTS:-}"
COUNT="${EVAL_COUNT:-50000}"
MANIFEST="${EVAL_MANIFEST:-data/processed/search_eval/searchqa_50k_seed42_hnb_first.jsonl}"
RETRIEVER_URL="${EVAL_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
OUTPUT="outputs/eval/v1/${RUN_ID}_50k.jsonl"

[[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 2; }
[[ -z "$V0_RESULTS" || -f "$V0_RESULTS" ]] || { echo "Missing v0 results: $V0_RESULTS" >&2; exit 2; }
for gpu in 0 1 2 3; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "$used" -le 1024 ]] || { echo "GPU $gpu is busy (${used} MiB)" >&2; exit 2; }
done

conda run --no-capture-output -n search-r1 python scripts/evaluate_qwen35_search_v1.py \
  --prepare-manifest-only \
  --input data/processed/nq_hotpotqa_train/test.parquet \
  --manifest "$MANIFEST" --count "$COUNT" --seed 42 \
  --source-order hotpotqa,nq,bamboogle

if ! curl -fsS -X POST "$RETRIEVER_URL" -H 'Content-Type: application/json' \
  -d '{"queries":["test"],"topk":1,"return_scores":true}' >/dev/null 2>&1; then
  echo "BM25 retrieval service is unavailable: $RETRIEVER_URL" >&2
  exit 2
fi

mkdir -p outputs/eval/v1
pids=()
shards=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait "${pids[@]}" 2>/dev/null || true
}
trap cleanup INT TERM
for gpu in 0 1 2 3; do
  shard="outputs/eval/v1/${RUN_ID}_50k.shard${gpu}.jsonl"
  shards+=("$shard")
  env EVAL_GPU="$gpu" EVAL_TP_SIZE=1 EVAL_COUNT="$COUNT" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}" EVAL_MANIFEST="$MANIFEST" \
    EVAL_RETRIEVER_URL="$RETRIEVER_URL" EVAL_RUN_ID="${RUN_ID}_shard${gpu}" \
    EVAL_MODEL="$MODEL_PATH" EVAL_OUTPUT="$shard" EVAL_V0_RESULTS="$V0_RESULTS" \
    EVAL_SHARD_INDEX="$gpu" EVAL_NUM_SHARDS=4 \
    bash scripts/run_eval_qwen35_search_v1.sh &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
trap - INT TERM
[[ "$status" -eq 0 ]] || exit "$status"

conda run --no-capture-output -n search-r1 python scripts/merge_eval_shards_v1.py \
  --manifest "$MANIFEST" --output "$OUTPUT" "${shards[@]}"
