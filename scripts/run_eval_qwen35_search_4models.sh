#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
COUNT="${EVAL_COUNT:-50000}"
LABEL="${EVAL_LABEL:-50k}"
MANIFEST="${EVAL_MANIFEST:-data/processed/search_eval/searchqa_50k_seed42.jsonl}"
BASE="/data3/xs/models/Qwen3.5-4B-Base"
LORA="/data0/xs/search/outputs/sft_qwen35_4b_lora_multiturn_masked_merged"
FULL="/data0/xs/search/outputs/sft_qwen35_4b_full_multiturn_masked"
TEACHER="/data0/xs/search/models/Qwen3.5-27B"
RETRIEVER_URL="${EVAL_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"

for path in "$BASE" "$LORA" "$FULL" "$TEACHER"; do
  [[ -d "$path" ]] || { echo "Missing model: $path" >&2; exit 2; }
done
for gpu in 0 1 2 3; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "$used" -le 1024 ]] || { echo "GPU $gpu is busy (${used} MiB)" >&2; exit 2; }
done

conda run --no-capture-output -n search-r1 python scripts/evaluate_qwen35_search.py \
  --prepare-manifest-only --input data/processed/nq_hotpotqa_train/test.parquet \
  --manifest "$MANIFEST" --count "$COUNT" --seed 42

mkdir -p outputs/eval
pids=()
server_pid=""
if ! curl -fsS -X POST "$RETRIEVER_URL" -H 'Content-Type: application/json' \
  -d '{"queries":["test"],"topk":1,"return_scores":true}' >/dev/null 2>&1; then
  bash scripts/run_bm25_server.sh > outputs/eval/bm25_server.log 2>&1 &
  server_pid="$!"
  for _ in $(seq 1 120); do
    if curl -fsS -X POST "$RETRIEVER_URL" -H 'Content-Type: application/json' \
      -d '{"queries":["test"],"topk":1,"return_scores":true}' >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi
curl -fsS -X POST "$RETRIEVER_URL" -H 'Content-Type: application/json' \
  -d '{"queries":["test"],"topk":1,"return_scores":true}' >/dev/null
cleanup() {
  [[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

launch() {
  local gpu="$1" run_id="$2" model="$3" adapter="${4:-}"
  local adapter_env=()
  [[ -z "$adapter" ]] || adapter_env+=(EVAL_LORA_ADAPTER="$adapter")
  env EVAL_GPU="$gpu" EVAL_COUNT="$COUNT" EVAL_MANIFEST="$MANIFEST" \
    EVAL_RETRIEVER_URL="$RETRIEVER_URL" \
    EVAL_RUN_ID="$run_id" EVAL_MODEL="$model" "${adapter_env[@]}" \
    EVAL_OUTPUT="outputs/eval/${run_id}_${LABEL}.jsonl" \
    bash scripts/run_eval_qwen35_search.sh > "outputs/eval/${run_id}_${LABEL}_launcher.log" 2>&1 &
  pids+=("$!")
  echo "$! $run_id GPU=$gpu"
}

launch 0 qwen35_4b_base "$BASE"
launch 1 qwen35_4b_lora "$LORA"
launch 2 qwen35_4b_full "$FULL"
launch 3 qwen35_27b_teacher "$TEACHER"
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
