#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MODEL_PATH="${EVAL_MODEL:-/data3/xs/models/Qwen3.5-4B-Base}"
LORA_ADAPTER="${EVAL_LORA_ADAPTER:-}"
RUN_ID="${EVAL_RUN_ID:-$(basename "$MODEL_PATH")}"
OUTPUT_PATH="${EVAL_OUTPUT:-outputs/eval/${RUN_ID}_50k.jsonl}"
GPU_IDS="${EVAL_GPUS:-${EVAL_GPU:-3}}"
TP_SIZE="${EVAL_TP_SIZE:-1}"
EVAL_COUNT="${EVAL_COUNT:-50000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_MANIFEST="${EVAL_MANIFEST:-data/processed/search_eval/searchqa_50k_seed42.jsonl}"
EVAL_RETRIEVER_URL="${EVAL_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export JAVA_HOME="${JAVA_HOME:-/data0/hcy/maple/jre.X86_64_LINUX}"
export PATH="$JAVA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
unset https_proxy http_proxy

IFS=',' read -r -a gpu_list <<< "$GPU_IDS"
if [[ "${#gpu_list[@]}" -ne "$TP_SIZE" ]]; then
  echo "EVAL_TP_SIZE ($TP_SIZE) must equal the number of EVAL_GPUS (${#gpu_list[@]})" >&2
  exit 2
fi
for gpu in "${gpu_list[@]}"; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$used" -gt 1024 ]]; then
    echo "GPU $gpu is not idle (memory.used=${used} MiB); refusing to start evaluation" >&2
    exit 2
  fi
done
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path does not exist: $MODEL_PATH" >&2
  exit 2
fi

extra=()
if [[ -n "$LORA_ADAPTER" ]]; then
  extra+=(--lora-adapter "$LORA_ADAPTER")
fi
mkdir -p "$(dirname "$OUTPUT_PATH")"
exec conda run --no-capture-output -n search-r1 python scripts/evaluate_qwen35_search.py \
  --model "$MODEL_PATH" "${extra[@]}" \
  --input data/processed/nq_hotpotqa_train/test.parquet \
  --manifest "$EVAL_MANIFEST" --resume \
  --retriever-url "$EVAL_RETRIEVER_URL" \
  --index data/index/bm25 --corpus data/corpus/wiki-18.jsonl \
  --output "$OUTPUT_PATH" --count "$EVAL_COUNT" --topk 3 --max-rounds 8 \
  --max-searches 8 --max-tokens 768 --temperature 0 --top-p 1 \
  --batch-size "$EVAL_BATCH_SIZE" --tensor-parallel-size "$TP_SIZE" --seed 42 \
  --shard-index "$EVAL_SHARD_INDEX" --num-shards "$EVAL_NUM_SHARDS" \
  2>&1 | tee "${OUTPUT_PATH%.jsonl}.log"
