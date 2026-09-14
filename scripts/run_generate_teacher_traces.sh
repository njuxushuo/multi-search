#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
TEACHER_GPUS="${TEACHER_GPUS:-3}"
export CUDA_VISIBLE_DEVICES="$TEACHER_GPUS"
export JAVA_HOME="${JAVA_HOME:-/data0/hcy/maple/jre.X86_64_LINUX}"
export PATH="$JAVA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
unset https_proxy http_proxy
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
# Strict answer/tool filtering can reject a large fraction of candidates; use
# a generous candidate pool so the exact 15k-train + 1k-eval gate can be met.
# Override with TEACHER_COUNT for smoke tests.
TEACHER_COUNT="${TEACHER_COUNT:-150000}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-128}"
TEACHER_MAX_ROUNDS="${TEACHER_MAX_ROUNDS:-8}"
TEACHER_MAX_SEARCHES="${TEACHER_MAX_SEARCHES:-100}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-768}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.9}"
TEACHER_SEED="${TEACHER_SEED:-42}"
TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}"
TEACHER_RESUME="${TEACHER_RESUME:-0}"
TEACHER_STOP_AT_SEARCH="${TEACHER_STOP_AT_SEARCH:-1}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-data/processed/teacher_traces_qwen35_27b.jsonl}"
TEACHER_SHARD_INDEX="${TEACHER_SHARD_INDEX:-0}"
TEACHER_NUM_SHARDS="${TEACHER_NUM_SHARDS:-1}"
TEACHER_LOG="${TEACHER_LOG:-outputs/teacher_trace_generation.log}"

IFS=',' read -r -a gpu_list <<< "$TEACHER_GPUS"
if [[ "${#gpu_list[@]}" -ne "$TEACHER_TP_SIZE" ]]; then
  echo "TEACHER_TP_SIZE ($TEACHER_TP_SIZE) must equal the number of TEACHER_GPUS (${#gpu_list[@]})" >&2
  exit 2
fi
for gpu in "${gpu_list[@]}"; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$used" -gt 1024 ]]; then
    echo "GPU $gpu is not idle (memory.used=${used} MiB); refusing to start" >&2
    exit 2
  fi
done

if compgen -G "models/Qwen3.5-27B/*.incomplete" > /dev/null; then
  echo "Qwen3.5-27B is still downloading; refusing to start teacher generation" >&2
  exit 2
fi
if [[ ! -f data/index/bm25/segments_1 ]]; then
  echo "BM25 index is not complete under data/index/bm25" >&2
  exit 2
fi

mkdir -p data/processed
conda run --no-capture-output -n search-r1 \
  python scripts/generate_teacher_traces.py \
    --teacher models/Qwen3.5-27B \
    --index data/index/bm25 \
    --output "$TEACHER_OUTPUT" \
    --count "$TEACHER_COUNT" \
    --batch-size "$TEACHER_BATCH_SIZE" \
    --max-rounds "$TEACHER_MAX_ROUNDS" \
    --max-searches "$TEACHER_MAX_SEARCHES" \
    --max-tokens "$TEACHER_MAX_TOKENS" \
    --temperature "$TEACHER_TEMPERATURE" \
    --top-p "$TEACHER_TOP_P" \
    --seed "$TEACHER_SEED" \
    --gpu-memory-utilization "$TEACHER_GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TEACHER_TP_SIZE" \
    --shard-index "$TEACHER_SHARD_INDEX" \
    --num-shards "$TEACHER_NUM_SHARDS" \
    $(if [[ "$TEACHER_RESUME" == "1" ]]; then echo --resume; fi) \
    $(if [[ "$TEACHER_STOP_AT_SEARCH" == "1" ]]; then echo --stop-at-search; fi) \
  2>&1 | tee "$TEACHER_LOG"
