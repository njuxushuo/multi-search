#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
GPUS="${TEACHER_GPUS:-0,1,2,3}"
IFS=',' read -r -a gpu_list <<< "$GPUS"
if [[ "${#gpu_list[@]}" -ne 4 ]]; then
  echo "TEACHER_GPUS must contain exactly four GPU ids: $GPUS" >&2
  exit 2
fi
QUESTION_COUNT="${TEACHER_QUESTION_COUNT:-2000}"
ROLLOUTS="${TEACHER_ROLLOUTS_PER_QUESTION:-4}"
OUT_DIR="${TEACHER_V3_OUT_DIR:-data/processed/searchqa_repro_v3_0_0/teacher_pilot}"
LOG_DIR="${TEACHER_V3_LOG_DIR:-outputs/v3/searchqa_repro_v3_0_0/teacher_pilot}"
MODEL="${TEACHER_MODEL:-models/Qwen3.5-27B}"
PILOT_MANIFEST="${TEACHER_PILOT_MANIFEST:-data/processed/searchqa_repro_v3_0_0/teacher_pilot_manifest_2k_seed42.jsonl}"
RETRIEVER_URL="${TEACHER_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
PYTHON_BIN="${TEACHER_PYTHON:-/data3/xs/conda/envs/search-r1/bin/python}"
PROTOCOL_CONFIG="${TEACHER_PROTOCOL_CONFIG:-configs/protocols/searchqa_repro_v3_0_0.json}"

[[ -d "$MODEL" ]] || { echo "Missing Teacher model: $MODEL" >&2; exit 2; }
[[ -f "$PILOT_MANIFEST" && -f "${PILOT_MANIFEST%.jsonl}.metadata.json" ]] || {
  echo "Missing frozen R3.0 Teacher pilot manifest or metadata: $PILOT_MANIFEST" >&2
  exit 2
}
if ! curl -fsS -X POST "$RETRIEVER_URL" -H 'Content-Type: application/json' \
  -d '{"queries":["R3.0 preflight"],"topk":3,"return_scores":true}' >/dev/null; then
  echo "BM25 retrieval service is unavailable: $RETRIEVER_URL" >&2
  exit 2
fi
for gpu in "${gpu_list[@]}"; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$used" -gt 1024 ]]; then
    echo "GPU $gpu is not idle (memory.used=${used} MiB); refusing to start R3.0 Teacher" >&2
    exit 2
  fi
done
"$PYTHON_BIN" scripts/verify_protocol_v3.py --protocol-config "$PROTOCOL_CONFIG" --tokenizer-check >/dev/null
mkdir -p "$OUT_DIR" "$LOG_DIR"

pids=()
for shard in 0 1 2 3; do
  gpu="${gpu_list[$shard]}"
  output="$OUT_DIR/shard${shard}.jsonl"
  log="$LOG_DIR/shard${shard}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
    scripts/generate_teacher_traces_v3.py \
    --protocol-config "$PROTOCOL_CONFIG" \
    --teacher "$MODEL" --pilot-manifest "$PILOT_MANIFEST" \
    --question-count "$QUESTION_COUNT" \
    --rollouts-per-question "$ROLLOUTS" \
    --shard-index "$shard" --num-shards 4 \
    --batch-size "${TEACHER_BATCH_SIZE:-32}" \
    --gpu-memory-utilization "${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}" \
    --retriever-url "$RETRIEVER_URL" \
    --output "$output" --resume \
    >"$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
