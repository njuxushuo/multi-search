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
mkdir -p "$OUT_DIR" outputs/v3/searchqa_repro_v3_0_0/teacher_pilot

pids=()
for shard in 0 1 2 3; do
  gpu="${gpu_list[$shard]}"
  output="$OUT_DIR/shard${shard}.jsonl"
  log="outputs/v3/searchqa_repro_v3_0_0/teacher_pilot/shard${shard}.log"
  CUDA_VISIBLE_DEVICES="$gpu" /data3/xs/conda/envs/search-r1/bin/python \
    scripts/generate_teacher_traces_v3.py \
    --question-count "$QUESTION_COUNT" \
    --rollouts-per-question "$ROLLOUTS" \
    --shard-index "$shard" --num-shards 4 \
    --batch-size "${TEACHER_BATCH_SIZE:-32}" \
    --gpu-memory-utilization "${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}" \
    --output "$output" --resume \
    >"$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
