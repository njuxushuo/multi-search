#!/usr/bin/env bash
set -euo pipefail

# Launch four independent TP1 workers on four explicitly selected idle GPUs.
# Existing shard files are resumed and later merged by watch_teacher_then_sft.sh.
cd "$(dirname "$0")/.."
GPUS="${TEACHER_GPUS_4WAY:-0,1,2,3}"
COUNT="${TEACHER_COUNT_4WAY:-150000}"
SEED="${TEACHER_SEED_4WAY:-42}"
IFS=',' read -r -a gpu_list <<< "$GPUS"
if [[ "${#gpu_list[@]}" -ne 4 ]]; then
  echo "TEACHER_GPUS_4WAY must contain exactly four GPU ids: $GPUS" >&2
  exit 2
fi
for gpu in "${gpu_list[@]}"; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$used" -gt 1024 ]]; then
    echo "GPU $gpu is not idle (memory.used=${used} MiB); refusing to launch" >&2
    exit 2
  fi
done

for i in 0 1 2 3; do
  gpu="${gpu_list[$i]}"
  output="data/processed/teacher_traces_qwen35_27b.shard${i}.jsonl"
  log="outputs/teacher_trace_shard${i}.supervisor.log"
  setsid bash -c "env TEACHER_GPUS=$gpu TEACHER_TP_SIZE=1 TEACHER_NUM_SHARDS=4 TEACHER_SHARD_INDEX=$i TEACHER_OUTPUT=$output TEACHER_LOG=$log TEACHER_COUNT=$COUNT TEACHER_SEED=$SEED TEACHER_BATCH_SIZE=128 TEACHER_MAX_ROUNDS=8 TEACHER_MAX_SEARCHES=100 TEACHER_MAX_TOKENS=768 TEACHER_RESUME=1 TEACHER_STOP_AT_SEARCH=1 TEACHER_GPU_MEMORY_UTILIZATION=0.85 bash scripts/run_generate_teacher_traces.sh" \
    > "${log}.launcher" 2>&1 < /dev/null &
  echo "launched shard $i on GPU $gpu (pid=$!)"
done

echo "Teacher generation launched; start scripts/watch_teacher_then_sft.sh once all workers are visible."
