#!/usr/bin/env bash
set -euo pipefail

# Wait for both parallel Qwen3.5-27B generators, then merge/revalidate and
# launch the exact 4B SFT job. This watcher never touches a GPU while the
# teacher processes are alive.
cd "$(dirname "$0")/.."
log="${WATCH_LOG:-outputs/teacher_to_sft_watcher.log}"
mkdir -p "$(dirname "$log")"
exec > >(tee -a "$log") 2>&1

echo "[$(date -Is)] waiting for teacher shards"
while pgrep -f 'generate_teacher_traces.py .*teacher_traces_qwen35_27b\.shard[0-3]\.jsonl' >/dev/null; do
  sleep 60
done

shopt -s nullglob
shards=(data/processed/teacher_traces_qwen35_27b.shard*.jsonl)
if ((${#shards[@]} < 4)); then
  echo "teacher shards are missing (expected 4, found ${#shards[@]})" >&2
  exit 2
fi

merged="data/processed/teacher_traces_qwen35_27b.merged.jsonl"
conda run --no-capture-output -n search-r1 python scripts/merge_teacher_shards.py \
  --output "$merged" \
  data/processed/teacher_traces_qwen35_27b.jsonl "${shards[@]}"

mkdir -p data/processed/search_sft_qwen35_4b
if ! conda run --no-capture-output -n search-r1 python scripts/validate_teacher_traces.py \
  "$merged" --output-dir data/processed/search_sft_qwen35_4b; then
  # A single stochastic pass may not yield the required 15k+1k strict
  # trajectories.  Retry the previously rejected questions with a new
  # deterministic shuffle/seed, preserving all accepted records and the
  # exact model/search/format protocol.
  if [[ "${TEACHER_RETRY_ON_INSUFFICIENT:-1}" != "1" ]]; then
    echo "strict trajectory gate not met and retry disabled" >&2
    exit 3
  fi
  echo "[$(date -Is)] strict gate not met; launching one supplemental seed pass"
  TEACHER_COUNT_4WAY="${TEACHER_RETRY_COUNT:-169615}" \
  TEACHER_SEED_4WAY="${TEACHER_RETRY_SEED:-43}" \
  TEACHER_GPUS_4WAY="${TEACHER_GPUS_4WAY:-0,1,2,3}" \
    setsid bash -c 'bash scripts/run_generate_teacher_traces_4way.sh' \
      > outputs/teacher_4way_retry_launcher.log 2>&1 < /dev/null &
  while pgrep -f 'generate_teacher_traces.py .*teacher_traces_qwen35_27b\.shard[0-3]\.jsonl' >/dev/null; do
    sleep 60
  done
  shards=(data/processed/teacher_traces_qwen35_27b.shard*.jsonl)
  conda run --no-capture-output -n search-r1 python scripts/merge_teacher_shards.py \
    --output "$merged" \
    data/processed/teacher_traces_qwen35_27b.jsonl "${shards[@]}"
  conda run --no-capture-output -n search-r1 python scripts/validate_teacher_traces.py \
    "$merged" --output-dir data/processed/search_sft_qwen35_4b
fi

# The SFT wrapper has its own four-GPU idle guard. Wait for those cards to become idle
# instead of racing any unrelated workload.
while true; do
  all_idle=1
  IFS=',' read -r -a sft_gpu_list <<< "${SFT_GPUS:-0,1,2,3}"
  for gpu in "${sft_gpu_list[@]}"; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$used" -gt 1024 ]]; then
      all_idle=0
      echo "[$(date -Is)] GPU $gpu busy (${used} MiB); retrying in 60s"
    fi
  done
  if [[ "$all_idle" -eq 1 ]]; then
    break
  fi
  sleep 60
done

exec scripts/run_sft_qwen35_4b_lora.sh
