#!/usr/bin/env bash
set -euo pipefail

# Finish the already-running Teacher v0 workers, replace the legacy
# Teacher->Full supervisor, and run only Teacher v1 with the frozen protocol.
cd "$(dirname "$0")/.."

LEGACY_SUPERVISOR_PID="${LEGACY_SUPERVISOR_PID:?LEGACY_SUPERVISOR_PID is required}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
MANIFEST="${EVAL_MANIFEST:-data/processed/search_eval/searchqa_50k_seed42_hnb_first.jsonl}"
V0_PREFIX="outputs/eval/qwen35_27b_teacher_50k_docnovelty"
V0_MERGED="${V0_PREFIX}.jsonl"
V1_RUN_ID="qwen35_27b_teacher_v1_neutral"
LOCK_FILE="outputs/eval/.teacher_v0_to_v1.lock"

mkdir -p outputs/eval/v1
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Teacher v0->v1 handoff is already running" >&2; exit 2; }

log() {
  echo "[$(date -Is)] $*"
}

if kill -0 "$LEGACY_SUPERVISOR_PID" 2>/dev/null; then
  legacy_command=$(ps -o args= -p "$LEGACY_SUPERVISOR_PID")
  case "$legacy_command" in
    *outputs/eval/run_teacher_full_4gpu.sh*) ;;
    *)
      echo "PID $LEGACY_SUPERVISOR_PID is not the legacy Teacher/Full supervisor: $legacy_command" >&2
      exit 2
      ;;
  esac
  kill -STOP "$LEGACY_SUPERVISOR_PID"
  log "legacy supervisor $LEGACY_SUPERVISOR_PID is stopped; Teacher v0 workers continue"
fi

while pgrep -f '[e]valuate_qwen35_search.py --model models/Qwen3.5-27B' >/dev/null; do
  completed=0
  for shard in "${V0_PREFIX}".shard{0,1,2,3}.jsonl; do
    if [[ -f "$shard" ]]; then
      rows=$(wc -l < "$shard")
      if (( rows > 0 )); then
        rows=$((rows - 1))
      fi
      completed=$((completed + rows))
    fi
  done
  log "waiting for Teacher v0 workers: ${completed}/50000 rows persisted"
  sleep 60
done

# The stopped supervisor cannot enter its Full SFT command. Once all of its
# Teacher workers are gone, discard only that shell and take over the merge.
if kill -0 "$LEGACY_SUPERVISOR_PID" 2>/dev/null; then
  kill -KILL "$LEGACY_SUPERVISOR_PID"
  log "removed stopped legacy supervisor; Full SFT will not be launched"
fi

shards=(
  "${V0_PREFIX}.shard0.jsonl"
  "${V0_PREFIX}.shard1.jsonl"
  "${V0_PREFIX}.shard2.jsonl"
  "${V0_PREFIX}.shard3.jsonl"
)
for shard in "${shards[@]}"; do
  [[ -f "$shard" ]] || { echo "Missing Teacher v0 shard: $shard" >&2; exit 2; }
done

log "strictly merging Teacher v0 shards"
"$CONDA_BIN" run --no-capture-output -n search-r1 \
  python scripts/merge_eval_shards.py \
  --manifest "$MANIFEST" --output "$V0_MERGED" "${shards[@]}"

log "validating the frozen v1 protocol tests"
"$CONDA_BIN" run --no-capture-output -n search-r1 \
  python -m unittest tests/test_eval_protocol_v1.py

while ! nvidia-smi -i 0,1,2,3 --query-gpu=memory.used \
  --format=csv,noheader,nounits | awk '$1 + 0 > 1024 { exit 1 }'; do
  log "waiting for GPUs 0-3 to release Teacher v0 memory"
  sleep 15
done

log "starting Teacher v1 with four TP=1 data-parallel workers"
exec env \
  EVAL_MODEL="models/Qwen3.5-27B" \
  EVAL_RUN_ID="$V1_RUN_ID" \
  EVAL_V0_RESULTS="$V0_MERGED" \
  EVAL_COUNT=50000 \
  EVAL_BATCH_SIZE=32 \
  EVAL_MANIFEST="$MANIFEST" \
  EVAL_RETRIEVER_URL="${EVAL_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}" \
  bash scripts/run_eval_qwen35_search_v1_4gpu.sh
