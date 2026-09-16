#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MODEL="${1:?Usage: $0 MODEL OUTPUT_NAME}"
NAME="${2:?Usage: $0 MODEL OUTPUT_NAME}"
GPUS="${EVAL_GPUS:-0,1,2,3}"
IFS=',' read -r -a gpu_list <<< "$GPUS"
if [[ "${#gpu_list[@]}" -ne 4 ]]; then
  echo "EVAL_GPUS must contain exactly four GPU ids: $GPUS" >&2
  exit 2
fi
OUT_DIR="outputs/v3/searchqa_repro_v3_0_0/eval"
INPUT="${EVAL_INPUT:-data/processed/nq_hotpotqa_train/test.parquet}"
MANIFEST="${EVAL_MANIFEST:-data/processed/search_eval/searchqa_50k_seed42_hnb_first.jsonl}"
mkdir -p "$OUT_DIR"
pids=()
for shard in 0 1 2 3; do
  output="$OUT_DIR/${NAME}.shard${shard}.jsonl"
  log="$OUT_DIR/${NAME}.shard${shard}.log"
  CUDA_VISIBLE_DEVICES="${gpu_list[$shard]}" /data3/xs/conda/envs/search-r1/bin/python \
    scripts/evaluate_qwen35_search_v3.py \
    --model "$MODEL" --input "$INPUT" --manifest "$MANIFEST" --output "$output" \
    --count "${EVAL_COUNT:-50000}" \
    --shard-index "$shard" --num-shards 4 \
    --batch-size "${EVAL_BATCH_SIZE:-32}" --resume \
    >"$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -eq 0 ]]; then
  /data3/xs/conda/envs/search-r1/bin/python scripts/merge_eval_shards_v3.py \
    --prefix "$OUT_DIR/$NAME" --num-shards 4 --count "${EVAL_COUNT:-50000}"
fi
exit "$status"
