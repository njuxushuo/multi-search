#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
log="outputs/lora_to_full_sft_watcher.log"
mkdir -p "$(dirname "$log")"
exec > >(tee -a "$log") 2>&1

echo "[$(date -Is)] waiting for LoRA SFT to finish"
while pgrep -f 'run_llamafactory_ddp_with_swanlab.py configs/sft_qwen35_4b_lora.yaml' >/dev/null; do
  sleep 300
done

# Do not silently launch full SFT after a failed or manually aborted LoRA run.
if [[ ! -f outputs/sft_qwen35_4b_lora_multiturn_masked/trainer_state.json ]] || \
   ! grep -q 'train_loss' outputs/sft_qwen35_4b_lora_multiturn_masked/trainer_state.json; then
  echo "[$(date -Is)] LoRA SFT did not produce a successful trainer_state; refusing full SFT" >&2
  exit 3
fi

free_kb=$(df --output=avail /data0 | tail -n 1 | tr -d ' ')
if [[ "$free_kb" -lt 100000000 ]]; then
  echo "[$(date -Is)] insufficient /data0 free space (${free_kb} KiB); refusing full SFT" >&2
  exit 4
fi

echo "[$(date -Is)] LoRA succeeded; launching four-GPU full SFT"
export SFT_GPUS="${SFT_GPUS:-0,1,2,3}"
exec setsid bash scripts/run_sft_qwen35_4b_full.sh
