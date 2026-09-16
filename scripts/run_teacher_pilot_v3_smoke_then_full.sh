#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${TEACHER_PYTHON:-/data3/xs/conda/envs/search-r1/bin/python}"
SMOKE_QUESTIONS="${TEACHER_SMOKE_QUESTIONS:-64}"
FULL_QUESTIONS="${TEACHER_QUESTION_COUNT:-2000}"
ROLLOUTS="${TEACHER_ROLLOUTS_PER_QUESTION:-4}"
ROOT_OUT="data/processed/searchqa_repro_v3_0_0"
ROOT_LOG="outputs/v3/searchqa_repro_v3_0_0"

"$PYTHON_BIN" scripts/prepare_teacher_pilot_manifest_v3.py
"$PYTHON_BIN" scripts/verify_protocol_v3.py --tokenizer-check

TEACHER_QUESTION_COUNT="$SMOKE_QUESTIONS" \
TEACHER_ROLLOUTS_PER_QUESTION="$ROLLOUTS" \
TEACHER_V3_OUT_DIR="$ROOT_OUT/teacher_smoke_${SMOKE_QUESTIONS}x${ROLLOUTS}" \
TEACHER_V3_LOG_DIR="$ROOT_LOG/teacher_smoke_${SMOKE_QUESTIONS}x${ROLLOUTS}" \
  bash scripts/run_generate_teacher_traces_v3_4way.sh

"$PYTHON_BIN" scripts/audit_teacher_pilot_v3.py \
  "$ROOT_OUT/teacher_smoke_${SMOKE_QUESTIONS}x${ROLLOUTS}"/shard{0,1,2,3}.jsonl \
  --question-count "$SMOKE_QUESTIONS" --rollouts-per-question "$ROLLOUTS" \
  --output "$ROOT_LOG/teacher_smoke_${SMOKE_QUESTIONS}x${ROLLOUTS}/audit.json" \
  --operational-gate

TEACHER_QUESTION_COUNT="$FULL_QUESTIONS" \
TEACHER_ROLLOUTS_PER_QUESTION="$ROLLOUTS" \
TEACHER_V3_OUT_DIR="$ROOT_OUT/teacher_pilot_${FULL_QUESTIONS}x${ROLLOUTS}" \
TEACHER_V3_LOG_DIR="$ROOT_LOG/teacher_pilot_${FULL_QUESTIONS}x${ROLLOUTS}" \
  bash scripts/run_generate_teacher_traces_v3_4way.sh

"$PYTHON_BIN" scripts/audit_teacher_pilot_v3.py \
  "$ROOT_OUT/teacher_pilot_${FULL_QUESTIONS}x${ROLLOUTS}"/shard{0,1,2,3}.jsonl \
  --question-count "$FULL_QUESTIONS" --rollouts-per-question "$ROLLOUTS" \
  --output "$ROOT_LOG/teacher_pilot_${FULL_QUESTIONS}x${ROLLOUTS}/audit.json" \
  --operational-gate
