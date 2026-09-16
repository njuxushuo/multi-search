#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
EVAL_PID_FILE="${EVAL_PID_FILE:-outputs/eval/v1/qwen35_4b_posttrained_v1_neutral_launcher.pid}"
EVAL_SUMMARY="${EVAL_SUMMARY:-outputs/eval/v1/qwen35_4b_posttrained_v1_neutral_50k.summary.json}"
PYTHON_BIN="${TEACHER_PYTHON:-/data3/xs/conda/envs/search-r1/bin/python}"

[[ -f "$EVAL_PID_FILE" ]] || { echo "Missing evaluation PID file: $EVAL_PID_FILE" >&2; exit 2; }
eval_pid=$(tr -d '[:space:]' <"$EVAL_PID_FILE")
[[ "$eval_pid" =~ ^[0-9]+$ ]] || { echo "Invalid evaluation PID: $eval_pid" >&2; exit 2; }
echo "Waiting for v1 evaluation launcher PID $eval_pid"
while kill -0 "$eval_pid" 2>/dev/null; do
  sleep 60
done

for _ in $(seq 1 10); do
  [[ -f "$EVAL_SUMMARY" ]] && break
  sleep 30
done
[[ -f "$EVAL_SUMMARY" ]] || {
  echo "v1 launcher exited without a merged summary: $EVAL_SUMMARY" >&2
  exit 2
}
"$PYTHON_BIN" - "$EVAL_SUMMARY" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if int(summary.get("examples", -1)) != 50000:
    raise SystemExit(f"v1 merged summary is incomplete: examples={summary.get('examples')}")
print(json.dumps({"v1_examples": summary["examples"], "v1_em": summary.get("em")}, ensure_ascii=False))
PY

echo "v1 evaluation is complete; starting gated R3.0 Teacher smoke/pilot"
exec bash scripts/run_teacher_pilot_v3_smoke_then_full.sh
