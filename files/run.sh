#!/usr/bin/env bash
# Minimal Plan -> Act -> Verify loop for the Vin Agent voice-agent harness.
# State lives on disk in files/IMPLEMENTATION_PLAN.md.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT_DIR="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/backend/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
mkdir -p files/logs backend/app/evals/reports/voice

while true; do
  timestamp=$(date +"%Y-%m-%d %H:%M")
  log_file="files/logs/$(date +%Y-%m-%d).log"

  "$PYTHON_BIN" -m compileall -q backend/app/evals backend/app/api >> "$log_file" 2>&1
  (cd backend && "$PYTHON_BIN" -m app.evals.run_voice_loop --mode local) >> "../$log_file" 2>&1
  (cd backend && "$PYTHON_BIN" -m app.evals.run_voice_loop --mode manifest) >> "../$log_file" 2>&1

  if [[ -n "${ELEVENLABS_API_KEY:-}" && -n "${ELEVENLABS_AGENT_ID:-}" && -n "${ELEVENLABS_TEST_ID:-}" ]]; then
    (cd backend && "$PYTHON_BIN" -m app.evals.run_voice_loop --mode elevenlabs       --repeat-count "${ELEVENLABS_REPEAT_COUNT:-5}" --test-id "$ELEVENLABS_TEST_ID")       >> "../$log_file" 2>&1
  fi

  echo "$timestamp iter ok" >> files/logs/loop-status.log

  if grep -q "^STATUS: done$" files/IMPLEMENTATION_PLAN.md; then
    echo "$timestamp STATUS: done - exiting loop" >> files/logs/loop-status.log
    break
  fi

  changed_files=$(git diff --name-only | wc -l)
  if [ "$changed_files" -gt 12 ]; then
    echo "$timestamp STOP condition hit: >12 files changed, needs human review"       >> files/logs/loop-status.log
    break
  fi

  sleep 5
done
