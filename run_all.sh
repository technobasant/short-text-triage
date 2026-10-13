#!/usr/bin/env bash
# Regenerates every figure in MEASUREMENTS.md. Order matters: later steps read
# the vectors and labels earlier steps write to out/.
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false TRIAGE_THREADS="${TRIAGE_THREADS:-4}"
for step in corpus dedupe bench_embed cluster prefix assign calls; do
  # Load average is recorded because the timed steps ran on a shared machine.
  load=$(sysctl -n vm.loadavg 2>/dev/null || cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo n/a)
  echo "== $step  $(date '+%H:%M:%S')  load: $load"
  uv run python -m "triage.$step"
done
