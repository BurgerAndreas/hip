#!/bin/bash

# Simple relauncher: keeps 4 agents running by resubmitting every ~71 hours.
# Runs in the background on the login node; submits SLURM jobs and exits.

set -euo pipefail

if [[ -z "${SWEEP_ID:-}" ]]; then
  echo "SWEEP_ID not set. Export SWEEP_ID or pass as first arg." >&2
  [[ $# -ge 1 ]] || exit 1
  export SWEEP_ID="$1"
  shift || true
fi

NUM_AGENTS=${NUM_AGENTS:-4}
SLEEP_HOURS=${SLEEP_HOURS:-71}

echo "Using SWEEP_ID=$SWEEP_ID, NUM_AGENTS=$NUM_AGENTS, SLEEP_HOURS=$SLEEP_HOURS"

while true; do
  echo "$(date): Submitting $NUM_AGENTS sweep agents"
  for i in $(seq 1 "$NUM_AGENTS"); do
    sbatch --export=ALL,SWEEP_ID="$SWEEP_ID" scripts/sweep_agent.sbatch || true
  done
  echo "$(date): Submitted. Sleeping ${SLEEP_HOURS}h"
  sleep $((SLEEP_HOURS*3600))
done &

disown
echo "Relaunch loop started in background (PID $!)."

