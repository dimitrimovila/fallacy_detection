#!/bin/bash
#SBATCH --job-name=pilot
#SBATCH --partition=owner1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#
# The pilot, one entry of configs/pilot.yaml at a time and in its order: launch the
# server of the entry, run the smoke test, make the calls of that entry into the run
# RUN_ID, stop the server. Submitted from the root of the repository, where
# runs/slurm/ must already exist for the log:
#
#   sbatch serving/slurm/pilot.sbatch 20260928T090000Z_pilot
#
# A failed smoke test stops the job before any call of that entry. Failed calls do
# not stop it: they are listed at the end, and resubmitting the job with the same
# RUN_ID makes only the calls still missing. The time is a first value, to be
# corrected once the real latency is known.

set -euo pipefail
RUN_ID=${1:?usage: sbatch serving/slurm/pilot.sbatch RUN_ID}
CONFIG=configs/pilot.yaml

cd "$SLURM_SUBMIT_DIR"
if [[ ! -f serving/slurm/common.sh ]]; then
    echo "submit the job from the root of the repository" >&2
    exit 1
fi
source serving/slurm/common.sh

entries=$(python -c 'import sys; from argfallacy.client import RunConfig; print(" ".join(RunConfig.load(sys.argv[1]).models))' "$CONFIG")
read -r -a ENTRIES <<< "$entries"
FAILED=()
for entry in "${ENTRIES[@]}"; do
    start_server "$entry"
    python serving/smoke_test.py --model "$entry"
    if ! argfallacy run execute "$CONFIG" --run-id "$RUN_ID" --model "$entry"; then
        FAILED+=("$entry")
    fi
    stop_server
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "calls failed for: ${FAILED[*]}; resubmit with the same RUN_ID to make them again" >&2
    exit 1
fi
echo "== run ${RUN_ID} complete: runs/${RUN_ID}/"
