#!/bin/bash
#SBATCH --job-name=pilot
#SBATCH --partition=owner1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#
# A pilot, one entry of the configuration at a time and in its order: launch the
# server of the entry, run the smoke test, make the calls of that entry into the run
# RUN_ID, stop the server. Submitted from the root of the repository, where
# runs/slurm/ must already exist for the log:
#
#   sbatch serving/slurm/pilot.sbatch configs/pilot2.yaml 20261005T090000Z_pilot2 qwen3_8_27b gemma4_31b
#
# The entries after RUN_ID restrict the job to them, in the order of the configuration;
# without any, every entry of the configuration runs. A run made in two times takes the
# same RUN_ID in both jobs. A failed smoke test stops the job before any call of that
# entry. Failed calls do not stop it: they are listed at the end, and resubmitting the
# job with the same arguments makes only the calls still missing, which is also how a
# job that ran out of its ten hours is resumed.

set -euo pipefail
USAGE="usage: sbatch serving/slurm/pilot.sbatch CONFIG RUN_ID [ENTRY ...]"
CONFIG=${1:?$USAGE}
RUN_ID=${2:?$USAGE}
shift 2

cd "$SLURM_SUBMIT_DIR"
if [[ ! -f serving/slurm/common.sh ]]; then
    echo "submit the job from the root of the repository" >&2
    exit 1
fi
source serving/slurm/common.sh

# an entry the configuration does not have stops the job here, before any server starts
entries=$(python -c 'import sys; from argfallacy.client import RunConfig, select_models; print(" ".join(select_models(RunConfig.load(sys.argv[1]), sys.argv[2:])))' "$CONFIG" "$@")
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
echo "== entries ${ENTRIES[*]} of run ${RUN_ID} complete: runs/${RUN_ID}/"
