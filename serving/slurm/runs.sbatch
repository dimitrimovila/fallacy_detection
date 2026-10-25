#!/bin/bash
#SBATCH --job-name=pilot
#SBATCH --partition=owner1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#
# Serve one entry once, then execute and check each configuration in order.
# Submit from the repository root; runs/slurm/ must already exist.

set -euo pipefail
USAGE="usage: sbatch serving/slurm/runs.sbatch ENTRY CONFIG RUN_ID [CONFIG RUN_ID ...]"
if [[ $# -lt 3 || $(( ($# - 1) % 2 )) -ne 0 ]]; then
    echo "$USAGE" >&2
    exit 1
fi
ENTRY=$1
shift
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f serving/slurm/common.sh ]]; then
    echo "submit the job from the root of the repository" >&2
    exit 1
fi
source serving/slurm/common.sh

# Validate every pair before loading the server.
python - "$ENTRY" "$@" <<'PY'
import sys
from argfallacy.client import RunConfig, select_models
entry = sys.argv[1]
for path in sys.argv[2::2]:
    select_models(RunConfig.load(path), [entry])
PY

start_server "$ENTRY"
python serving/smoke_test.py --model "$ENTRY"
while [[ $# -gt 0 ]]; do
    CONFIG=$1
    RUN_ID=$2
    shift 2
    argfallacy run plan "$CONFIG" --model "$ENTRY"
    argfallacy run execute "$CONFIG" --run-id "$RUN_ID" --model "$ENTRY"
    argfallacy parse "$RUN_ID"
    argfallacy report format "$RUN_ID" --model "$ENTRY"
done
stop_server
