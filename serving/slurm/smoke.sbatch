#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=owner1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#
# The smoke test of one entry of serving/models.yaml: launch its server, wait for it,
# run serving/smoke_test.py, stop the server. Submitted from the root of the
# repository, where runs/slurm/ must already exist for the log:
#
#   sbatch serving/slurm/smoke.sbatch qwen3_8_27b
#
# The time is a first value, to be corrected once the real loading time is known.

set -euo pipefail
ENTRY=${1:?usage: sbatch serving/slurm/smoke.sbatch ENTRY}

cd "$SLURM_SUBMIT_DIR"
if [[ ! -f serving/slurm/common.sh ]]; then
    echo "submit the job from the root of the repository" >&2
    exit 1
fi
source serving/slurm/common.sh

start_server "$ENTRY"
python serving/smoke_test.py --model "$ENTRY"
stop_server
