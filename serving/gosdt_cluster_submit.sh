#!/bin/bash
# The caller explicitly submits both jobs; a failed preflight cancels its dependent run.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/slurm
if [[ ! -x .venv-gosdt/bin/python || ! -d scratch/gosdt-worker-vendor/gosdt ]]; then
    echo "Complete the dedicated runtime setup before submitting jobs." >&2
    exit 1
fi
if [[ -e evaluations/gosdt_schemes2_diagnostic_20261030 ]]; then
    echo "Experiment output exists; inspect it and use an explicit resume job." >&2
    exit 1
fi
sbatch --test-only --account=dm_warm_up --partition=allgroups --qos=normal \
    --time=01:00:00 serving/slurm/gosdt_cpu.sbatch check
sbatch --test-only --account=dm_warm_up --partition=allgroups --qos=normal \
    --time=2-00:00:00 serving/slurm/gosdt_cpu.sbatch run
check_job=$(sbatch --parsable --account=dm_warm_up --partition=allgroups --qos=normal \
    --time=01:00:00 serving/slurm/gosdt_cpu.sbatch check)
check_job=${check_job%%;*}
[[ "$check_job" =~ ^[0-9]+$ ]] || { echo "Unexpected preflight job identifier" >&2; exit 1; }
echo "Synthetic preflight: $check_job; log runs/slurm/gosdt-$check_job.out"
if ! run_job=$(sbatch --parsable --account=dm_warm_up --partition=allgroups --qos=normal \
    --time=2-00:00:00 --dependency="afterok:$check_job" --kill-on-invalid-dep=yes \
    serving/slurm/gosdt_cpu.sbatch run); then
    echo "Experiment submission failed; cancelling the just-created preflight $check_job." >&2
    scancel "$check_job"
    exit 1
fi
run_job=${run_job%%;*}
[[ "$run_job" =~ ^[0-9]+$ ]] || { echo "Unexpected experiment job identifier" >&2; exit 1; }
echo "Conditional experiment: $run_job; log runs/slurm/gosdt-$run_job.out"
echo "The experiment starts only after preflight succeeds and its code/input receipt matches."
