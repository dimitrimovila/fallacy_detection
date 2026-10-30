#!/bin/bash
#SBATCH --job-name=gosdt-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=runs/slurm/gosdt-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mode=${1:?Use check, run or resume}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv-gosdt/bin/python serving/gosdt_cluster.py "$mode" \
    --vendor "$PWD/scratch/gosdt-worker-vendor"
