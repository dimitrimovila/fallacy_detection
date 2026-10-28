#!/usr/bin/env bash
# CPU only. Supply partition, account, CPUs, memory and time to sbatch explicitly.
set -euo pipefail
if (( $# < 7 )); then
    echo "usage: postprocess.sbatch SITE PYTHON ENVIRONMENT REPO MODE INPUT OUTPUT [replay options]" >&2
    exit 2
fi
site=$1
python_bin=$2
environment=$3
repo=$4
mode=$5
input=$6
output=$7
shift 7
[[ -x "$python_bin" ]] || { echo "PYTHON must be an executable absolute path" >&2; exit 2; }
[[ "$python_bin" = /* && "$repo" = /* && "$input" = /* && "$output" = /* ]] || {
    echo "PYTHON, REPO, INPUT and OUTPUT must be absolute paths" >&2; exit 2;
}
cd "$repo"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$python_bin" serving/postprocess.py --site "$site" --environment "$environment" \
    --mode "$mode" --input "$input" --output "$output" "$@"
