#!/bin/bash
# Package/download setup only; no fit and no scheduler submission.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scratch runs/slurm
if [[ -e .venv-gosdt || -e scratch/gosdt-worker-vendor ]]; then
    echo "Existing runtime found; preserve it and inspect before repeating setup." >&2
    exit 1
fi
python3 -m venv scratch/gosdt-bootstrap
scratch/gosdt-bootstrap/bin/python -m pip install uv
export UV_PYTHON_INSTALL_DIR="$PWD/scratch/gosdt-python"
export UV_PYTHON_BIN_DIR="$PWD/scratch/gosdt-python-bin"
scratch/gosdt-bootstrap/bin/uv python install 3.12
scratch/gosdt-bootstrap/bin/uv venv --python 3.12 .venv-gosdt
scratch/gosdt-bootstrap/bin/uv pip install --python .venv-gosdt/bin/python \
    -r serving/gosdt_driver_requirements.txt -e '.[dev]'
scratch/gosdt-bootstrap/bin/uv pip install --python .venv-gosdt/bin/python \
    --target scratch/gosdt-worker-vendor --only-binary=:all: \
    -r serving/gosdt_worker_requirements.txt
scratch/gosdt-bootstrap/bin/uv pip freeze --python .venv-gosdt/bin/python \
    > scratch/gosdt-driver-freeze.txt
echo "Setup complete. Submit the synthetic preflight before the experiment."
