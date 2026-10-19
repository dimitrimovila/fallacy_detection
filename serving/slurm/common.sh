# Shared by smoke.sbatch and pilot.sbatch, which source it after `set -euo pipefail`
# from the root of the repository. Sourcing it prepares the job (steps 1 to 4);
# start_server and stop_server are step 5, one entry of serving/models.yaml at a time.

# The settings below depend on the cluster. Each is read from the environment, then
# from the .env at the root of the repository, and falls back to the value written here,
# that of the cluster of the Department of Mathematics. They are read with plain shell
# because conda, and with it the client that reads the rest of the .env, is not active yet.
env_setting() {
    local value
    value=$(sed -n "s/^[[:space:]]*$1=//p" .env 2> /dev/null | tail -n 1 | tr -d '\r' || true)
    value=${value%\"}; value=${value#\"}; value=${value%\'}; value=${value#\'}
    printf '%s' "${value:-$2}"
}
SERVER_ENV=${SERVER_ENV:-$(env_setting SERVER_ENV vllm-0.30.0)}   # vLLM, step 1 of serving/README.md
CLIENT_ENV=${CLIENT_ENV:-$(env_setting CLIENT_ENV argfallacy)}    # this repository, `pip install -e .`
CONDA_ROOT=${CONDA_ROOT:-$(env_setting CONDA_ROOT /conf/shared-software/anaconda)}
# seconds for a server to load its weights and answer /health; raise it where the weights
# are read slowly (a launch reads 50 to 65 GB)
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-$(env_setting HEALTH_TIMEOUT 1800)}
if [[ ! $HEALTH_TIMEOUT =~ ^[0-9]+$ ]]; then
    echo "HEALTH_TIMEOUT must be a number of seconds, found '$HEALTH_TIMEOUT'" >&2
    exit 1
fi
RELEASE_TIMEOUT=120      # seconds for a stopped server to give the GPUs back
# on a cluster without conda: the Apptainer image of serving/apptainer/vllm.def, which
# holds vLLM and the libraries of the client; empty, the two conda environments above
CONTAINER=${CONTAINER:-$(env_setting CONTAINER "")}

# conda's activation scripts are not written for `set -u`
use_env() {
    set +u
    conda activate "$1"
    set -u
}

# 1. the software, conda or the image, and the settings of the .env with the rules of the
#    client: a variable already in the environment wins over the file
if [[ -n "$CONTAINER" ]]; then
    if [[ ! -f "$CONTAINER" ]]; then
        echo "no image at CONTAINER=$CONTAINER: build it as serving/apptainer/vllm.def says" >&2
        exit 1
    fi
    # the command that runs inside the image: it sees the GPUs, the repository (the
    # working directory) and, once the .env is read, the folder of the weights; the client
    # is imported from src/, so the image needs no installation of it
    image_command() {
        IMAGE_CMD=(apptainer exec --nv --bind "$PWD")
        if [[ -n "${HF_HOME:-}" ]]; then IMAGE_CMD+=(--bind "$HF_HOME"); fi
        if [[ -n "${TIKTOKEN_ENCODINGS_BASE:-}" ]]; then
            IMAGE_CMD+=(--bind "$TIKTOKEN_ENCODINGS_BASE")
        fi
        IMAGE_CMD+=(--env "PYTHONPATH=$PWD/src" "$CONTAINER")
    }
    in_image() {
        image_command
        "${IMAGE_CMD[@]}" "$@"
    }
    # the server and the client share the image, so there is no environment to switch
    use_env() { :; }
    python() { in_image python3 "$@"; }
    argfallacy() { in_image python3 -m argfallacy.cli "$@"; }
else
    if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
        echo "no conda at CONDA_ROOT=$CONDA_ROOT: set CONDA_ROOT in the .env to the folder that" \
             "\`conda info --base\` prints on this cluster, or CONTAINER to an Apptainer image" >&2
        exit 1
    fi
    set +u
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    set -u
    use_env "$CLIENT_ENV"
fi
# (assigned first: a failure inside `eval "$(...)"` would not stop the job)
settings=$(python -c 'import shlex; from argfallacy.env import load_env; print("\n".join(f"export {k}={shlex.quote(v)}" for k, v in load_env().items()))')
eval "$settings"
: "${HF_HOME:?set HF_HOME in the .env: the folder under /extra that holds the weights}"
: "${LLM_API_KEY:?set LLM_API_KEY in the .env: any string, vLLM does not check it}"

# 2. what the job runs on
echo "== job ${SLURM_JOB_ID} on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi
use_env "$SERVER_ENV"
python -c 'import torch, vllm; print(f"vLLM {vllm.__version__}, PyTorch {torch.__version__} built for CUDA {torch.version.cuda}, GPUs seen: {torch.cuda.device_count()} (available: {torch.cuda.is_available()})")'
use_env "$CLIENT_ENV"

# 3. the weights of the pinned revisions are already under HF_HOME, downloaded from
#    the front-end: the job does not depend on the network of the node
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
# FlashInfer compiles its sampling kernels on first use with the nvcc it finds in the
# environment, and the environment has none (a CUDA 13 compiler there gave kernels that
# a CUDA 12.9 driver cannot run). The PyTorch sampler draws from the same top-k and
# top-p distribution.
export VLLM_USE_FLASHINFER_SAMPLER=0

# 4. the node is shared: a port of this job, and a server that listens only on the node
PORT=$((20000 + SLURM_JOB_ID % 20000))
export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:?the job sees no GPU: submit it with --gres=gpu:N}"
NGPU=${#GPUS[@]}
echo "== port ${PORT}, ${NGPU} GPUs, weights in ${HF_HOME}"

# 5. one server at a time
SERVER_PID=""

start_server() {
    local entry=$1 line log
    local -a serve
    # an entry whose revision is still a placeholder stops the job here
    line=$(python serving/serve_command.py "$entry")
    # the line is quoted for the shell (shlex): a flag whose value is JSON, with spaces and
    # quotes, must come back as one argument, which `read -a` would split
    eval "serve=(${line})"
    log="runs/slurm/${SLURM_JOB_ID}-${entry}-vllm.log"
    echo "== ${entry}: vllm serve ${line} (log: ${log})"
    use_env "$SERVER_ENV"
    # setsid runs a program, not a shell function, so the command of the image is spelled out
    local -a launch=(vllm)
    if [[ -n "$CONTAINER" ]]; then
        image_command
        launch=("${IMAGE_CMD[@]}" vllm)
    fi
    # its own process group, so that stopping it reaches the workers too
    setsid "${launch[@]}" serve "${serve[@]}" --host 127.0.0.1 --port "$PORT" \
        --tensor-parallel-size "$NGPU" --dtype auto > "$log" 2>&1 &
    SERVER_PID=$!
    use_env "$CLIENT_ENV"

    local waited=0
    until python -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' \
            "http://127.0.0.1:${PORT}/health" 2> /dev/null; do
        if ! kill -0 "$SERVER_PID" 2> /dev/null; then
            echo "the server of ${entry} exited before answering; the end of its log:" >&2
            tail -n 40 "$log" >&2
            exit 1
        fi
        if [[ $waited -ge $HEALTH_TIMEOUT ]]; then
            echo "the server of ${entry} did not answer /health in ${HEALTH_TIMEOUT} s (HEALTH_TIMEOUT in the .env)" >&2
            tail -n 40 "$log" >&2
            exit 1
        fi
        sleep 10
        waited=$((waited + 10))
    done
    echo "== ${entry}: server ready after ${waited} s"
}

stop_server() {
    [[ -n "$SERVER_PID" ]] || return 0
    local pid=$SERVER_PID waited=0
    SERVER_PID=""
    echo "== stopping the server"
    kill -TERM -- "-${pid}" 2> /dev/null || true
    while kill -0 "$pid" 2> /dev/null && [[ $waited -lt $RELEASE_TIMEOUT ]]; do
        sleep 5
        waited=$((waited + 5))
    done
    kill -KILL -- "-${pid}" 2> /dev/null || true
    wait "$pid" 2> /dev/null || true
    # the next launch needs the whole memory of the GPUs
    waited=0
    until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
        if [[ $waited -ge $RELEASE_TIMEOUT ]]; then
            echo "the GPUs are still in use ${RELEASE_TIMEOUT} s after the server stopped" >&2
            nvidia-smi >&2
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "== GPUs released"
}

# a job that stops half way, cancelled or out of time, still stops its server
trap stop_server EXIT
trap 'exit 143' TERM INT
