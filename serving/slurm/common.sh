# Shared by smoke.sbatch and pilot.sbatch, which source it after `set -euo pipefail`
# from the root of the repository. Sourcing it prepares the job (steps 1 to 4);
# start_server and stop_server are step 5, one entry of serving/models.yaml at a time.

SERVER_ENV=vllm-0.30.0   # vLLM, as in step 1 of serving/README.md
CLIENT_ENV=argfallacy    # this repository, installed with `pip install -e .`
CONDA_ROOT=/conf/shared-software/anaconda
HEALTH_TIMEOUT=1800      # seconds for a server to load its weights and answer /health
RELEASE_TIMEOUT=120      # seconds for a stopped server to give the GPUs back

# conda's activation scripts are not written for `set -u`
use_env() {
    set +u
    conda activate "$1"
    set -u
}

# 1. conda, and the settings of the .env with the rules of the client: a variable
#    already in the environment wins over the file
set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
set -u
use_env "$CLIENT_ENV"
# (assigned first: a failure inside `eval "$(...)"` would not stop the job)
settings=$(python -c 'import shlex; from argfallacy.env import load_env; print("\n".join(f"export {k}={shlex.quote(v)}" for k, v in load_env().items()))')
eval "$settings"
: "${HF_HOME:?set HF_HOME in the .env: the folder under /storage that holds the weights}"
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
    read -r -a serve <<< "$line"
    log="runs/slurm/${SLURM_JOB_ID}-${entry}-vllm.log"
    echo "== ${entry}: vllm serve ${line} (log: ${log})"
    use_env "$SERVER_ENV"
    # its own process group, so that stopping it reaches the workers too
    setsid vllm serve "${serve[@]}" --host 127.0.0.1 --port "$PORT" \
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
            echo "the server of ${entry} did not answer /health in ${HEALTH_TIMEOUT} s" >&2
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
