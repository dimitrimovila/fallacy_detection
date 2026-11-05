#!/bin/bash
#SBATCH --job-name=act-pilot
#SBATCH --partition=pgpu_dm
#SBATCH --account=dm_warm_up
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#
# The ACT stage of the pilot on node22 (UPSCALE), in one job with four GPUs: the answerer
# (qwen3_8_27b) on the first two GPUs of the job, port 8000, and the optimizer
# (gpt_oss_120b) on the other two, port 8001, each with the serve_args of
# serving/models.yaml and tensor parallelism 2. When both answer /health, the job runs
# `act-pilot run-act` and stops: the evaluation is run later, once the content validity
# of the questions has been rated. Submitted from the root of the repository, where
# runs/slurm/ must already exist for the log:
#
#   sbatch act_pilot/slurm/pilot_node22.sbatch --max-calls 21000
#
# The arguments are passed to run-act (--max-calls, --boundary, --folds, --seeds). Every
# reply is cached under act_pilot/runs/act_pilot/cache/, so a job that stops half way is
# resumed by submitting it again with the same arguments.
#
# The .env at the root, read by serving/slurm/common.sh, must hold CONTAINER, HF_HOME,
# TIKTOKEN_ENCODINGS_BASE, LLM_API_KEY and HEALTH_TIMEOUT as for the full runs, plus
# ACT_PILOT_DATA (the folder with items.csv and folds.csv) and ACT_PILOT_DEPS (the folder
# with the libraries of act_pilot that the image lacks, installed as act_pilot/README.md
# says).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f serving/slurm/common.sh || ! -f act_pilot/configs/pilot.yaml ]]; then
    echo "submit the job from the root of the repository" >&2
    exit 1
fi
CONFIG=act_pilot/configs/pilot.yaml
ANSWERER=qwen3_8_27b
OPTIMIZER=gpt_oss_120b
ANSWERER_PORT=8000
OPTIMIZER_PORT=8001

# steps 1 to 4 of common.sh: the image, the settings of the .env, the offline variables,
# the GPUs of the job; its single-server functions are not used here
source serving/slurm/common.sh
trap - EXIT
if [[ -z "$CONTAINER" ]]; then
    echo "node22 has no conda: set CONTAINER in the .env to the Apptainer image" >&2
    exit 1
fi
: "${ACT_PILOT_DATA:?set ACT_PILOT_DATA in the .env: the folder with items.csv and folds.csv}"
: "${ACT_PILOT_DEPS:?set ACT_PILOT_DEPS in the .env: see act_pilot/README.md}"
if [[ $NGPU -ne 4 ]]; then
    echo "the job sees ${NGPU} GPUs, it needs 4: submit it with --gres=gpu:4" >&2
    exit 1
fi
export LLM_BASE_URL="http://127.0.0.1:${ANSWERER_PORT}/v1"
export OPTIMIZER_BASE_URL="http://127.0.0.1:${OPTIMIZER_PORT}/v1"
export ACT_PILOT_DATA

# the client in the image, with act_pilot and its libraries on the path; no GPU
client() {
    apptainer exec --bind "$PWD" --bind "$ACT_PILOT_DATA" --bind "$ACT_PILOT_DEPS" \
        --env "PYTHONPATH=$PWD/act_pilot/src:$ACT_PILOT_DEPS" "$CONTAINER" "$@"
}
act_pilot() { client python3 -m act_pilot "$@"; }

# configuration, data and libraries are checked before any server starts (no call)
act_pilot estimate --config "$CONFIG"

# a port in use would send the requests of this job to another server
for port in "$ANSWERER_PORT" "$OPTIMIZER_PORT"; do
    if python -c 'import socket, sys; s = socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)' "$port"; then
        echo "port ${port} is already in use on $(hostname)" >&2
        exit 1
    fi
done

PIDS=()
LOGS=()

# start_server ENTRY PORT GPU,GPU: in the background, in its own process group
start_server() {
    local entry=$1 port=$2 gpus=$3 line log
    local -a serve
    line=$(python serving/serve_command.py "$entry")
    eval "serve=(${line})"
    log="runs/slurm/${SLURM_JOB_ID}-${entry}-vllm.log"
    echo "== ${entry} on GPUs ${gpus}, port ${port}: vllm serve ${line} (log: ${log})"
    image_command
    CUDA_VISIBLE_DEVICES=$gpus setsid "${IMAGE_CMD[@]}" vllm serve "${serve[@]}" \
        --host 127.0.0.1 --port "$port" --tensor-parallel-size 2 --dtype auto \
        > "$log" 2>&1 &
    PIDS+=($!)
    LOGS+=("$log")
}

stop_servers() {
    [[ ${#PIDS[@]} -gt 0 ]] || return 0
    local pid waited=0
    echo "== stopping the servers"
    for pid in "${PIDS[@]}"; do kill -TERM -- "-${pid}" 2> /dev/null || true; done
    for pid in "${PIDS[@]}"; do
        while kill -0 "$pid" 2> /dev/null && [[ $waited -lt $RELEASE_TIMEOUT ]]; do
            sleep 5
            waited=$((waited + 5))
        done
        kill -KILL -- "-${pid}" 2> /dev/null || true
        wait "$pid" 2> /dev/null || true
    done
    PIDS=()
    waited=0
    until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
        if [[ $waited -ge $RELEASE_TIMEOUT ]]; then
            echo "the GPUs are still in use ${RELEASE_TIMEOUT} s after the servers stopped" >&2
            nvidia-smi >&2
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "== GPUs released"
}
# a job that stops half way, cancelled or out of time, still stops its servers
trap stop_servers EXIT
trap 'exit 143' TERM INT

start_server "$ANSWERER" "$ANSWERER_PORT" "${GPUS[0]},${GPUS[1]}"
start_server "$OPTIMIZER" "$OPTIMIZER_PORT" "${GPUS[2]},${GPUS[3]}"

# both servers load at the same time; the job waits for both, HEALTH_TIMEOUT in all
PORTS=("$ANSWERER_PORT" "$OPTIMIZER_PORT")
waited=0
for k in 0 1; do
    port=${PORTS[$k]}
    until python -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' \
            "http://127.0.0.1:${port}/health" 2> /dev/null; do
        if ! kill -0 "${PIDS[$k]}" 2> /dev/null; then
            echo "the server on port ${port} exited before answering; the end of its log:" >&2
            tail -n 40 "${LOGS[$k]}" >&2
            exit 1
        fi
        if [[ $waited -ge $HEALTH_TIMEOUT ]]; then
            echo "the server on port ${port} did not answer /health in ${HEALTH_TIMEOUT} s (HEALTH_TIMEOUT in the .env)" >&2
            tail -n 40 "${LOGS[$k]}" >&2
            exit 1
        fi
        sleep 10
        waited=$((waited + 10))
    done
    echo "== server on port ${port} ready after ${waited} s"
done

# each port must serve the model that the configuration names for its role
act_pilot_check='
import json, sys, urllib.request
from act_pilot.config import load_config
llm = load_config(sys.argv[1]).llm
for role, url in ((llm.answerer, sys.argv[2]), (llm.optimizer, sys.argv[3])):
    served = [m["id"] for m in json.load(urllib.request.urlopen(url + "/models", timeout=10))["data"]]
    if role.model not in served:
        sys.exit(f"{url} serves {served}, the {role.name} of the configuration is {role.model}")
'
client python3 -c "$act_pilot_check" "$CONFIG" "$LLM_BASE_URL" "$OPTIMIZER_BASE_URL"

act_pilot run-act --config "$CONFIG" "$@"
echo "== ACT stage done: act_pilot/runs/act_pilot/act/; rate the content validity before evaluating"
