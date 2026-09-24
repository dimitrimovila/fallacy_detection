# Runbook: serving a model and talking to it

The steps in the order in which they are run. On the cluster, steps 2 to 6 run inside a
job, one entry at a time (section *On the cluster* below); by hand they serve an
interactive session.

1. **Install vLLM 0.30.0**, once, with pip in a conda environment of its own. It is the
   first version that supports K2 Horizon, and it serves all the models of
   `serving/models.yaml`: every model and both reasoning conditions run on this version.
   Before installing, read the driver of the node with `nvidia-smi` (`Driver Version` in
   the header; on the cluster, with the short interactive job of the preparation below).
   The standard wheel is built for CUDA 13.0 and needs an NVIDIA driver 580 or later:
   ```
   conda create -n vllm-0.30.0 python=3.12 -y
   conda activate vllm-0.30.0
   pip install vllm==0.30.0
   ```
   With an older driver, the last line is replaced by the CUDA 12.9 variant, with the
   command of the vLLM documentation for a specific CUDA version, the version fixed
   instead of read from the latest release:
   ```
   export VLLM_VERSION=0.30.0
   export CUDA_VERSION=129
   export CPU_ARCH=$(uname -m)
   pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl" --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
   ```
   Only Python comes from conda: PyTorch arrives with vLLM through pip, because a
   PyTorch installed with conda links NCCL statically and can break vLLM.
2. **Launch vLLM** on the node with the GPU, one entry of `serving/models.yaml` per launch:
   ```
   vllm serve $(python serving/serve_command.py <entry>) --dtype auto --port 8000
   ```
   `serve_command.py` prints the part of the command that comes from `models.yaml`:
   `<model_id> --revision <revision>`, the `serve_args` of the entry, and
   `--max-model-len 8192` when they do not set it. It refuses an entry whose revision is
   `PLACEHOLDER`, and it needs only PyYAML, so it runs in the vLLM environment. A model
   that does not fit on one GPU takes `--tensor-parallel-size` equal to the GPUs used.
   The OpenAI-compatible API and structured output are on by default, with no flag.
   Qwen and Gemma have one entry with reasoning off and one with reasoning on
   (`_think`), served by two separate launches of the same model. The entry with reasoning off is
   launched without a reasoning parser: with the parser on and no reasoning, vLLM from
   0.19.0 onwards can silently skip the JSON schema constraint. The `_think` entry is
   launched with the reasoning parser of the model and with `--max-model-len 16384`,
   because it has `max_tokens` 8192; without a parser, structured output constrains the
   JSON from the first token and the model does not reason. A call sent to the server of
   the other entry would be accepted, since the `model_id` is the same, and answered in
   the wrong condition: that is why the smoke test comes before every run of an entry.
3. **Declare the endpoint** in the `.env` at the root of the repository, never in the code:
   ```
   LLM_BASE_URL=http://localhost:8000/v1
   LLM_API_KEY=any-string
   ```
   vLLM does not check the key, but the `openai` client insists that there is one.
4. **Check the entry** before any run of it:
   ```
   python serving/smoke_test.py --model <entry>
   ```
   It prints the answer, the answer token inside the final JSON with its
   position, the `top_logprobs` at that token and the latency. If there is
   reasoning before the JSON (gpt oss, `_think` entries) and a naive reader would have ended up there, it says so. On a
   `_think` entry it also checks that there was reasoning and that the answer was not cut off. If the logprobs do not arrive, it says so plainly: then `supports_logprobs` in
   `models.yaml` must be set to `false` and `p_logprob` will stay empty for that model.
   The exit code is zero only if the entry can be run: no reasoning or a cut-off answer
   on a `_think` entry, and no logprobs on an entry with `supports_logprobs: true`, give
   another code, and a job stops on it.
5. **Count the calls** before making them:
   `argfallacy run plan configs/pilot.yaml --model <entry>`. Without `--model`, the calls of
   every entry of the configuration.
6. **Run** the entry the server is serving:
   ```
   argfallacy run execute configs/pilot.yaml --run-id <run_id> --model <entry>
   ```
   The invocations for the other entries, each after its own launch of the server, take
   the same `--run-id` and write into the same run: one `raw.jsonl`, one manifest that
   describes the whole configuration. A name that is not among the models of the
   configuration stops the command before any call. It can be interrupted: relaunching
   the same command resumes from the missing calls, without duplicating rows.

## On the cluster

Server and client run in the same job, on the node of the GPUs: no SSH tunnel, and the
personal computer can be off. The jobs are in `serving/slurm/`, for the node
`labdasan0` (partition `owner1`, two 48 GB GPUs): neither model of the pilot fits on
one of them, so every launch splits the model over the two.

**Preparation, once, on the front-end.**

1. Clone the repository, and create the folder of the logs, which must exist before a
   job starts:
   ```
   mkdir -p runs/slurm
   ```
2. Read the driver of the node with a short interactive job, and install vLLM (step 1
   above) with the matching wheel, in the environment `vllm-0.30.0`:
   ```
   srun --partition=owner1 --account=thesis --gres=gpu:1 --mem=4G --time=00:05:00 nvidia-smi
   ```
3. Create the environment of the client, `argfallacy`, from the root of the repository:
   ```
   conda create -n argfallacy python=3.12 -y
   conda activate argfallacy
   pip install -e .
   ```
   The names of the two environments are written in `serving/slurm/common.sh`; keeping
   them apart means the dependencies of the client cannot move those of vLLM.
4. Write the `.env` of the cluster, at the root of the repository: `HF_HOME`, a folder
   under `/storage` for the weights, and `LLM_API_KEY`, any string. `LLM_BASE_URL` is
   not needed: the job sets it.
   ```
   HF_HOME=/storage/<user>/huggingface
   LLM_API_KEY=any-string
   ```
5. Download the weights of the pinned revisions, with the Hugging Face command line
   of the vLLM environment and the same `HF_HOME`; `<model_id>` and `<revision>` are
   those of `serving/models.yaml`, and the pilot needs Qwen3.8 and Gemma 4, each for
   its two entries. The space under `/storage` is shared: check it first with `df -h`.
   ```
   export HF_HOME=/storage/<user>/huggingface
   hf download <model_id> --revision <revision>
   ```
   The job runs with `HF_HUB_OFFLINE=1`: it reads the weights from there and does not
   depend on the network of the node.

**Running.** From the root of the repository on the front-end:

```
sbatch serving/slurm/smoke.sbatch qwen3_8_27b          # one entry
squeue -u $USER                                        # the job and its node
sbatch serving/slurm/pilot.sbatch <run_id>             # after the four smoke tests
```

The smoke test is run on each of the four entries of the pilot before the pilot job.
A `<run_id>` of the usual form is `$(date -u +%Y%m%dT%H%M%SZ)_pilot`; resubmitting the
pilot job with the same `<run_id>` resumes from the missing calls. The log of the job
is `runs/slurm/<job name>-<job id>.out`, and every launch of vLLM has its own,
`runs/slurm/<job id>-<entry>-vllm.log`.

What a job does, in order: it activates conda and exports the settings of the `.env`
(a variable already set wins); it prints `nvidia-smi`, the versions of vLLM and
PyTorch, the CUDA version PyTorch was built for and the GPUs it sees; it sets
`HF_HUB_OFFLINE=1`; it takes a port derived from the job id, because the node is
shared, and exports `LLM_BASE_URL=http://127.0.0.1:<port>/v1`, with the server
listening only on that address. Then, for each entry, it launches `vllm serve` with the
command of step 2 plus `--tensor-parallel-size` equal to the GPUs of the job, waits for
`/health` (and stops if the server exits), runs the smoke test, and stops the server,
waiting until the GPUs are free. `smoke.sbatch` stops there; `pilot.sbatch` runs step 6
between the smoke test and the stop, for the entries of `configs/pilot.yaml` in their
order. A failed smoke test stops the job before any call of that entry; calls that
fail do not stop it, and the job ends listing the entries that have some.

After the pilot job, `argfallacy parse <run_id>` gives the tables (see the README).

