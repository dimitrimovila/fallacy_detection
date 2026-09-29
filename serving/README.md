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
   conda create -n vllm-0.30.0 -c conda-forge --override-channels python=3.12 -y
   conda activate vllm-0.30.0
   python -m pip install vllm==0.30.0
   ```
   Python comes from conda-forge alone: on the cluster `conda create` with the default
   Anaconda channels stops with `CondaToSNonInteractiveError`, because their Terms of
   Service are not accepted, and without the environment `pip` installs into the shared
   base Python and writes into `~/.local`. Before every installation, `which python`
   must point to `envs/vllm-0.30.0`, and pip is run as `python -m pip`. Only Python
   comes from conda: PyTorch comes through pip, because a PyTorch installed with conda
   links NCCL statically and can break vLLM.

   With an older driver the CUDA 12.9 variant of the same version is installed. On
   `labdasan0` the driver is 575.57.08, so this is the variant used there. The command of
   the vLLM documentation for a specific CUDA version (the `+cu129` wheel with
   `--extra-index-url https://download.pytorch.org/whl/cu129`) left there a PyTorch built
   for CUDA 13.0 and the standard vLLM of PyPI, whose extension needs `libcudart.so.13`
   and does not import. The `+cu129` wheel itself is right: it requires `torch==2.13.0`,
   `torchvision==0.28.0`, `torchaudio==2.11.0` and `flashinfer-python==0.6.18.post1`, and
   its extension needs `libcudart.so.12`. These three commands, in this order, gave a
   working environment:
   ```
   python -m pip install "torch==2.13.0+cu129" "torchvision==0.28.0+cu129" "torchaudio==2.11.0+cu129" --index-url https://download.pytorch.org/whl/cu129
   python -m pip install --no-deps --force-reinstall "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
   python -m pip install "cuda-python~=12.9.0"
   ```
   They repaired an environment where the other dependencies of vLLM were already
   installed: `--no-deps` keeps pip from replacing the PyTorch of the first line, and
   installs nothing else. Whether a new environment reaches the same state with the same
   order, without `--no-deps` on the second line and without the CUDA 13 libraries, has
   not been tried; `pip check` at the end says what is missing. `cuda-python` stays in
   the environment, at 12.9: `flashinfer-python` and the `nvidia-cutlass-dsl-libs-*`
   packages need it, and the 13.x series clashes with `cuda-bindings` 12.9.

   Do not mix wheels built for CUDA 12 and for CUDA 13 in the same environment. The
   `-cu12` and `-cu13` packages of cuDNN, NCCL, NVSHMEM and cuSPARSELt write the same
   files, and those files are registered under the package installed last. Removing the
   `-cu13` ones then removes libraries of PyTorch too, which stops importing
   (`libcudnn.so.9: cannot open shared object file`); neither `pip check` nor the
   `RECORD` files of the packages show it, `ldd` on `libtorch_cuda.so` does, with
   `not found`. If it happens, after removing the `-cu13` packages reinstall the `-cu12`
   ones that share the files, at the versions of the installation (those of PyTorch
   2.13.0+cu129 on the cluster), without touching anything else:
   ```
   python -m pip install --no-deps --force-reinstall nvidia-cudnn-cu12==9.20.0.48 nvidia-nccl-cu12==2.29.7 nvidia-nvshmem-cu12==3.4.5 nvidia-cusparselt-cu12==0.8.1
   ```

   The check, whatever the variant (the versions shown are those of the CUDA 12.9 one):
   ```
   python -m pip show vllm     # Version: 0.30.0+cu129
   python -m pip check         # no broken requirements
   python -c 'import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda)'
   ```
   The last line prints `0.30.0 2.13.0+cu129 12.9`. On the cluster the environment is
   written to the network home, and the whole installation takes hours: the wheel of
   vLLM alone is 545 MB, 1.6 GB and about 5,100 files once installed.
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
   gpt oss always reasons (the effort takes only low, medium or high) and is launched like
   a `_think` entry: `max_tokens` 8192, `--max-model-len 16384` and `--reasoning-parser
   openai_gptoss`. For this model vLLM moves the JSON schema to the `final` channel of
   the harmony format, leaving the `analysis` channel free for the reasoning, only when
   a parser is configured; without it the schema constrains the JSON from the first
   token and the model does not reason.
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
   reasoning before the JSON (gpt oss, K2 Horizon, `_think` entries) and a naive reader would have ended up there, it says so. On an
   entry that reasons (`enable_thinking` true, or `is_reasoning` true where the entry does not set `enable_thinking`: the `_think` entries, K2 Horizon, gpt oss) it also checks that there was reasoning and that the answer was not cut off. The reasoning is read from `usage` and from the reasoning field of the message (`reasoning_content` or `reasoning`), and the script prints its length and where it was found: `usage` can say zero while the field holds the thinking. It also checks that the prompt reached the model: `prompt_tokens` below a quarter of an estimate at four characters per token means the chat template dropped the content, as it did for K2 Horizon before `--chat-template-content-format string`. If the logprobs do not arrive, it says so plainly: then `supports_logprobs` in
   `models.yaml` must be set to `false` and `p_logprob` will stay empty for that model.
   The exit code is zero only if the entry can be run: a prompt not read, no reasoning or
   a cut-off answer on an entry that reasons, and no logprobs on an entry with
   `supports_logprobs: true`, give another code, and a job stops on it.
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
   srun --partition=owner1 --gres=gpu:1 --mem=4G --time=00:05:00 nvidia-smi
   ```
   The front-end closes idle SSH sessions: the installation and the downloads of points
   5 and 6 run inside `tmux`. Jobs submitted with `sbatch` do not depend on the session.
3. Create the environment of the client, `argfallacy`, from the root of the repository:
   ```
   conda create -n argfallacy -c conda-forge --override-channels python=3.12 -y
   conda activate argfallacy
   python -m pip install -e .
   ```
   The names of the two environments are written in `serving/slurm/common.sh`; keeping
   them apart means the dependencies of the client cannot move those of vLLM.
4. Write the `.env` of the cluster, at the root of the repository: `HF_HOME`, a folder
   under `/extra` for the weights, `TIKTOKEN_ENCODINGS_BASE`, the folder of the
   vocabularies of gpt oss (point 6), and `LLM_API_KEY`, any string. `LLM_BASE_URL` is
   not needed: the job sets it.
   ```
   HF_HOME=/extra/<user>/huggingface
   TIKTOKEN_ENCODINGS_BASE=/extra/<user>/tiktoken_encodings
   LLM_API_KEY=any-string
   ```
   The weights go under `/extra`, not `/storage`: `/storage` is shared by the whole
   department and was nearly full (98 per cent) when the weights of the pilot were
   downloaded, while `/extra` had 23 TB free. `labdasan0` mounts the same `/extra`.
   The job exports every setting of this file before it launches the server, so
   `vllm serve` sees them as the client does.
5. Download the weights of the pinned revisions, with the Hugging Face command line
   of the vLLM environment and the same `HF_HOME`; `<model_id>` and `<revision>` are
   those of `serving/models.yaml`, and the pilot needs Qwen3.8 and Gemma 4, each for
   its two entries. The space under `/extra` is shared: check it first with `df -h`.
   ```
   export HF_HOME=/extra/<user>/huggingface
   hf download <model_id> --revision <revision>
   ```
   No token is needed: neither repository of the pilot is gated. Each model takes a
   little more than an hour, and the two of the pilot occupy 111 GB. When `hf` suggests
   upgrading `huggingface_hub`, the suggestion is ignored: it would change a library of
   the vLLM environment. The job runs with `HF_HUB_OFFLINE=1`: it reads the weights from
   there and does not depend on the network of the node. The first launch of a model
   reads its weights at about 110 MB/s (eight minutes for Qwen3.8) and compiles it; the
   compilation is cached in `~/.cache/vllm/torch_compile_cache`, so the later launches of
   the same model skip it.

   The repositories of gpt oss hold the weights three times: at the root, the files
   vLLM reads, and in `original/` and `metal/` the copies for the reference
   implementation and for Apple Metal, which vLLM does not read. Those two folders are
   left out:
   ```
   hf download openai/gpt-oss-20b --revision <revision> --exclude "original/*" --exclude "metal/*"
   hf download openai/gpt-oss-120b --revision <revision> --exclude "original/*" --exclude "metal/*"
   ```
   13.8 GB for the 20b and 65.3 GB for the 120b, against 41.3 and 195.8 GB for the whole
   repositories; neither is gated. With `hf` 1.33.0 each pattern takes its own
   `--exclude`; `--dry-run` lists the files and the total without downloading anything.
6. For gpt oss, the vocabularies of its tokenizer. vLLM reads the harmony format of gpt
   oss through the `openai-harmony` library, which downloads `o200k_base` and
   `cl100k_base` on first use unless `TIKTOKEN_ENCODINGS_BASE` names a folder that
   holds them. They are downloaded once from the front-end, so that the server does not
   depend on the network of the node:
   ```
   mkdir -p /extra/<user>/tiktoken_encodings
   wget -O /extra/<user>/tiktoken_encodings/o200k_base.tiktoken https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
   wget -O /extra/<user>/tiktoken_encodings/cl100k_base.tiktoken https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
   ```
   and the folder goes into the `.env` as `TIKTOKEN_ENCODINGS_BASE` (point 4).

**Running.** From the root of the repository on the front-end, in the environment
`argfallacy`:

```
argfallacy run plan configs/pilot.yaml                 # counts the calls, makes none
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)_pilot
echo $RUN_ID | tee -a ~/run_ids.txt                    # kept for a resubmission
sbatch serving/slurm/pilot.sbatch $RUN_ID
squeue -u $USER                                        # the job and its node
```

The pilot job runs the smoke test on every entry before its calls, so no separate smoke
test is needed first: a failed one stops the job before any call of that entry, the
calls already made stay, and resubmitting the job with the same `RUN_ID` resumes from
the missing calls. `smoke.sbatch` tries one entry without creating a run, for a new
model, a new revision or another node:

```
sbatch serving/slurm/smoke.sbatch qwen3_8_27b
```

In `squeue`, a `TIME` of `INVALID` in the first seconds of a job is not an error. The
log of the job is `runs/slurm/<job name>-<job id>.out`, and every launch of vLLM has
its own, `runs/slurm/<job id>-<entry>-vllm.log`.

What a job does, in order: it activates conda and exports the settings of the `.env`
(a variable already set wins); it prints `nvidia-smi`, the versions of vLLM and
PyTorch, the CUDA version PyTorch was built for and the GPUs it sees; it sets
`HF_HUB_OFFLINE=1`, and `VLLM_USE_FLASHINFER_SAMPLER=0`, because FlashInfer compiles
its sampling kernels on first use with the `nvcc` of the environment, and the
environment has none (a CUDA 13 compiler there gave kernels that need a driver newer
than that of `labdasan0`; the sampler of PyTorch draws from the same top-k and top-p
distribution); it takes a port derived from
the job id, because the node is shared, and exports `LLM_BASE_URL=http://127.0.0.1:<port>/v1`, with the server
listening only on that address. Then, for each entry, it launches `vllm serve` with the
command of step 2 plus `--tensor-parallel-size` equal to the GPUs of the job, waits for
`/health` (and stops if the server exits), runs the smoke test, and stops the server,
waiting until the GPUs are free. `smoke.sbatch` stops there; `pilot.sbatch` runs step 6
between the smoke test and the stop, for the entries of `configs/pilot.yaml` in their
order. A failed smoke test stops the job before any call of that entry; calls that
fail do not stop it, and the job ends listing the entries that have some.

After the pilot job, `argfallacy parse <run_id>` gives the tables (see the README).

