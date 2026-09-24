# Runbook: serving a model and talking to it

Ten lines, in the order in which they are run.

1. **Install vLLM 0.30.0**, once, with pip in a conda environment of its own. It is the
   first version that supports K2 Horizon, and it serves all the models of
   `serving/models.yaml`: every model and both reasoning conditions run on this version.
   Before installing, the first job reads the driver of the node with `nvidia-smi`
   (`Driver Version` in the header). The standard wheel is built for CUDA 13.0 and needs
   an NVIDIA driver 580 or later:
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
2. **Launch vLLM** on the node with the GPU. `<model_id>` and `<revision>` come from
   `serving/models.yaml`; while `revision` is `PLACEHOLDER` the client does not start:
   ```
   vllm serve <model_id> --revision <revision> --dtype auto --max-model-len 8192 --port 8000
   ```
   The OpenAI-compatible API and structured output are on by default, with no flag.
   If the notes of the model in `models.yaml` list extra flags, they must be added here.
   Qwen and Gemma have one entry with reasoning off and one with reasoning on
   (`_think`), served by two separate launches of the same model. The entry with reasoning off is
   launched without a reasoning parser: with the parser on and no reasoning, vLLM from
   0.19.0 onwards can silently skip the JSON schema constraint. The `_think` entry is
   launched with the reasoning parser given in the notes of the model and with
   `--max-model-len 16384`, because it has `max_tokens` 8192; without a parser, structured output
   constrains the JSON from the first token and the model does not reason.
3. **Declare the endpoint** in the `.env` at the root of the repository, never in the code:
   ```
   LLM_BASE_URL=http://localhost:8000/v1
   LLM_API_KEY=any-string
   ```
   vLLM does not check the key, but the `openai` client insists that there is one.
4. **Check the model** before any run:
   ```
   python serving/smoke_test.py
   ```
   It prints the answer, the answer token inside the final JSON with its
   position, the `top_logprobs` at that token and the latency. If there is
   reasoning before the JSON (gpt oss, `_think` entries) and a naive reader would have ended up there, it says so. On a
   `_think` entry it also checks that there was reasoning and that the answer was not cut off. If the logprobs do not arrive, it says so plainly: then `supports_logprobs` in
   `models.yaml` must be set to `false` and `p_logprob` will stay empty for that model.
5. **Count the calls** before making them: `argfallacy run plan configs/pilot.yaml`.
6. **Run**: `argfallacy run execute configs/pilot.yaml`. It can be interrupted: relaunching
   the same command resumes from the missing calls, without duplicating rows.

## >>> TO BE COMPLETED <<<

This section depends on the cluster. One line is needed for each of the two
points, with the exact commands:

* **How to reach the endpoint from one's own computer.** If the GPU node cannot be reached
  directly, the SSH tunnel has this form, to be completed with the real node, user and port:
  ```
  ssh -N -L 8000:<gpu-node>:8000 <user>@<login-host>
  ```
  With the tunnel open, `LLM_BASE_URL` stays `http://localhost:8000/v1`.

* **How to launch it as a job.** If the cluster uses a scheduler (Slurm or another), the
  submission script goes here: partition, GPUs requested, maximum time, module or container to
  load, and how to find out which node the job landed on, in order to open the tunnel to it.

Until these two lines are there, everything runs against the fake backend
(`tests/fakes/llm.py`), which is what the tests use: no network calls.
