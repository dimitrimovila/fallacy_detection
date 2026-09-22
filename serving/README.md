# Runbook: serving a model and talking to it

Ten lines, in the order in which they are run.

1. **Launch vLLM** on the node with the GPU. `<model_id>` and `<revision>` come from
   `serving/models.yaml`; while `revision` is `PLACEHOLDER` the client does not start:
   ```
   vllm serve <model_id> --revision <revision> --dtype auto --max-model-len 8192 --port 8000
   ```
   The OpenAI-compatible API and structured output are on by default; if your
   version of vLLM asks for it to be enabled, add `--guided-decoding-backend outlines`.
   If the notes of the model in `models.yaml` list extra flags, they must be added here.
   Qwen and Gemma have one entry with reasoning off and one with reasoning on
   (`_think`), served by two separate launches of the same model. The entry with reasoning off is
   launched without a reasoning parser: with the parser on and no reasoning, vLLM from
   0.19.0 onwards can silently skip the JSON schema constraint. The `_think` entry is
   launched with the reasoning parser given in the notes of the model and with
   `--max-model-len 16384`, because it has `max_tokens` 8192; without a parser, structured output
   constrains the JSON from the first token and the model does not reason.
2. **Declare the endpoint** in the `.env` at the root of the repository, never in the code:
   ```
   LLM_BASE_URL=http://localhost:8000/v1
   LLM_API_KEY=any-string
   ```
   vLLM does not check the key, but the `openai` client insists that there is one.
3. **Check the model** before any run:
   ```
   python serving/smoke_test.py
   ```
   It prints the answer, the answer token inside the final JSON with its
   position, the `top_logprobs` at that token and the latency. If there is
   reasoning before the JSON (gpt oss, `_think` entries) and a naive reader would have ended up there, it says so. On a
   `_think` entry it also checks that there was reasoning and that the answer was not cut off. If the logprobs do not arrive, it says so plainly: then `supports_logprobs` in
   `models.yaml` must be set to `false` and `p_logprob` will stay empty for that model.
4. **Count the calls** before making them: `argfallacy run plan configs/pilot.yaml`.
5. **Run**: `argfallacy run execute configs/pilot.yaml`. It can be interrupted: relaunching
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
