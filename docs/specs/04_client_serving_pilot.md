# Spec 04. Interviewer, serving, minimal parser and pilot

Version 2, 22 September 2026. Components: `src/argfallacy/client/`, `src/argfallacy/parse/` (minimal version), `serving/`.
Depends on: 01, 03 and the data (`docs/data.md`). Produces: `runs/<run_id>/`.

## 1. Serving on the cluster (`serving/`)

* `serving/models.yaml`: one entry per model with `tier`, `model_id` (the name vLLM serves it under), `revision` (the Hugging Face commit), `display_name`, `enabled`, `supports_logprobs`, `is_reasoning`, `reasoning`, `max_concurrency`, `notes`, and optionally `max_tokens`. Tier 1: Qwen3.8 27B, Gemma 4 31B, gpt oss 20b; tier 2: gpt oss 120b, K2 Horizon 32B; tier 3: GPT 5.5, disabled. The exact names of the Hugging Face repositories are fixed in `serving/models.yaml`.
* Reasoning conditions. Qwen3.8 and Gemma 4 switch reasoning on and off over the same weights, so each of them has two entries with the same `model_id` and `revision` and a different `reasoning`: `qwen3_8_27b` and `gemma4_31b` with `enable_thinking: false`, `qwen3_8_27b_think` and `gemma4_31b_think` with `enable_thinking: true` and the template's default level. The main condition is reasoning off. The `reasoning` goes into the cache key, so the two entries share no answers.
* The `max_tokens` of an entry applies instead of the configuration's. The entries with reasoning on need it, since they write their reasoning before the JSON: for them 8192. Being a generation parameter, it goes into the cache key and into the manifest.
* `serving/README.md`: a ten-line runbook. Launching vLLM with the OpenAI-compatible API and structured output enabled, with an example:
  ```
  vllm serve <model_id> --dtype auto --max-model-len 8192 --port 8000
  ```
  For Qwen and Gemma the two conditions are served by two launches of the same model, same vLLM version and same weights. The entry with reasoning off is served without a reasoning parser: with the parser on and no reasoning, vLLM from 0.19.0 onwards can silently skip the JSON schema constraint. The `_think` entry is served with vLLM's reasoning parser and with a `--max-model-len` large enough for `max_tokens` plus the prompt (16384): without a parser, structured output constrains the JSON from the very first token and the model cannot reason. The run proceeds model by model, so restarting the server between the two entries is enough.
  Then the instructions to reach the endpoint from one's own computer if the cluster requires an SSH tunnel, and the way to launch it as a job if the cluster uses a scheduler. The cluster-specific part is still to be completed.
* `serving/smoke_test.py`: sends a chat completion with `logprobs=true`, `top_logprobs=20` and `response_format` with the stage-two JSON schema on a test item, and prints the answer, the top_logprobs at the position of the first token of `answer`, and the latency. If the logprobs do not arrive, it says so plainly. On an entry with reasoning on it also says whether there was any reasoning (reasoning tokens greater than zero) and whether the answer was cut off (`finish_reason` equal to `length`). This is the first command to run on every model and on every new entry.

Environment variables: `LLM_BASE_URL` (for example `http://localhost:8000/v1`), `LLM_API_KEY` (any string for vLLM; the real key for a commercial provider). None in the code.

## 2. The client (`argfallacy.client`)

### 2.1 Call
A function `ask(request) -> RawResponse` built on the `openai` package (chat completions), with: `model_id`, messages, `temperature`, `seed`, `max_tokens`, `response_format` (JSON Schema), `logprobs` and `top_logprobs` when the model supports them. `RawResponse` keeps the full response exactly as it arrives (content, logprobs block, usage, `finish_reason`), plus latency and timestamp. Never summarised, never cleaned up.

### 2.2 Cache
SQLite in `runs/cache.sqlite`, a table with key = sha256 of (`model_id`, `prompt_version`, text of the rendered prompt, generation parameters, `sample_index`, `revision`, `tier`, `reasoning`) and value = the serialised `RawResponse`. The generation parameters are `temperature`, `max_tokens`, `seed`, `logprobs`, `top_logprobs` (only when logprobs are requested) and whether a JSON schema is sent, not the schema itself. `model_id` is the identifier the server serves, not the key or the `display_name` of `models.yaml`. The item and CQ identifiers are not in the key: the rendered prompt stands for them, so two items with the same text share their answers, and a template edit invalidates the entry even if the version string was not bumped. The cache is consulted before every call; a call already made is not repeated. The cache is never cleared automatically. Changing the prompt changes the key, so a new prompt version generates new calls only for that prompt.

### 2.3 Plan and execution
* `argfallacy run plan CONFIG` reads a YAML configuration file (items to include, stage, scheme condition `gold` or `predicted`, models, number of samples, prompt version) and prints how many calls are needed per model, how many are already in the cache, and an estimate of the duration given the concurrency. It calls nothing.
* `argfallacy run execute CONFIG` creates `runs/<run_id>/` with `run_id` = date and time plus the name of the configuration, writes `manifest.json`, makes the missing calls and appends every response to `raw.jsonl`. Deterministic order: by model, by item, by CQ, by sample.
* Concurrency with a limit per model (from `models.yaml`), retries with growing waits on 429 and 5xx errors, a timeout per call. A definitive error produces a line in `raw.jsonl` with `error` set, and the key does not enter the cache, so the call is retried on resume.
* Resume: relaunching the same `execute` after an interruption resumes from the missing calls, without duplicating lines in `raw.jsonl` (only what is not already in the cache is written).
* `predicted` condition: the plan uses the stage-one predictions of the model itself and generates stage-two calls only for the items whose predicted scheme differs from the gold one; for the others the results of the `gold` condition are valid here too, and the parser knows it.

### 2.4 Manifest
`manifest.json` with: `run_id`, full configuration, `model_id` and `display_name`, prompt version, schemes version (tag and hash of the content of `schemes/`), generation parameters, number of samples, start and end timestamps, number of calls planned, executed, served from the cache, failed, and the note `logprobs_available` per model.

### 2.5 `raw.jsonl`
One line per call: `run_id`, `item_id`, `stage`, `scheme_condition`, `scheme`, `cq_id` (empty for stage one), `sample_index`, `model`, `model_id`, `revision`, `tier`, `reasoning`, `prompt_version`, `json_schema` (the schema sent to the model with that call), `params`, `cache_key`, `from_cache`, `raw_response`, `latency_s`, `error`. The file is append-only.

## 3. Minimal parser (`argfallacy.parse`, pilot version)

`argfallacy parse RUN_ID` reads `raw.jsonl` and produces `answers.csv`, one row per call: the keys above, plus `answer` (canonical or `invalid`), `confidence`, `justification`, `p_logprob` for every admitted answer, `logprob_partial`, `parse_ok`, `finish_reason`. `parse_ok` is true only if the object of the answer respects the `json_schema` of its row (spec 03 section 5): an answer outside the enumeration, with `confidence` outside 0 and 100, missing a required field or with an extra field is `invalid`, and the row stays. Then `summary.csv`, one row per (item, stage, scheme, cq_id, model): hard answer from sample 0, `p_logprob`, `p_verbal`, `p_sample`, `n_samples_ok`, `idk`, `invalid_count`, according to spec 03 section 6. The full version of the parser is spec 05; this one must only suffice for the pilot report and goes no further.

## 4. The pilot

`configs/pilot.yaml`: 50 items chosen from `items.csv` with a gold scheme other than `none`, stratified by scheme in proportion with at least 4 per scheme, fixed seed; stage one and stage two in the `gold` condition; two models, Qwen3.8 27B and Gemma 4 31B, each in the two reasoning conditions, so four entries (`qwen3_8_27b`, `qwen3_8_27b_think`, `gemma4_31b`, `gemma4_31b_think`); five samples. Calls: 6960, per entry 250 of stage one and 1490 of stage two.

`items.csv` contains neither the near duplicates nor the removed items (`docs/data.md`, section 5), so the pilot draws from all the items with a gold scheme, that is, the test set of the experiments. The seeded draw runs within each stratum (same scheme).

`argfallacy pilot report RUN_ID...` produces `pilot_report.md` with, per model:
* stage one: accuracy on the gold scheme and a reduced confusion matrix;
* for every CQ: distribution of the answers, share of `cannot_be_determined`, share of `na`, share of `invalid`;
* agreement between the answer of sample 0 and the majority of the five;
* correlation between `p_logprob`, `p_verbal` and `p_sample` per CQ;
* how many traversals per scheme end incomplete because an answer is `na` (spec 03), with the node where they stop;
* mean latency and tokens per call, separating the reasoning tokens from those of the answer, share of truncated answers (`finish_reason` equal to `length`), and the time projection for the full run (all the items of `items.csv`, today 773, all the CQs, five samples).

Then, for each of the two models, the comparison between the two reasoning conditions:
* cost: ratio between the tokens and between the latencies of the two conditions, and the time projection of phase 2 with and without the reasoning-on condition;
* agreement between the answers of sample 0 in the two conditions, per CQ;
* shape of `p_logprob`: share of values above 0.99 or below 0.01, per CQ and overall. With reasoning on, the answer comes after a text that has already decided it, and the probability could concentrate at the extremes and lose information;
* stage-one accuracy and correlation between `p_logprob`, `p_verbal` and `p_sample`, already computed per entry, side by side.

Stopping rule written in the report: every CQ with more than 50 per cent `cannot_be_determined` or more than 10 per cent `invalid` on a model must be listed at the top of the report; the same 50 per cent threshold applies to `na`. The decision on what to do is not automatic: it is taken by reading the report. The same holds for the reasoning-on condition: whether it enters phase 2 is decided on the comparison above.

## 5. Acceptance tests

Since 20 September 2026 `tests/` keeps only the tests that defend a result, check the data or help the work. The points without a note have an automatic test; the marked points remain requirements, but today they have no test, or only a partial one.

All of them with the fake backend in `tests/fakes/llm.py`, which returns deterministic answers as a function of the prompt and of the `sample_index`, with a realistic logprobs block; no network calls in the tests.

1. Cache: two identical `ask` produce a single call to the backend; changing `prompt_version`, a parameter or `sample_index` produces another one.
2. Resume: a run interrupted half way (the fake backend raises an error after N calls) and relaunched completes the plan without duplicate lines in `raw.jsonl` and without redoing the successful calls.
3. Plan: on 10 test items with a known gold scheme, `plan` counts calls = sum over the CQs of the scheme of every item, times 5 samples, times the models, plus the stage-one calls.
4. Manifest: all the fields of section 2.4 present; the schemes hash changes if a byte of a YAML file is modified. *Today only the part on the fields has an automatic test.*
5. Parser: from a fake `raw.jsonl` with valid answers, invalid ones and partial logprobs, `answers.csv` and `summary.csv` have the expected values; `p_logprob` is renormalised over the admitted answers; an output that does not respect the schema is `invalid` and does not drop the row. *Today the valid answer, the out-of-schema one that stays as `invalid`, and the three probabilities of the summary have an automatic test; the case with partial logprobs does not.*
6. `predicted` condition: the plan generates stage-two calls only for the items whose predicted scheme differs from the gold one. *Today without an automatic test.*
7. `smoke_test.py` runs against the fake backend (`--model fake`, the default value) and prints the logprobs block. *Today without an automatic test: it is run by hand.*
8. `max_tokens`: an entry of `models.yaml` with `max_tokens` produces calls with that value instead of the configuration's, and a cache key different from that of an identical entry without it.

## 6. What not to do
* No parsing inside the client: the client saves, and that is all.
* No automatic cleanup of `runs/`.
* No prompt written in the code: only the templates of `prompts/`.
* No calls to the models in the tests.

## 7. Definition of done
Lint and tests green (commands in the README); `plan` and `execute` work against the fake backend and against vLLM on the cluster with one model (checked with `smoke_test.py`); `pilot_report.md` produced on 50 items, 2 models and their two reasoning conditions; summary with the numbers of the report and the CQs above threshold.
