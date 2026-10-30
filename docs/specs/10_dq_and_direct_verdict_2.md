# Spec 10. D+Q and the direct verdict with the 2.0 menus

Version 1, 25 October 2026, written before any D+Q call and before any result of the direct verdict with the 2.0 menus. Components: the stage `dq` of the client, with `prompts/dq_v1.md` and `prompts/schemas/dq_v1.json`; the zero-shot stage on schemes 2.0; the parser rows of `dq`; the format check `argfallacy report format`; the job script `serving/slurm/runs.sbatch`; the configurations `configs/dq_pilot_v2.yaml`, `configs/dq_pilot_v3.yaml`, `configs/dq_full_v2.yaml`, `configs/dq_full_v3.yaml` and `configs/zeroshot_schemes_v2.yaml`; the D+Q part of `argfallacy evaluate configs/evaluation_round2.yaml`.
Depends on: 01, 03, 04, 05, 06, 07, 08, 09. Used by: 08.

## 0. Purpose

The direct verdict (zero-shot, spec 03, section 8) reads the text, the scheme and the verdicts of the diagram, and picks the verdict without the CQs. On full run v2 it reaches macro F1 0.478, above every route through the CQs computed so far. D+Q gives the model the same prompt plus the CQs of the scheme, with their options and the definitions of their answers, and asks it to answer every CQ before the verdict. One call gives three results:
* the verdict with the CQs in view;
* the answers through the diagram, an algorithmic verdict with no extra call;
* the consistency between the verdict declared and the verdict of the path.

The question is how the CQs should enter a system built on LLMs (question 2 of the thesis): here, inside the prompt of the verdict, as a guide (Castagna, Sassoon and Parsons 2024).

D+Q has a matched control: the direct verdict rendered on schemes 2.0. It differs from zero-shot v1 only in the menus of correlation to cause, example and slippery slope, 192 items. On those three schemes the verdicts of the 2.0 diagrams replace those of the 1.1 diagrams. Correlation to cause also loses its variable C.

Version 2.0 of D+Q is the primary one. Version 1.1 is the control, because the 2.0 questions were written while looking at the errors of full run v2 on the test set (section 6.3).

Two results of round 2, computed before this spec, shape the reading. They use no D+Q output.
* S-soft lost to `A2.pruned` (H4, −0.065 [−0.122, −0.003]). Its control without history placed the loss on the list of all the CQs in the prompt, not on the earlier answers: `B4.nohistory` against `A2.pruned`, −0.057 [−0.094, −0.011].
* With the list in view, 9.1 per cent of the 3,162 answers of `qwen3_8_27b` differ from those of full run v3, in both directions: 152 `no` became `yes`, 103 `yes` became `no`. The largest moves are on analogy CQ4 (`yes` from 28 to 52 per cent), example CQ3 (from 56 to 27 per cent), popular opinion CQ1 (from 17 to 30 per cent), expert opinion CQ2 (from 65 to 81 per cent) and slippery slope CQ3 (from 78 to 94 per cent). `na` on expert opinion CQ5.1 goes from 0 to 13 per cent.
* The list of the other questions changes how some questions are read, with no single direction.

D+Q shows every CQ by construction: the list is part of the treatment. The primary comparison is on the verdict, so it stays as planned. The answers of D+Q through the diagram are expected to lose against answers given in isolation. Their right reference is `B4.nohistory`, which has the same list and separate calls (section 6.2).

The reference model is that of spec 08, P10, `qwen3_8_27b`.

## 1. Common rules

* Condition and metrics: the 601 items of the test set, gold scheme condition, fine space (spec 08, P1 and P2). The deciding metric is macro F1. A missing or invalid verdict counts as wrong.
* Resamples, folds and seed: those of spec 08, P12 and P13, seed 20261019, 10,000 resamples. Every comparison is paired on the same resamples as round 2.
* Version of the schemes and of the answer definitions. A D+Q run on `schemes: v1` uses `prompts/stage2_v2_answers.yaml`; one on `schemes: v2` uses `prompts/stage2_v3_answers.yaml`. The burden table of the soft traversal is `schemes/burden/v1.yaml` for v1 and `schemes/burden/v2.yaml` for v2.
* Server. Every run of this spec is made on one server: one launch of the reference model on the UPSCALE cluster, in one job (section 4.4).
* Names of the systems, as in round 2: no suffix for schemes 1.1, `@2.0` for schemes 2.0:
  * `zeroshot:<entry>@2.0`, the direct verdict with the 2.0 menus;
  * `zeroshot.rerun:<entry>`, zero-shot v1 made again on the server of the D+Q runs;
  * `DQ:<entry>` and `DQ:<entry>@2.0`, the verdict declared by D+Q;
  * `DQ.path:<entry>` and `DQ.path:<entry>@2.0`, the hard path of the D+Q answers;
  * `DQ.soft:<entry>` and `DQ.soft:<entry>@2.0`, their soft traversal.

## 2. The prompt `dq_v1`

### 2.1 The template

`prompts/dq_v1.md` is `prompts/zeroshot_v1.md` with four changes, so that the only difference from the zero-shot is the CQs:

1. The task sentence becomes: "Your task is to answer every critical question of the scheme about it and then to choose the **one** verdict that best describes the argument, the way a careful and fair reader with a good general education would."
2. Between the text and the verdicts, a section:

~~~~text
## The critical questions

Answer each of these questions about the argument in the text. Besides the options
listed under a question, every question admits two more answers:

* `cannot_be_determined`: use this only if you have no basis at all for answering with
  one of the options listed under the question, even after using everything listed in
  the section "What to do". The text not stating the answer explicitly is not, by
  itself, a reason to use it. If you lean towards one option, however slightly, give it.
* `na`: {{na_definition}}

{{questions}}
~~~~

3. In "What to do":
   * "Where a verdict calls for an assessment" becomes "Where a question or a verdict calls for an assessment".
   * "Do not let your own view on whether the conclusion is true decide the answer." becomes "Do not let your own view on whether the conclusion is true decide an answer, unless a question asks about truth."
   * The last sentence becomes "Answer every critical question first, in the order listed. Then, with your answers in view, choose the single verdict, among those listed, that best describes the argument."
4. "Your answer" asks for the object of section 2.3, with the keys in the order shown. It describes the CQ keys and the verdict: "Each key `CQ...` takes one of the answers of its question, written exactly as listed; `verdict` is one of the verdicts above, written exactly as listed." The sentences on `confidence` and `justification` are those of `zeroshot_v1.md`, with "your answer" read as "your verdict".

`{{na_definition}}` is the `na` text of the answer definitions of the version, word for word. The two files have the same text today. The `cannot_be_determined` text is that of the definitions with two edits:
* `{options}` becomes "with one of the options listed under the question";
* "and lower your confidence" is dropped, since the answers carry no confidence of their own.

`{{questions}}` lists every CQ of the scheme in the order of the YAML, word for word:

~~~~text
**CQ1.1.** <text of the question>
Options: `yes`, `no`.
* `yes`: <its own definition>
* `no`: <its own definition>
~~~~

* The options are the `answer_space` of the diagram, in the order of `listed_options` (spec 03, section 4).
* The bullet lines appear only for a CQ with an entry of its own under `per_cq`. The `default` definitions of `yes` and `no` are not printed: they only restate the question.
* The verdicts are rendered by the function of the zero-shot (spec 03, section 8), so the block of the verdicts is the same text in the two prompts.

Nothing in the prompt says where an answer leads. It names no arc, no terminal and no rule from answers to verdicts. It does not ask the verdict to follow the answers: a prompt that asked for it would make the consistency of section 5.3 an empty measure.

### 2.2 Rendering

`render_dq(item, scheme, definitions)` renders `dq_v1`, where `definitions` is the version of the answer definitions of section 1. The scheme block, the text and the verdicts are filled by the same functions as the zero-shot. The function is deterministic.

### 2.3 Answer format

```json
{"CQ1": "no", "CQ1.1": "yes", "CQ2": "yes", "CQ3": "yes", "CQ4": "no", "verdict": "Weak Analogy", "confidence": 70, "justification": "..."}
```

`prompts/schemas/dq_v1.json` holds `verdict`, `confidence` and `justification`, as `zeroshot_v1.json` holds `answer`, `confidence` and `justification`. For each scheme the renderer builds the schema of the call:
* it puts one property per CQ in front, in the order of the YAML, each with the enum of `answer_options` (the `answer_space` plus `cannot_be_determined` and `na`);
* it narrows `verdict` to the names shown for the terminals;
* every property is required, and `additionalProperties` is false.

The properties are written in this order and the server generates them in this order (xgrammar, compact layout, spec 04, section 1). The format check of section 4.3 verifies it on every call.

## 3. The direct verdict with the 2.0 menus

* `configs/zeroshot_schemes_v2.yaml`: `zeroshot_v1.yaml` with `name: zeroshot_schemes_v2` and `schemes: v2`. Same five entries, same `prompt_version: v1`.
* The template, the verdict names and the definitions are those of zero-shot v1. Only the scheme files change.
* On the frozen data, the rendered prompt is identical to that of zero-shot v1 on the 409 items outside correlation to cause, example and slippery slope. The cache key is therefore the same (spec 04, section 2.2). The two runs differ on the other 192 items.
* The pairing rule of `RunConfig` (`schemes: v2` only with `prompt_version: v3`, and the reverse) applies to the stages `stage1` and `stage2` only. `zeroshot` and `dq` accept `schemes: v1` and `schemes: v2` with `prompt_version: v1`, as `judge` already does. `ssoft` keeps its own rule, `schemes: v2` only. Nothing else in the rule changes.
* The repeat of zero-shot v1, `zeroshot.rerun`: the run of the existing `configs/zeroshot_v1.yaml` with `--model qwen3_8_27b`, made in the same job after the direct verdict with the 2.0 menus. The 409 items whose prompt is identical come from the cache of that run. The other 192 are new calls, on the same server as the D+Q runs.

## 4. The stage `dq` in the client

### 4.1 Plan and raw lines

* A configuration with `stages: [dq]` and `prompt_version: v1` plans one call per item under its gold scheme, with the samples of each entry, as the zero-shot does. It cannot be mixed with other stages.
* The definitions follow `schemes` (section 1). The manifest records their file and its sha256.
* The lines of `raw.jsonl` have `stage` `dq`, the `scheme` of the prompt and an empty `cq_id`. Nothing else in the line changes.
* The cache key keeps its fields (spec 04, section 2.2). The CQs, their definitions and the verdicts are in the rendered prompt.

### 4.2 Configurations

* `configs/dq_full_v3.yaml`: `stages: [dq]`, `schemes: v2`, the 601 items of the test set (`exclude_none_scheme: true`), the five entries of `zeroshot_v1.yaml` in its order, `prompt_version: v1`.
* `configs/dq_full_v2.yaml`: the same with `schemes: v1`.
* `configs/dq_pilot_v3.yaml` and `configs/dq_pilot_v2.yaml`: the same two, on the 50 items of `configs/pilot2.yaml` (same `n`, `seed` and `min_per_scheme`), with `qwen3_8_27b` only.

`argfallacy run plan` counts:
* 601 calls for `qwen3_8_27b` on each full configuration;
* 50 on each pilot;
* 601 on `configs/zeroshot_schemes_v2.yaml` for `qwen3_8_27b`.

The calls of a pilot are the calls of the full run on the same items, so the full run takes them from the cache.

### 4.3 The format check

`argfallacy report format RUN_ID [--model ENTRY]` reads `answers.csv`, `manifest.json` and `raw.jsonl` of a parsed `zeroshot` or `dq` run. It prints one table per entry and exits with a code other than zero if any check fails. The checks, each on every planned call of the entry:
1. the call has a line without `error`;
2. the answer is `parse_ok`;
3. `finish_reason` is not `length`;
4. the keys of the object, in the content after any reasoning, are in the order of the properties of the `json_schema` of the line;
5. for an entry with logprobs (`logprobs_available` in the manifest):
   * every verdict row has a `p_label`;
   * every CQ row of a `dq` call has its `p_logprob_<answer>` masses, that is, the position of its value was found.

The check reads no gold, no annotation and no accuracy. The share of `cannot_be_determined` and `na` per CQ is printed as information and stops nothing.

### 4.4 The job

`serving/slurm/runs.sbatch ENTRY CONFIG RUN_ID [CONFIG RUN_ID ...]` serves one entry once and runs every pair in order. After the server is up and the smoke test passes, for each pair it runs:

1. `argfallacy run plan CONFIG --model ENTRY`, which prints the cached and missing calls in the log;
2. `argfallacy run execute CONFIG --run-id RUN_ID --model ENTRY`;
3. `argfallacy parse RUN_ID`;
4. `argfallacy report format RUN_ID --model ENTRY`.

A failed execute, parse or format check stops the job before the next pair. A resubmission with the same arguments makes only the missing calls. The `#SBATCH` lines are those of `pilot.sbatch`; on another cluster partition, account and resources are passed on the command line. The script shares `serving/slurm/common.sh` and launches nothing by itself.

The order for this spec is:
1. `dq_pilot_v3`;
2. `dq_pilot_v2`;
3. `zeroshot_schemes_v2`;
4. `zeroshot_v1` (the repeat);
5. `dq_full_v3`;
6. `dq_full_v2`.

A pilot that fails its format check stops the job before any full run.

## 5. Reading the answers of D+Q

### 5.1 Parser

The parser expands a `dq` line into one row of `answers.csv` per CQ of the scheme and one verdict row. The rows share the line's `cache_key`, `parse_ok` and `finish_reason`.

* **The verdict row:** `cq_id` empty. `answer` is the terminal id of the name in `verdict`, read back through `labels/fallacies.yaml`, as at the zero-shot stage. `p_label` is computed with the key `verdict` (spec 04, section 4.2). It carries `confidence` and `justification`.
* **A CQ row:** `cq_id` set and `answer` the value under that key. The `p_logprob_<answer>` masses are read at the first token of that value, through `answer_position` and `logprob_masses` with the CQ id as the key. `confidence` and `justification` are empty.
* An object that fails its schema makes every row of the line `invalid`.
* The rows of a `dq` run also get `keys_in_order` (check 4 of section 4.3). The column is written only when the raw input includes the stage `dq`, as `path` and `depth` for `ssoft`. The tables of the other runs do not change.

`summary.csv` groups as before, by item, stage, condition, scheme, CQ and model:
* **The verdict row** follows the verdict stages (spec 05, section 5): `p_label`, `p_verbal`, and `p_sample` with several samples.
* **A CQ row** follows stage two (spec 05, section 4): `p_logprob_<answer>`, `p_logprob`, `p_sample_<answer>` and `p_sample`. `p_verbal` is empty, since no CQ has a stated confidence. With several samples the hard answer of a CQ is the majority of its valid samples, CQ by CQ, with the tie rule of spec 04, section 3.

The probability of a CQ is the probability of its answer given the prompt and the answers written before it in the same object: the model writes the CQs in order. At temperature 0 the earlier answers are the greedy ones.

### 5.2 The answers through the diagram

* `DQ.path`: the rule of A0 on the hard answers of the D+Q rows. An answer outside the arcs (`cannot_be_determined`, `na`, `invalid`) stops the path with no verdict. No call is added.
* `DQ.soft`: the traversal of `A2.pruned` (spec 09, section 6.2), with the burden table of the version of the schemes and threshold 0.02. The masses come from the source of each entry as A4 reads it: `p_logprob` for `qwen3_8_27b` and `gemma4_31b`, `p_sample` for the two gpt oss. K2 Horizon has no soft source on the CQ rows (no stated confidence per CQ, answer-token logprobs at 1.00), so it gets no `DQ.soft`.
* The masses of a later CQ were computed under the earlier answers actually written. On a branch the model did not take they are an approximation; the soft traversal treats them as given.

### 5.3 Consistency

For each D+Q run and entry, on the 601 items:
* **Consistency**, the main measure: the share of items whose `DQ.path` has a verdict equal to the verdict declared. An item whose path stops counts as not consistent.
* **Beside it:**
  * the share of items whose path stops;
  * the agreement on the items where the path has a verdict;
  * the mean mass that `DQ.soft` gives to the declared verdict;
  * the share of items where the declared verdict is the most probable terminal of `DQ.soft`;
  * the accuracy of the declared verdict and of `DQ.path` on the consistent items and on the others.
* **The departure node.** On the items that are not consistent, the departure node is the first node of the hard path whose chosen arc leaves no route to the declared verdict. If the path stops at an answer outside the arcs while the declared verdict is still reachable, the departure is that stop. `dq_departures.csv` counts the departures per scheme, CQ and answer.

Every share has a 95 per cent interval on the resamples. The table is given in total, by scheme and by gold origin.

Consistency is necessary for a faithful explanation, not sufficient: a verdict can agree with a path whose answers were bent toward it (Jacovi and Goldberg 2020; Turpin et al. 2023). Section 6.2 measures the answers themselves against Dumitru's labels.

## 6. Comparisons

### 6.1 The primary comparison

H_DQ:
* `DQ:qwen3_8_27b@2.0` against `zeroshot:qwen3_8_27b@2.0`, paired, macro F1, fine space, gold scheme condition, 601 items.
* The interval is at 95 per cent, the 2.5 and 97.5 percentiles of the paired bootstrap. It is a family of its own, with one comparison, fixed here before the runs. It is not part of the m = 4 of round 2.
* An improvement is claimed only if the interval lies above zero. An interval below zero is reported as the cost of the CQs in the prompt. An interval that contains zero is reported as no measurable difference, not as equivalence. With 601 items the interval is about ±0.05 to ±0.07 (Card et al. 2020), so only large effects are visible.
* Reported beside, without deciding:
  * the same difference in accuracy and in balanced accuracy;
  * the comparison by gold origin, by scheme (schemes under 30 items flagged) and without NLAS (spec 08, P4), with 95 per cent intervals.

### 6.2 Secondary and exploratory comparisons

All are paired, on macro F1 with 95 per cent intervals unless stated, and none decides anything. The first two are the secondary comparisons.

1. D+Q 1.1 against the direct verdict on the same server: `DQ:qwen3_8_27b` against `zeroshot.rerun:qwen3_8_27b`.
2. The consistency of section 5.3, for both versions.
3. The answers of D+Q through the diagram against answers given in isolation:
   * `DQ.path@2.0` against `A0:qwen3_8_27b@2.0`;
   * `DQ.soft@2.0` against `A2.pruned`;
   * on schemes 1.1, `DQ.path` against `A0:qwen3_8_27b`.
4. The answers of D+Q through the diagram against S-soft: `DQ.soft@2.0` against `B4`, and `DQ.path@2.0` against `B4.path`.
5. `DQ.soft@2.0` against `B4.nohistory`. The two have the same list of the CQs; they differ in that D+Q answers every CQ in one generation, each after the earlier ones, with the verdicts in view. The comparison separates this part of D+Q from the effect of the list, already measured by `B4.nohistory` against `A2.pruned`.
6. The answers at the level of each CQ (spec 09, section 2.5): on the 2,656 binary labels of schemes 1.1, the agreement of the hard answers of `DQ:qwen3_8_27b` and the Brier score of their `p_logprob`. They are set against those of `qwen3_8_27b` in full run v2 on the same pairs. Every pair of an item moves together in the resamples. Reported in total and by scheme. This measures with far more power than 601 verdicts whether answering inside the prompt of the verdict changes the answers.
7. The answers that change, descriptive, per scheme and CQ: the share of items whose hard answer differs between `DQ@2.0` and full run v3, and between `DQ@2.0` and `B4.nohistory`; on schemes 1.1, between `DQ` and full run v2. Each share comes with the share of the first admitted answer in each source.
8. The menus and the server, descriptive and exploratory:
   * `zeroshot:qwen3_8_27b@2.0` against the zero-shot v1 of round 2, on the 601 items and on the 192 items where the menus differ;
   * the verdict agreement between the two on the 409 items whose prompts are identical;
   * the agreement between `zeroshot.rerun` and the zero-shot v1 of round 2 on the 601 items.

   The 409 items are found by rendering, not written in the code. When the cache did not hold the earlier calls, the agreement on them measures the change of server.
9. Abstention (spec 08, P11, with the rules of the round 2 implementation): AUGRC, AURC and the selective accuracy at 100, 95, 90, 80 and 70 per cent for:
   * `DQ@2.0` and `DQ` (confidence `p_label`);
   * `zeroshot@2.0` and `zeroshot.rerun` (`p_label`);
   * `DQ.soft@2.0` and `DQ.soft` (the mass of the verdict).

   The paired differences in AUGRC, D+Q against its control, are exploratory.

### 6.3 Reading

* **The bias of the 2.0 questions.** The 2.0 questions were written while looking at the errors of full run v2 on the test set. The primary comparison gives D+Q the tuned questions and its control only the 2.0 menus, so it leans in favour of D+Q. Rule fixed now: an improvement in H_DQ is read as robust only if the point estimate of comparison 1 (D+Q 1.1 against `zeroshot.rerun`) is also above zero. Otherwise it is reported as possibly due to the tuning. It is a rule of reading, not a second test. An interval of H_DQ at or below zero needs no such check, since the bias goes in favour of D+Q.
* **D+Q above the direct verdict:** the CQs help as a guide to the verdict (Castagna, Sassoon and Parsons 2024).
* **D+Q level with it and consistent:** the same accuracy, with an explanation that leads to the verdict. The share of consistent items is the share of verdicts explained at no cost. It is read beside the curve of B2, where 61 per cent of the verdicts come from the diagram at the macro F1 of the direct verdict.
* **D+Q level with it or below, with low consistency:** the CQ answers do not explain the verdict (Turpin et al. 2023).
* **D+Q below:** imposing the CQs hurts the verdict. With the result of S-soft, the likely mechanism is the reading of each question with its siblings in view.
  * If consistency is high and the answers through the diagram are poor (comparisons 3 to 6), the verdict followed answers made worse by the list.
  * If consistency is low, the verdict did not follow them.
* **The answers through the diagram:**
  * Below the diagram 2.0: the effect of the list already measured by `B4.nohistory` is expected.
  * Comparison 5 says whether answering in one generation with the verdicts in view adds to it or recovers part of it.
  * Comparison 6 says whether the answers move away from Dumitru's labels. Lower agreement together with high consistency means answers bent toward the verdict.

## 7. D+Q in B2

Exploratory, as in spec 09, section 3. Two families:
* **Schemes 1.1:** `B2.dq`, with the sides B1, `zeroshot:qwen3_8_27b` (zero-shot v1 of round 2) and `DQ:qwen3_8_27b`.
* **Schemes 2.0:**
  * `B2@2.0`, with B1@2.0 (stacking (b) on full run v3, spec 09, section 2.4) and `zeroshot:qwen3_8_27b@2.0`, threshold 0, with its curve;
  * `B2.dq@2.0`, with B1@2.0, `zeroshot:qwen3_8_27b@2.0`, `DQ:qwen3_8_27b@2.0` and `B4`.

S-soft enters only the family of schemes 2.0, whose menus and diagrams are its own. This replaces the earlier plan of S-soft as a third side next to B1 on schemes 1.1, which the terminal `false_premise` of example 2.0 prevented.

* **Features.** Each side added beyond the two of spec 09 brings its own confidence as a feature: `p_label` for D+Q, the mass of the verdict for B4. A missing value takes the mean of the training items, with an indicator, as `p_label` in spec 09. D+Q also brings whether its verdict is consistent with its path. With more than two sides there is one regression per side and the side with the highest estimate is chosen, as spec 09 states; the curve exists only for two sides.
* **Explained.** A final verdict counts as explained when it equals the verdict of B1, of B4, or of a consistent D+Q.
* **Comparisons, exploratory:**
  * `B2@2.0` against `zeroshot:qwen3_8_27b@2.0`;
  * `B2.dq` against `zeroshot:qwen3_8_27b`;
  * `B2.dq@2.0` against `zeroshot:qwen3_8_27b@2.0` and against `DQ:qwen3_8_27b@2.0`.
* **Out of fold.** The inputs of B1 are cross-fitted as in spec 09, section 1, and no verdict depends on the labels of its own item.

## 8. Evaluation

### 8.1 Configuration

`configs/evaluation_round2.yaml` gets four keys under `runs`, each a list of run IDs, empty until the run exists:

```yaml
  zeroshot_schemes_2: []   # the run of configs/zeroshot_schemes_v2.yaml
  zeroshot_rerun: []       # the run of configs/zeroshot_v1.yaml repeated with the D+Q runs
  dq: []                   # the run of configs/dq_full_v2.yaml (schemes 1.1)
  dq_schemes_2: []         # the run of configs/dq_full_v3.yaml (schemes 2.0)
```

A configuration without these keys is still accepted, and nothing else in the configuration changes.

Each run must name the expected folder of `schemes/`, with the content hash of the current files: `v1` for `zeroshot_rerun` and `dq`, `v2` for the other two. `zeroshot_schemes_2` and `zeroshot_rerun` must also have the prompt version `zeroshot_v1`, and the two D+Q runs `dq_v1`. A mismatch stops the command.

A run in which the reference model has fewer than 601 verdict rows is incomplete. Its systems and comparisons are left out and listed as missing with the item ids, and the rest is computed.

### 8.2 The results of round 2 do not change

The D+Q part adds files and sections; it rewrites none of the existing ones. Every CSV that the evaluation of round 2 writes today is byte for byte the same in three cases:
* without the four keys;
* with the four keys empty;
* with the D+Q runs present.

These CSVs are `deciding.csv`, `deciding_breakdowns.csv`, `exploratory.csv`, `main_table.csv`, `verdicts.csv`, `cq_combined.csv`, `cq_level.csv`, `cq_level_differences.csv`, `human_path.csv`, `routing_curve.csv`, `ssoft_changes.csv`, `abstention.csv`, `abstention_curves.csv`, `abstention_composition.csv`, `conformal.csv` and `round2_choices.csv`.

The systems of this spec, those of section 7 included, appear only in the new files.

`manifest.json` keeps its existing keys and adds `dq`, with:
* the four run lists and the run of each system;
* the family of H_DQ and its level;
* the systems of the spec and what is missing.

The four entries of the round 2 `missing` list that wait for this spec ("B2 with S-soft as another side", "B2 with D+Q as another side", "B2 on schemes 2.0", "B4 against the zero-shot with the 2.0 menus") point to the D+Q section when the corresponding system is computed. The comparison of B4 against the zero-shot with the 2.0 menus (spec 09, section 6.4) is computed in `dq_exploratory.csv`. `summary.html` keeps its sections and adds one at the end.

### 8.3 Output

`argfallacy evaluate configs/evaluation_round2.yaml` also writes into `evaluations/round2/`:
* `dq_verdicts.csv`: the format of `verdicts.csv`, with the systems of sections 1 and 7;
* `dq_main_table.csv`: the columns of `main_table.csv`;
* `dq_deciding.csv`: H_DQ, with the difference, its interval, the outcome, and the reading rule of section 6.3 (`robust`, `possibly tuning`, or `not applicable`). It is always written; without the runs its outcome is `missing` with the reason;
* `dq_deciding_breakdowns.csv`: what section 6.1 reports beside;
* `dq_exploratory.csv`: the comparisons of sections 6.2 and 7;
* `dq_consistency.csv` and `dq_departures.csv`: section 5.3;
* `dq_cq_level.csv`: section 6.2, comparison 6;
* `dq_cq_changes.csv`: section 6.2, comparison 7;
* `dq_replication.csv`: section 6.2, comparison 8;
* `dq_abstention.csv` and `dq_abstention_curves.csv`: section 6.2, comparison 9;
* `dq_routing_curve.csv`: the curve of `B2@2.0`, as `routing_curve.csv`;
* `dq_choices.csv`: the hyperparameters of section 7;
* the D+Q section of `summary.html`, in this order:
  1. the primary comparison;
  2. what is reported beside it;
  3. consistency and departures;
  4. the level of each CQ;
  5. the exploratory comparisons;
  6. the abstention curves as a figure;
  7. B2;
  8. what was not computed.

The systems are computed for every entry present in the D+Q and zero-shot runs. The comparisons named in sections 6 and 7 are computed for the reference model.

## 9. Acceptance tests

On the diagrams of `schemes/v1/` and `schemes/v2/`, on lines and tables written by hand and on the fake backend; no run and no network. The fake backend writes, for a schema with several enum properties, an object with every property in the order of the schema and alternatives at the first token of every value.

1. **Prompt:**
   * rendering is deterministic;
   * every CQ of the scheme appears word for word, in the order of the YAML, with its options;
   * the definitions of `per_cq` appear under their CQ and the `default` definitions nowhere;
   * `cannot_be_determined` and `na` are defined once;
   * the scheme, the text and the verdicts are the same text as in the zero-shot prompt of the same item;
   * the definitions file follows the schemes folder;
   * no prompt contains a terminal id or the word "diagram".
2. **Schema:** the properties are the CQs in the order of the YAML, then `verdict`, `confidence` and `justification`, all required. Each CQ has the enum of `answer_options` (`positive` and `negative` for ad hominem CQ1). `verdict` has the names shown for the scheme.
3. **Pairing rule and zero-shot 2.0:**
   * `stage2` with `schemes: v2` and `prompt_version: v2` is still refused;
   * `zeroshot` and `dq` are accepted with both folders and `prompt_version: v1`;
   * on the frozen data the zero-shot prompt and its cache key are identical under the two folders exactly outside correlation to cause, example and slippery slope (409 items), and differ on the other 192;
   * `configs/zeroshot_schemes_v2.yaml` plans 601 calls for `qwen3_8_27b`.
4. **Plan, on the fake backend:**
   * the D+Q configurations plan 601 calls (full) and 50 (pilot) for `qwen3_8_27b`, all with `stage` `dq` and an empty `cq_id`;
   * an entry with five samples plans five per item;
   * a run interrupted and resubmitted completes without duplicate lines;
   * a pilot followed by its full run takes the 50 calls of the pilot from the cache.
5. **Parser:**
   * a `dq` line gives one row per CQ and one verdict row;
   * on a line whose CQs have different alternatives, each CQ gets the masses of its own position;
   * the key of a CQ is matched exactly: on a scheme with both `CQ1` and `CQ1.1`, and both `CQ2` and `CQ2.1`, each gets the masses of its own value, whatever their order in the object;
   * `p_label` is read under `verdict`;
   * an object that fails the schema makes every row `invalid`;
   * an object with keys out of order is `parse_ok` with `keys_in_order` false;
   * the tables of a run without `dq` have the columns they have today;
   * with five samples the hard answer of a CQ is its majority and the verdict the majority of the verdicts.
6. **Format check:** it passes on a clean fake run. It exits with a code other than zero on each of: a call with only an error line, an invalid object, a truncated answer, keys out of order, a CQ without masses, a verdict without `p_label`. It reads no gold.
7. **Traversals and consistency:**
   * `DQ.path` equals A0 on the same hard answers;
   * `DQ.soft` equals `A2.pruned` on the same masses within 1e-12;
   * on hand cases, consistency counts a stopped path as not consistent, and the departure node is the first arc that leaves no route to the declared verdict, or the stop.
8. **Round 2 unchanged:** on the hand-made runs of the round 2 tests, every CSV of section 8.2 is byte for byte the same without the four keys, with them empty, and with D+Q and zero-shot 2.0 runs present; the new files are only the `dq_` files of section 8.3.
9. **Deciding comparison:**
   * H_DQ uses the percentiles 2.5 and 97.5;
   * two identical systems give [0, 0] and no improvement;
   * the reading rule gives `robust` only with a positive point estimate of comparison 1;
   * without the runs, `dq_deciding.csv` has the row with outcome `missing`.
10. **CQ level:** on hand-written labels the agreement and the Brier score of the D+Q answers are those computed by hand, and all the pairs of an item move together.
11. **B2 with more sides:**
    * changing the gold of an item changes no verdict of that item;
    * equal verdicts on every side give that verdict, counted as explained;
    * the explained rule of section 7 holds;
    * S-soft is never a side of a family of schemes 1.1.
12. **Determinism:** the same seed gives the same outputs.

## 10. What not to do

* No change to `schemes/v1/`, `schemes/v2/`, the burden tables, the interpreter, the loader, the aggregators of rounds 1 and 2, `evaluations/decision_v2/`, and no change to the CSVs of `evaluations/round2/` (section 8.2).
* No change to `prompts/zeroshot_v1.md`, `judge_v1.md`, `ssoft_v1.md`, the stage two templates, the answer definitions or their JSON schemas. No change to the fields of the cache key.
* No sentence in `dq_v1.md` that ties an answer to a verdict or asks the verdict to follow the answers.
* No change to the primary comparison after a result: a new idea becomes an exploratory comparison, named as such.
* The new human annotations (direct verdict, re-annotation of schemes 1.1, annotation of schemes 2.0) enter no comparison of this spec.
* No label or scheme name written in the code; the 409 and 192 items are found by rendering.
* The format check reads no gold and no accuracy.
* No job launched on the cluster.

## 11. Documents to align in the same commit

* README: the stage `dq`, the zero-shot on schemes 2.0, the five configurations, `argfallacy report format`, `serving/slurm/runs.sbatch`, the four keys of `configs/evaluation_round2.yaml`.
* Spec 03: a section on `dq_v1` after the one on `ssoft_v1`. In section 8, the zero-shot on schemes 2.0.
* Spec 04: the stage `dq`, the pairing rule restricted to the CQ stages, the format check and `runs.sbatch`.
* Spec 05: the rows of `dq` and the column `keys_in_order`.
* Spec 08, "Round 2": the D+Q part of the evaluation and its family of one comparison.
* Spec 09, section 3: one sentence saying that the families of B2 with D+Q and S-soft are those of this spec, section 7.

## 12. Definition of done

* Lint and tests green, with the commands of the README.
* `argfallacy run plan` counts 601 calls for `qwen3_8_27b` on `configs/dq_full_v3.yaml`, `configs/dq_full_v2.yaml` and `configs/zeroshot_schemes_v2.yaml`, and 50 on each pilot.
* `bash -n serving/slurm/runs.sbatch` passes.
* `argfallacy evaluate configs/evaluation_round2.yaml` on the real runs leaves the CSVs of section 8.2 unchanged and writes `dq_deciding.csv` with outcome `missing` until the runs exist.
* After the runs: the systems and comparisons of sections 6 and 7 computed.
* Summary in Italian.

## References

* Card, D., Henderson, P., Khandelwal, U., Jia, R., Mahowald, K. and Jurafsky, D. (2020). With little power comes great responsibility. *EMNLP*, 9263–9274.
* Castagna, F., Sassoon, I. and Parsons, S. (2024). Critical-questions-of-thought: steering LLM reasoning with argumentative querying. arXiv:2412.15177.
* Jacovi, A. and Goldberg, Y. (2020). Towards faithfully interpretable NLP systems: how should we define and evaluate faithfulness? *ACL*, 4198–4205.
* Turpin, M., Michael, J., Perez, E. and Bowman, S. R. (2023). Language models don't always say what they think: unfaithful explanations in chain-of-thought prompting. *NeurIPS*.
