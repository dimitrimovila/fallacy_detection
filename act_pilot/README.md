# ACT pilot: rewriting two critical questions

This folder holds a self-contained Python package, `act_pilot`, for a small experiment of the
thesis. Two critical questions (CQ) of Walton's schemes do not separate classes that they
should separate. The pilot looks for better wordings of those two questions, written by hand
and obtained with a reimplementation of the loop of ACT (Agentic Classification Tree, Grari et
al., arXiv:2509.26433) at depth one, and checks them against lexical baselines on outer
cross-validation folds.

No experiment has been run with this code yet: it has been tested with fake models and
synthetic data only.

## The two boundaries

| id | scheme | positive class ("yes") | negative class ("no") | original question |
|---|---|---|---|---|
| B1 | `popular_opinion` | `good_argumentation` | `ad_populum` | Is the fact that A is considered to be true (or false) by many people a good reason to believe that A is in fact true (or false)? |
| B2 | `ad_hominem` | `poisoning_the_well` | `ad_hominem_abusive` | Is a direct attack on the other person used as a pretext to discredit them even before the discussion has begun? |

Polarity is fixed: the answer "yes" stands for the positive class. Both boundaries are defined
in `configs/pilot.yaml` (scheme text, role of the question, class descriptions shown to the
optimizer, forbidden terms), so a third boundary is a new YAML block, not new code.

## Input files

Both are prepared outside this repository and pointed to by `data.items` and `data.folds`.

`items.csv`, one row per item:

| column | meaning |
|---|---|
| `item_id` | unique id |
| `text` | the argument |
| `scheme` | scheme id, for example `popular_opinion` (compared in lower case) |
| `gold` | gold class, for example `ad_populum` (compared in lower case) |
| `source` | corpus of origin: `NLAS`, `EthiX`, `LOGIC`, `Reddit`, ... |

Other columns are ignored. If the names differ (the repository's `data/items.csv` has
`gold_scheme` and `gold_fallacy`), `data.columns` in the configuration maps them:
`columns: {scheme: gold_scheme, gold: gold_fallacy}`.

`folds.csv`, one row per item: `item_id`, `outer_fold` (an integer from 0 to 4). Every item of
a boundary must have a fold; items of other schemes or classes are ignored.

## What a run does

1. **Estimate.** Before any call, the number of calls per boundary is counted
   (`cost_estimate.md`): answerer calls with and without the cache, optimizer calls from the
   minimum to the maximum, and a rough token count. `--max-calls` stops the run when the
   estimate is above a limit; `--dry-run` stops after the estimate.
2. **ACT stage**, for each boundary, outer fold and seed (`act/`):
   - the outer training folds are split into an inner training set and an inner validation
     set (20%, stratified by class and source; the seed of fold `f` is `split_seed + f`);
   - the seed of the loop is the original CQ;
   - at each of the `k = 10` steps the answerer answers the current question on every inner
     training and validation item; weighted Gini impurity and balanced accuracy are recorded
     on both;
   - for each group (answered yes, answered no) up to `m = 20` correct and `m` wrong training
     examples are sampled with the run's seed; the optimizer writes a critique, then a
     rewritten question;
   - the rewrite prompt states the constraints (a single yes/no question; a reformulation of
     the original CQ with the same role in the scheme; no name of the fallacy, of the class or
     of the scheme; no topic, person or word specific to the examples; at most two logical
     conjunctions); the same constraints are also checked in code (`constraints.py`), and a
     violating proposal is sent back with the reasons, at most three times;
   - all candidate questions are saved; the selected one is the best on the **inner
     validation** set (ties to the earliest step), never on training;
   - three seeds (0, 1, 2) change the examples shown to the optimizer.
3. **Evaluation** on the outer folds (`evaluation/`, `report.md`):
   - candidates: the original CQ, the hand-written rewrites (`manual_rewrites.yaml`), the
     question selected by ACT for each seed (one per outer fold, chosen without that fold),
     and four generic control questions;
   - primary metric: balanced accuracy of the pooled out-of-fold predictions; paired,
     class-stratified bootstrap (5000 resamples) of the difference with the original CQ, with a
     95% interval; exact McNemar test;
   - baselines on the same folds: TF-IDF + CART (depth by inner cross-validation) and TF-IDF +
     RuleFit (imodels, MIT licence); a source-only baseline is added for context;
   - everything again within each source that has at least 10 items per class; other sources
     are described only (counts and class recalls);
   - lexical predictability: out-of-fold AUC of a TF-IDF + logistic regression that predicts the
     **answer** of the LLM to each question, with the most weighted terms;
   - acceptance of each manual and ACT candidate (below).
4. **Content-validity template**: `content_validity_template.csv` lists every question to be
   rated by hand.

## Acceptance criterion

`acceptance.assess` accepts a candidate only if:

1. the paired bootstrap interval of its difference with the original CQ lies above zero;
2. its balanced accuracy is not below the best lexical baseline, globally and within every
   evaluable source;
3. every question of the candidate (one per fold for ACT) has content validity of at least 2
   on a 1 to 3 scale from every rater in `content_validity.csv`.

It is rejected as soon as one condition fails, and pending while (1) and (2) hold and the
ratings are missing. `content_validity.csv` is filled in by hand from the template, with the
columns `boundary`, `question`, `content_validity` (1, 2 or 3; empty means not rated) and,
optionally, `rater` and `notes`. Questions are matched ignoring case and spacing.

## Choices to review before the real run

- **"cannot determine" and unreadable answers.** The prompt asks for yes or no. A reply that
  says the answer cannot be determined is kept as a third category and counted per question;
  for the metrics it counts as "no" (`evaluation.undetermined_policy: negative`), because only
  "yes" stands for the positive class. The alternative `positive` exists; excluding such items
  would make the paired comparisons run on different item sets, so it is not offered.
- **Seeds and small groups.** When a group has fewer than `m` correct or wrong examples, all of
  them are shown and the seed only changes their order. With about 100 items per class the
  groups are larger than 20 and the seeds also change which examples are shown.
- **The rewrite prompt includes the earlier questions** with their training balanced accuracy
  (never the validation one). At temperature 0 with a cache, a step after a rejected one would
  otherwise send the same prompt and get the same rejected reply.
- **Source confound.** If no source has both classes, the per-source analysis is only
  descriptive and the report says so; `source_only` shows how much of the boundary the source
  alone explains.
- **Differences from the ACT paper**: depth one; seed = the original CQ instead of a generic
  question; selection on inner validation instead of training impurity; constraints checked in
  code; `m = 20` instead of 50. The algorithm was reimplemented from the paper; no code of the
  original repository (which has no licence) is used.

## Installation and tests

Python 3.11 or later. From this folder:

```bash
python -m pip install --editable ".[dev]"
```

```bash
python -m pytest
```

The tests use fake models and synthetic data and make no network call; the end-to-end test
runs the whole pipeline in about ten seconds.

## Running on the cluster with vLLM

The two roles can use different models, each behind its own OpenAI-compatible server. The
defaults of `configs/pilot.yaml` are Qwen3.8 27B (answerer) and Gemma 4 31B (optimizer), both
with reasoning off, as in `serving/models.yaml`. vLLM is installed and launched as described in
`serving/README.md`; `serving/serve_command.py <entry>` prints the model, its pinned revision
and its flags. In an interactive job with GPUs, start the two servers:

```bash
vllm serve $(python ../serving/serve_command.py qwen3_8_27b) --port 8000 --host 127.0.0.1
```

```bash
vllm serve $(python ../serving/serve_command.py gemma4_31b) --port 8001 --host 127.0.0.1
```

Then, in the client environment (this package installed), point the roles at the servers and
at the data:

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1 OPTIMIZER_BASE_URL=http://127.0.0.1:8001/v1 ACT_PILOT_DATA=/path/to/pilot_data
```

Count the calls first:

```bash
act-pilot estimate --config configs/pilot.yaml
```

Run the ACT stage, with a guard on the number of calls (adjust the limit to the estimate):

```bash
act-pilot run-act --config configs/pilot.yaml --max-calls 60000
```

A single boundary, fold or seed can be run with `--boundary B1`, `--folds 0 1`, `--seeds 0`;
results of other folds and seeds already on disk are kept. Every reply is cached under
`runs/act_pilot/cache/`, so an interrupted run resumes where it stopped, and every request is
logged in `runs/act_pilot/calls.jsonl`.

Evaluate and write the report:

```bash
act-pilot evaluate --config configs/pilot.yaml
```

Write the template of the ratings, copy it to `content_validity.csv`, fill in the scores and
evaluate again (only the acceptance changes; all answers come from the cache):

```bash
act-pilot validity-template --config configs/pilot.yaml
```

`act-pilot all --config configs/pilot.yaml` runs the three stages in sequence. Outputs go to
`runs/act_pilot/`, which git ignores.

## Outputs

```
runs/act_pilot/
  cost_estimate.md                 the estimate, with the split sizes per fold
  calls.jsonl                      one line per request (role, model, cache hit, attempts, reply)
  cache/                           the reply cache
  act/<B>_act_candidates.csv       every candidate of every ACT run, with scores and violations
  act/<B>_act_selected.csv         the selected question per outer fold and seed
  evaluation/<B>_items.csv         one row per item and candidate or baseline
  evaluation/<B>_questions.csv     one row per candidate or baseline: metrics and acceptance
  evaluation/<B>_sources.csv       the same per source
  content_validity_template.csv    the questions to rate
  report.md                        the report
```

## Licences of the dependencies

scikit-learn (BSD-3-Clause) and imodels (MIT). imodels is pinned with scikit-learn below 1.10
because RuleFit still passes `penalty="l1"`, which scikit-learn deprecates in 1.8 and removes
in 1.10.
