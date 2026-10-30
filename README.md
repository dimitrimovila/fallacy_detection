# Explainable fallacy detection through Walton's schemes

## Context

Master's thesis in Data Science, University of Padova.
Author: Dumitru Movila.
Supervisor: Prof. Giovanni Da San Martino.
Academic year: 2025/2026.

## Goal

Douglas Walton's approach to fallacy detection recognises the argumentation scheme a
text follows, then asks the critical questions of that scheme, and reaches a verdict
from the answers. This thesis starts from eight of Walton's schemes, turned into flow
diagrams (identification question, ordered critical questions, and terminals that are
either a specific fallacy or good argumentation) in the thesis of Enrico Bergamasco,
*Schemi argomentativi e domande critiche nei Large Language Models come strumento di
individuazione delle fallacie* [Argumentation schemes and critical questions in Large
Language Models as a tool for fallacy detection] (Master's thesis in Linguistics, University
of Padova, academic year 2025/2026). A first stage asks which of the eight schemes a text
follows, or none. A second stage puts each critical question of that scheme to several
large language models, one question per call; every answer comes both as a plain
answer (yes, no, cannot be determined from the text) and as a probability, from the
token log-probabilities, from the stated confidence and, for the models asked several
times, from the frequency over the samples. Six aggregators, from the fixed diagram to learned models, combine these
uncertain answers into a fallacy verdict, or an abstention, together with an
explanation: the path through the diagram, or the weights of the model that decided.

## Research results and decisions

Read [the research findings register](docs/research_findings.md) for completed results, limitations, annotation/provenance lessons and confirmed methodological choices. It includes the negative C3@2.0 diagnostic and distinguishes proposals from decisions. The narrative is versioned; raw runs, full evaluation tables and local audit ledgers are not included.

## Repository structure

```
configs/          run configurations (the pilot: items, stages, models, samples)
data/             the data: items.csv and annotations.csv
docs/             specifications per component, data log
labels/           label dictionaries: fallacies and schemes
prompts/          versioned prompt templates, JSON Schemas of the answers, answer definitions
schemes/          the eight scheme diagrams as YAML, one folder per version: v1/ (1.1, the default), v2/ (2.0);
                  burden/, the burden of proof of each critical question, one table per version
serving/          the models, how to serve them with vLLM, a smoke test, the cluster jobs
src/argfallacy/   the Python package: schemes, labels, annotations, prompts, client, parse, aggregate, eval, CLI
tests/            pytest suite, with a fake model backend (no network calls)
runs/             created at run time: raw JSONL answers and manifests (not versioned)
```

## Installation

Python 3.11 or later.

```bash
python -m pip install --editable ".[dev]"
```

The paths to data outside the repository are read from a `.env` file at the repository
root, which git ignores; no module holds a path. Create it with one `KEY=VALUE` per line,
Windows paths without quotes:

```
PRIOR_RUNS_DIR=<folder of the earlier runs, each with results.csv and metrics.csv>
WORKBOOK_PATH=<the annotation workbook, Dati_da_annotare.xlsx>
```

The model endpoint goes in the same file, as `LLM_BASE_URL` and `LLM_API_KEY`
(see `serving/README.md`).

## Data

The raw files are not included: they belong to the research group. They are the
results of earlier runs of four models on a test set of 607 rows, and the annotation
workbook. Both were frozen once into two tables:

* `data/items.csv`: 773 items, one row per text, with a fixed identifier. 601 are the
  test set of the experiments: the 607 rows hold 606 distinct texts, and five pairs of
  near duplicates (the same argument written almost identically) keep one item each.
  The other 172 items are only in the workbook.
* `data/annotations.csv`: one row per annotated cell of the workbook, with the
  annotator, the sheet and row, what the cell annotates, the canonical value and the
  cell as written.

`docs/data.md` records where every part comes from, what was corrected or
dropped and why, and how to read each value. The earlier results are rescored on the
same 601 items; reproducing their published numbers on all 607 rows is only an internal
check of the scorer.

## How to reproduce

Tests and lint:

```bash
python -m ruff check src tests serving
python -m pytest tests
for f in serving/slurm/*; do bash -n "$f"; done     # syntax of the cluster jobs
```

Updating the rows of the `Annotazione Dumitru` sheets in `data/annotations.csv` from the
workbook at `WORKBOOK_PATH` (each of those sheets carries an `item_id` column):

```bash
argfallacy annotations update
argfallacy annotations update --skip "Annotazione Dumitru correlation"   # leaves that sheet's rows as they are
```

Drawing the samples of the human annotation (the CQ annotation on schemes 2.0 and the
direct verdict, disjoint, with the seed of the configuration) and writing the sheet of the
direct verdict for a local annotation app. The folder must be outside the repository, since
the texts come from corpora with their own terms; it receives `direct_verdict_sheet.jsonl`
(one item per line: position, item, scheme name, text and the verdict menu of the zero-shot
prompt), `samples.csv` (every drawn item with its sample, and round and position for the
direct verdict) and `manifest.json` (seed, counts, commit, sha256 of the files). The same
seed gives the same bytes.

```bash
argfallacy annotations samples configs/annotation_samples.yaml --out /path/outside/the/repository
```

Serving one entry of `serving/models.yaml`, then checking it answers with
log-probabilities (the smoke test exits with a code other than zero if the entry cannot
be run):

```bash
vllm serve $(python serving/serve_command.py <entry>) --dtype auto --port 8000
python serving/smoke_test.py --model <entry>
```

The models are served with vLLM 0.30.0, installed with pip in a conda environment of its
own; the standard wheel needs an NVIDIA driver 580 or later, and `serving/README.md` gives
the CUDA 12.9 variant for older drivers. The model revisions in `serving/models.yaml` are
pinned Hugging Face commits, except the placeholder of the commercial model, disabled.
Calls refuse to start while a revision is a placeholder. Qwen and Gemma each have a second entry
with reasoning on (`_think`), served by a separate launch with the reasoning parser and
`--max-model-len 34816`; the entry with reasoning off is served without the parser. gpt oss
and K2 Horizon, whose reasoning does not switch off, are served in the same way as the
`_think` entries, with their own reasoning parser and the same context. The flags of each
entry are its `serve_args`; the servers of the runs compact the JSON
(`--structured-outputs-config`) and ignore the sampling defaults of the model
(`--generation-config vllm`). Each entry also states how many samples it takes and how they
are drawn: one at temperature 0 for Qwen and Gemma with reasoning off, five at the
producer's temperature for gpt oss and K2 Horizon. See `serving/README.md`.

Pilot (50 items, prompts v2; Qwen and Gemma with reasoning off, gpt oss 20b and K2 Horizon,
each with its own samples). A server serves one entry at a time, so each entry is run
after its own launch, and `--run-id` gathers the four into one run:

```bash
argfallacy run plan configs/pilot2.yaml                    # counts the calls, makes none
argfallacy run execute configs/pilot2.yaml --run-id <run_id> --model <entry>
argfallacy parse <run_id>                                  # answers.csv and summary.csv
argfallacy report run <run_id>                             # run_report.md
```

The report is computed from the files of the parser, the manifest, `data/` and the folder of
`schemes/` the run was planned on.
The token counts and the layout of the JSON come from `raw.jsonl`, read one line at a
time: without it (a copy of the run that has only the tables) the report says so and the
rest is unchanged, and `--no-raw` skips the reading. The same command reports any run:
only the items of the test set with a gold label are scored, the projection of the full
run appears for a run that covers part of the test set, and the comparison of the
reasoning conditions only when a model is in both. `argfallacy pilot report <run_id>`, the
earlier name, writes the same report as `pilot_report.md`.

The zero-shot condition gives each item of the test set, with its gold scheme, the verdicts
of that scheme's diagram and their definitions, and asks for the verdict directly, without
the critical questions: `configs/zeroshot_v1.yaml` (601 items, the five entries of the full
run) and `configs/zeroshot_pilot.yaml` (the 50 pilot items, two entries), run and parsed with
the same commands. The parser writes the verdict as its terminal id, with the probability of
the label chosen in `p_label`.

`configs/zeroshot_schemes_v2.yaml` uses the same zero-shot prompt and five entries
with `schemes: v2`. For the 601 test items, rendering gives 409 identical prompts
and cache keys across the two folders, and 192 different prompts.

The `dq` stage answers every critical question and then declares a verdict in one
generation. `configs/dq_full_v3.yaml` uses schemes 2.0; `configs/dq_full_v2.yaml`
uses schemes 1.1. Both contain the five zero-shot entries and plan 601 calls for
`qwen3_8_27b`. `configs/dq_pilot_v3.yaml` and `configs/dq_pilot_v2.yaml` contain
only that entry and the same 50 items as `configs/pilot2.yaml`. The definitions
follow the schemes folder and their file and SHA-256 enter the manifest.
The CQ keys precede `verdict`, `confidence` and `justification` in the schema.
Parsing writes one row per CQ and one verdict row, with `keys_in_order` in both
tables only for runs containing `dq`. CQ rows have their own logprob masses and
no stated confidence; the verdict row has `p_label` and the stated confidence.

```bash
argfallacy run plan configs/dq_full_v3.yaml --model qwen3_8_27b
argfallacy report format <run_id> --model qwen3_8_27b
sbatch serving/slurm/runs.sbatch qwen3_8_27b <config> <run_id> [<config> <run_id> ...]
```

The format report reads a parsed `dq` or `zeroshot` run, without gold labels or
accuracy. It fails on absent or failed calls, invalid or truncated answers,
keys outside schema order and, for entries with logprobs, missing CQ masses or
verdict probabilities. Outside-arc answer shares per scheme and CQ are informational.
`runs.sbatch` starts the entry once, runs the smoke test, then runs plan, execute,
parse and the format check for each pair in order, stopping at the first failure.
For D+Q the order is pilot v3, pilot v2, zero-shot schemes v2, the existing
zero-shot v1 repeated, full v3 and full v2. Resubmission reuses successful calls.

The D+Q evaluation reads the optional run lists `zeroshot_schemes_2`,
`zeroshot_rerun`, `dq` and `dq_schemes_2` under `runs` in
`configs/evaluation_round2.yaml`, initially empty. It checks schemes folder,
content hash, prompt version and item coverage. Incomplete runs exclude their
systems and name the missing items. D+Q adds only `dq_` CSVs, `manifest.dq` and a
final section of `summary.html`; existing round-two CSVs remain unchanged.
It computes the declared verdict, hard and soft CQ traversals, consistency,
CQ-level diagnostics, replication and abstention. H_DQ is its own family of one
comparison at 95%, with the reading rule based on the 1.1 control. Round-two
models are released before D+Q and its bootstrap uses the existing item draws.
Exploratory routing adds `C2@2.0` and its `dq_routing_curve.csv`, `C2.dq` on
schemes 1.1, and `C2.dq@2.0` with D+Q and S-soft on schemes 2.0. The selected
penalties, including nested C1 fits, are recorded in `dq_choices.csv`.

The judge stage (`stages: [judge]`) gives the same prompt plus the answers of one entry of a
run to the critical questions of the scheme, read from the `summary.csv` of that run, which
must have been parsed first: `configs/judge_full_v2.yaml` (601 items, `qwen3_8_27b` reading its own
answers in full run v2 with their justifications) and `configs/judge_full_v2_answers_only.yaml`
(the same without justifications), run and parsed with the same commands.

For full run v3, `configs/judge_full_v3.yaml` and
`configs/judge_full_v3_answers_only.yaml` use the same `judge_v1` prompt, with
`schemes: v2` and the answers of `20261019T151336Z_full_v3`. Each plans 601 calls.
The source manifest must name the same schemes folder; when it records a content hash,
that hash must match the current scheme files. A mismatch stops the plan before calls.
A legacy source manifest without a folder means `schemes/v1/`.
The v3 pairing rule applies to the CQ prompts, not to a judge-only run.

A configuration asks the CQs of `schemes/v1/` (version 1.1) unless it sets `schemes: v2`
(version 2.0). For CQ runs, this goes only with `prompt_version: v3`, and v3 only with it:
`configs/full_v3.yaml` asks the CQs of version 2.0 at stage two, under the gold scheme, with
the five entries of the full run. The manifest records the folder, and the parser and the
report read the diagrams from there; a manifest without it means `schemes/v1/`. On a run of
version 2.0 the report leaves out the agreement with the per-CQ annotation, which answers the
CQs of version 1.1.

On the cluster two Slurm jobs do this, launching and stopping the servers themselves
(preparation in `serving/README.md`):

```bash
sbatch serving/slurm/smoke.sbatch <entry>      # the smoke test of one entry
sbatch serving/slurm/pilot.sbatch configs/pilot2.yaml <run_id> [<entry> ...]   # in order
```

Scoring the earlier runs (the folder at `PRIOR_RUNS_DIR`) in both modes of the scorer:

```bash
argfallacy score prior                         # writes runs/scores/prior/
argfallacy score prior --out <folder>
```

The canonical mode rescores every run on the 601 items of the test set, with the gold of
`data/items.csv`: `metrics.csv` and `classwise.csv`, in the fine and the symmetric collapsed
label space, and `baselines.csv`, with the majority class per scheme and the information
ceiling of the answer vector. The compat mode counts like the scoring code of the earlier
runs, and its three files (`compat_*.csv`) reproduce theirs; it only checks the scorer and
gives no number of the thesis. `report.md` lists the rows that stand for no item, the gold
labels that the diagram of their gold scheme does not produce, and the predictions that are
no class of the label space.

The aggregators are functions of `argfallacy.aggregate`: A0 (the diagram), A1 (the diagram
with its blocks of questions in the order chosen on the other folds), A2 (the diagram traversed
with soft answers, the answers outside the arcs following the burden of proof of each critical
question), A3 (a logistic regression per scheme on the soft answers, trained on the other
folds) and A4 (the soft answers of several models averaged, then traversed as in A2). A5, the
LLM judge, is the verdict of a run of the judge stage. The five folds, stratified by scheme,
come from one seed (`argfallacy.eval.assign_folds`).

The evaluation compares them with A0 of the reference model:

```bash
argfallacy evaluate configs/evaluation_v2.yaml             # evaluations/decision_v2/
```

It reads `summary.csv` and `manifest.json` of the runs the configuration names, never
`raw.jsonl`, so the runs must have been parsed by the current parser. It writes, under
`evaluations/<name>/` (not versioned): `verdicts.csv` (one row per system and item, with the
verdict, the confidence, the abstention and the explanation), `a1_orders.csv` and
`a3_choices.csv` (the order of A1 and the C of A3 chosen in each fold and scheme), one CSV per
table (the main table, the deciding, diagnostic and exploratory comparisons, and the main
table and deciding comparisons without NLAS), `manifest.json` and `summary.md`. Every metric
has a 95 per cent bootstrap interval, on 10,000 resamples of the items drawn once from the
seed; the deciding comparisons, each primary configuration against A0 of the reference model
in macro F1, have their interval at level 1 - 0.05/m. The command stops if the reference
model, the entry whose A0 has the highest macro F1, is not the one the configuration names.

The runs of the judge stage enter the evaluation through the key `judge` of its
configuration. `runs.zeroshot` accepts one run ID or a list of runs containing disjoint
model entries with the same scheme and prompt versions. The manifest records the source
run of each zero-shot entry. The full v2 configuration includes K2 and both judge runs;
its output is `evaluations/decision_v2/`, replacing the previous output.

CQ diagnostics and human agreement are included in the evaluation folder as
`cq_report.md`, with distributions, nominal alpha, Brier scores, pooled reliability
diagrams and diagram sensitivity. They can also be refreshed without recomputing
the aggregators:

```bash
argfallacy report cqs 20261005T193017Z_full_v2 --out evaluations/decision_v2
argfallacy report cqs 20261019T151336Z_full_v3 --out evaluations/cq_full_v3
```

The evaluation also writes a self-contained `summary.html` with coloured tables,
intervals, plain-language method names, CQ filters and embedded reliability plots.
To include the four-judge audit and revised questions in the same page, refresh it
from existing tables (no inference or metric recomputation):

```bash
argfallacy report summary evaluations/decision_v2 --judge-audit evaluations/judge_audit_20261021 --revised-cqs evaluations/cq_full_v3
```

The page can be copied on its own and opened offline. `evaluate` regenerates the
core evaluation page; rerun the presentation command to include the optional audits.
The earlier `scratch/report_prototype/summary_report.py` is an exploratory report
for diagram and zero-shot predictions, not the current evaluation renderer.

Human CQ comparisons require the original schemes; revised questions are not joined
by CQ number. Identical duplicate annotation rows count once; conflicting CQ ratings
stop the report. Conflicting or coarse human verdicts are excluded and counted.
Unavailable second-annotator CQ ratings and blind repeats are named in the report.

Not implemented yet: the predicted scheme condition, the collapsed space, and the breakdowns
by scheme and corpus and the abstention curves of the first stage (round two has its own, see
below).

## Combining critical-question answers

Round-two C1 combines the five entries before traversing the diagram. Its primary
configuration is centred stacking supervised by the binary answers on the gold
scheme's `Annotazione Dumitru` sheet, with soft traversal on schemes 1.1.
`argfallacy.aggregate.c1_inputs.build_dataset` reads supplied `summary.csv` and
manifest frames; `argfallacy.eval.c1.cq_labels` selects those labels. The annotation
row must also judge the gold scheme present; answers from a row judging another
scheme do not enter the labelled population or the human path reference.

The offline API takes these prepared inputs, the shared folds and the evaluation
seed; it does not open a run or launch inference:

```python
from argfallacy.aggregate.c1 import c1_out_of_fold, configurations
from argfallacy.eval.c1 import cq_evaluation, entry_answers
import pandas as pd

configs = configurations(excluded_label_schemes=(
    "example", "cause_to_effect", "correlation_to_cause",
))
results = {c.name: c1_out_of_fold(data, labels, folds, seed, c) for c in configs}
predictions = pd.concat([entry_answers(data), *[r.combined for r in results.values()]])
cq_report = cq_evaluation(data, labels, predictions, indices=bootstrap_indices)
```

The named alternatives cover majority, Dawid–Skene, CQ-weighted A4 and stacking
supervised by gold verdicts, each with soft and hard traversal, including
`C1.hard`. Labelled fits use the other four outer folds and choose hyperparameters
with four inner folds split by item. Dawid–Skene instead fits all items once,
without labels. Weighted A4 chooses its signals jointly by inner Brier score;
ties prefer equal weights, then each entry's distribution signal and the fixed
logprob/sample/verbal order. Missing alternative signals do not contribute to
that split; the mean outside mass remains fixed.

All combiners decide only `p_first`. A hard answer uses its larger arc, with
ties going to the reference answer when it is on an arc, otherwise to the first
arc; outside mass above 0.5 gives an indeterminate answer. Fitted stacking and
Dawid–Skene can disagree with unanimous hard inputs; no unanimity override is
applied. The stacking centre and the two supervised objectives share all features.

`configurations(schemes_version="2.0")` supplies the unsupervised and gold-fitted
alternatives on revised schemes. `transfer_c1(result, revised_data)` applies each
original held-out fold's CQ-stacking coefficients without fitting revised labels.
`configurations(include_unstable=True)` enables the optional re-annotation
exclusion; pass binary paired labels as `repeats`. Its exclusions are estimated
within each training partition. `baseline_aggregations(data)` supplies unchanged
A2 of the reference entry and A4 for comparison.

The CQ report includes agreement, Brier scores and paired 95 per cent bootstrap
intervals overall and by scheme; all CQs of an item move together.
`human_path_reference` reports the human diagram path as a descriptive reference,
including coverage over annotated items and the test set. The evaluation of round
two runs all of this on the real runs (see "Round-two evaluation").

## Routing and learned trees

C2 routes between C1 and the reference zero-shot. C3 fits one decision tree of
depth at most three per scheme on C1's combined probabilities and outside masses.
Their offline APIs share a `C1CrossFitter`, which caches fits by their exact
training items. Training inputs come from four inner item folds; held-out inputs
come from C1 fitted on the complete training partition. Hyperparameter validation
rebuilds both levels inside each split, including C1's own parameter selection.
Globally out-of-fold C1 predictions are not reused as second-level training data.

```python
from argfallacy.aggregate import C1CrossFitter, Side, c2_out_of_fold, c3_out_of_fold
from argfallacy.eval.routing import routing_curve

source = C1CrossFitter(data, labels, seed)
zero = Side.from_summary(zeroshot_summary, f"zeroshot:{data.reference}", data.reference)
c2 = c2_out_of_fold(source, zero, folds)  # primary tau = 0
c3 = c3_out_of_fold(source, folds)
curve = routing_curve(c2, data.items, indices=bootstrap_indices)
```

The correctness regressions of C2 fit only disagreements between available
verdicts. Agreement selects the diagram; one available verdict is used directly;
no available verdict means abstention. Each regression chooses C by pooled inner
log loss. Its features are diagram confidence, terminal margin, abstention,
mean binary entropy in bits along the most probable path, zero-shot `p_label`
with a missing indicator, and the scheme. Missing probabilities use the mean of
all training items, or 0.5 if none is observed. A correctness target with one class
uses that constant probability; with no disagreement observations it uses 0.5.
The selected side's distribution is retained, while C2 confidence is its estimated
probability of correctness. The explanation includes the side and both estimates.

`c2.at_tau(tau)` applies another threshold to the same held-out estimates.
`routing_curve` reports every change in the curve between -1 and 1 and the
thresholds attaining at least 60, 70, 80, 90 and 100 per cent explained items.
The denominator includes every test item. Discrete jumps retain the achieved
share; unavailable diagram verdicts can make a target unreachable, which is
reported explicitly. The 95 per cent intervals reuse the evaluation's item
resamples with thresholds fixed from the pooled estimates.

C3 uses Gini splits and balanced class weights. Its terminal distribution is the
weighted leaf distribution; confidence is the unweighted share of its chosen
class among the training items of the leaf. Explanations list CQ features and
thresholds along the tree path. Tied validation scores prefer smaller depth,
then larger leaves. One training class gives that class; no eligible training
item gives an abstention. Gold labels absent from their scheme's terminals stay
out of tree and recalibration training.

Exploratory APIs accept `ReferenceInputs(data, seed, hard=True)` as C2's A0 side,
or `ReferenceInputs(data, seed)` for C3's tree on the reference hard answers.
That tree encodes the first/second arc as 1/0 and outside answers as a missing
split with outside mass 1. `c3_out_of_fold(source, folds, kind="pooled")` instead
fits one shared terminal-mass slope and one bias per label, shrinking toward
identity with L-BFGS and choosing C by inner log loss. It normalises over the
terminals of each item's scheme.

For revised schemes, `C1CrossFitter(revised_data, labels, seed, fit_data=data)`
fits only the original CQ labels and predicts the revised signals; supply the
zero-shot with matching revised menus. Additional fixed predictions, including
D+Q and S-soft when available, enter C2 as `extra_sides=(Side(name, predictions,
p_label=confidence, consistent=flags, schemes_version=version),)`. D+Q supplies
consistency flags; S-soft instead sets `explained=True`. Pass `schemes_version`
to the router as well. Each added confidence uses training-mean imputation and a
missing indicator; D+Q also supplies its consistency with its path. Each side
gets a correctness regression on the shared features; the largest
estimate among available sides wins, with ties to the diagram then input order.
A final verdict counts as explained when it equals C1, C4 or a consistent D+Q.
Use the `name` argument to identify exploratory results and parameter choices.

## Sequential critical questions

The `ssoft` stage asks questions along the revised diagrams, showing all the CQs
and the earlier answers on the current path. Each node takes one call. The shared
function `argfallacy.aggregate.expand` routes probabilities with `schemes/burden/v2.yaml`
and expands node arcs whose cumulative mass is at least 0.02. Terminal arcs are
never pruned. Different histories at the same CQ remain separate calls.
`argfallacy.aggregate.a2_pruned` applies the same expansion to independent CQ answers.
The result keeps terminal, abstention and pruned masses separate; unanswered nodes
are reported as pending. A probability transformation can be supplied before routing
and pruning; the default leaves probabilities unchanged.

```bash
argfallacy run plan configs/ssoft_full_v3.yaml            # 601 first calls
argfallacy run plan configs/ssoft_nohistory_full_v3.yaml  # 3,162 calls
```

The first configuration plans further levels during execution. The second sets
`history: false` and asks every CQ independently, with the same list of questions.
Both use Qwen, the gold scheme and original probabilities, without calibration.
The manifest records the expansion policy, prompt and burden hashes; its planned
count grows as nodes are discovered. Resume requires the same policy and inputs.
It reuses successful records and the cache, and retries only missing calls.

From the repository root on the cluster, using the existing job script:

```bash
mkdir -p runs/slurm
stamp=$(date -u +%Y%m%dT%H%M%SZ)
sbatch serving/slurm/pilot.sbatch configs/ssoft_full_v3.yaml "${stamp}_ssoft_full_v3" qwen3_8_27b
sbatch serving/slurm/pilot.sbatch configs/ssoft_nohistory_full_v3.yaml "${stamp}_ssoft_nohistory_full_v3" qwen3_8_27b
```

Keep the same run ID when resubmitting an interrupted job. `argfallacy parse RUN_ID`
writes `ssoft.csv`, with the stage-two probability columns plus `path` and `depth`.
These rows are separate from `summary.csv`; `answers.csv` retains each raw attempt.
The no-history control has an empty path and depth. With history, depth starts at zero.

Out-of-fold temperatures, their use during expansion and the four exploratory S-soft
configurations are deferred; calibrating an existing run cannot recover nodes that its
original pruning did not call.

## Round-two evaluation

```bash
argfallacy evaluate configs/evaluation_round2.yaml        # evaluations/round2/
```

A configuration whose `runs` name `answers_schemes_2`, `ssoft` or `ssoft_nohistory` is
evaluated as round two. The command computes C1 (every combiner, supervision and
traversal on full run v2, the unsupervised, gold-fitted and transferred ones on full run
v3), C2 and its curve, C3 with the pooled recalibration and the tree on the reference
answers, A2.pruned, and C4, C4.path and C4.nohistory from the `ssoft.csv` of the runs
listed under `ssoft` and `ssoft_nohistory`; with those lists empty, C4 and its
comparisons are listed as missing and the rest is computed. The zero-shot run of the
reference model must have `p_label`, a feature of C2 and the confidence of its abstention
curve: a `summary.csv` without it is refused, so parse that run again where its
`raw.jsonl` is. `cq_label_exclusion` names the schemes whose CQ labels the exploratory
`C1.annotation_exclusion` leaves out; it is scored on the other schemes.

Added after the results, all exploratory and marked "(added after the results)" in their
description: `C1.stacking.gold_balanced.soft` and `.hard` and `C3.pooled.balanced`, the
models fitted on the gold verdicts with balanced class weights; `A2.prior` and `C1.prior`,
each terminal mass divided by the frequency of its class in the other folds; and
`A3.combined`, A3 of round 1 on the combined answers of the majority of the five entries,
beside `A3` itself. The command also scores every system and comparison on the macro F1
over the frequent classes (at least 10 gold verdicts in the test set), and puts the binary
human CQ answers in place of the reference model's, CQ by CQ and at random shares of the
disagreements, through the diagram 1.1.

The four deciding comparisons (C1, C2, C3 and C4 against their references, and the second
comparison of C1 and C3 against the zero-shot) have their interval at level 1 - 0.05/m with
m = 4. A second comparison is marked as not read when the first does not pass, and a
component not computed keeps its place in m. Beside them, at 95 per cent: accuracy and
balanced accuracy, the same comparisons by gold origin, by scheme and without NLAS, and the
exploratory comparisons. Abstention curves (AUGRC, AURC, selective accuracy at 100, 95,
90, 80 and 70 per cent coverage, the composition of the abstained items) cover every system
with a confidence; conformal sets at alpha 0.10, with thresholds from the other four folds,
cover the systems with a distribution over the terminals.

It writes `manifest.json` (runs, seed, folds, m, primary configurations, grids, what was
not computed and why), `verdicts.csv`, `main_table.csv`, `deciding.csv`,
`deciding_breakdowns.csv`, `exploratory.csv`, `cq_combined.csv`, `round2_choices.csv`,
`cq_level.csv`, `cq_level_differences.csv`, `human_path.csv`, `routing_curve.csv`,
`ssoft_changes.csv`, `abstention.csv`, `abstention_curves.csv`,
`abstention_composition.csv`, `conformal.csv`, `frequent_classes.csv`,
`frequent_classes_differences.csv`, `substitution.csv`, `substitution_curve.csv` and
`summary.html`, which
`argfallacy report summary evaluations/round2` redraws from those tables. The nested
cross-fitting of C2 and C3 refits C1 several hundred times: on runs of the size of the
test set the command takes about ten minutes on a laptop, most of it in C3.

## License

The code is released under the MIT License (see `LICENSE`). The files in `data/`
are not covered by it: the texts come from existing corpora, each under its own terms.
