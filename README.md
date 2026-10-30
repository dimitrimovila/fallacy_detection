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

Read [the research findings register](docs/research_findings.md) for completed results, limitations, annotation/provenance lessons and confirmed methodological choices. It includes the negative B3@2.0 diagnostic and distinguishes proposals from decisions. The narrative is versioned; raw runs, full evaluation tables and local audit ledgers are not included.

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
Exploratory routing adds `B2@2.0` and its `dq_routing_curve.csv`, `B2.dq` on
schemes 1.1, and `B2.dq@2.0` with D+Q and S-soft on schemes 2.0. The selected
penalties, including nested B1 fits, are recorded in `dq_choices.csv`.

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
argfallacy report cqs 20261019T151336Z_full_v3 --out scratch/cq_full_v3
```

The evaluation also writes a self-contained `summary.html` with coloured tables,
intervals, plain-language method names, CQ filters and embedded reliability plots.
To include the four-judge audit and revised questions in the same page, refresh it
from existing tables (no inference or metric recomputation):

```bash
argfallacy report summary evaluations/decision_v2 --judge-audit evaluations/_archive/judge_audit_20261021 --revised-cqs evaluations/_archive/cq_full_v3
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

Round-two B1 combines the five entries before traversing the diagram. Its primary
configuration is centred stacking supervised by the binary answers on the gold
scheme's `Annotazione Dumitru` sheet, with soft traversal on schemes 1.1.
`argfallacy.aggregate.b1_inputs.build_dataset` reads supplied `summary.csv` and
manifest frames; `argfallacy.eval.b1.cq_labels` selects those labels. The annotation
row must also judge the gold scheme present; answers from a row judging another
scheme do not enter the labelled population or the human path reference.

The offline API takes these prepared inputs, the shared folds and the evaluation
seed; it does not open a run or launch inference:

```python
from argfallacy.aggregate.b1 import b1_out_of_fold, configurations
from argfallacy.eval.b1 import cq_evaluation, entry_answers
import pandas as pd

configs = configurations(excluded_label_schemes=(
    "example", "cause_to_effect", "correlation_to_cause",
))
results = {c.name: b1_out_of_fold(data, labels, folds, seed, c) for c in configs}
predictions = pd.concat([entry_answers(data), *[r.combined for r in results.values()]])
cq_report = cq_evaluation(data, labels, predictions, indices=bootstrap_indices)
```

The named alternatives cover majority, Dawid–Skene, CQ-weighted A4 and stacking
supervised by gold verdicts, each with soft and hard traversal, including
`B1.hard`. Labelled fits use the other four outer folds and choose hyperparameters
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
alternatives on revised schemes. `transfer_b1(result, revised_data)` applies each
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

B2 routes between B1 and the reference zero-shot. B3 fits one decision tree of
depth at most three per scheme on B1's combined probabilities and outside masses.
Their offline APIs share a `B1CrossFitter`, which caches fits by their exact
training items. Training inputs come from four inner item folds; held-out inputs
come from B1 fitted on the complete training partition. Hyperparameter validation
rebuilds both levels inside each split, including B1's own parameter selection.
Globally out-of-fold B1 predictions are not reused as second-level training data.

```python
from argfallacy.aggregate import B1CrossFitter, Side, b2_out_of_fold, b3_out_of_fold
from argfallacy.eval.routing import routing_curve

source = B1CrossFitter(data, labels, seed)
zero = Side.from_summary(zeroshot_summary, f"zeroshot:{data.reference}", data.reference)
b2 = b2_out_of_fold(source, zero, folds)  # primary tau = 0
b3 = b3_out_of_fold(source, folds)
curve = routing_curve(b2, data.items, indices=bootstrap_indices)
```

The correctness regressions of B2 fit only disagreements between available
verdicts. Agreement selects the diagram; one available verdict is used directly;
no available verdict means abstention. Each regression chooses C by pooled inner
log loss. Its features are diagram confidence, terminal margin, abstention,
mean binary entropy in bits along the most probable path, zero-shot `p_label`
with a missing indicator, and the scheme. Missing probabilities use the mean of
all training items, or 0.5 if none is observed. A correctness target with one class
uses that constant probability; with no disagreement observations it uses 0.5.
The selected side's distribution is retained, while B2 confidence is its estimated
probability of correctness. The explanation includes the side and both estimates.

`b2.at_tau(tau)` applies another threshold to the same held-out estimates.
`routing_curve` reports every change in the curve between -1 and 1 and the
thresholds attaining at least 60, 70, 80, 90 and 100 per cent explained items.
The denominator includes every test item. Discrete jumps retain the achieved
share; unavailable diagram verdicts can make a target unreachable, which is
reported explicitly. The 95 per cent intervals reuse the evaluation's item
resamples with thresholds fixed from the pooled estimates.

B3 uses Gini splits and balanced class weights. Its terminal distribution is the
weighted leaf distribution; confidence is the unweighted share of its chosen
class among the training items of the leaf. Explanations list CQ features and
thresholds along the tree path. Tied validation scores prefer smaller depth,
then larger leaves. One training class gives that class; no eligible training
item gives an abstention. Gold labels absent from their scheme's terminals stay
out of tree and recalibration training.

Exploratory APIs accept `ReferenceInputs(data, seed, hard=True)` as B2's A0 side,
or `ReferenceInputs(data, seed)` for B3's tree on the reference hard answers.
That tree encodes the first/second arc as 1/0 and outside answers as a missing
split with outside mass 1. `b3_out_of_fold(source, folds, kind="pooled")` instead
fits one shared terminal-mass slope and one bias per label, shrinking toward
identity with L-BFGS and choosing C by inner log loss. It normalises over the
terminals of each item's scheme.

For revised schemes, `B1CrossFitter(revised_data, labels, seed, fit_data=data)`
fits only the original CQ labels and predicts the revised signals; supply the
zero-shot with matching revised menus. Additional fixed predictions, including
D+Q and S-soft when available, enter B2 as `extra_sides=(Side(name, predictions,
p_label=confidence, consistent=flags, schemes_version=version),)`. D+Q supplies
consistency flags; S-soft instead sets `explained=True`. Pass `schemes_version`
to the router as well. Each added confidence uses training-mean imputation and a
missing indicator; D+Q also supplies its consistency with its path. Each side
gets a correctness regression on the shared features; the largest
estimate among available sides wins, with ties to the diagram then input order.
A final verdict counts as explained when it equals B1, B4 or a consistent D+Q.
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
evaluated as round two. The command computes B1 (every combiner, supervision and
traversal on full run v2, the unsupervised, gold-fitted and transferred ones on full run
v3), B2 and its curve, B3 with the pooled recalibration and the tree on the reference
answers, A2.pruned, and B4, B4.path and B4.nohistory from the `ssoft.csv` of the runs
listed under `ssoft` and `ssoft_nohistory`; with those lists empty, B4 and its
comparisons are listed as missing and the rest is computed. The zero-shot run of the
reference model must have `p_label`, a feature of B2 and the confidence of its abstention
curve: a `summary.csv` without it is refused, so parse that run again where its
`raw.jsonl` is. `cq_label_exclusion` names the schemes whose CQ labels the exploratory
`B1.annotation_exclusion` leaves out; it is scored on the other schemes.

Added after the results, all exploratory and marked "(added after the results)" in their
description: `B1.stacking.gold_balanced.soft` and `.hard` and `B3.pooled.balanced`, the
models fitted on the gold verdicts with balanced class weights; `A2.prior` and `B1.prior`,
each terminal mass divided by the frequency of its class in the other folds; and
`A3.combined`, A3 of round 1 on the combined answers of the majority of the five entries,
beside `A3` itself. The command also scores every system and comparison on the macro F1
over the frequent classes (at least 10 gold verdicts in the test set), and puts the binary
human CQ answers in place of the reference model's, CQ by CQ and at random shares of the
disagreements, through the diagram 1.1.

The four deciding comparisons (B1, B2, B3 and B4 against their references, and the second
comparison of B1 and B3 against the zero-shot) have their interval at level 1 - 0.05/m with
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
cross-fitting of B2 and B3 refits B1 several hundred times: on runs of the size of the
test set the command takes about ten minutes on a laptop, most of it in B3.

## Analyses of saved evaluations

```bash
argfallacy analyze configs/analyses_v1.yaml  # evaluations/analyses_v1/
argfallacy report summary evaluations/analyses_v1 --format analysis
```

This independent, version-1 configuration reads saved out-of-fold predictions
from both rounds. All paths, including the output and optional inputs, resolve
relative to the configuration file's directory. `--out DIR` explicitly overrides
the output root (relative to the current directory). Output overlapping an input
or source evaluation is rejected. Seed, resamples, frozen population and folds
must agree with the source manifests. The command performs no fitting or inference.

The current component writes `main_table.csv`, `comparisons.csv`, `per_class.csv`,
`rare_classes.csv`, `integrity.csv`, `stage1.csv`, `routing_curve.csv`,
`confidence.csv` and `manifest.json`. It also writes `cq_level.csv`,
`cq_level_differences.csv`, `human_agreement.csv`, `human_path.csv`,
`abstention.csv`, `abstention_curves.csv`, `abstention_thresholds.csv`,
`abstention_differences.csv`, `abstention_composition.csv` and `difficulty.csv`.
Main metrics include percentile intervals,
item and supported-class denominators, absent classes and undefined flags.
Scheme and corpus breakdowns retain the small-group rules. Comparisons use the
common declared eligible items; intentionally restricted systems keep their own
populations. Additional restrictions can be declared per source with
`populations: {SYSTEM: {exclude_schemes: [...]}}` or `{item_ids: [...]}`.
The recorded `B1.annotation_exclusion` rule is read from its source manifest.
Original full-population deciding levels and reading gates are retained; new
views and diagnostic comparisons are exploratory at 95%.

`bootstrap_batch_size` defaults to 128 (valid range 1 to 4096). Systems are
processed sequentially. The same random streams are restarted in deterministic
batches, retaining scalar replicate statistics only. Subgroups filter whole-test
draws; without NLAS uses the existing generator 2. The manifest records input
hashes, source manifests and commits, analysis code identity, item order, folds,
versions, denominators, elapsed time and peak process memory. Tables count omitted
empty replicates per metric. The rare-class list uses gold support below 20 on the full frozen test
set, before any subset is selected.

The saved B2 curve has rounded thresholds and cannot reconstruct origin views.
Without exact side records, the new routing table has headers and a pending
manifest status. `argfallacy.eval.routing.routing_curve` accepts opt-in
`export_dir` and `source_evaluation` arguments; alternatively call
`argfallacy.eval.analysis_routing.export_routing_records` on an already computed
primary `B2Result`. This verifies saved verdicts and exports full-precision
side probabilities, verdicts and folds into a separate analysis/replay directory.
Configure its `routing_records.csv` under the source's `routing_records` key,
using a different final output root. Every origin then uses the pooled threshold
grid and the same thresholds selected for requested explained shares. Any fitted
replay must be separately profiled with unchanged settings; `analyze` never
starts it or reconstructs terminal distributions from chosen labels.

`annotations` optionally names the existing frozen annotations table (null by
default); the published configuration uses `data/annotations.csv`. It supplies
the historical 1.1 CQ/verdict ratings and Enrico's unchanged difficulty levels
1 to 5. B1's binary labels retain the positive scheme-judgement selection of
`annotations.gold_scheme_rows`; P7 also includes the gold-scheme sheet's negative
scheme judgements and nonbinary answers. CQ tables identify these populations,
item counts and pair counts separately, with pooled, scheme and individual-CQ
views in total and within each origin. Displayed hard human agreement requires
20 annotated items. Rewritten 2.0 CQs have no 1.1 human targets; S-soft rows count
observed path occurrences and distinct item/CQ pairs separately, excluding
unvisited nodes. Historical cross-sheet verdicts are separate from human paths
under the gold scheme. Missing second-rater CQ ratings and blind repeats stay pending.

Ordered abstention indices use P11's discrete means, item-id ties and missing-
verdict errors. Fixed cuts use half-up rounding. Each bootstrap draw is filtered
to the declared population and reranked, with all of an item's CQs moving together.
Threshold rows keep ties together, require a verdict and finite confidence, and
include an explicit empty selection at `+inf`. Thresholds retain full precision;
missing confidence stays missing. Coverage denominators, missingness and maximum
operational coverage are explicit. Systems without confidence are reference
points. The margin check is pending unless genuine terminal distributions exist
in a source's `verdicts.csv` `distribution` column (JSON terminal-to-mass mapping,
validated against saved confidence and verdict); distributions are never invented.
Only the four exploratory AUGRC pairs specified in spec 08 are exported.
Composition reports group rejection rates and availability among all, kept and
rejected items. Its Python adapter accepts optional aligned attributes, with null
values unobserved. Difficulty uses rated, operationally eligible items for
Spearman correlation with `1-confidence`; constant or insufficient samples and
unusable bootstrap replicates remain explicit.

`direct_verdict.answers` and `direct_verdict.samples` default to null. Both null
means pending with no human file reads; supplying only one is an error. Keep
external human paths in `configs/analyses_v1.local.yaml` (ignored). The reader
accepts the unchanged app CSV, including an explicitly selected final export
whose revisions start above zero. It selects the greatest revision, breaking
equal revisions by physical row order, and rejects conflicting repeats of any
revision. It validates timezone-aware timestamps, positions, sample membership,
gold schemes and canonical terminal ids in the item's 1.1 menu. The sample CSV
uses the existing `direct` identifier; CQ rows need no direct positions or rounds.
Partial samples, incomplete rounds and zero answers are accepted. The manifest
reports planned, answered and missing counts, round completeness, scheme/origin
counts and whether 200 answers have been reached.

`human_direct.csv` keeps direct-versus-gold disagreement separate from human paths
and historical raters, includes paired direct/path scores against gold, and scores
available systems against the direct human verdict as a second reference. It never
replaces gold in primary tables or supplies labels to fitted systems.
`uncertainty_disagreement.csv` reports confidence by observed agreement group,
AUROC of `1-confidence` (half credit for ties), and disagreement prevalence among
all, kept and rejected observed items. Selection ranks the global eligible system
population before intersecting human observations or origins, on the shared whole-
test item draws. Empty groups, single-class AUROC and unusable replicates stay
undefined. `abstention_composition.csv` adds the human attribute with explicit
known/missing denominators. These are associations with observed disagreement.

`dq` is an optional source evaluation directory containing `manifest.json` and
the dedicated `dq_*.csv` tables. The example selects `../evaluations/round2`;
`dq: null` skips dedicated D+Q tables and their parsed runs. The directory must
match an explicitly selected round-two source; no other evaluation is scanned or
pooled, including `round2_recomputed_spec10_20261026`. The four registered full
runs currently contain Qwen outputs; pilots and configured model lists add no
test populations. Exact `manifest.dq` mappings keep declared DQ, DQ.path, DQ.soft,
the version-2.0 zero-shot, the same-server rerun and exploratory B2 families
distinct. Parsed verdicts must agree before recovering `p_label` and its lower-
bound flag; path/soft/routing confidence is retained. Compatible 1.1 parsed D+Q
CQs enter the existing item-clustered diagnostics; 2.0 human targets remain pending.

The analysis imports the fourteen source tables under their original names:
`dq_verdicts.csv`, `dq_main_table.csv`, `dq_deciding.csv`,
`dq_deciding_breakdowns.csv`, `dq_exploratory.csv`, `dq_consistency.csv`,
`dq_departures.csv`, `dq_cq_level.csv`, `dq_cq_changes.csv`, `dq_replication.csv`,
`dq_abstention.csv`, `dq_abstention_curves.csv`, `dq_routing_curve.csv` and
`dq_choices.csv`. Their source values and rounding are preserved, with source
evaluation/table/hash/row keys. Imported abstention summaries remain separate
from new analysis curves. New breakdowns from saved predictions are exploratory.
H_DQ retains its own m=1 family at 95%, its same-server robustness reading and
the original round-two m=4 and round-one m=5 families. New AUGRC pairs are still
restricted to the exact four above. Rounded D+Q routing curves/choices cannot
supply exact per-item decisions; new replay-dependent results stay pending.

`source_conformal.csv` preserves the existing saved B5 aggregate summaries,
versions and source hashes even when exact per-item inputs are unavailable. It
does not reconstruct sets or add human-disagreement estimates.
`conformal_items.csv` contains genuine B5 sets, folds, terminal ids, sizes, gold
inclusion and full-precision calibration thresholds. Only systems selected by
the saved B5 calculation qualify. Genuine saved `distribution` columns or an
explicit source `conformal_records` directory can supply them. `conformal.csv`
is recomputed from those sets and checked against saved aggregates at their
serialization precision; human agreement groups are added only with explicit
human inputs. Rounded source aggregates and `round2_choices.csv` are insufficient.
Missing exact sources leave these tables pending without blocking other tables.

The opt-in Python API `analysis_conformal.export_conformal` accepts already
calculated genuine aggregations and exports them to a new, separate directory.
`round2.evaluate_round2(..., conformal_export_dir=DIR)` provides a companion hook
on the existing B5 calculation, with unchanged folds, alpha and terminal rules.
It verifies saved predictions and aggregates and records hashes and resource
measurements. Configure that directory as `evaluations.<identity>.conformal_records`
for a later analysis. `analyze` never initiates fitting, deterministic replay or
exports; any fitted replay needs its own explicit run, dependency audit and
separate resource measurements. Source commit metadata that is absent stays
unavailable. No real analysis or replay has been run as implementation validation.
`argfallacy report summary DIR --format analysis` regenerates `DIR/summary.html`
from that directory's statistical CSVs and manifest only. Automatic dispatch uses
`manifest.report_format: thesis_analyses_v1`; `--format evaluation` retains the
existing round-one/round-two renderer and its optional audit arguments. Analysis
reports reject evaluation audit arguments. They do not load `.env`, run analysis,
score predictions, fit systems, parse runs or read source annotations. Required
columns, numeric values, row counts and SHA-256 fingerprints are checked against
the saved table contracts. A missing optional table is pending only with an explicit
manifest reason. Table search and pagination work offline; the CSV links and full
source manifest preserve every row and denominator. HTML regeneration leaves the
statistical CSVs unchanged. Existing conformal aggregates remain visible beside
independently pending exact sets. Selected saved frequent-class summaries are imported
as post-results exploratory tables; the `<20` rare-class diagnosis remains distinct
from the `>=10` frequent-class cutoff. No substitution simulations are repeated.

### CPU post-processing and replay readiness

The three operations have distinct commands and resource scopes:

```powershell
# Saved-result analysis: no fitting; direct_verdict paths remain null.
python serving/postprocess.py --site local --environment current-python --mode analysis --input configs/analyses_v1.yaml --output evaluations/analyses_v1
# Report only: no external source content is opened.
python serving/postprocess.py --site local --environment current-python --mode report --input evaluations/analyses_v1 --output evaluations/analyses_v1/summary.html
# Readiness only: validates inputs for a separate fitted replay, without fitting.
python serving/postprocess.py --site local --environment current-python --mode replay --input configs/analyses_v1.yaml --output evaluations/analysis_replay --evaluation round2 --check-only
# Explicit heavy replay: run manually only after readiness and resource review.
python serving/postprocess.py --site local --environment current-python --mode replay --input configs/analyses_v1.yaml --output evaluations/analysis_replay --evaluation round2
```

Replay reconstructs only primary B2 and the primary B1 conformal companion. It
uses the selected source's original folds and grids, requires complete genuine
zero-shot `p_label`, and verifies saved predictions before accepting exports.
It never invokes a full evaluation. It profiles fitting separately and writes to
a new directory. Other B5 systems are outside this replay command and remain
pending unless they already have genuine saved distributions or companion exports.
If `p_label` is unavailable, recover the parsed source on the machine holding its
raw run before replay; rounded routing curves cannot repair it. Never overwrite
the selected evaluation or replace frozen predictions to obtain agreement.

The selected `round2` and `round2_recomputed_spec10_20261026` have identical saved
verdicts, main tables and deciding tables. Their base-system confidences differ
beyond serialization precision (maximum absolute difference 0.0014600673485322124
for B2 and 0.000009284341750293024 for B1); the dedicated D+Q verdict table is
byte-identical. The analysis keeps the selected original source and records its
hashes. Confidence-based results must not silently switch to the recomputed copy.
The replay reader uses the native evaluator's numeric missing-value convention
for fitting, while source-contract validation preserves literal empty fields.

For an analysis after a verified replay, copy `configs/analyses_v1.yaml` to an
ignored `configs/analyses_replay.local.yaml`, set the round-two `routing_records`
to the returned `routing/routing_records.csv`, `conformal_records` to the returned
`conformal/` directory, and choose a different analysis output root. Keep both
direct-human paths null. External human input has a separate, explicitly opted-in
local configuration; D+Q can be enabled independently with `dq: ../evaluations/round2`
or disabled with `dq: null`. Do not mix the recomputed or separate aggregator directories.

`serving/slurm/postprocess.sbatch` takes explicit site, Python executable,
environment identity, repository, mode, input and output arguments. It requests no
GPU, starts no server, sources no serving configuration and has no site defaults.
The following is a manual deployment template; all bracketed values are unresolved:

```bash
sbatch --partition='<CPU_PARTITION>' --account='<ACCOUNT>' --cpus-per-task='<CPUS>' \
  --mem='<MEMORY>' --time='<TIME>' serving/slurm/postprocess.sbatch \
  '<SITE>' '<ABSOLUTE_CPU_PYTHON>' '<ENVIRONMENT_ID>' '<ABSOLUTE_REPO>' \
  replay '<ABSOLUTE_ANALYSIS_CONFIG>' '<NEW_ABSOLUTE_REPLAY_OUTPUT>' --evaluation round2
# Use mode analysis with a separate output root, or mode report with INPUT/summary.html.
```

Transfer the committed code and dependencies, the chosen saved evaluations and
their manifests/CSV tables, frozen `data/items.csv` and `data/annotations.csv`,
scheme/label/burden files, and only the configured parsed run manifests and
`summary.csv`, `answers.csv` and `ssoft.csv` where used. Preserve hashes and
config-relative layout, or update the local configuration explicitly. Analysis
and report need no raw responses, model weights, new human logs, exports or sample
sheets. A replay blocked by missing parsed probabilities needs a separately
verified parsed export first. Mathematics is a deployment option, not a verified
account/location; the saved spec-10 runs used UPSCALE. The template supplies neither
site's credentials, paths or partitions. No cluster invocation is made here.

### Reproducing the full synthetic resource measurement

```powershell
python serving/benchmark_analysis.py --workdir ../scratch/report-benchmark-128 --batch-size 128
# Optional second full profile, against exactly the same invented source exports:
python serving/benchmark_analysis.py --workdir ../scratch/report-benchmark-64 --batch-size 64 --compare ../scratch/report-benchmark-128
```

Use new directories for each invocation. Both fresh processes use 601 invented
items, all eight schemes, 10,000 resamples, 21 round-one and 43 round-two verdict
systems, eleven imported D+Q/control/routing systems, five-model CQ grids, combined
CQ vectors and synthetic historical human/difficulty ratings. They run the actual
saved-result pipeline and the saved-table report, then compare statistical values
exactly after excluding only source hashes/row provenance. The second process
copies the same frozen invented inputs and changes only the batch boundary.
`benchmark.json`
records stage times, peak process memory, workload counts and limitations. It does
not cover fitting/replay, real external humans, S-soft path occurrences or genuine
conformal exports. The default remains batch size 128. A bootstrap-only profile
does not establish the full pipeline's resource use, and synthetic analysis memory
does not establish that fitted replay fits locally.

Acceptance checks compare every saved scalar estimate and interval across batch
boundaries on complete invented end-to-end cases, including optional humans,
D+Q and conformal sets. A separate 601-item, 10,000-resample check compares shared
draw fingerprints and scalar intervals at batch sizes 64 and 128, including omitted
subgroup replicates and the 95%, 98.75% and 99% levels. This draw/score check is not
the full-pipeline memory measurement.

The recorded full profile is in [docs/analysis_resources.json](docs/analysis_resources.json):
7,355.07 seconds for synthetic preparation and saved-result analysis, 8.89 seconds
for HTML, 7,363.96 seconds overall, and a peak Windows working set of 1,000.01 MiB.
It produced 39 CSV tables and a 105,919,803-byte self-contained report. Timing
excludes module imports and overlaps other validation work during part of the run;
peak memory includes the process lifetime. The synthetic parsed entries have one
sample each, and imported D+Q diagnostic summaries are sparse; the record names
these limits, source-code hashes and software versions. Actual-source analysis
and fitted replay require their own measurements. Batch size 128 is retained.

Collapsed main tables and the predicted-scheme verdict condition are deferred
to level 3; stage-one scheme recognition is reported separately.

## License

The code is released under the MIT License (see `LICENSE`). The files in `data/`
are not covered by it: the texts come from existing corpora, each under its own terms.


### Reading saved evaluations

Start with [evaluations/README.md](evaluations/README.md) or the browser entry
[evaluations/index.html](evaluations/index.html). They contain a question-to-table
map and selected saved point estimates. Five active collections remain at the
top level: `decision_v2`, `round2`, `exploratory_aggregators_v3`, `analyses_v1`,
and `b3_schemes2_diagnostic_20261029`.
Historical audits, standalone CQ 2.0 diagnostics and the alternative round-2
recomputation are preserved in `evaluations/_archive/`, with a relocation map.
Recorded historical provenance paths are unchanged.

`python serving/evaluation_catalog.py` refreshes the entry points and collection
guides using saved aggregate tables only. It does not fit or score systems.
The analysis HTML now embeds labelled previews of up to 25 rows per table; search
covers only the preview. Complete CSVs remain authoritative, and report validation
still checks every row and table hash. The previous complete embedded HTML is
preserved in `evaluations/_archive/reports/analyses_v1_complete_html_20261029.zip`.
For targeted reading, filter the CSV by system, version and population rather
than opening all detailed reports or importing all item-level records.

### Isolated B3 diagnostic on schemes 2.0

`argfallacy evaluate-b3 configs/b3_schemes2_diagnostic.yaml` runs only the
approved exploratory B3@2.0 versus transferred B1.hard@2.0 experiment. The
original CART grid and nested B1 isolation are retained; Qwen A0/A2 are saved
contextual references. The protocol is in
[spec 11](docs/specs/11_b3_schemes2_diagnostic.md). Results go to a new directory
under evaluations/; the command refuses to overwrite existing outputs.
It saves full trees, B1 fits, inner scores, exact OOF paths, paired IC95%,
common-coverage and origin/class sensitivities, and protected-input hashes.
The +0.03 practical-interest flag is separate from the interval and from
argumentative validity. Run artifacts contain no new LLM calls or annotations.

### Matched GOSDT/CART diagnostic on schemes 2.0

The fixed protocol is in [spec 12](docs/specs/12_gosdt_structure_learning.md).
The optional native GOSDT dependency is loaded by a separate worker from an
explicitly supplied interpreter and vendor directory; the command never installs
packages or creates cluster allocations. Supervision supports Windows Job Objects
and numbered Linux Slurm steps with verified cgroup-v2 memory enforcement.

```powershell
argfallacy evaluate-gosdt configs/gosdt_schemes2_diagnostic.yaml --worker-python <python-path> --worker-vendor <vendor-directory> --check-only
```

The readiness check uses invented inputs only and stops on inadequate free memory.
After execution is authorized and readiness is verified, omit `--check-only` to
run the experiment. Outputs go to a new evaluations directory; `--resume` verifies
recorded hashes before reusing completed checkpoints. The run retains all 601
arguments in canonical scoring and saves full trees, held-out library/replay
comparisons, candidate scores, selected configurations, native status/bounds and
resource records. Missing required solver fits make the scientific comparison
incomplete. The [CPU deployment runbook](docs/gosdt_cluster_runbook.md) describes
private runtime setup, verified input transfer and the synthetic preflight that
gates cluster execution. Native cluster enforcement remains a deployment check,
not a claim established by local tests.
