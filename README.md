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
individuazione delle fallacie* (Master's thesis in Linguistics, University of Padova,
academic year 2025/2026). A first stage asks which of the eight schemes a text
follows, or none. A second stage puts each critical question of that scheme to several
large language models, one question per call; every answer comes both as a plain
answer (yes, no, cannot be determined from the text) and as a probability, from the
token log-probabilities, from the stated confidence and from the frequency over five
samples. Six aggregators, from the fixed diagram to learned models, combine these
uncertain answers into a fallacy verdict, or an abstention, together with an
explanation: the path through the diagram, or the weights of the model that decided.

## Repository structure

```
configs/          run configurations (the pilot: items, stages, models, samples)
data/             the data: items.csv and annotations.csv
docs/             specifications per component, data log, proposals (in Italian)
labels/           label dictionaries: fallacies and schemes
prompts/          versioned prompt templates, JSON Schemas of the answers, answer definitions
schemes/          the eight scheme diagrams as YAML
serving/          the models, how to serve them with vLLM, a smoke test
src/argfallacy/   the Python package: schemes, labels, annotations, prompts, client, parse, CLI
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

`docs/dati.md` (in Italian) records where every part comes from, what was corrected or
dropped and why, and how to read each value. The earlier results are rescored on the
same 601 items; reproducing their published numbers on all 607 rows is only an internal
check of the scorer.

## How to reproduce

Tests and lint:

```bash
python -m ruff check src tests serving/smoke_test.py
python -m pytest tests
```

Updating the rows of the `Annotazione Dumitru` sheets in `data/annotations.csv` from the
workbook at `WORKBOOK_PATH` (each of those sheets carries an `item_id` column):

```bash
argfallacy annotations update
```

Serving a model on the cluster, then checking it answers with log-probabilities:

```bash
vllm serve <model_id> --dtype auto --max-model-len 8192 --port 8000
python serving/smoke_test.py --model <name in serving/models.yaml>
```

The model revisions in `serving/models.yaml` must be pinned before any real run:
calls refuse to start while a revision is a placeholder. Qwen and Gemma are served with
their reasoning parser and `--max-model-len 16384`, because each has a second entry with
reasoning on (`_think`); see `serving/README.md`.

Pilot (50 items, two models, each with reasoning off and on, five samples):

```bash
argfallacy run plan configs/pilot.yaml         # counts the calls, makes none
argfallacy run execute configs/pilot.yaml      # makes the calls, writes runs/<run_id>/
argfallacy parse <run_id>                      # answers.csv and summary.csv
```

Not implemented yet: the pilot report, the full runs, the aggregators and the evaluation.

## License

The code is released under the MIT License (see `LICENSE`). The files in `data/`
are not covered by it: the texts come from existing corpora, each under its own terms.
