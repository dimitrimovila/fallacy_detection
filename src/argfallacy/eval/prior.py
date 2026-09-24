"""The prior runs as input to the scorer, and ``argfallacy score prior``.

Each prior run is a folder ``<model>/<condition>/`` under ``PRIOR_RUNS_DIR`` with
a ``results.csv`` (one row per critical question for the pipeline runs, one per
text for the zero-shot ones) and the ``metrics.csv`` it was scored into.  The
files are only read.

Two uses, kept apart.  Compat mode scores the rows as they are, to reproduce the
files of the runs.  The canonical mode maps the rows onto the items of the test
set through ``eleni_id`` and scores them against the gold of ``data/items.csv``.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..annotations import ITEMS_FILE
from ..labels import SchemeError, load_scheme_vocabulary, normalize_label
from ..schemes import load_vocabulary
from .baselines import majority_by
from .canonical import (
    SCHEME_SPACE,
    Score,
    gold_scheme_mismatches,
    predicted_scheme_class,
    score_fallacies,
    score_schemes,
)
from .compat import compat_classwise, compat_metrics, compat_per_source

SPACES = ("fine", "collapsed_symmetric")
PIPELINE = "pipeline"
ZERO_SHOT = "zero-shot"
ID_SEPARATOR = ";"


@dataclass(frozen=True)
class PriorRun:
    """One prior run: model and condition from its path, its rows as strings."""

    model: str
    folder: str
    path: Path
    long: pd.DataFrame
    """``results.csv`` as written, every value a string."""

    @property
    def rows(self) -> pd.DataFrame:
        """One row per text, the first kept: 607 in every run."""
        return self.long.drop_duplicates(subset="text", keep="first")

    @property
    def pipeline(self) -> bool:
        return "predicted_scheme" in self.long.columns

    @property
    def condition(self) -> str:
        return PIPELINE if self.pipeline else ZERO_SHOT

    def answer_vectors(self) -> pd.Series:
        """Prior row id -> the hard answers to the CQs, as (cq_number, answer) pairs.

        Answers stripped and lower-cased; an unusable answer is kept as written.
        """
        if not self.pipeline:
            raise SchemeError(f"{self.model}/{self.folder} is not a pipeline run: no CQ answers")
        answers = self.long.assign(answer=self.long["cq_answer"].str.strip().str.lower())
        answers = answers.drop_duplicates(subset=["id", "cq_number"], keep="first")
        return answers.groupby("id", sort=False).apply(
            lambda part: tuple(sorted(zip(part["cq_number"], part["answer"], strict=True))),
            include_groups=False,
        )


def read_prior_runs(prior_dir: str | Path) -> list[PriorRun]:
    """Every folder under ``prior_dir`` with both a ``results.csv`` and a ``metrics.csv``."""
    prior_dir = Path(prior_dir)
    runs = []
    for results in sorted(prior_dir.rglob("results.csv")):
        folder = results.parent
        if not (folder / "metrics.csv").exists():
            continue
        long = pd.read_csv(results, dtype=str, keep_default_na=False)
        runs.append(PriorRun(folder.parent.name, folder.name, folder, long))
    return runs


def load_test_set(path: str | Path = ITEMS_FILE) -> pd.DataFrame:
    """The items of the test set of the experiments, every value a string."""
    items = pd.read_csv(path, dtype=str, keep_default_na=False)
    return items[items["in_test_set"] == "True"].reset_index(drop=True)


def prior_items(run: PriorRun, items: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The run's rows on the items of the test set, and the prior ids left out.

    An item stands for the first of its ``eleni_id`` values; the prior rows that
    stand for no item are returned as dropped.
    """
    first = items["eleni_id"].str.split(ID_SEPARATOR).str[0]
    rows = run.rows.set_index("id")
    if missing := sorted(set(first) - set(rows.index)):
        raise SchemeError(f"{run.model}/{run.folder}: no row for the prior ids {missing}")
    columns = ["predicted_fallacy"] + (["predicted_scheme"] if run.pipeline else [])
    table = rows.loc[first.tolist(), columns].reset_index(names="prior_id")
    table.insert(0, "item_id", items["item_id"].tolist())
    dropped = sorted(set(rows.index) - set(first), key=int)
    return table, dropped


# ------------------------------------------------------------------ outputs
METRICS_COLUMNS = ["run", "condition", "task", "space", "n_items", "n_classes_gold",
                   "accuracy", "macro_precision", "macro_recall", "macro_f1"]
CLASSWISE_COLUMNS = ["run", "condition", "task", "space",
                     "class", "support", "precision", "recall", "f1"]
BASELINE_COLUMNS = ["baseline", "partition", "run", "condition", "space", "n_items",
                    "n_classes_gold", "accuracy", "macro_precision", "macro_recall", "macro_f1"]


def _predictions(table: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.DataFrame({"item_id": table["item_id"], "predicted": table[column]})


def _canonical_scores(
    run: PriorRun, table: pd.DataFrame, items: pd.DataFrame,
    vocab: dict[str, Any], scheme_vocab: dict[str, Any],
) -> list[tuple[str, str, Score]]:
    """(task, space, score) for the tasks of the run, on the 601 items."""
    fallacies = _predictions(table, "predicted_fallacy")
    scores = [("final_label", space, score_fallacies(fallacies, items, space, vocabulary=vocab))
              for space in SPACES]
    if run.pipeline:
        schemes = _predictions(table, "predicted_scheme")
        scores.append(("scheme", SCHEME_SPACE,
                       score_schemes(schemes, items, vocabulary=scheme_vocab)))
        predicted = table["predicted_scheme"].map(
            lambda raw: predicted_scheme_class(raw, scheme_vocab)[0])
        correct = table.loc[predicted.to_numpy() == items["gold_scheme"].to_numpy(), "item_id"]
        scores += [("final_label (scheme-correct)", space,
                    score_fallacies(fallacies, items, space, items=correct, vocabulary=vocab))
                   for space in SPACES]
    return scores


def _baselines(
    runs: list[tuple[PriorRun, pd.DataFrame]], items: pd.DataFrame,
    vocab: dict[str, Any], scheme_vocab: dict[str, Any],
) -> list[dict[str, Any]]:
    gold = [normalize_label(g, vocab).id for g in items["gold_fallacy"]]
    lines: list[tuple[str, str, str, str, list[Hashable]]] = [
        ("majority", "gold_scheme", "", "", items["gold_scheme"].tolist())
    ]
    for run, table in runs:
        if not run.pipeline:
            continue
        schemes = [predicted_scheme_class(s, scheme_vocab)[0] for s in table["predicted_scheme"]]
        vectors = run.answer_vectors().reindex(table["prior_id"]).tolist()
        lines.append(("majority", "predicted_scheme", run.model, run.condition, schemes))
        lines.append(("ceiling", "predicted_scheme", run.model, run.condition,
                      list(zip(schemes, vectors, strict=True))))
    out = []
    for baseline, partition, name, condition, keys in lines:
        predictions = pd.DataFrame({"item_id": items["item_id"],
                                    "predicted": majority_by(keys, gold)})
        for space in SPACES:
            score = score_fallacies(predictions, items, space, vocabulary=vocab)
            out.append({"baseline": baseline, "partition": partition, "run": name,
                        "condition": condition, "space": space, **score.summary()})
    return out


def _report(
    items: pd.DataFrame, dropped: dict[str, list[str]], mismatches: pd.DataFrame,
    scored: list[tuple[PriorRun, str, str, Score]],
) -> str:
    lines = ["# Scores of the prior runs", "",
             f"Canonical rescoring on the {len(items)} items of the test set of `data/items.csv`.",
             "", "## Prior rows that stand for no item", ""]
    lines += [f"* {name}: {', '.join(ids)}" for name, ids in dropped.items()]
    lines += ["", "## Gold terminals the diagram of the gold scheme does not produce", "",
              "Kept and scored like any other item.", "",
              "| item_id | eleni_id | gold_scheme | gold_fallacy |", "| --- | --- | --- | --- |"]
    lines += [f"| {r.item_id} | {r.eleni_id} | {r.gold_scheme} | {r.gold_fallacy} |"
              for r in mismatches.itertuples()]
    lines += ["", "## Predictions that are no class of the space", "",
              "| run | condition | task | space | n_items | accuracy | macro_f1 | not a class |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run, task, space, score in scored:
        kinds = ", ".join(f"{k} {n}" for k, n in score.not_a_class.items()) or "0"
        lines.append(f"| {run.model} | {run.condition} | {task} | {space} | {score.n_items} "
                     f"| {score.accuracy:.3f} | {score.macro_f1:.3f} | {kinds} |")
    lines += ["", "## Classes left out of the macro averages", ""]
    left_out: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for run, task, space, score in scored:
        left_out.setdefault((space, score.left_out), []).append(
            f"{run.model}/{run.condition} {task}")
    for (space, classes), where in left_out.items():
        lines.append(f"* {space}, {', '.join(classes) or 'none'}: {'; '.join(where)}")
    return "\n".join(lines) + "\n"


def score_prior(
    prior_dir: str | Path, out_dir: str | Path, items_path: str | Path = ITEMS_FILE
) -> dict[str, Path]:
    """Score every prior run in both modes and write the files; return their paths."""
    runs = read_prior_runs(prior_dir)
    if not runs:
        raise SchemeError(f"no results.csv with a metrics.csv beside it under {prior_dir}")
    vocab = load_vocabulary(check_schemes=False)
    scheme_vocab = load_scheme_vocabulary()
    items = load_test_set(items_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {
        "compat_metrics": pd.concat([compat_metrics(r, vocab) for r in runs]),
        "compat_classwise": pd.concat([compat_classwise(r, vocab) for r in runs]),
        "compat_per_source": pd.concat([compat_per_source(r, vocab) for r in runs]),
    }

    tables, dropped, scored = [], {}, []
    for run in runs:
        table, left = prior_items(run, items)
        tables.append((run, table))
        dropped[f"{run.model}/{run.folder}"] = left
        scored += [(run, task, space, score)
                   for task, space, score in _canonical_scores(run, table, items, vocab,
                                                               scheme_vocab)]
    frames["metrics"] = pd.DataFrame(
        [{"run": r.model, "condition": r.condition, "task": t, "space": s, **score.summary()}
         for r, t, s, score in scored], columns=METRICS_COLUMNS)
    frames["classwise"] = pd.concat(
        [score.classes.assign(run=r.model, condition=r.condition, task=t, space=s)
         for r, t, s, score in scored])[CLASSWISE_COLUMNS]
    frames["baselines"] = pd.DataFrame(_baselines(tables, items, vocab, scheme_vocab),
                                       columns=BASELINE_COLUMNS)

    written = {}
    for name, frame in frames.items():
        written[name] = out_dir / f"{name}.csv"
        frame.to_csv(written[name], index=False)
    written["report"] = out_dir / "report.md"
    report = _report(items, dropped, gold_scheme_mismatches(items, vocab), scored)
    written["report"].write_text(report, encoding="utf-8")
    return written
