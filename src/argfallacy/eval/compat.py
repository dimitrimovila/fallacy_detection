"""Compat mode of the scorer: count like the scoring code of the prior runs.

It exists only to show that our counting reproduces the files those runs wrote,
on the same input.  None of its numbers goes into the thesis, and none of its
rules is used by the canonical mode.

The rules, all of them the prior code's:

* one row per text, the first kept (``PriorRun.rows``);
* the fallacy tasks skip the rows whose gold fallacy is empty or contains ``?``,
  the scheme task the rows whose gold scheme is empty;
* labels go through ``compat_normalize`` and ``compat_normalize_scheme``;
* the classes are every normalized label seen in the gold or in the predictions,
  so each distinct wrong string a model writes is a class with F1 zero and lowers
  the macro averages;
* ``n_classes_gold`` is the number of distinct normalized gold labels.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..labels import COMPAT_SPACE, compat_normalize, compat_normalize_scheme
from ..schemes import load_vocabulary
from .metrics import accuracy, per_class

if TYPE_CHECKING:
    from .prior import PriorRun

METRICS_COLUMNS = ["prompt_name", "model", "task", "n_samples", "n_classes_gold",
                   "accuracy", "micro_f1", "macro_f1", "precision", "recall"]
CLASSWISE_COLUMNS = ["prompt_name", "model", "task", "class_label",
                     "precision", "recall", "f1-score", "support"]
PER_SOURCE_COLUMNS = ["model", "pipeline_type", "source", "n", "micro_f1"]
ALL_SOURCES = "ALL"


def _mapped(values: pd.Series, normalize) -> list[str]:
    """``normalize`` applied to every value, computed once per distinct string."""
    table = {value: normalize(value) for value in set(values)}
    return [table[value] for value in values]


def _fallacy_rows(frame: pd.DataFrame) -> pd.DataFrame:
    gold = frame["gold_fallacy"]
    return frame[(gold.str.strip() != "") & ~gold.str.contains("?", regex=False)]


def _fallacy_pair(frame: pd.DataFrame, space: str, vocab: dict[str, Any]):
    rows = _fallacy_rows(frame)

    def norm(value: str) -> str:
        return compat_normalize(value, vocab, space=space)

    return _mapped(rows["gold_fallacy"], norm), _mapped(rows["predicted_fallacy"], norm)


def _scheme_pair(frame: pd.DataFrame, vocab: dict[str, Any]):
    rows = frame[frame["gold_scheme"].str.strip() != ""]

    def norm(value: str) -> str:
        return compat_normalize_scheme(value, vocab)

    return _mapped(rows["gold_scheme"], norm), _mapped(rows["predicted_scheme"], norm)


def _tasks(run: PriorRun, vocab: dict[str, Any]) -> Iterator[tuple[str, list, list]]:
    """(task, gold, predicted) in the order of the prior ``metrics.csv`` files."""
    frame = run.rows
    yield "final_label", *_fallacy_pair(frame, "fine", vocab)
    if run.pipeline:
        yield "scheme", *_scheme_pair(frame, vocab)
    yield "final_label (collapsed)", *_fallacy_pair(frame, COMPAT_SPACE, vocab)
    if run.pipeline:
        def norm(value: str) -> str:
            return compat_normalize_scheme(value, vocab)

        predicted = _mapped(frame["predicted_scheme"], norm)
        correct = frame[[p == g for p, g in
                         zip(predicted, _mapped(frame["gold_scheme"], norm), strict=True)]]
        yield "final_label (scheme-correct)", *_fallacy_pair(correct, "fine", vocab)
        yield ("final_label (scheme-correct, collapsed)",
               *_fallacy_pair(correct, COMPAT_SPACE, vocab))


def _union_table(gold: list, predicted: list) -> pd.DataFrame:
    return per_class(gold, predicted, sorted(set(gold) | set(predicted)))


def compat_metrics(run: PriorRun, vocabulary: dict[str, Any] | None = None) -> pd.DataFrame:
    """The rows of the run's ``metrics.csv``, recomputed."""
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    rows = []
    for task, gold, predicted in _tasks(run, vocab):
        table = _union_table(gold, predicted)
        share = accuracy(gold, predicted)
        rows.append({
            "prompt_name": run.folder, "model": run.model, "task": task,
            "n_samples": len(gold), "n_classes_gold": len(set(gold)),
            "accuracy": share, "micro_f1": share,
            "macro_f1": table["f1"].mean(),
            "precision": table["precision"].mean(),
            "recall": table["recall"].mean(),
        })
    return pd.DataFrame(rows, columns=METRICS_COLUMNS)


def compat_classwise(run: PriorRun, vocabulary: dict[str, Any] | None = None) -> pd.DataFrame:
    """The rows of the run's ``classwise_metrics.csv``, recomputed."""
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    wanted = ("final_label", "final_label (collapsed)", "scheme")
    parts = []
    for task, gold, predicted in _tasks(run, vocab):
        if task not in wanted:
            continue
        table = _union_table(gold, predicted).rename(
            columns={"class": "class_label", "f1": "f1-score"})
        table.insert(0, "task", task)
        table.insert(0, "model", run.model)
        table.insert(0, "prompt_name", run.folder)
        parts.append(table[CLASSWISE_COLUMNS])
    return pd.concat(parts, ignore_index=True)


def source_group(source: str) -> str:
    """The corpus of a ``source`` value: the part before ``_`` or `` (``."""
    return re.split(r"_| \(", source, maxsplit=1)[0]


def compat_per_source(run: PriorRun, vocabulary: dict[str, Any] | None = None) -> pd.DataFrame:
    """The run's rows of ``per_dataset_micro_f1.csv``: fine accuracy per corpus, then all."""
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    rows = _fallacy_rows(run.rows)
    gold, predicted = _fallacy_pair(rows, "fine", vocab)
    frame = pd.DataFrame({"group": rows["source"].map(source_group).tolist(),
                          "gold": gold, "predicted": predicted})
    out = [
        {"source": group, "n": len(part),
         "micro_f1": accuracy(part["gold"].tolist(), part["predicted"].tolist())}
        for group, part in frame.groupby("group", sort=True)
    ]
    out.append({"source": ALL_SOURCES, "n": len(frame), "micro_f1": accuracy(gold, predicted)})
    table = pd.DataFrame(out)
    table.insert(0, "pipeline_type", run.folder)
    table.insert(0, "model", run.model)
    return table[PER_SOURCE_COLUMNS]
