"""The canonical scorer: the rules the thesis numbers are computed with.

Gold and prediction are read with the dictionary, so a known variant spelling is
the label it stands for.  The gold must be a terminal.  A prediction is correct
when it lands on the same class of the label space as the gold; a prediction
that is no class of the space (an invalid output, no verdict, a family label in
the fine space, a coarse or out-of-scope label) is wrong, and is counted by kind.

Macro averages run over the classes with support in the gold of the items
scored.  The other classes of the space are still reported, and listed as left
out; a prediction of one of them is an error like any other.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..labels import (
    SchemeError,
    collapse,
    label_space,
    light_normalize,
    load_scheme_vocabulary,
    normalize_label,
    normalize_scheme_label,
    resolve_for_scheme,
    scheme_ids,
)
from ..schemes import load_all, load_vocabulary
from .metrics import accuracy, per_class

NO_VERDICT = "no_verdict"
"""Kind of a prediction that is missing: an empty cell, or an abstention."""

SCHEME_SPACE = "schemes"
"""Name of the label space of stage one: the eight schemes and ``none``."""


@dataclass(frozen=True)
class Score:
    """The metrics of one set of predictions in one label space."""

    n_items: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    classes: pd.DataFrame
    """Every class of the space: class, support, precision, recall, f1."""
    left_out: tuple[str, ...]
    """Classes of the space without gold support, outside the macro averages."""
    not_a_class: dict[str, int] = field(default_factory=dict)
    """Predictions that are no class of the space, counted by kind."""

    @property
    def n_classes_gold(self) -> int:
        return len(self.classes) - len(self.left_out)

    def summary(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "n_classes_gold": self.n_classes_gold,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
        }


def _is_missing(raw: object, markers: Iterable[str]) -> bool:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return True
    return light_normalize(raw) in {light_normalize(m) for m in markers}


def predicted_fallacy_class(
    raw: object, space: str = "fine", vocabulary: dict[str, Any] | None = None
) -> tuple[str | None, str]:
    """(class of ``space`` or None, kind) for one raw fallacy prediction.

    The kind is ``class`` when the prediction is a class of the space; otherwise
    it says what the prediction was.  An unknown string raises.
    """
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    if _is_missing(raw, vocab.get("missing_markers", [])):
        return None, NO_VERDICT
    label = normalize_label(str(raw), vocab)
    if label.kind in ("terminal", "family"):
        cls = collapse(label.id, space, vocab)
        if cls in label_space(space, vocab):
            return cls, "class"
    return None, label.kind


def predicted_scheme_class(
    raw: object, vocabulary: dict[str, Any] | None = None
) -> tuple[str | None, str]:
    """(scheme id or None, kind) for one raw stage-one prediction."""
    vocab = vocabulary or load_scheme_vocabulary()
    special = vocab.get("special_values", {})
    if _is_missing(raw, special.get("missing", [])):
        return None, NO_VERDICT
    if light_normalize(raw) in {light_normalize(m) for m in special.get("idk", [])}:
        return None, "idk"
    return normalize_scheme_label(str(raw), vocab), "class"


def _aligned(
    predictions: pd.DataFrame, gold: pd.DataFrame, column: str, items: Iterable[str] | None
) -> pd.DataFrame:
    """The prediction rows (optionally a subset), each with its gold, one per item."""
    if not predictions["item_id"].is_unique:
        repeated = sorted(predictions.loc[predictions["item_id"].duplicated(), "item_id"])
        raise SchemeError(f"more than one prediction for the items {repeated}")
    frame = predictions[["item_id", "predicted"]]
    if items is not None:
        wanted = set(items)
        frame = frame[frame["item_id"].isin(wanted)]
        if missing := sorted(wanted - set(frame["item_id"])):
            raise SchemeError(f"no prediction for the items {missing}")
    merged = frame.merge(gold[["item_id", column]], on="item_id", how="left", indicator=True)
    if unknown := sorted(merged.loc[merged["_merge"] != "both", "item_id"]):
        raise SchemeError(f"no gold for the items {unknown}")
    return merged.drop(columns="_merge")


def _score(
    gold: list[str], predicted: list[str | None], kinds: list[str], classes: Sequence[str]
) -> Score:
    table = per_class(gold, predicted, classes)
    supported = table[table["support"] > 0]
    left_out = tuple(table.loc[table["support"] == 0, "class"])

    def macro(column: str) -> float:
        return float(supported[column].mean()) if len(supported) else 0.0

    return Score(
        n_items=len(gold),
        accuracy=accuracy(gold, predicted),
        macro_precision=macro("precision"),
        macro_recall=macro("recall"),
        macro_f1=macro("f1"),
        classes=table,
        left_out=left_out,
        not_a_class=dict(sorted(Counter(k for k in kinds if k != "class").items())),
    )


def _gold_terminal(item_id: str, raw: object, vocab: dict[str, Any]) -> str:
    try:
        label = normalize_label(str(raw), vocab)
    except SchemeError as error:
        raise SchemeError(f"item {item_id}: gold fallacy {raw!r}: {error}") from None
    if label.kind != "terminal":
        raise SchemeError(
            f"item {item_id}: gold fallacy {raw!r} is a {label.kind} label, not a terminal"
        )
    return label.id


def score_fallacies(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    space: str = "fine",
    items: Iterable[str] | None = None,
    vocabulary: dict[str, Any] | None = None,
) -> Score:
    """Score raw fallacy predictions against the gold in a label space.

    ``predictions`` has ``item_id`` and ``predicted`` (the raw label, or empty for
    no verdict); ``gold`` has ``item_id`` and ``gold_fallacy``, as in
    ``data/items.csv``.  ``items`` restricts the scoring to a subset.
    """
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    frame = _aligned(predictions, gold, "gold_fallacy", items)
    gold_ids = [collapse(_gold_terminal(i, g, vocab), space, vocab)
                for i, g in zip(frame["item_id"], frame["gold_fallacy"], strict=True)]
    resolved = [predicted_fallacy_class(p, space, vocab) for p in frame["predicted"]]
    return _score(gold_ids, [c for c, _ in resolved], [k for _, k in resolved],
                  label_space(space, vocab))


def score_schemes(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    items: Iterable[str] | None = None,
    vocabulary: dict[str, Any] | None = None,
) -> Score:
    """Score raw stage-one predictions against ``gold_scheme``; classes: schemes and ``none``."""
    vocab = vocabulary or load_scheme_vocabulary()
    frame = _aligned(predictions, gold, "gold_scheme", items)
    gold_ids = []
    for item_id, raw in zip(frame["item_id"], frame["gold_scheme"], strict=True):
        try:
            sid, kind = predicted_scheme_class(raw, vocab)
        except SchemeError as error:
            raise SchemeError(f"item {item_id}: gold scheme {raw!r}: {error}") from None
        if kind != "class":
            raise SchemeError(f"item {item_id}: gold scheme {raw!r} is {kind}, not a scheme")
        gold_ids.append(sid)
    resolved = [predicted_scheme_class(p, vocab) for p in frame["predicted"]]
    return _score(gold_ids, [c for c, _ in resolved], [k for _, k in resolved],
                  scheme_ids(vocab))


def gold_scheme_mismatches(
    gold: pd.DataFrame, vocabulary: dict[str, Any] | None = None
) -> pd.DataFrame:
    """The items whose gold terminal the diagram of their gold scheme does not produce.

    They are kept and scored like any other item: this only lists them, because
    no aggregator built on the diagrams can get them right in the gold scheme
    condition.
    """
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    schemes = load_all()
    rows = [
        row for row in gold.itertuples(index=False)
        if row.gold_scheme in schemes
        and resolve_for_scheme(row.gold_fallacy, schemes[row.gold_scheme], schemes,
                               vocab).status == "scheme_mismatch"
    ]
    return pd.DataFrame(rows, columns=gold.columns).reset_index(drop=True)
