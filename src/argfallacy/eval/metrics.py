"""The counting shared by the two modes: per-class precision, recall and F1.

Both modes hand this function two aligned lists of labels and the classes to
report.  What differs between them is only what they hand in: the labels (compat
strings or canonical ids) and the classes (every label seen, or the label space).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence

import pandas as pd

CLASS_COLUMNS = ["class", "support", "precision", "recall", "f1"]


def _ratio(numerator: int, denominator: int) -> float:
    """A zero denominator gives zero, as in the scoring code of the prior runs."""
    return numerator / denominator if denominator else 0.0


def per_class(
    gold: Sequence[Hashable],
    predicted: Sequence[Hashable | None],
    classes: Sequence[Hashable],
) -> pd.DataFrame:
    """One row per class: support, precision, recall, F1.

    ``predicted`` may hold values that are not among ``classes`` (``None`` for a
    prediction that is no class at all): such a prediction counts in no class's
    precision and lowers the recall of the gold class of its item.
    """
    if len(gold) != len(predicted):
        raise ValueError(f"{len(gold)} gold labels but {len(predicted)} predictions")
    support = Counter(gold)
    emitted = Counter(predicted)
    correct = Counter(g for g, p in zip(gold, predicted, strict=True) if g == p)
    rows = []
    for label in classes:
        tp, n_gold, n_predicted = correct[label], support[label], emitted[label]
        rows.append({
            "class": label,
            "support": n_gold,
            "precision": _ratio(tp, n_predicted),
            "recall": _ratio(tp, n_gold),
            "f1": _ratio(2 * tp, n_gold + n_predicted),
        })
    return pd.DataFrame(rows, columns=CLASS_COLUMNS)


def accuracy(gold: Sequence[Hashable], predicted: Sequence[Hashable | None]) -> float:
    """Share of items whose prediction equals the gold.  Zero on no items."""
    if len(gold) != len(predicted):
        raise ValueError(f"{len(gold)} gold labels but {len(predicted)} predictions")
    return _ratio(sum(g == p for g, p in zip(gold, predicted, strict=True)), len(gold))
