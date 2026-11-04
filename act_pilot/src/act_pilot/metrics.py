"""Split quality and paired comparisons.

Labels and predictions are 0/1 integer sequences; 1 is the positive class.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


def gini(labels: Sequence[int]) -> float:
    """The Gini impurity of a group of binary labels; 0 for an empty group."""
    n = len(labels)
    if n == 0:
        return 0.0
    p = sum(labels) / n
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


def weighted_gini(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """The impurity of the split induced by a question, weighted by group size."""
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions differ in length")
    n = len(labels)
    if n == 0:
        return float("nan")
    yes = [y for y, p in zip(labels, predictions, strict=True) if p == 1]
    no = [y for y, p in zip(labels, predictions, strict=True) if p == 0]
    return len(yes) / n * gini(yes) + len(no) / n * gini(no)


def recalls(labels: Sequence[int], predictions: Sequence[int]) -> tuple[float, float]:
    """The recall of the positive and of the negative class (NaN for an absent class)."""
    y = np.asarray(labels)
    p = np.asarray(predictions)
    pos = y == 1
    neg = y == 0
    r_pos = float((p[pos] == 1).mean()) if pos.any() else float("nan")
    r_neg = float((p[neg] == 0).mean()) if neg.any() else float("nan")
    return r_pos, r_neg


def balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """The mean of the two class recalls; NaN unless both classes are present."""
    r_pos, r_neg = recalls(labels, predictions)
    if math.isnan(r_pos) or math.isnan(r_neg):
        return float("nan")
    return (r_pos + r_neg) / 2.0


@dataclass(frozen=True)
class PairedDifference:
    """``value`` is BA(candidate) minus BA(reference); ``low`` and ``high`` bound the CI."""

    value: float
    low: float
    high: float
    n_boot: int
    level: float

    @property
    def excludes_zero_in_favour(self) -> bool:
        return not math.isnan(self.low) and self.low > 0.0


def paired_bootstrap_ba(
    labels: Sequence[int],
    pred_candidate: Sequence[int],
    pred_reference: Sequence[int],
    *,
    n_boot: int = 5000,
    seed: int = 12345,
    level: float = 0.95,
) -> PairedDifference:
    """A paired, class-stratified percentile bootstrap of the difference in balanced accuracy.

    Items are resampled with replacement within each class, so every resample keeps the
    class sizes and both balanced accuracies stay defined. The same resampled items are used
    for the candidate and the reference, which makes the comparison paired.
    """
    y = np.asarray(labels)
    a = np.asarray(pred_candidate)
    b = np.asarray(pred_reference)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("labels and predictions differ in length")
    value = balanced_accuracy(y, a) - balanced_accuracy(y, b)
    if math.isnan(value):
        return PairedDifference(float("nan"), float("nan"), float("nan"), n_boot, level)
    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_boot)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        correct_a = (a[idx] == cls).astype(float)
        correct_b = (b[idx] == cls).astype(float)
        draws = rng.integers(0, len(idx), size=(n_boot, len(idx)))
        diffs += (correct_a[draws].mean(axis=1) - correct_b[draws].mean(axis=1)) / 2.0
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(diffs, [alpha, 1.0 - alpha])
    return PairedDifference(float(value), float(low), float(high), n_boot, level)


@dataclass(frozen=True)
class McNemar:
    """``b``: items only the candidate gets right; ``c``: items only the reference gets right."""

    b: int
    c: int
    p_value: float


def mcnemar_exact(
    labels: Sequence[int], pred_candidate: Sequence[int], pred_reference: Sequence[int]
) -> McNemar:
    """The exact two-sided McNemar test on the discordant pairs."""
    y = np.asarray(labels)
    right_a = np.asarray(pred_candidate) == y
    right_b = np.asarray(pred_reference) == y
    b = int((right_a & ~right_b).sum())
    c = int((~right_a & right_b).sum())
    if b + c == 0:
        return McNemar(b, c, 1.0)
    return McNemar(b, c, float(binomtest(b, b + c, 0.5).pvalue))


def binomial_halfwidth(n: int, p: float = 0.5, z: float = 1.96) -> float:
    """The normal-approximation half-width of a proportion's 95% interval, for planning."""
    return z * math.sqrt(p * (1 - p) / n) if n > 0 else float("nan")
