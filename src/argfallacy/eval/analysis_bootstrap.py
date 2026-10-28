"""Replay the evaluation's item draws in bounded, deterministic batches."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluate import Scorer, weights
from .folds import generators


@dataclass(frozen=True)
class Bootstrap:
    seed: int
    resamples: int
    batch_size: int = 128
    _scorers: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def scorer(self, gold):
        """Reuse vocabulary and label encodings, never per-system resample arrays."""
        key = tuple(gold)
        if key not in self._scorers:
            self._scorers[key] = Scorer(gold, np.empty((0, len(gold))))
        return self._scorers[key]

    def indices(self, n: int, generator: int = 1):
        """Restart the same stream for each system, without retaining B by N."""
        rng = generators(self.seed)[generator]
        for start in range(0, self.resamples, self.batch_size):
            count = min(self.batch_size, self.resamples - start)
            yield rng.integers(0, n, size=(count, n))


def weighted_metrics(scorer: Scorer, predicted, counts):
    """Reuse the canonical vectorized scorer with only the current batch's counts."""
    scorer.weights = counts
    scorer.total = counts.sum(axis=1)
    scorer.support = counts @ scorer.gold_onehot
    with np.errstate(invalid="ignore", divide="ignore"):
        values = scorer.metrics(predicted)
    empty = scorer.total == 0
    for value in values.values():
        value[empty] = np.nan
    return values


def score_population(gold, predicted, eligible, bootstrap: Bootstrap, generator=1):
    """Filter whole-population draws; return scalar replicate statistics only.

    For generator 2 the caller supplies the ordered without-NLAS population.
    Zero Good Argumentation denominators keep the scorer's numeric convention.
    A replicate with no eligible items has no metric and remains NaN.
    """
    mask = np.asarray(eligible, dtype=bool)
    scorer = bootstrap.scorer(gold)
    observed = {
        k: float(v[0])
        for k, v in weighted_metrics(scorer, predicted, mask.astype(float)[None, :]).items()
    }
    spread = {k: np.full(bootstrap.resamples, np.nan) for k in observed}
    start = 0
    for indices in bootstrap.indices(len(gold), generator):
        counts = weights(indices, len(gold))
        counts *= mask
        values = weighted_metrics(scorer, predicted, counts)
        stop = start + len(indices)
        for metric, value in values.items():
            spread[metric][start:stop] = value
        start = stop
    # Retain structural encodings only. The next population owns the sole workspace.
    scorer.weights = np.empty((0, len(gold)))
    scorer.support = np.empty((0, len(scorer.classes)))
    scorer.total = np.empty(0)
    return observed, spread


def percentile(values, level=0.95):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan, 0
    alpha = 1 - level
    low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high), len(finite)


def contribution_ratios(item_positions, numerators, denominators, n_items, bootstrap):
    """Sum contributions within items before applying the shared item draws.

    Inputs can contain many CQs or path occurrences per item. The only matrix
    involving replicates is the current batch by the complete item population.
    """
    positions = np.asarray(item_positions, dtype=int)
    top = np.bincount(positions, weights=numerators, minlength=n_items)
    bottom = np.bincount(positions, weights=denominators, minlength=n_items)
    observed = top.sum() / bottom.sum() if bottom.sum() else np.nan
    spread = np.full(bootstrap.resamples, np.nan)
    if not bottom.any():
        return float(observed), spread
    start = 0
    for indices in bootstrap.indices(n_items):
        counts = weights(indices, n_items)
        # Fixed row-wise summation also fixes floating-point order across batch sizes.
        a, b = (counts * top).sum(axis=1), (counts * bottom).sum(axis=1)
        stop = start + len(indices)
        spread[start:stop] = np.divide(a, b, out=np.full(len(a), np.nan), where=b > 0)
        start = stop
    return float(observed), spread
