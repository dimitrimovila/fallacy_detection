"""C5: abstention curves (spec 08, P11) and conformal prediction sets fitted by fold.

A curve sorts the items by decreasing confidence, ties by ``item_id``; items with a
verdict but no confidence follow every item with one, and items without a verdict
come last and count as errors, as in the main table.  Every resample is sorted by
the same order, so the curves of a resample are those of its items drawn as many
times as the resample draws them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..aggregate.interface import TIE, Aggregation
from .evaluate import interval

ALPHA = 0.10
"""The miscoverage of the conformal sets."""
LEVELS = (1.0, 0.95, 0.9, 0.8, 0.7)
"""The coverage levels of the selective accuracy, fixed before the results."""


# ------------------------------------------------------------------ curves


def _key(aggregation: Aggregation, kind: str) -> float:
    if aggregation.verdict is None:
        return -2.0
    if kind == "margin":
        top = sorted(aggregation.distribution.values(), reverse=True) + [0.0, 0.0]
        return top[0] - top[1]
    confidence = aggregation.confidence
    return -1.0 if confidence is None or not math.isfinite(confidence) else float(confidence)


def ranking(ids: Sequence[str], aggregations: Mapping[str, Aggregation],
            kind: str = "confidence") -> np.ndarray:
    """The positions of the items in ``ids``, most confident first, ties by ``item_id``.

    ``kind`` is ``confidence`` or ``margin``, the difference between the two most
    probable terminals, computed as a check.
    """
    keys = np.array([_key(aggregations[i], kind) for i in ids])
    names = np.argsort(np.argsort(np.array(ids)))
    return np.lexsort((names, -keys))


def cumulative_errors(wrong: np.ndarray, order: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """e(k), the errors among the first k items drawn, for every row of ``indices``.

    A row of ``indices`` is a sample of item positions; ``np.arange(N)`` is the
    observed sample.
    """
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    ranked = np.sort(rank[np.asarray(indices)], axis=1)
    return np.cumsum(np.asarray(wrong, dtype=float)[order][ranked], axis=1)


def _cut(level: float, n: int) -> int:
    # round(c * N), halves up
    return max(1, int(math.floor(level * n + 0.5)))


def abstention(name: str, ids: Sequence[str], wrong: np.ndarray,
               aggregations: Mapping[str, Aggregation], indices: np.ndarray,
               levels: Sequence[float] = LEVELS, margin: bool = False,
               ) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """AUGRC, AURC and the selective accuracy at each level, with 95 per cent intervals.

    Returns the summary row, the observed curve (one row per k) and the order of the
    items.  With ``margin`` the row also has the AUGRC of the margin, as a check.
    """
    n = len(ids)
    order = ranking(ids, aggregations)
    observed = cumulative_errors(wrong, order, np.arange(n)[None])
    resampled = cumulative_errors(wrong, order, indices)
    k = np.arange(1, n + 1)
    row: dict = {"system": name, "n_items": n,
                 "with_verdict": float(np.mean([aggregations[i].verdict is not None
                                                for i in ids]))}
    values = {"augrc": lambda e: e.mean(axis=1) / n, "aurc": lambda e: (e / k).mean(axis=1)}
    for level in levels:
        cut = _cut(level, n)
        values[f"selective_accuracy_{round(100 * level)}"] = (
            lambda e, cut=cut: 1 - e[:, cut - 1] / cut)
    for metric, compute in values.items():
        low, high = interval(compute(resampled))
        row.update({metric: float(compute(observed)[0]), f"{metric}_low": low,
                    f"{metric}_high": high})
    if margin:
        check = cumulative_errors(wrong, ranking(ids, aggregations, "margin"),
                                  np.arange(n)[None])
        row["augrc_margin"] = float(check.mean() / n)
    curve = pd.DataFrame({"system": name, "k": k, "coverage": k / n,
                          "selective_risk": observed[0] / k,
                          "generalized_risk": observed[0] / n})
    return row, curve, order


def composition(name: str, ids: Sequence[str], order: np.ndarray,
                groups: Mapping[str, Mapping[str, str]],
                levels: Sequence[float] = LEVELS) -> list[dict]:
    """What the abstained items are, at every level below full coverage: for each kind
    of group (Good Argumentation, scheme, corpus, gold origin), the count of each value."""
    rows = []
    n = len(ids)
    for level in levels:
        cut = _cut(level, n)
        abstained = [ids[p] for p in order[cut:]]
        if not abstained:
            continue
        for kind, of in groups.items():
            counts = pd.Series([of[i] for i in abstained]).value_counts()
            for value, count in sorted(counts.items()):
                rows.append({"system": name, "coverage": level, "n_abstained": len(abstained),
                             "kind": kind, "value": value, "count": int(count),
                             "share": count / len(abstained)})
    return rows


# --------------------------------------------------------------- conformal


def conformal_threshold(scores: Sequence[float], alpha: float = ALPHA) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest of the n scores; infinite when that
    rank exceeds n, so that the set holds every terminal."""
    ordered = np.sort(np.asarray(scores, dtype=float))
    # the rounding keeps (n + 1)(1 - alpha) from landing a hair above an integer
    rank = math.ceil(round((len(ordered) + 1) * (1 - alpha), 9))
    return float(ordered[rank - 1]) if 1 <= rank <= len(ordered) else math.inf


def terminal_probabilities(aggregation: Aggregation,
                           terminals: Sequence[str]) -> dict[str, float] | None:
    """The distribution renormalised over the terminals of the scheme, abstention and
    pruned mass removed; None when no terminal has mass."""
    masses = {t: float(aggregation.distribution.get(t, 0.0)) for t in terminals}
    total = sum(masses.values())
    return None if total <= 0 else {t: m / total for t, m in masses.items()}


def conformal_sets(
    aggregations: Mapping[str, Aggregation],
    terminals: Mapping[str, Sequence[str]],
    gold: Mapping[str, str],
    folds: Mapping[str, int],
    alpha: float = ALPHA,
) -> tuple[dict[str, tuple[str, ...]], dict[int, float]]:
    """The set of every item and the threshold of every fold.

    The score of a terminal is 1 - P(t).  The threshold of fold f comes from the
    scores of the gold verdicts of the items of the other folds; a gold verdict
    without mass, or outside the terminals of its scheme, scores 1.  The set holds
    the terminals with a score at most the threshold, every terminal when no
    terminal has mass.
    """
    probabilities = {i: terminal_probabilities(aggregations[i], terminals[i])
                     for i in aggregations}
    gold_score = {i: 1.0 - (probabilities[i] or {}).get(gold[i], 0.0) for i in aggregations}
    sets, thresholds = {}, {}
    for fold in sorted({folds[i] for i in aggregations}):
        q = conformal_threshold([s for i, s in gold_score.items() if folds[i] != fold], alpha)
        thresholds[fold] = q
        for item in (i for i in aggregations if folds[i] == fold):
            given = probabilities[item]
            sets[item] = tuple(terminals[item]) if given is None else tuple(
                t for t in terminals[item] if 1.0 - given[t] <= q + TIE)
    return sets, thresholds


def conformal_summary(name: str, sets: Mapping[str, tuple[str, ...]], gold: Mapping[str, str],
                      groups: Mapping[str, Mapping[str, str]]) -> list[dict]:
    """Coverage, mean size and share of sets with one verdict, in total and by group."""
    def row(kind, subset, items):
        sizes = np.array([len(sets[i]) for i in items])
        return {"system": name, "kind": kind, "subset": subset, "n_items": len(items),
                "coverage": float(np.mean([gold[i] in sets[i] for i in items])),
                "mean_size": float(sizes.mean()), "singleton_share": float(np.mean(sizes == 1))}

    items = sorted(sets)
    rows = [row("all", "all", items)]
    for kind, of in groups.items():
        for value in sorted({of[i] for i in items}):
            rows.append(row(kind, value, [i for i in items if of[i] == value]))
    return rows
