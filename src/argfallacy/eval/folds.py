"""The folds of the out-of-fold estimation, and the generators that come from the seed.

One seed gives three generators: the first draws the folds, the second the
resamples of the test set, the third the resamples of the items without NLAS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FOLDS = 5


def generators(seed: int) -> list[np.random.Generator]:
    """The three generators of the evaluation: folds, resamples, resamples without NLAS."""
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(3)]


def assign_folds(items: pd.DataFrame, seed: int, folds: int = FOLDS) -> dict[str, int]:
    """``item_id`` -> fold, stratified by gold scheme.

    The items, in order of ``item_id``, are taken one gold scheme at a time, the
    schemes in alphabetical order; the items of each scheme are shuffled and dealt
    in turn to the folds, continuing from one scheme to the next, so the sizes of
    the folds, and of each scheme in the folds, differ by at most one.
    """
    rng = generators(seed)[0]
    ordered = items.sort_values("item_id")
    out: dict[str, int] = {}
    position = 0
    for scheme in sorted(ordered["gold_scheme"].unique()):
        own = ordered.loc[ordered["gold_scheme"] == scheme, "item_id"].tolist()
        for index in rng.permutation(len(own)):
            out[own[index]] = position % folds
            position += 1
    return out
