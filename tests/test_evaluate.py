"""The evaluation of spec 08, first stage, on data written by hand."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from argfallacy.eval import assign_folds


def _test_set() -> pd.DataFrame:
    schemes = ["analogy"] * 11 + ["example"] * 7 + ["ad_hominem"] * 5
    return pd.DataFrame({"item_id": [f"{n:016x}" for n in range(len(schemes))][::-1],
                         "gold_scheme": schemes})


def test_the_folds_are_drawn_from_the_seed_and_balanced():
    items = _test_set()
    folds = assign_folds(items, seed=20261019)
    assert folds == assign_folds(items.sample(frac=1, random_state=1), seed=20261019)
    assert folds != assign_folds(items, seed=1)
    assert set(folds) == set(items["item_id"]) and set(folds.values()) == set(range(5))

    sizes = Counter(folds.values())
    assert max(sizes.values()) - min(sizes.values()) <= 1
    for scheme, own in items.groupby("gold_scheme"):
        per_fold = Counter(folds[item] for item in own["item_id"])
        counts = [per_fold.get(fold, 0) for fold in range(5)]
        assert max(counts) - min(counts) <= 1, scheme
