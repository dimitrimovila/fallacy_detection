"""CQ inputs fitted within the training partition of the consuming model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..schemes import SchemeError
from .c1 import PRIMARY_CONFIG, C1Config, aggregate_combined, fit_c1
from .c1_inputs import C1Dataset
from .diagram import a0
from .interface import Aggregation


def inner_partitions(ids: Sequence[str], seed: int):
    """Four shuffled item folds; smaller partitions use leave-one-out folds."""
    ids = sorted(set(ids))
    if len(ids) < 2:
        return [((), tuple(ids))] if ids else []
    return [
        (tuple(ids[i] for i in train), tuple(ids[i] for i in test))
        for train, test in KFold(min(4, len(ids)), shuffle=True, random_state=seed).split(ids)
    ]


@dataclass(frozen=True)
class CQInputs:
    data: C1Dataset
    p: np.ndarray
    aggregations: Mapping[str, Aggregation]
    hard_tree: bool = False

    def rows(self, ids: Sequence[str]) -> pd.DataFrame:
        rows = self.data.pairs.copy()
        rows["p_first"], rows["outside"] = self.p, self.data.outside
        return rows[rows.item_id.isin(set(ids))]

    def tree_features(self, ids: Sequence[str]) -> np.ndarray:
        rows = self.rows(ids).copy()
        if self.hard_tree:
            e = self.data.entries.index(self.data.reference)
            observed = self.data.observations[rows.index, e]
            rows["p_first"] = np.where(observed == 0, 1.0, np.where(observed == 1, 0.0, 0.5))
            rows["outside"] = (~np.isin(observed, [0, 1])).astype(float)
        by_item = {i: own for i, own in rows.groupby("item_id", sort=False)}
        return np.array(
            [by_item[i][["p_first", "outside"]].to_numpy().ravel() for i in ids], dtype=float
        )


class C1CrossFitter:
    """Cache by exact training items, including C1's own hyperparameter selection.

    ``fit_data`` supplies original-scheme inputs for cross-version transfer. Only
    the caller's training items are labelled; predictions use the target signals.
    """

    def __init__(
        self,
        data: C1Dataset,
        labels: pd.DataFrame,
        seed: int,
        config: C1Config = PRIMARY_CONFIG,
        *,
        fit_data: C1Dataset | None = None,
    ):
        if config.combiner != "stacking" or config.supervision != "cq":
            raise SchemeError("C2/C3 learned inputs require CQ-supervised C1 stacking")
        if config.exclude_schemes or config.unstable:
            raise SchemeError("C2/C3 require the primary C1 label population")
        self.data, self.labels, self.seed = data, labels, seed
        self.config, self.fit_data = config, data if fit_data is None else fit_data
        if not set(data.items.item_id) <= set(self.fit_data.items.item_id):
            raise SchemeError("target items are missing from the original C1 inputs")
        self.models, self.choices, self._fitted, self._crossed = {}, [], {}, {}

    def _inputs(self, p, ids):
        wanted = set(ids)
        aggregation = aggregate_combined(self.data, p, "soft")
        return CQInputs(self.data, p, {i: a for i, a in aggregation.items() if i in wanted})

    def fit(self, ids: Sequence[str]) -> CQInputs:
        key = tuple(sorted(set(ids)))
        if key not in self._fitted:
            model, choice = fit_c1(self.fit_data, self.labels, key, self.seed, self.config)
            self.models[key] = model
            self.choices.append({"training_items": key, **choice})
            self._fitted[key] = self._inputs(model.predict(self.data), self.data.items.item_id)
        return self._fitted[key]

    def cross_fit(self, ids: Sequence[str]) -> CQInputs:
        key = tuple(sorted(set(ids)))
        if key not in self._crossed:
            p = np.full(len(self.data.pairs), 0.5)
            for train, test in inner_partitions(key, self.seed):
                selected = self.data.pair_indices(test)
                p[selected] = self.fit(train).p[selected]
            self._crossed[key] = self._inputs(p, key)
        return self._crossed[key]

    def partition(self, train: Sequence[str], test: Sequence[str]) -> tuple[CQInputs, CQInputs]:
        if set(train) & set(test):
            raise ValueError("training and held-out items overlap")
        return self.cross_fit(train), self.fit(train)


class ReferenceInputs:
    """Independent reference probabilities for routing, hard answers for its tree."""

    def __init__(self, data: C1Dataset, seed: int, hard: bool = False):
        self.seed, self.hard = seed, hard
        e = data.entries.index(data.reference)
        self.data = replace(data, outside=1 - data.arc_mass[:, e])
        p = data.base_splits[:, e].copy()
        predictions = aggregate_combined(self.data, p, "soft")
        if hard:
            predictions = {}
            for item, group in data.pairs.groupby("item_id", sort=False):
                scheme = data.schemes[group.scheme.iloc[0]]
                rows = {row.cq_id: {"answer": data.hard[i, e]} for i, row in group.iterrows()}
                predictions[item] = a0(scheme, rows)
        self.inputs = CQInputs(self.data, p, predictions, hard_tree=True)

    def fit(self, ids: Sequence[str]) -> CQInputs:
        return self.inputs

    def partition(self, train: Sequence[str], test: Sequence[str]) -> tuple[CQInputs, CQInputs]:
        if set(train) & set(test):
            raise ValueError("training and held-out items overlap")
        return self.inputs, self.inputs
