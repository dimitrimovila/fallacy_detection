"""C3: small scheme-specific trees and a pooled recalibration of terminal masses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.tree import DecisionTreeClassifier

from ..eval.folds import assign_folds
from ..schemes import SchemeError
from .c1 import C_GRID
from .crossfit import CQInputs, inner_partitions
from .interface import TIE, Aggregation, decide, macro_f1

# Shallower trees, then larger leaves, give stronger shrinkage on tied scores.
TREE_GRID = tuple(product((1, 2, 3), (5, 3, 1)))


@dataclass(frozen=True)
class CQTree:
    model: DecisionTreeClassifier | None
    names: tuple[str, ...]
    raw_leaves: Mapping[int, Mapping[str, int]]

    @classmethod
    def fit(cls, scheme, x, y, depth, leaf, seed):
        names = tuple(
            f"{cq} {feature}" for cq in scheme.cq_order for feature in ("p_first", "outside")
        )
        if not len(y):
            return cls(None, names, {})
        x = np.asarray(x, dtype=float)
        model = DecisionTreeClassifier(
            criterion="gini",
            class_weight="balanced",
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=seed,
        )
        model.fit(x, y)
        leaves = {}
        for node, label in zip(model.apply(x), y, strict=True):
            counts = leaves.setdefault(int(node), {})
            counts[label] = counts.get(label, 0) + 1
        return cls(model, names, leaves)

    def predict(self, row) -> Aggregation:
        if self.model is None:
            return decide({}, 1.0, "no eligible training items for this scheme")
        row = np.asarray(row, dtype=float)
        probabilities = self.model.predict_proba(row.reshape(1, -1))[0]
        distribution = dict(zip(self.model.classes_, probabilities, strict=True))
        result = decide(distribution, 0.0, "")
        node, steps = 0, []
        while self.model.tree_.children_left[node] != -1:
            feature, threshold = self.model.tree_.feature[node], self.model.tree_.threshold[node]
            left = row[feature] <= threshold
            steps.append(
                f"{self.names[feature]} {row[feature]:.6f} {'<=' if left else '>'} {threshold:.6f}"
            )
            node = (
                self.model.tree_.children_left[node]
                if left
                else self.model.tree_.children_right[node]
            )
        counts = self.raw_leaves[int(node)]
        confidence = counts.get(result.verdict, 0) / sum(counts.values())
        path = " > ".join(steps) or "root leaf"
        return replace(
            result,
            confidence=confidence,
            explanation=f"{path}; {result.verdict}; raw leaf share "
            f"{confidence:.6f} ({counts.get(result.verdict, 0)}/{sum(counts.values())})",
        )


def _eligible(data, ids):
    items = data.items.set_index("item_id")
    return sorted(
        i
        for i in ids
        if items.loc[i, "gold_fallacy"] in data.schemes[items.loc[i, "gold_scheme"]].terminals
    )


@dataclass(frozen=True)
class CalibrationLoss:
    labels: tuple[str, ...]
    ids: tuple[str, ...]
    logs: np.ndarray
    allowed: np.ndarray
    gold: np.ndarray

    def __call__(self, coef):
        if not self.ids:
            return 0.0, np.zeros_like(coef)
        logits = np.where(self.allowed, coef[0] * self.logs + coef[1:], -np.inf)
        normal = logsumexp(logits, axis=1)
        probability = np.exp(logits - normal[:, None])
        row = np.arange(len(self.ids))
        value = np.mean(normal - logits[row, self.gold])
        probability[row, self.gold] -= 1
        return value, np.r_[
            np.mean((probability * self.logs).sum(axis=1)), probability.mean(axis=0)
        ]


@dataclass(frozen=True)
class PooledCalibration:
    coef: np.ndarray
    labels: tuple[str, ...]
    c: float
    objective_value: float

    @staticmethod
    def loss(inputs: CQInputs, ids) -> CalibrationLoss:
        data = inputs.data
        labels = tuple(sorted(set().union(*(s.terminals for s in data.schemes.values()))))
        ids = tuple(_eligible(data, ids))
        items = data.items.set_index("item_id")
        allowed = np.zeros((len(ids), len(labels)), dtype=bool)
        logs, gold = np.zeros_like(allowed, dtype=float), []
        for i, item in enumerate(ids):
            scheme = data.schemes[items.loc[item, "gold_scheme"]]
            for terminal in scheme.terminals:
                j = labels.index(terminal)
                allowed[i, j] = True
                logs[i, j] = np.log(
                    inputs.aggregations[item].distribution.get(terminal, 0.0) + 1e-6
                )
            gold.append(labels.index(items.loc[item, "gold_fallacy"]))
        return CalibrationLoss(labels, ids, logs, allowed, np.asarray(gold, dtype=int))

    @classmethod
    def fit(cls, inputs: CQInputs, ids, c):
        loss = cls.loss(inputs, ids)
        centre = np.r_[1.0, np.zeros(len(loss.labels))]
        if not loss.ids:
            return cls(centre, loss.labels, c, 0.0)

        def penalised(coef):
            value, gradient = loss(coef)
            delta = coef - centre
            return value + delta @ delta / (2 * c * len(loss.ids)), gradient + delta / (
                c * len(loss.ids)
            )

        result = minimize(
            penalised,
            centre,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
        )
        if not np.isfinite(result.fun) or (
            not result.success and np.max(np.abs(result.jac)) > 1e-5
        ):
            raise SchemeError(f"pooled calibration optimisation failed: {result.message}")
        return cls(result.x, loss.labels, c, float(result.fun))

    def predict(self, inputs: CQInputs, ids) -> dict[str, Aggregation]:
        metadata = inputs.data.items.set_index("item_id")
        result = {}
        for item in ids:
            scheme = inputs.data.schemes[metadata.loc[item, "gold_scheme"]]
            terminals = sorted(scheme.terminals)
            logs = np.array(
                [
                    np.log(inputs.aggregations[item].distribution.get(t, 0.0) + 1e-6)
                    for t in terminals
                ]
            )
            logits = self.coef[0] * logs + [self.coef[1 + self.labels.index(t)] for t in terminals]
            probabilities = np.exp(logits - logsumexp(logits))
            distribution = dict(zip(terminals, probabilities, strict=True))
            result[item] = decide(
                distribution,
                0.0,
                f"pooled terminal calibration; slope {self.coef[0]:.6f}; "
                f"C {self.c:g}; {inputs.aggregations[item].explanation}",
            )
        return result


def _tree_choice(source, outer_train, eligible, scheme):
    if len(eligible) < 4:
        return TREE_GRID[0]
    gold = source.data.items.set_index("item_id").gold_fallacy
    splits = []
    for train, validation in inner_partitions(eligible, source.seed):
        context = sorted(set(outer_train) - set(validation))
        fitting, testing = source.partition(context, validation)
        splits.append(
            (
                fitting.tree_features(train),
                [gold[i] for i in train],
                testing.tree_features(validation),
                [gold[i] for i in validation],
            )
        )
    chosen, best = TREE_GRID[0], -np.inf
    for depth, leaf in TREE_GRID:
        truth, predicted = [], []
        for x, y, v, target in splits:
            model = CQTree.fit(scheme, x, y, depth, leaf, source.seed)
            truth.extend(target)
            predicted.extend(model.predict(row).verdict for row in v)
        score = macro_f1(truth, predicted)
        if score > best + TIE:
            chosen, best = (depth, leaf), score
    return chosen


def _calibration_choice(source, train):
    eligible = _eligible(source.data, train)
    if len(eligible) < 4:
        return C_GRID[0]
    splits = []
    for fitting, validation in inner_partitions(eligible, source.seed):
        context = sorted(set(train) - set(validation))
        a, b = source.partition(context, validation)
        splits.append((a, fitting, b, validation))
    chosen, best = C_GRID[0], np.inf
    for c in C_GRID:
        loss_sum, n = 0.0, 0
        for a, fitting, b, validation in splits:
            model = PooledCalibration.fit(a, fitting, c)
            value, _ = PooledCalibration.loss(b, validation)(model.coef)
            loss_sum, n = loss_sum + value * len(validation), n + len(validation)
        if n and loss_sum / n < best - TIE:
            chosen, best = c, loss_sum / n
    return chosen


@dataclass(frozen=True)
class C3Result:
    aggregations: Mapping[str, Aggregation]
    choices: list[dict]
    models: Mapping[int, object]
    upstream: Mapping[str, Aggregation]
    folds: Mapping[str, int]
    kind: str
    name: str


def c3_out_of_fold(source, folds=None, *, kind="tree", name=None) -> C3Result:
    if kind not in {"tree", "pooled"}:
        raise ValueError(f"unknown C3 model family {kind!r}")
    data, seed = source.data, source.seed
    if name is None:
        name = "C3.pooled" if kind == "pooled" else "C3"
    folds = dict(folds) if folds is not None else assign_folds(data.items, seed)
    metadata = data.items.set_index("item_id")
    predictions, models, upstream, choices = {}, {}, {}, []
    for fold in sorted({folds[i] for i in data.items.item_id}):
        train = sorted(i for i in data.items.item_id if folds[i] != fold)
        test = sorted(i for i in data.items.item_id if folds[i] == fold)
        training, testing = source.partition(train, test)
        upstream.update({i: testing.aggregations[i] for i in test})
        if kind == "pooled":
            c = _calibration_choice(source, train)
            model = PooledCalibration.fit(training, train, c)
            models[fold] = model
            predictions.update(model.predict(testing, test))
            choices.append({"fold": fold, "component": name, "C": c})
            continue
        models[fold] = {}
        for sid, scheme in sorted(data.schemes.items()):
            own_test = [i for i in test if metadata.loc[i, "gold_scheme"] == sid]
            if not own_test:
                continue
            fitting = _eligible(data, [i for i in train if metadata.loc[i, "gold_scheme"] == sid])
            classes = {metadata.loc[i, "gold_fallacy"] for i in fitting}
            depth, leaf = (
                _tree_choice(source, train, fitting, scheme) if len(classes) > 1 else TREE_GRID[0]
            )
            model = CQTree.fit(
                scheme,
                training.tree_features(fitting),
                [metadata.loc[i, "gold_fallacy"] for i in fitting],
                depth,
                leaf,
                seed,
            )
            models[fold][sid] = model
            choices.append(
                {
                    "fold": fold,
                    "component": name,
                    "scheme": sid,
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                }
            )
            predictions.update(
                {
                    i: model.predict(row)
                    for i, row in zip(own_test, testing.tree_features(own_test), strict=True)
                }
            )
    return C3Result(predictions, choices, models, upstream, folds, kind, name)
