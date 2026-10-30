"""Four CQ combiners; fitted parameters contain no held-out labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp

from ..schemes import SchemeError, enumerate_paths
from .b1_inputs import B1Dataset, Signal
from .burden import SPLIT

EPS = 1e-12
GOLD_EPS = 1e-6


def majority(data: B1Dataset) -> np.ndarray:
    """The share of arc votes on the first arc; outside votes do not enter the split."""
    first, second = (data.observations == 0).sum(axis=1), (data.observations == 1).sum(axis=1)
    return np.divide(first, first + second, out=np.full(len(first), 0.5), where=first + second > 0)


def majority_codes(data: B1Dataset) -> np.ndarray:
    """Initialise EM with the majority split and the common outside-mass rule."""
    return combined_codes(data, majority(data))


def combined_codes(data: B1Dataset, p: np.ndarray) -> np.ndarray:
    """Outside mass above one half abstains; arc ties prefer the reference arc."""
    out = np.where(np.asarray(p) >= 0.5, 0, 1)
    ref = data.observations[:, data.entries.index(data.reference)]
    out[(np.abs(np.asarray(p) - 0.5) <= EPS) & (ref == 1)] = 1
    out[data.outside > 0.5] = 2
    return out


@dataclass(frozen=True)
class WeightedPool:
    weights: Mapping[tuple[str, str], np.ndarray]
    selections: tuple[str, ...]
    entries: tuple[str, ...]
    shrinkage: float = 0.0

    @classmethod
    def equal(cls, data: B1Dataset) -> WeightedPool:
        return cls(
            {q: np.full(5, 0.2) for q in data.questions},
            tuple(data.sources[e] for e in data.entries),
            data.entries,
        )

    @classmethod
    def fit(cls, data, labels, ids, shrinkage=0.0, selections=None):
        target = data.targets(labels)
        training = data.pairs.item_id.isin(set(ids)).to_numpy() & np.isfinite(target)
        qindex, weights = data.question_index, {}
        for q, question in enumerate(data.questions):
            learned = np.zeros(5)
            for e in range(5):
                valid = training & (qindex == q) & (data.observations[:, e] >= 0)
                n = valid.sum()
                correct = (data.observations[valid, e] == np.where(target[valid] == 1, 0, 1)).sum()
                learned[e] = max(0.0, logit((correct + 1) / (n + 2)))
            learned = learned / learned.sum() if learned.sum() else np.full(5, 0.2)
            weights[question] = (1 - shrinkage) / 5 + shrinkage * learned
        return cls(
            weights,
            selections or tuple(data.sources[e] for e in data.entries),
            data.entries,
            shrinkage,
        )

    def predict(self, data: B1Dataset) -> np.ndarray:
        if data.entries != self.entries:
            raise SchemeError("weighted pool entries differ from its fitted entries")
        split = np.zeros_like(data.base_splits)
        effective = data.arc_mass.copy()
        for e, (entry, source) in enumerate(zip(data.entries, self.selections, strict=True)):
            if source == data.sources[entry]:
                # The distribution includes the same hard fallback as A4.
                split[:, e] = data.base_splits[:, e]
            else:
                values = data.splits[:, data.signals.index(Signal(entry, source))]
                effective[~np.isfinite(values), e] = 0.0
                split[:, e] = np.nan_to_num(values, nan=0.5)
        weights = np.array(
            [
                self.weights.get(q, np.full(5, 0.2))
                for q in zip(data.pairs.scheme, data.pairs.cq_id, strict=True)
            ]
        )
        weighted_mass = weights * effective
        denominator = weighted_mass.sum(axis=1)
        return np.divide(
            (weighted_mass * split).sum(axis=1),
            denominator,
            out=np.full(len(split), 0.5),
            where=denominator > 0,
        )


@dataclass(frozen=True)
class DawidSkene:
    confusion: np.ndarray
    priors: Mapping[tuple[str, str], np.ndarray]
    iterations: int = 0
    log_likelihood: float = np.nan

    def posterior(self, data):
        priors = np.array(
            [
                self.priors.get(q, np.array([0.5, 0.5]))
                for q in zip(data.pairs.scheme, data.pairs.cq_id, strict=True)
            ]
        )
        joint = np.log(np.maximum(priors, EPS))
        for e in range(5):
            valid = data.observations[:, e] >= 0
            joint[valid] += np.log(
                np.maximum(self.confusion[e][:, data.observations[valid, e]].T, EPS)
            )
        norm = logsumexp(joint, axis=1)
        return np.exp(joint - norm[:, None]), float(norm.sum())

    def predict(self, data):
        return self.posterior(data)[0][:, 0]

    @classmethod
    def fit(cls, data):
        initial = majority_codes(data)
        posterior = np.column_stack(
            [
                np.where(initial == 0, 1.0, np.where(initial == 1, 0.0, 0.5)),
                np.where(initial == 1, 1.0, np.where(initial == 0, 0.0, 0.5)),
            ]
        )
        qindex, previous = data.question_index, -np.inf
        for iteration in range(1, 501):
            confusion = np.zeros((5, 2, 3))
            for e in range(5):
                for observed in range(3):
                    confusion[e, :, observed] = posterior[data.observations[:, e] == observed].sum(
                        axis=0
                    )
            confusion = np.maximum(confusion, EPS)
            confusion /= confusion.sum(axis=2, keepdims=True)
            priors = {}
            for q, question in enumerate(data.questions):
                counts = np.maximum(posterior[qindex == q].sum(axis=0), EPS)
                priors[question] = counts / counts.sum()
            model = cls(confusion, priors, iteration)
            posterior, likelihood = model.posterior(data)
            model = cls(confusion, priors, iteration, likelihood)
            if abs(likelihood - previous) < 1e-8:
                break
            previous = likelihood
        return model


def balanced_weights(labels: Sequence[str]) -> np.ndarray:
    """The weight of each item, n / (K n_label) as in ``class_weight="balanced"``: every
    label weighs the same in total, and the weights have mean 1."""
    labels = list(labels)
    if not labels:
        return np.zeros(0)
    counts = Counter(labels)
    return np.array([len(labels) / (len(counts) * counts[label]) for label in labels])


class DiagramLoss:
    """Vectorised gold-terminal path probabilities and their exact derivatives.

    With ``balanced`` every item weighs n / (K n_label), computed on the gold verdicts of
    the items of the loss, so that each verdict weighs the same in total.
    """

    def __init__(self, data: B1Dataset, ids: Sequence[str], balanced: bool = False):
        self.data = data
        lookup = {
            (row.item_id, row.cq_id): i for i, row in enumerate(data.pairs.itertuples(index=False))
        }
        wanted = set(ids)
        self.ids, paths, path_items = [], [], []
        for item in data.items.itertuples(index=False):
            scheme = data.schemes[item.gold_scheme]
            if item.item_id not in wanted or item.gold_fallacy not in scheme.terminals:
                continue
            number = len(self.ids)
            self.ids.append(item.item_id)
            for steps, terminal in enumerate_paths(scheme):
                if terminal == item.gold_fallacy:
                    paths.append(
                        [
                            (lookup[(item.item_id, cq)], scheme.answer_space(cq).index(a))
                            for cq, a in steps
                        ]
                    )
                    path_items.append(number)
        width = max((len(path) for path in paths), default=0)
        self.indices = np.zeros((len(paths), width), dtype=int)
        self.arcs = np.zeros_like(self.indices)
        self.mask = np.zeros_like(self.indices, dtype=bool)
        for i, path in enumerate(paths):
            for j, (index, arc) in enumerate(path):
                self.indices[i, j], self.arcs[i, j], self.mask[i, j] = index, arc, True
        self.path_items = np.array(path_items, dtype=int)
        gold = data.items.set_index("item_id").gold_fallacy
        self.item_weights = (balanced_weights([gold[i] for i in self.ids]) if balanced
                             else None)
        self.added = np.zeros((len(data.pairs), 2))
        for i, pair in enumerate(data.pairs.itertuples(index=False)):
            route = data.routing[pair.scheme][pair.cq_id]
            if route == SPLIT:
                self.added[i] = 0.5
            elif route in data.schemes[pair.scheme].answer_space(pair.cq_id):
                self.added[i, data.schemes[pair.scheme].answer_space(pair.cq_id).index(route)] = 1.0

    def _paths(self, p):
        mass = np.column_stack([p, 1 - p]) * (1 - self.data.outside[:, None])
        mass += self.added * self.data.outside[:, None]
        factors = np.where(self.mask, mass[self.indices, self.arcs], 1.0)
        probability = np.bincount(
            self.path_items, weights=factors.prod(axis=1), minlength=len(self.ids)
        )
        return probability, factors

    def log_mass(self, p):
        return np.log(self._paths(p)[0] + GOLD_EPS)

    def objective(self, coef, x):
        p = expit(x @ coef)
        probability, factors = self._paths(p)
        prefix = np.column_stack([np.ones(len(factors)), np.cumprod(factors, axis=1)])[:, :-1]
        suffix = np.column_stack(
            [np.cumprod(factors[:, ::-1], axis=1)[:, ::-1], np.ones(len(factors))]
        )[:, 1:]
        derivative = prefix * suffix * (1 - 2 * self.arcs) * (1 - self.data.outside[self.indices])
        derivative *= (-1 / (probability[self.path_items] + GOLD_EPS))[:, None] / len(self.ids)
        if self.item_weights is not None:
            derivative *= self.item_weights[self.path_items][:, None]
        gradient_p = np.bincount(
            self.indices[self.mask], weights=derivative[self.mask], minlength=len(p)
        )
        losses = -np.log(probability + GOLD_EPS)
        value = losses.mean() if self.item_weights is None else (
            self.item_weights * losses).mean()
        return value, x.T @ (gradient_p * p * (1 - p))


@dataclass(frozen=True)
class Stacking:
    coef: np.ndarray
    signals: tuple[Signal, ...]
    c: float | None = None

    @classmethod
    def at_centre(cls, data):
        coef = np.zeros(data.x.shape[1])
        coef[1 + np.array(data.primary_signals)] = 0.2
        return cls(coef, data.signals)

    def predict(self, data):
        if data.signals != self.signals:
            raise SchemeError("stacking signals differ from its fitted signal order")
        return expit(data.x @ self.coef)

    @classmethod
    def _optimise(cls, data, objective, n, c):
        centre = cls.at_centre(data).coef
        if not n:
            return cls(centre, data.signals, c)

        def penalised(coef):
            value, gradient = objective(coef)
            delta = coef - centre
            return value + delta @ delta / (2 * c * n), gradient + delta / (c * n)

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
            raise SchemeError(f"stacking optimisation failed: {result.message}")
        return cls(np.asarray(result.x), data.signals, c)

    @classmethod
    def fit_cq(cls, data, labels, ids, c):
        target = data.targets(labels)
        selected = data.pairs.item_id.isin(set(ids)).to_numpy() & np.isfinite(target)
        x, y = data.x[selected], target[selected]

        def loss(coef):
            z = x @ coef
            return np.mean(np.logaddexp(0, z) - y * z), x.T @ (expit(z) - y) / len(y)

        return cls._optimise(data, loss, len(y), c)

    @classmethod
    def fit_gold(cls, data, ids, c, balanced=False):
        loss = DiagramLoss(data, ids, balanced)
        return cls._optimise(data, lambda coef: loss.objective(coef, data.x), len(loss.ids), c)
