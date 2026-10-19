"""A3: a weighted checklist per scheme, a logistic regression on the masses of the arcs.

One model per scheme, trained on the items of that scheme in the other folds.
The features are, for each CQ in the order of the diagram, the masses of its two
arcs; what is missing to 1 is the mass outside them.  The class weights are
balanced, which aligns the training with macro F1: without them a model with weak
features drifts toward the most frequent class of the scheme.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from ..schemes import Scheme
from .interface import TIE, Aggregation, decide, macro_f1
from .soft import LOGPROB, Rows, answer_masses

C_GRID = (0.01, 0.1, 1.0, 10.0)
INNER_FOLDS = 4


def features(scheme: Scheme, rows: Rows, source: str = LOGPROB) -> list[float]:
    """The masses of the two arcs of every CQ, in the order of the diagram and of its answers.

    A CQ without masses takes its hard answer as 1 and 0; an answer outside the arcs
    gives 0 and 0.
    """
    out = []
    for cq in scheme.cq_order:
        masses = answer_masses(scheme, cq, rows.get(cq), source) or {}
        out += [masses.get(answer, 0.0) for answer in scheme.answer_space(cq)]
    return out


def _names(scheme: Scheme) -> list[str]:
    return [f"{cq} {answer}" for cq in scheme.cq_order for answer in scheme.answer_space(cq)]


def _fit(x: np.ndarray, y: Sequence[str], c: float | None, seed: int) -> LogisticRegression | str:
    """The fitted model, or the class itself when the training items have one class."""
    classes = sorted(set(y))
    if len(classes) == 1:
        return classes[0]
    # the default penalty of lbfgs is L2; no scaling, the masses are already in [0, 1]
    model = LogisticRegression(C=c, solver="lbfgs", class_weight="balanced",
                               max_iter=10000, random_state=seed)
    return model.fit(x, list(y))


def _predict(model: LogisticRegression | str, x: np.ndarray) -> list[str]:
    if isinstance(model, str):
        return [model] * len(x)
    return [str(label) for label in model.predict(x)]


def choose_c(x: Sequence[Sequence[float]], y: Sequence[str], seed: int) -> float:
    """The C of the grid with the best macro F1 of the pooled predictions of an inner
    cross validation; ties go to the smaller C."""
    x, y = np.asarray(x, dtype=float), list(y)
    if len(y) < INNER_FOLDS:
        return C_GRID[0]
    # not stratified: some classes have one or two items
    splits = list(KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed).split(x))
    best_score, best_c = None, C_GRID[0]
    for c in C_GRID:
        predicted: list[str | None] = [None] * len(y)
        for train, test in splits:
            model = _fit(x[train], [y[i] for i in train], c, seed)
            for index, label in zip(test, _predict(model, x[test]), strict=True):
                predicted[index] = label
        score = macro_f1(y, predicted)
        if best_score is None or score > best_score + TIE:
            best_score, best_c = score, c
    return best_c


def _explain(model: LogisticRegression, names: list[str], row: np.ndarray, verdict: str,
             probability: float, c: float) -> str:
    """The three largest contributions, coefficient times feature, to the class chosen."""
    classes = [str(label) for label in model.classes_]
    if len(classes) == 2:
        # one row of coefficients, for the second class against the first
        weights = model.coef_[0] if verdict == classes[1] else -model.coef_[0]
    else:
        weights = model.coef_[classes.index(verdict)]
    contributions = sorted(zip(weights * row, names, strict=True), key=lambda x: -x[0])[:3]
    parts = ", ".join(f"{name} {value:+.2f}" for value, name in contributions)
    return f"{verdict} {probability:.2f}: {parts}; C {c:g}"


def a3_out_of_fold(
    schemes: Mapping[str, Scheme], items: pd.DataFrame, rows_by_item: Mapping[str, Rows],
    folds: Mapping[str, int], seed: int, source: str = LOGPROB,
) -> tuple[dict[str, Aggregation], list[dict[str, Any]]]:
    """A3 on every item, from a model trained on the items of its scheme in the other folds.

    ``items`` has ``item_id``, ``gold_scheme`` and ``gold_fallacy``.  An item whose
    gold is not a terminal of its scheme stays out of training.  Returns the
    aggregation of every item and, per fold and scheme, the C chosen (None with one
    class among the training items).
    """
    out: dict[str, Aggregation] = {}
    choices: list[dict[str, Any]] = []
    for scheme_id, own in sorted(items.groupby("gold_scheme"), key=lambda pair: pair[0]):
        scheme = schemes[scheme_id]
        own = own.sort_values("item_id")
        ids = own["item_id"].tolist()
        gold = dict(zip(ids, own["gold_fallacy"], strict=True))
        x = {item: np.asarray(features(scheme, rows_by_item[item], source)) for item in ids}
        names = _names(scheme)
        for fold in sorted(set(folds.values())):
            train = [item for item in ids
                     if folds[item] != fold and gold[item] in scheme.terminals]
            test = [item for item in ids if folds[item] == fold]
            if not test:
                continue
            x_train = np.asarray([x[item] for item in train])
            y_train = [gold[item] for item in train]
            c = choose_c(x_train, y_train, seed) if len(set(y_train)) > 1 else None
            model = _fit(x_train, y_train, c, seed)
            choices.append({"fold": fold, "scheme": scheme_id, "C": c})
            for item in test:
                if isinstance(model, str):
                    out[item] = decide({model: 1.0}, 0.0,
                                       f"{model} 1.00: the only class among the training items")
                    continue
                probabilities = model.predict_proba(x[item].reshape(1, -1))[0]
                distribution = {str(label): float(p)
                                for label, p in zip(model.classes_, probabilities, strict=True)}
                result = decide(distribution, 0.0, "")
                explanation = _explain(model, names, x[item], result.verdict,
                                       result.confidence, c)
                out[item] = decide(distribution, 0.0, explanation)
    choices.sort(key=lambda record: (record["fold"], record["scheme"]))
    return out, choices
