"""B2: estimate correctness on disagreements and choose an available side."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..eval.folds import assign_folds
from ..parse import INVALID
from ..schemes import SchemeError
from .b1 import C_GRID
from .crossfit import CQInputs, inner_partitions
from .interface import TIE, Aggregation, decide
from .soft import OUTSIDE, _most_probable_path, _route

FEATURES = ("confidence", "margin", "abstention", "entropy", "p_label", "scheme")


@dataclass(frozen=True)
class Side:
    name: str
    aggregations: Mapping[str, Aggregation]
    p_label: Mapping[str, float] | None = None
    consistent: Mapping[str, bool] | None = None
    explained: bool = False
    schemes_version: str | None = None

    def __post_init__(self):
        for item, value in (self.p_label or {}).items():
            if value is None or pd.isna(value):
                continue
            if not np.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise SchemeError(f"{self.name} {item}: p_label outside [0, 1]: {value}")

    @classmethod
    def from_summary(cls, summary, name, model, stage="zeroshot"):
        rows = summary[(summary.model == model) & (summary.stage == stage)]
        if "scheme_condition" in rows:
            rows = rows[rows.scheme_condition == "gold"]
        if rows.item_id.duplicated().any():
            item = rows.loc[rows.item_id.duplicated(), "item_id"].iloc[0]
            raise ValueError(f"{name}: duplicate verdict rows for {item}")
        predictions, probabilities = {}, {}
        for row in rows.to_dict("records"):
            item, label = row["item_id"], row["answer"]
            p = row.get("p_label", np.nan)
            p = np.nan if p is None or pd.isna(p) else float(p)
            if np.isfinite(p) and not 0 <= p <= 1:
                raise SchemeError(f"{name} {item}: p_label outside [0, 1]")
            absent = label is None or pd.isna(label) or label in {"", INVALID}
            probabilities[item] = p
            predictions[item] = Aggregation(
                None if absent else label,
                None if not np.isfinite(p) else p,
                {} if absent else {label: 1.0},
                float(absent),
                row.get("justification", ""),
            )
        return cls(name, predictions, probabilities)


@dataclass(frozen=True)
class RoutingChoice:
    aggregation: Aggregation
    side: str | None
    explained: bool


def _pick(prediction, probability, side, estimates):
    explanation = f"{side}; " + ", ".join(f"P({s})={p:.6f}" for s, p in estimates.items())
    return RoutingChoice(
        replace(
            prediction,
            confidence=float(probability),
            explanation=f"{explanation}; {prediction.explanation}",
        ),
        side,
        side == "diagram",
    )


def route(
    diagram: Aggregation, zero: Aggregation, p_d: float, p_z: float, tau: float = 0.0
) -> RoutingChoice:
    """Forced agreement/fallback rules precede the threshold on disagreements."""
    estimates = {"diagram": p_d, "zero-shot": p_z}
    if diagram.verdict is None and zero.verdict is None:
        return RoutingChoice(decide({}, 1.0, "no side has a verdict"), None, False)
    if diagram.verdict is not None and (zero.verdict is None or diagram.verdict == zero.verdict):
        return _pick(diagram, p_d, "diagram", estimates)
    if diagram.verdict is None:
        return _pick(zero, p_z, "zero-shot", estimates)
    if p_d + tau >= p_z - TIE:
        return _pick(diagram, p_d, "diagram", estimates)
    return _pick(zero, p_z, "zero-shot", estimates)


def _multi_route(predictions, probabilities):
    available = [j for j, a in enumerate(predictions) if a.verdict is not None]
    if not available:
        return RoutingChoice(decide({}, 1.0, "no side has a verdict"), None, False)
    if len({predictions[j].verdict for j in available}) == 1:
        selected = available[0]
    else:
        best = max(probabilities[j] for j in available)
        selected = next(j for j in available if probabilities[j] >= best - TIE)
    return selected


def routing_features(inputs: CQInputs, zero: Side, ids, *, extra_sides=()):
    """Entropy in bits on the most probable diagram path; no label is read."""
    metadata = inputs.data.items.set_index("item_id")
    by_item = dict(tuple(inputs.rows(ids).groupby("item_id", sort=False)))
    records = []
    for item in ids:
        scheme = inputs.data.schemes[metadata.loc[item, "gold_scheme"]]
        own = by_item[item]
        p = dict(zip(own.cq_id, own.p_first, strict=True))
        outside = dict(zip(own.cq_id, own.outside, strict=True))
        arcs = {}
        for cq in scheme.cq_order:
            a, b = scheme.answer_space(cq)
            masses = {
                a: (1 - outside[cq]) * p[cq],
                b: (1 - outside[cq]) * (1 - p[cq]),
                OUTSIDE: outside[cq],
            }
            arcs[cq] = _route(scheme, cq, masses, inputs.data.routing[scheme.scheme_id][cq])[0]
        mass, steps, _ = _most_probable_path(scheme, arcs)
        entropy = (
            [
                -(p[cq] * np.log2(p[cq]) + (1 - p[cq]) * np.log2(1 - p[cq]))
                if 0 < p[cq] < 1
                else 0.0
                for cq, _, _ in steps
            ]
            if mass > 0
            else []
        )
        prediction = inputs.aggregations[item]
        highest = sorted(prediction.distribution.values(), reverse=True) + [0.0, 0.0]
        confidence = prediction.confidence
        records.append(
            {
                "confidence": highest[0] if confidence is None else confidence,
                "margin": highest[0] - highest[1],
                "abstention": prediction.abstention,
                "entropy": float(np.mean(entropy)) if entropy else 0.0,
                "p_label": (zero.p_label or {}).get(item, np.nan),
                "scheme": scheme.scheme_id,
            }
        )
    frame = pd.DataFrame(records, columns=FEATURES)
    for side in extra_sides:
        confidence = (
            side.p_label
            if side.p_label is not None
            else {i: a.confidence for i, a in side.aggregations.items()}
        )
        frame[f"{side.name}.confidence"] = [confidence.get(i, np.nan) for i in ids]
        if side.consistent is not None:
            frame[f"{side.name}.consistent"] = [bool(side.consistent.get(i, False)) for i in ids]
    return frame


@dataclass(frozen=True)
class Router:
    models: tuple
    mean_p_label: float
    schemes: tuple[str, ...]
    extra_means: tuple[tuple[str, float], ...] = ()
    extra_binary: tuple[str, ...] = ()

    def transform(self, frame):
        values = frame[list(FEATURES[:-1])].to_numpy(dtype=float, copy=True)
        missing = ~np.isfinite(values[:, -1])
        values[missing, -1] = self.mean_p_label
        onehot = (
            np.column_stack([(frame.scheme == s).to_numpy() for s in self.schemes])
            if self.schemes
            else np.empty((len(frame), 0))
        )
        extra = []
        for column, mean in self.extra_means:
            own = frame[column].to_numpy(dtype=float, copy=True)
            absent = ~np.isfinite(own)
            own[absent] = mean
            extra.extend([own, absent])
        extra.extend(frame[c].to_numpy(dtype=float) for c in self.extra_binary)
        return np.column_stack([values, missing, onehot, *extra])

    @classmethod
    def fit(cls, frame, targets, c, seed):
        observed = pd.to_numeric(frame.p_label, errors="coerce").dropna()
        mean = float(observed.mean()) if len(observed) else 0.5
        extra_means = []
        for column in frame.columns:
            if column.endswith(".confidence"):
                observed = pd.to_numeric(frame[column], errors="coerce")
                observed = observed[np.isfinite(observed)]
                extra_means.append((column, float(observed.mean()) if len(observed) else 0.5))
        binary = tuple(c for c in frame.columns if c.endswith(".consistent"))
        skeleton = cls((), mean, tuple(sorted(frame.scheme.unique())), tuple(extra_means), binary)
        x, targets = skeleton.transform(frame), np.asarray(targets, dtype=float)
        if targets.ndim != 2 or len(targets) != len(frame):
            raise ValueError("router targets must have one row per training item")
        penalties = (c,) * targets.shape[1] if np.isscalar(c) else c
        models = []
        for j, penalty in enumerate(penalties):
            valid = np.isfinite(targets[:, j])
            y = targets[valid, j]
            if not len(y) or len(set(y)) == 1:
                models.append(float(y.mean()) if len(y) else 0.5)
            else:
                model = LogisticRegression(
                    C=penalty, solver="lbfgs", max_iter=10000, random_state=seed
                )
                models.append(model.fit(x[valid], y))
        return replace(skeleton, models=tuple(models))

    def predict(self, frame):
        x = self.transform(frame)
        return np.column_stack(
            [
                np.full(len(frame), m) if isinstance(m, float) else m.predict_proba(x)[:, 1]
                for m in self.models
            ]
        )


def _sides(inputs, fixed, ids):
    return [
        [
            inputs.aggregations[i],
            *[s.aggregations.get(i, decide({}, 1.0, "missing")) for s in fixed],
        ]
        for i in ids
    ]


def _targets(inputs, fixed, ids):
    gold = inputs.data.items.set_index("item_id").gold_fallacy
    output = np.full((len(ids), len(fixed) + 1), np.nan)
    for i, (item, predictions) in enumerate(zip(ids, _sides(inputs, fixed, ids), strict=True)):
        if len({p.verdict for p in predictions if p.verdict is not None}) < 2:
            continue
        for j, prediction in enumerate(predictions):
            if prediction.verdict is not None:
                output[i, j] = float(prediction.verdict == gold[item])
    return output


def _choose(source, fixed, train):
    losses, counts = np.zeros((len(C_GRID), len(fixed) + 1)), np.zeros(len(fixed) + 1)
    for fitting, validation in inner_partitions(train, source.seed):
        training, testing = source.partition(fitting, validation)
        x, v = (
            routing_features(training, fixed[0], fitting, extra_sides=fixed[1:]),
            routing_features(testing, fixed[0], validation, extra_sides=fixed[1:]),
        )
        y, truth = _targets(training, fixed, fitting), _targets(testing, fixed, validation)
        valid = np.isfinite(truth)
        counts += valid.sum(axis=0)
        for k, c in enumerate(C_GRID):
            model = Router.fit(x, y, c, source.seed)
            p = np.clip(model.predict(v), 1e-15, 1 - 1e-15)
            losses[k] += np.where(valid, -truth * np.log(p) - (1 - truth) * np.log(1 - p), 0.0).sum(
                axis=0
            )
    choices = []
    for j in range(len(fixed) + 1):
        selected, best = C_GRID[0], np.inf
        for k, c in enumerate(C_GRID):
            if counts[j] and losses[k, j] / counts[j] < best - TIE:
                selected, best = c, losses[k, j] / counts[j]
        choices.append(selected)
    return tuple(choices)


@dataclass(frozen=True)
class B2Result:
    aggregations: Mapping[str, Aggregation]
    records: pd.DataFrame
    models: Mapping[int, Router]
    choices: list[dict]
    upstream: Mapping[str, Aggregation]
    fixed_sides: tuple[Side, ...]
    folds: Mapping[str, int]
    name: str

    def at_tau(self, tau):
        if len(self.fixed_sides) != 1:
            raise ValueError("the tau curve requires exactly two sides")
        predictions, sides, explained = {}, [], []
        # the columns are assigned once: the curve calls this for every threshold
        for row in self.records.itertuples(index=False):
            choice = route(
                self.upstream[row.item_id],
                self.fixed_sides[0].aggregations.get(row.item_id, decide({}, 1.0, "missing")),
                row.p_d,
                row.p_z,
                tau,
            )
            predictions[row.item_id] = choice.aggregation
            sides.append(choice.side)
            explained.append(choice.explained)
        records = self.records.assign(side=sides, explained=explained)
        return replace(self, aggregations=predictions, records=records)


def b2_out_of_fold(
    source, zero: Side, folds=None, *, tau=0.0, extra_sides=(), name="B2", schemes_version=None
) -> B2Result:
    data = source.data
    folds = dict(folds) if folds is not None else assign_folds(data.items, source.seed)
    fixed = (zero, *extra_sides)
    names = ("diagram", "zero-shot", *[s.name for s in extra_sides])
    if len(set(names)) != len(names) or (extra_sides and tau != 0):
        raise ValueError("additional sides need distinct names and tau zero")
    metadata = data.items.set_index("item_id")
    for side in fixed:
        if side.schemes_version is not None and schemes_version != side.schemes_version:
            raise SchemeError(
                f"{side.name}: schemes {side.schemes_version} cannot enter {schemes_version}"
            )
        for item in data.items.item_id:
            prediction = side.aggregations.get(item)
            if (
                prediction
                and prediction.verdict is not None
                and prediction.verdict
                not in (data.schemes[metadata.loc[item, "gold_scheme"]].terminals)
            ):
                raise SchemeError(
                    f"{side.name} {item}: unknown scheme terminal {prediction.verdict!r}"
                )
    predictions, upstream, models, choices, records = {}, {}, {}, [], []
    for fold in sorted({folds[i] for i in data.items.item_id}):
        train = sorted(i for i in data.items.item_id if folds[i] != fold)
        test = sorted(i for i in data.items.item_id if folds[i] == fold)
        training, testing = source.partition(train, test)
        penalties = _choose(source, fixed, train)
        model = Router.fit(
            routing_features(training, zero, train, extra_sides=extra_sides),
            _targets(training, fixed, train),
            penalties,
            source.seed,
        )
        models[fold] = model
        choices.append(
            {"fold": fold, "component": name, "C": dict(zip(names, penalties, strict=True))}
        )
        probability = model.predict(routing_features(testing, zero, test, extra_sides=extra_sides))
        for item, given, p in zip(test, _sides(testing, fixed, test), probability, strict=True):
            if not extra_sides:
                choice = route(*given, *p, tau)
            else:
                selected = _multi_route(given, p)
                if isinstance(selected, RoutingChoice):
                    choice = selected
                else:
                    j = selected
                    choice = _pick(given[j], p[j], names[j], dict(zip(names, p, strict=True)))
                    verdict = choice.aggregation.verdict
                    explained = verdict == given[0].verdict or any(
                        verdict == prediction.verdict
                        and (side.explained or bool((side.consistent or {}).get(item, False)))
                        for side, prediction in zip(fixed, given[1:], strict=True)
                    )
                    choice = replace(choice, explained=explained)
            upstream[item], predictions[item] = given[0], choice.aggregation
            records.append(
                {
                    "item_id": item,
                    "fold": fold,
                    "p_d": p[0],
                    "p_z": p[1],
                    "side": choice.side,
                    "explained": choice.explained,
                    "probabilities": dict(zip(names, p, strict=True)),
                }
            )
    return B2Result(
        predictions,
        pd.DataFrame(records).sort_values("item_id").reset_index(drop=True),
        models,
        choices,
        upstream,
        fixed,
        folds,
        name,
    )
