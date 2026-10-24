"""Out-of-fold CQ combination and the hard/soft diagram conditions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..eval.folds import assign_folds
from ..prompts import CANNOT_BE_DETERMINED
from ..schemes import SchemeError
from .c1_inputs import C1Dataset
from .combiners import DawidSkene, Stacking, WeightedPool, combined_codes, majority
from .diagram import a0
from .interface import TIE, Aggregation, decide, macro_f1
from .soft import LOGPROB, OUTSIDE, _explain, a2, a4, traverse_soft

C_GRID = (0.01, 0.1, 1.0, 10.0)
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class C1Config:
    name: str = "C1"
    combiner: str = "stacking"
    supervision: str = "cq"
    traversal: str = "soft"
    exclude_schemes: tuple[str, ...] = ()
    unstable: bool = False

    def __post_init__(self):
        valid = {
            ("majority", "none"),
            ("dawid_skene", "none"),
            ("weighted", "cq"),
            ("stacking", "cq"),
            ("stacking", "gold"),
        }
        if (self.combiner, self.supervision) not in valid or self.traversal not in {"soft", "hard"}:
            raise SchemeError("unsupported C1 combiner, supervision or traversal")
        if (self.exclude_schemes or self.unstable) and self.supervision != "cq":
            raise SchemeError("CQ annotation exclusions require CQ supervision")


PRIMARY_CONFIG = C1Config()


def configurations(
    *, schemes_version="1.1", excluded_label_schemes=(), include_unstable=False
) -> tuple[C1Config, ...]:
    """Named fits; CQ-supervised transfer to 2.0 uses transfer_c1 instead of refitting."""
    if schemes_version not in {"1.1", "2.0"}:
        raise SchemeError(f"unsupported C1 schemes version {schemes_version!r}")
    out = []
    for combiner, supervision in [
        ("majority", "none"),
        ("dawid_skene", "none"),
        ("weighted", "cq"),
        ("stacking", "cq"),
        ("stacking", "gold"),
    ]:
        for traversal in ("soft", "hard"):
            if schemes_version == "2.0" and supervision == "cq":
                continue
            primary = combiner == "stacking" and supervision == "cq"
            name = (
                ("C1" if traversal == "soft" else "C1.hard")
                if primary
                else (f"C1.{combiner}.{supervision}.{traversal}")
            )
            if schemes_version == "2.0":
                name += "@2.0"
            out.append(C1Config(name, combiner, supervision, traversal))
    if schemes_version == "1.1":
        if excluded_label_schemes:
            out.append(
                C1Config("C1.annotation_exclusion", exclude_schemes=tuple(excluded_label_schemes))
            )
        if include_unstable:
            out.append(C1Config("C1.stable_cqs", unstable=True))
    return tuple(out)


@dataclass(frozen=True)
class Majority:
    def predict(self, data):
        return majority(data)


def training_labels(data, labels, ids, config, repeats=None):
    """Estimate an instability exclusion using training labels only, including inner fits."""
    if labels is None:
        raise SchemeError("CQ-supervised C1 needs binary CQ labels")
    unknown = set(config.exclude_schemes) - set(data.schemes)
    if unknown:
        raise SchemeError(f"unknown schemes in CQ label exclusions: {sorted(unknown)}")
    own = labels[labels.item_id.isin(set(ids))].copy()
    excluded = {q for q in data.questions if q[0] in config.exclude_schemes}
    if config.unstable:
        if repeats is None:
            raise SchemeError("the unstable-CQ exploratory condition needs paired re-annotations")
        keys = ["item_id", "scheme", "cq_id"]
        paired = own.merge(
            repeats[keys + ["target"]], on=keys, suffixes=("", "_repeat"), validate="one_to_one"
        )
        for q, group in paired.groupby(["scheme", "cq_id"]):
            if len(group) >= 10 and (group.target == group.target_repeat).mean() < 0.6:
                excluded.add(q)
    keep = [(s, q) not in excluded for s, q in zip(own.scheme, own.cq_id, strict=True)]
    return own.loc[keep], excluded


def _fit(data, labels, ids, config, parameter, selections=None, repeats=None):
    if config.supervision == "cq":
        labels, _ = training_labels(data, labels, ids, config, repeats)
        if config.combiner == "weighted":
            return WeightedPool.fit(data, labels, ids, parameter, selections)
        return Stacking.fit_cq(data, labels, ids, parameter)
    return Stacking.fit_gold(data, ids, parameter)


def fit_c1(data, labels, ids, seed, config=PRIMARY_CONFIG, repeats=None):
    """Fit one training partition; suitable for nested cross-fitting by later components."""
    ids = sorted(set(ids))
    if config.supervision == "none":
        model = Majority() if config.combiner == "majority" else DawidSkene.fit(data)
        return model, {"parameter": None, "signals": None}
    if config.supervision == "gold":
        items = data.items.set_index("item_id")
        ids = [
            i
            for i in ids
            if items.loc[i, "gold_fallacy"] in data.schemes[items.loc[i, "gold_scheme"]].terminals
        ]
    if config.combiner == "weighted":
        signal_grids = []
        for entry in data.entries:
            available = [s.source for s in data.signals if s.entry == entry]
            signal_grids.append(
                [data.sources[entry], *[s for s in available if s != data.sources[entry]]]
            )
        candidates = [(p, s) for p in LAMBDA_GRID for s in product(*signal_grids)]
    else:
        candidates = [(p, None) for p in C_GRID]
    splits = list(KFold(4, shuffle=True, random_state=seed).split(ids)) if len(ids) >= 4 else []
    best, selected = None, candidates[0]
    targets = data.targets(labels) if config.supervision == "cq" else None
    questions = list(zip(data.pairs.scheme, data.pairs.cq_id, strict=True))
    weight_cache = {}
    for parameter, signals in candidates:
        truth, prediction = [], []
        for inner_fold, (train, test) in enumerate(splits):
            training, validation = [ids[i] for i in train], [ids[i] for i in test]
            if config.combiner == "weighted":
                key = parameter, inner_fold
                if key not in weight_cache:
                    weight_cache[key] = _fit(
                        data, labels, training, config, parameter, repeats=repeats
                    )
                model = replace(weight_cache[key], selections=signals)
            else:
                model = _fit(data, labels, training, config, parameter, signals, repeats)
            p = model.predict(data)
            if config.supervision == "cq":
                _, excluded = training_labels(data, labels, training, config, repeats)
                selected_rows = data.pair_indices(validation)
                selected_rows = [
                    i
                    for i in selected_rows
                    if np.isfinite(targets[i]) and questions[i] not in excluded
                ]
                truth.extend(targets[selected_rows])
                prediction.extend(p[selected_rows])
            else:
                verdicts = aggregate_combined(data, p, "soft")
                gold = data.items.set_index("item_id").gold_fallacy
                truth.extend(gold[i] for i in validation)
                prediction.extend(verdicts[i].verdict for i in validation)
        if not truth:
            continue
        if config.supervision == "gold":
            criterion = -macro_f1(truth, prediction)
        else:
            y, p = np.asarray(truth), np.asarray(prediction)
            criterion = (
                float(np.mean((p - y) ** 2))
                if config.combiner == "weighted"
                else float(
                    np.mean(
                        -y * np.log(np.maximum(p, 1e-15))
                        - (1 - y) * np.log(np.maximum(1 - p, 1e-15))
                    )
                )
            )
        # Grids put stronger shrinkage first, then the fixed signal order.
        if best is None or criterion < best - TIE:
            best, selected = criterion, (parameter, signals)
    parameter, signals = selected
    model = _fit(data, labels, ids, config, parameter, signals, repeats)
    choice = {"parameter": float(parameter), "signals": signals, "inner_loss": best}
    if config.supervision == "cq":
        _, excluded = training_labels(data, labels, ids, config, repeats)
        choice["excluded_cqs"] = sorted(excluded)
    return model, choice


def hard_codes(data, p):
    return combined_codes(data, p)


def aggregate_combined(
    data: C1Dataset, p: Sequence[float], traversal: str
) -> dict[str, Aggregation]:
    """A0 or A2 on the combined split and the fixed mean mass outside the arcs."""
    p = np.asarray(p, dtype=float)
    if len(p) != len(data.pairs) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise SchemeError("invalid combined CQ probabilities")
    codes, output = hard_codes(data, p), {}
    for item, group in data.pairs.groupby("item_id", sort=False):
        scheme = data.schemes[group.scheme.iloc[0]]
        masses, answers = {}, {}
        for i, pair in group.iterrows():
            a, b = scheme.answer_space(pair.cq_id)
            masses[pair.cq_id] = {
                a: (1 - data.outside[i]) * p[i],
                b: (1 - data.outside[i]) * (1 - p[i]),
                OUTSIDE: data.outside[i],
            }
            answers[pair.cq_id] = {
                "answer": (a if codes[i] == 0 else b) if codes[i] != 2 else CANNOT_BE_DETERMINED
            }
        if traversal == "hard":
            output[item] = a0(scheme, answers)
        elif traversal == "soft":
            terminals, abstention, arcs = traverse_soft(
                scheme, masses, data.routing[scheme.scheme_id]
            )
            by_cq = {row.cq_id: i for i, row in group.iterrows()}

            def note(cq, by_cq=by_cq):
                i = by_cq[cq]
                given = ", ".join(
                    f"{entry} {data.hard[i, j] or 'missing'}"
                    for j, entry in enumerate(data.entries)
                )
                return f" ({given}; p_first {p[i]:.4f})"

            explanation = _explain(scheme, terminals, abstention, arcs, note)
            output[item] = decide(terminals, abstention, explanation)
        else:
            raise SchemeError(f"unknown C1 traversal {traversal!r}")
    return output


@dataclass(frozen=True)
class C1Result:
    config: C1Config
    aggregations: Mapping[str, Aggregation]
    combined: pd.DataFrame
    choices: list[dict]
    models: Mapping[int | None, object]
    folds: Mapping[str, int]


def _result(data, p, config, choices, models, folds):
    combined = data.pairs.copy()
    combined["p_first"], combined["outside"] = p, data.outside
    codes = hard_codes(data, p)
    combined["answer"] = [
        data.schemes[row.scheme].answer_space(row.cq_id)[code] if code < 2 else CANNOT_BE_DETERMINED
        for row, code in zip(data.pairs.itertuples(index=False), codes, strict=True)
    ]
    combined["system"], combined["combiner"], combined["supervision"] = (
        config.name,
        config.combiner,
        config.supervision,
    )
    combined["fold"] = combined.item_id.map(folds) if config.supervision != "none" else None
    aggregations = aggregate_combined(data, p, config.traversal)
    if config.exclude_schemes:
        combined = combined[~combined.scheme.isin(config.exclude_schemes)].reset_index(drop=True)
        keep = set(combined.item_id)
        aggregations = {i: v for i, v in aggregations.items() if i in keep}
    return C1Result(config, aggregations, combined, choices, models, dict(folds))


def c1_out_of_fold(
    data: C1Dataset,
    labels: pd.DataFrame | None,
    folds: Mapping[str, int] | None,
    seed: int,
    config: C1Config = PRIMARY_CONFIG,
    repeats=None,
) -> C1Result:
    """Fit labelled parameters only on other items; unsupervised DS fits all items once."""
    folds = dict(folds) if folds is not None else assign_folds(data.items, seed)
    if missing := set(data.items.item_id) - set(folds):
        raise SchemeError(f"C1 items have no fold: {sorted(missing)}")
    if config.supervision == "none":
        model, _ = fit_c1(data, None, data.items.item_id, seed, config)
        return _result(data, model.predict(data), config, [], {None: model}, folds)
    p, models, choices = np.full(len(data.pairs), np.nan), {}, []
    for fold in sorted(set(folds.values())):
        train = [i for i in data.items.item_id if folds[i] != fold]
        test = [i for i in data.items.item_id if folds[i] == fold]
        if not test:
            continue
        model, choice = fit_c1(data, labels, train, seed, config, repeats)
        models[fold] = model
        choices.append({"fold": fold, "component": config.name, **choice})
        selected = data.pair_indices(test)
        p[selected] = model.predict(data)[selected]
    return _result(data, p, config, choices, models, folds)


def transfer_c1(source: C1Result, target: C1Dataset, name=None) -> C1Result:
    """Apply the original CQ-supervised coefficients of each item's held-out fold."""
    if source.config.combiner != "stacking" or source.config.supervision != "cq":
        raise SchemeError("cross-version C1 transfer requires CQ-supervised stacking")
    if name is None:
        name = source.config.name + "@2.0"
    p = np.full(len(target.pairs), np.nan)
    for fold, model in source.models.items():
        ids = [i for i in target.items.item_id if source.folds.get(i) == fold]
        selected = target.pair_indices(ids)
        p[selected] = model.predict(target)[selected]
    return _result(
        target, p, replace(source.config, name=name), source.choices, source.models, source.folds
    )


def baseline_aggregations(data: C1Dataset) -> dict[str, dict[str, Aggregation]]:
    """A2 of the reference entry and A4, using the unchanged round-one aggregators."""
    output = {f"A2:{data.reference}": {}, "A4": {}}
    for item, group in data.pairs.groupby("item_id", sort=False):
        scheme = data.schemes[group.scheme.iloc[0]]
        rows = {entry: {} for entry in data.entries}
        for i, pair in group.iterrows():
            arcs = scheme.answer_space(pair.cq_id)
            for e, entry in enumerate(data.entries):
                if data.observations[i, e] < 0:
                    continue
                r, p = data.arc_mass[i, e], data.base_splits[i, e]
                rows[entry][pair.cq_id] = {
                    "answer": data.hard[i, e],
                    f"p_logprob_{arcs[0]}": r * p,
                    f"p_logprob_{arcs[1]}": r * (1 - p),
                    f"p_logprob_{CANNOT_BE_DETERMINED}": 1 - r,
                    "p_logprob_na": 0.0,
                }
        routing = data.routing[scheme.scheme_id]
        output[f"A2:{data.reference}"][item] = a2(scheme, rows[data.reference], routing, LOGPROB)
        output["A4"][item] = a4(scheme, rows, dict.fromkeys(data.entries, LOGPROB), routing)
    return output
