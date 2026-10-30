"""Nested CQ inputs, routing rules, small trees and pooled terminal calibration."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from argfallacy.aggregate.b1 import b1_out_of_fold
from argfallacy.aggregate.interface import decide
from argfallacy.paths import REPO_ROOT
from argfallacy.schemes import load_all
from test_b1 import example_data


def two_schemes():
    from argfallacy.aggregate.b1_inputs import build_dataset

    data, _, summary, manifest = example_data(10)
    second = load_all(REPO_ROOT / "schemes/v1")["expert_opinion"]
    cq = next(q for q, edges in second.nodes.items() if all(e.is_terminal for e in edges.values()))
    second = replace(
        second, cqs={cq: second.cqs[cq]}, cq_order=[cq], start=cq, nodes={cq: second.nodes[cq]}
    )
    items = data.items.copy()
    items.item_id = "j" + items.item_id
    items.gold_scheme = second.scheme_id
    items.gold_fallacy = [second.nodes[cq]["yes" if j % 2 else "no"].target for j in range(10)]
    rows = summary.copy()
    rows.item_id, rows.scheme, rows.cq_id = "j" + rows.item_id, second.scheme_id, cq
    schemes = {**data.schemes, second.scheme_id: second}
    return build_dataset(
        schemes,
        pd.concat([data.items, items]),
        pd.concat([summary, rows]),
        manifest,
        "e0",
        {**data.routing, second.scheme_id: {cq: "no"}},
    )


def zero_side(data):
    from argfallacy.aggregate.routing import Side

    scheme = next(iter(data.schemes.values()))
    terminals = list(scheme.terminals)
    predictions = {
        item: decide({terminals[j % 2]: 1.0}, 0.0, "zero-shot")
        for j, item in enumerate(data.items.item_id)
    }
    return Side(
        "zeroshot:e0",
        predictions,
        {i: (0.7 if j % 3 else np.nan) for j, i in enumerate(predictions)},
    )


def test_routing_agreement_missing_verdicts_extremes_and_monotone_share():
    from argfallacy.aggregate.routing import route

    data, _, _, _ = example_data(2)
    a, b = list(next(iter(data.schemes.values())).terminals)[:2]
    yes, no, absent = decide({a: 1}, 0, "diagram"), decide({b: 1}, 0, "zero"), decide({}, 1, "none")
    assert route(yes, yes, 0.1, 0.9, -2).aggregation.verdict == a
    assert route(yes, yes, 0.1, 0.9, -2).explained
    assert route(absent, no, 1, 0, 2).aggregation.verdict == b
    assert route(yes, absent, 0, 1, -2).explained
    assert route(absent, absent, 0.5, 0.5, 0).aggregation.verdict is None
    assert route(yes, no, 0.2, 0.8, -2).side == "zero-shot"
    assert route(yes, no, 0.2, 0.8, 2).side == "diagram"
    assert route(yes, no, 0.2, 0.8, 0.6).side == "diagram"
    shares = [
        np.mean([route(yes, no, p, 1 - p, tau).explained for p in np.linspace(0, 1, 11)])
        for tau in np.linspace(-2, 2, 21)
    ]
    assert np.diff(shares).min() >= 0


def test_cross_fitted_inputs_exclude_their_labels_and_match_outer_b1():
    from argfallacy.aggregate.crossfit import B1CrossFitter

    data, labels, _, _ = example_data(10)
    folds = {i: j % 5 for j, i in enumerate(data.items.item_id)}
    source = B1CrossFitter(data, labels, seed=23)
    outer = b1_out_of_fold(data, labels, folds, 23)
    train = [i for i in folds if folds[i] != 0]
    held = [i for i in folds if folds[i] == 0]
    training, testing = source.partition(train, held)
    for item in held:
        assert testing.aggregations[item] == outer.aggregations[item]
    mutated = labels.copy()
    mutated.loc[mutated.item_id == train[0], "target"] = (
        1 - mutated.loc[mutated.item_id == train[0], "target"]
    )
    other_training = B1CrossFitter(data, mutated, 23).partition(train, held)[0]
    row = data.pair_indices([train[0]])
    np.testing.assert_equal(training.p[row], other_training.p[row])
    with pytest.raises(ValueError, match="overlap"):
        source.partition(train, train[:1])


def test_changed_own_gold_or_cq_label_changes_no_b1_b2_or_b3_verdict():
    from argfallacy.aggregate.crossfit import B1CrossFitter
    from argfallacy.aggregate.routing import b2_out_of_fold
    from argfallacy.aggregate.tree import b3_out_of_fold

    data, labels, _, _ = example_data(10)
    folds = {i: j % 5 for j, i in enumerate(data.items.item_id)}
    original_items = data.items.copy()
    mutated_items = data.items.copy()
    terminals = list(next(iter(data.schemes.values())).terminals)
    own = "i000"
    mutated_items.loc[mutated_items.item_id == own, "gold_fallacy"] = next(
        t for t in terminals if t != original_items.set_index("item_id").loc[own, "gold_fallacy"]
    )
    mutated_labels = labels.copy()
    mutated_labels.loc[mutated_labels.item_id == own, "target"] = (
        1 - mutated_labels.loc[mutated_labels.item_id == own, "target"]
    )
    verdicts = []
    fixed_zero = zero_side(data)
    for items, ratings in [
        (original_items, labels),
        (mutated_items, labels),
        (original_items, mutated_labels),
        (mutated_items, mutated_labels),
    ]:
        dataset = replace(data, items=items)
        source = B1CrossFitter(dataset, ratings, 23)
        b1 = b1_out_of_fold(dataset, ratings, folds, 23)
        b2 = b2_out_of_fold(source, fixed_zero, folds)
        b3 = b3_out_of_fold(source, folds)
        calibrated = b3_out_of_fold(source, folds, kind="pooled")
        verdicts.append([r.aggregations[own] for r in (b1, b2, b3, calibrated)])
        assert b2.upstream[own] == b1.aggregations[own] == b3.upstream[own]
    assert all(v == verdicts[0] for v in verdicts[1:])


def test_tree_separates_one_cq_has_bounded_depth_and_reports_raw_leaf_share():
    from argfallacy.aggregate.tree import CQTree

    data, _, _, _ = example_data(20)
    scheme = next(iter(data.schemes.values()))
    labels = data.items.gold_fallacy.to_numpy()
    x = np.column_stack([np.arange(20) % 2, np.zeros(20)])
    model = CQTree.fit(scheme, x, labels, depth=3, leaf=1, seed=23)
    assert model.model.get_depth() == 1
    assert model.model.tree_.feature[0] == 0
    predictions = [model.predict(row) for row in x]
    assert [p.verdict for p in predictions] == list(labels)
    assert all(p.confidence == 1 and "CQ4" in p.explanation for p in predictions)
    # Class weighting chooses the minority of a leaf; confidence is its raw share.
    a, b = list(scheme.terminals)[:2]
    raw = [a] * 8 + [b] * 2 + [a] + [b]
    x = np.array([[0.0, 0.0]] * 10 + [[1.0, 0.0]] * 2)
    model = CQTree.fit(scheme, x, raw, depth=1, leaf=1, seed=23)
    answer = model.predict(x[-1])
    assert answer.verdict == b and answer.confidence == 0.5


def test_tree_models_are_per_scheme_and_reference_inputs_are_independent_of_labels():
    from argfallacy.aggregate.crossfit import ReferenceInputs
    from argfallacy.aggregate.tree import b3_out_of_fold

    data = two_schemes()
    source = ReferenceInputs(data, seed=23)
    features = source.fit(data.items.item_id).tree_features(list(data.items.item_id)[:10])
    assert set(features[:, 0]) == {0.0, 1.0}
    result = b3_out_of_fold(source, {i: j % 5 for j, i in enumerate(data.items.item_id)})
    assert all(set(models) == set(data.schemes) for models in result.models.values())
    assert all(
        model.model.get_depth() <= 3
        for models in result.models.values()
        for model in models.values()
    )


def test_router_imputation_and_single_class_fallback_use_training_items_only():
    from argfallacy.aggregate.routing import Router

    frame = pd.DataFrame(
        {
            "confidence": [0.2, 0.3, 0.8],
            "margin": [0.1, 0.2, 0.7],
            "abstention": [0.0, 0.0, 0.0],
            "entropy": [0.3, 0.4, 0.5],
            "p_label": [0.2, np.nan, 0.8],
            "scheme": ["s"] * 3,
        }
    )
    model = Router.fit(frame.iloc[:2], np.array([[1, 0], [1, 0]]), c=(0.01, 0.01), seed=23)
    assert model.mean_p_label == 0.2
    np.testing.assert_equal(model.predict(frame), np.tile([1.0, 0.0], (3, 1)))


def test_pooled_recalibration_has_shared_parameters_and_exact_gradient():
    from argfallacy.aggregate.crossfit import ReferenceInputs
    from argfallacy.aggregate.tree import PooledCalibration

    data = two_schemes()
    inputs = ReferenceInputs(data, 23).fit(data.items.item_id)
    model = PooledCalibration.fit(inputs, data.items.item_id, c=1)
    centre = np.zeros_like(model.coef)
    centre[0] = 1
    objective = PooledCalibration.loss(inputs, data.items.item_id)
    value, gradient = objective(centre)
    direction = np.random.default_rng(2).normal(size=len(centre))
    numerical = (objective(centre + 1e-6 * direction)[0] - value) / 1e-6
    assert gradient @ direction == pytest.approx(numerical, abs=2e-5)
    assert model.objective_value <= value + 1e-8
    assert len(model.coef) == 1 + len(set().union(*(s.terminals for s in data.schemes.values())))
    assert all(
        sum(p.distribution.values()) == pytest.approx(1)
        for p in model.predict(inputs, data.items.item_id).values()
    )


def test_routing_curve_reuses_item_resamples_and_marks_unattainable_shares():
    from argfallacy.aggregate.crossfit import ReferenceInputs
    from argfallacy.aggregate.routing import Side, b2_out_of_fold
    from argfallacy.eval.routing import routing_curve

    data, _, _, _ = example_data(10)
    result = b2_out_of_fold(
        ReferenceInputs(data, 23),
        zero_side(data),
        {i: j % 5 for j, i in enumerate(data.items.item_id)},
    )
    indices = np.tile(np.arange(10), (3, 1))
    curve = routing_curve(result, data.items, indices)
    targets = curve[curve.requested_share.notna()]
    assert set(targets.requested_share) == {0.6, 0.7, 0.8, 0.9, 1.0}
    assert (targets.accuracy == targets.accuracy_low).all()
    assert (targets.macro_f1 == targets.macro_f1_high).all()
    predictions = {i: decide({}, 1.0, "missing") for i in data.items.item_id}
    result = b2_out_of_fold(
        ReferenceInputs(data, 23),
        Side("missing", predictions),
        {i: j % 5 for j, i in enumerate(data.items.item_id)},
    )
    # The diagram is always available, so the zero-shot absence forces it at all tau.
    assert result.records.explained.all()
    hard = data.hard.copy()
    hard[:2, 0] = "cannot_be_determined"
    unavailable = replace(data, hard=hard)
    result = b2_out_of_fold(
        ReferenceInputs(unavailable, 23, hard=True),
        zero_side(data),
        {i: j % 5 for j, i in enumerate(data.items.item_id)},
    )
    curve = routing_curve(result, data.items, indices)
    unreachable = curve[curve.requested_share == 1.0].iloc[0]
    assert not unreachable.attainable and pd.isna(unreachable.accuracy)
    assert unreachable.explained_share == pytest.approx(0.8)


def test_additional_sides_and_summary_reader_are_supported():
    from argfallacy.aggregate.crossfit import ReferenceInputs
    from argfallacy.aggregate.routing import Side, b2_out_of_fold

    data, _, _, _ = example_data(10)
    side = zero_side(data)
    summary = pd.DataFrame(
        [
            {
                "item_id": i,
                "model": "e0",
                "stage": "zeroshot",
                "scheme_condition": "gold",
                "answer": a.verdict,
                "p_label": 0.7,
                "justification": "answer",
            }
            for i, a in side.aggregations.items()
        ]
    )
    parsed = Side.from_summary(summary, "zero", "e0")
    assert parsed.p_label["i000"] == 0.7
    folds = {i: j % 5 for j, i in enumerate(data.items.item_id)}
    extra = Side("ssoft", side.aggregations)
    result = b2_out_of_fold(
        ReferenceInputs(data, 23, hard=True), parsed, folds, extra_sides=(extra,)
    )
    assert all(len(model.models) == 3 for model in result.models.values())
    assert set(result.records.side) <= {"diagram", "zero-shot", "ssoft"}
    with pytest.raises(ValueError, match="duplicate"):
        Side.from_summary(pd.concat([summary, summary]), "zero", "e0")
    with pytest.raises(ValueError, match="p_label"):
        Side("bad", side.aggregations, {"i000": np.inf})


def test_extra_side_with_the_highest_estimate_is_selected():
    from argfallacy.aggregate.crossfit import ReferenceInputs
    from argfallacy.aggregate.routing import Side, b2_out_of_fold

    data, _, _, _ = example_data(10)
    scheme = next(iter(data.schemes.values()))
    a, b = scheme.answer_space(scheme.start)
    hard, probabilities = data.hard.copy(), data.base_splits.copy()
    hard[:, 0], probabilities[:, 0] = b, 0.2
    data = replace(data, hard=hard, base_splits=probabilities)
    fixed = Side(
        "zero",
        {
            i: decide({scheme.nodes[scheme.start][a].target: 1.0}, 0.0, "fixed")
            for i in data.items.item_id
        },
    )
    extra = Side(
        "ssoft",
        {
            row.item_id: decide({row.gold_fallacy: 1.0}, 0.0, "third side")
            for row in data.items.itertuples(index=False)
        },
    )
    result = b2_out_of_fold(
        ReferenceInputs(data, 23, hard=True),
        fixed,
        {i: j % 5 for j, i in enumerate(data.items.item_id)},
        extra_sides=(extra,),
    )
    assert set(result.records.side) == {"ssoft"}


def test_agreement_items_do_not_fit_the_correctness_models():
    from argfallacy.aggregate.crossfit import ReferenceInputs
    from argfallacy.aggregate.routing import Side, b2_out_of_fold

    data, _, _, _ = example_data(10)
    source = ReferenceInputs(data, 23)
    terminals = list(next(iter(data.schemes.values())).terminals)
    predictions = {}
    for j, (item, prediction) in enumerate(source.inputs.aggregations.items()):
        label = (
            prediction.verdict if j < 5 else next(t for t in terminals if t != prediction.verdict)
        )
        predictions[item] = decide({label: 1.0}, 0.0, "fixed")
    fixed = Side("zero", predictions)
    folds = {i: j % 5 for j, i in enumerate(data.items.item_id)}
    before = b2_out_of_fold(source, fixed, folds)
    items = data.items.copy()
    items.loc[items.item_id == "i000", "gold_fallacy"] = next(
        t for t in terminals if t != predictions["i000"].verdict
    )
    after = b2_out_of_fold(ReferenceInputs(replace(data, items=items), 23), fixed, folds)
    pd.testing.assert_frame_equal(before.records, after.records)


def test_cross_version_inputs_fit_original_labels_and_predict_target_signals():
    from argfallacy.aggregate import load_burden, routes
    from argfallacy.aggregate.b1_inputs import build_dataset
    from argfallacy.aggregate.crossfit import B1CrossFitter

    data, labels, summary, manifest = example_data(10)
    schemes = load_all(REPO_ROOT / "schemes/v2")
    scheme = schemes[next(iter(data.schemes))]
    rows = pd.concat([summary.assign(cq_id=cq) for cq in scheme.cq_order])
    target = build_dataset(
        {scheme.scheme_id: scheme},
        data.items,
        rows,
        manifest,
        "e0",
        {scheme.scheme_id: routes(scheme, load_burden("v2", schemes)[scheme.scheme_id])},
    )
    source = B1CrossFitter(target, labels, 23, fit_data=data)
    ids = sorted(data.items.item_id)[:8]
    fitted = source.fit(ids)
    np.testing.assert_equal(fitted.p, source.models[tuple(ids)].predict(target))
