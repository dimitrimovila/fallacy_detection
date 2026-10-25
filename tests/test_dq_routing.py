"""Exploratory routing: held-out labels, added signals and explained verdicts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from argfallacy.aggregate import load_burden, routes
from argfallacy.aggregate.c1_inputs import build_dataset
from argfallacy.aggregate.crossfit import C1CrossFitter, ReferenceInputs
from argfallacy.aggregate.interface import decide
from argfallacy.aggregate.routing import Router, Side, c2_out_of_fold, routing_features
from argfallacy.paths import REPO_ROOT
from argfallacy.schemes import SchemeError, load_all
from test_c1 import example_data
from test_c2_c3 import zero_side


def transferred_inputs(data, summary, manifest):
    schemes = load_all(REPO_ROOT / "schemes/v2")
    scheme = schemes[next(iter(data.schemes))]
    rows = pd.concat([summary.assign(cq_id=cq) for cq in scheme.cq_order])
    return build_dataset(
        {scheme.scheme_id: scheme},
        data.items,
        rows,
        manifest,
        "e0",
        {scheme.scheme_id: routes(scheme, load_burden("v2", schemes)[scheme.scheme_id])},
    )


@pytest.mark.parametrize("version", ["1.1", "2.0"])
def test_multiside_own_gold_and_cq_labels_never_change_own_prediction(version):
    data, labels, summary, manifest = example_data(10)
    target = data if version == "1.1" else transferred_inputs(data, summary, manifest)
    ids = list(data.items.item_id)
    folds = {i: j % 5 for j, i in enumerate(ids)}
    zero = zero_side(target)
    dq = Side(
        "DQ",
        zero.aggregations,
        {i: 0.8 for i in ids},
        consistent={i: j % 2 == 0 for j, i in enumerate(ids)},
    )
    extras = (dq,)
    if version == "2.0":
        extras += (
            Side(
                "C4",
                zero.aggregations,
                {i: 0.7 for i in ids},
                explained=True,
                schemes_version="2.0",
            ),
        )
    own = ids[0]
    before = c2_out_of_fold(
        C1CrossFitter(target, labels, 23, fit_data=data),
        zero,
        folds,
        extra_sides=extras,
        schemes_version=version,
    )
    changed_items, changed_labels = data.items.copy(), labels.copy()
    other = next(
        t
        for t in next(iter(target.schemes.values())).terminals
        if t != changed_items.set_index("item_id").loc[own, "gold_fallacy"]
    )
    changed_items.loc[changed_items.item_id == own, "gold_fallacy"] = other
    changed_labels.loc[changed_labels.item_id == own, "target"] = (
        1 - changed_labels.loc[changed_labels.item_id == own, "target"]
    )
    for ratings in (labels, changed_labels):
        after = c2_out_of_fold(
            C1CrossFitter(
                replace(target, items=changed_items),
                ratings,
                23,
                fit_data=replace(data, items=changed_items),
            ),
            zero,
            folds,
            extra_sides=extras,
            schemes_version=version,
        )
        assert before.aggregations[own] == after.aggregations[own]
        pd.testing.assert_frame_equal(
            before.records[before.records.item_id == own],
            after.records[after.records.item_id == own],
        )


@pytest.mark.parametrize(
    "matches,consistent,expected",
    [
        ("all", False, True),
        ("C1", False, True),
        ("DQ", True, True),
        ("DQ", False, False),
        ("C4", False, True),
        ("none", True, False),
    ],
)
def test_explained_depends_on_equal_verdicts_not_the_selected_side(
    monkeypatch,
    matches,
    consistent,
    expected,
):
    data, _, _, _ = example_data(10)
    source = ReferenceInputs(data, 23)
    ids = list(data.items.item_id)
    a, b = list(next(iter(data.schemes.values())).terminals)[:2]

    def predictions(label):
        return {i: decide({label: 1.0}, 0, "fixed") for i in ids}

    source.inputs = replace(
        source.inputs, aggregations=predictions(a if matches in {"all", "C1"} else b)
    )
    zero = Side("zero", predictions(a))
    dq = Side(
        "DQ",
        predictions(a if matches in {"all", "DQ"} else b),
        consistent={i: consistent for i in ids},
    )
    c4 = Side("C4", predictions(a if matches in {"all", "C4"} else b), explained=True)
    # On disagreements the zero-shot wins, even when another side explains its label.
    monkeypatch.setattr(
        Router, "predict", lambda self, frame: np.tile([0.1, 0.9, 0.2, 0.3], (len(frame), 1))
    )
    result = c2_out_of_fold(
        source, zero, {i: j % 5 for j, i in enumerate(ids)}, extra_sides=(dq, c4)
    )
    assert {p.verdict for p in result.aggregations.values()} == {a}
    assert set(result.records.explained) == {expected}
    if matches != "all":
        assert set(result.records.side) == {"zero-shot"}


def test_added_confidence_and_missing_indicators_use_training_means():
    data, _, _, _ = example_data(10)
    source = ReferenceInputs(data, 23)
    ids = list(data.items.item_id)
    zero = zero_side(data)
    dq = Side(
        "DQ",
        zero.aggregations,
        {ids[0]: 0.2, ids[1]: np.nan, ids[2]: 0.9},
        consistent={i: i == ids[0] for i in ids},
    )
    c4 = Side("C4", zero.aggregations, {ids[0]: np.nan, ids[1]: 0.4, ids[2]: 0.95}, explained=True)
    frame = routing_features(source.inputs, zero, ids, extra_sides=(dq, c4))
    assert list(frame["DQ.consistent"][:3]) == [True, False, False]
    model = Router.fit(frame.iloc[:2], np.array([[1, 0, 1, 0], [1, 0, 1, 0]]), 0.01, 23)
    absent = frame.iloc[[2]].copy()
    absent[["DQ.confidence", "C4.confidence"]] = np.nan
    imputed = absent.copy()
    imputed[["DQ.confidence", "C4.confidence"]] = [0.2, 0.4]
    delta = model.transform(absent) - model.transform(imputed)
    assert sorted(delta[0]) == [0] * (delta.shape[1] - 2) + [1, 1]


def test_ssoft_version_cannot_enter_a_family_of_schemes_1_1():
    data, _, _, _ = example_data(10)
    zero = zero_side(data)
    c4 = Side("C4", zero.aggregations, explained=True, schemes_version="2.0")
    with pytest.raises(SchemeError, match="2.0.*1.1"):
        c2_out_of_fold(ReferenceInputs(data, 23), zero, extra_sides=(c4,), schemes_version="1.1")
