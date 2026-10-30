"""Protocol-specific isolation, exact exported decisions and conditional scoring."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from tests.test_c1 import example_data

from argfallacy.aggregate.c1_inputs import build_dataset
from argfallacy.aggregate.crossfit import C1CrossFitter
from argfallacy.aggregate.tree import CQTree, c3_out_of_fold
from argfallacy.eval.c3_diagnostic import (
    BASELINE,
    C3,
    _peak_memory,
    export_tree,
    replay_tree,
    score_tables,
)
from argfallacy.schemes import load_all


def test_full_tree_export_replays_float32_boundaries_and_weighted_leaves():
    data, _, _, _ = example_data(10)
    scheme = next(iter(data.schemes.values()))
    labels = sorted(scheme.terminals)[:2]
    x = np.zeros((10, len(scheme.cq_order) * 2))
    x[:, 0] = np.arange(10) / 10
    tree = CQTree.fit(scheme, x, [labels[0]] * 7 + [labels[1]] * 3, 3, 1, 23)
    exported = export_tree(tree, scheme)
    threshold = tree.model.tree_.threshold[0]
    rows = np.vstack([x, x[0], x[0]])
    rows[-2:, 0] = [threshold, np.nextafter(threshold, np.inf)]
    for row in rows:
        verdict, distribution, leaf, _ = replay_tree(exported, row)
        actual = tree.predict(row)
        assert verdict == actual.verdict
        assert distribution == pytest.approx(actual.distribution)
        assert leaf == tree.model.apply([row])[0]
    assert sum(r["raw_support"] for r in exported["rules"]) == len(x)
    assert len(exported["nodes"]) == tree.model.tree_.node_count


def test_canonical_scoring_keeps_abstentions_and_good_false_positives():
    items = pd.DataFrame({
        "item_id": ["a", "b", "c", "d"],
        "gold_fallacy": ["good_argumentation", "weak_analogy", "weak_analogy", "non_sequitur"],
        "gold_scheme": ["analogy"] * 4, "source": ["nlas", "logic", "logic", "reddit"],
        "origin": ["scheme", "fallacy", "fallacy", "fallacy"],
        "reachable_gold": [True, True, True, False],
    })
    predictions = pd.DataFrame([
        {"system": s, "item_id": i, "verdict": p}
        for s, labels in ((C3, ["weak_analogy", "weak_analogy", "weak_analogy", "weak_analogy"]),
                          (BASELINE, ["good_argumentation", "weak_analogy", None, None]))
        for i, p in zip(items.item_id, labels, strict=True)
    ])
    metrics, differences, _ = score_tables(items, predictions, 23, 50)
    c3 = metrics[(metrics.system == C3) & (metrics.subset == "all")].set_index("metric")
    assert c3.loc["accuracy", "estimate"] == 0.5
    # Weak analogy: TP=2, FP=2, FN=0. Non sequitur: F1=0.
    assert c3.loc["macro_f1_without_good_class", "estimate"] == pytest.approx(1 / 3)
    assert set(metrics[metrics.subset == "common_coverage"].n_items) == {2}
    assert set(metrics[metrics.subset == "all"].n_items) == {4}
    assert set(metrics[metrics.subset == "unreachable_gold"].n_items) == {1}
    assert len(differences[differences.role == "primary"]) == 1


def test_cross_version_c3_capture_preserves_predictions_and_heldout_isolation():
    from argfallacy.aggregate import load_burden, routes
    from argfallacy.paths import REPO_ROOT

    data, labels, summary, manifest = example_data(6)
    schemes = load_all(REPO_ROOT / "schemes/v2")
    scheme = schemes[next(iter(data.schemes))]
    rows = pd.concat([summary.assign(cq_id=cq) for cq in scheme.cq_order])
    target = build_dataset(
        {scheme.scheme_id: scheme}, data.items, rows, manifest, "e0",
        {scheme.scheme_id: routes(scheme, load_burden("v2", schemes)[scheme.scheme_id])})
    folds = {i: n % 2 for n, i in enumerate(data.items.item_id)}
    events = []
    original = c3_out_of_fold(C1CrossFitter(target, labels, 23, fit_data=data), folds)
    captured = c3_out_of_fold(C1CrossFitter(target, labels, 23, fit_data=data), folds,
                              audit=events.append)
    assert captured.aggregations == original.aggregations
    own = data.items.item_id.iloc[0]
    changed = labels.copy()
    changed.loc[changed.item_id == own, "target"] = 1 - changed.loc[
        changed.item_id == own, "target"]
    item_table = data.items.copy()
    item_table.loc[item_table.item_id == own, "gold_fallacy"] = sorted(scheme.terminals)[-1]
    mutated = c3_out_of_fold(C1CrossFitter(
        replace(target, items=item_table), changed, 23, fit_data=replace(data, items=item_table)),
        folds)
    assert mutated.aggregations[own] == original.aggregations[own]
    assert any(e["phase"] == "outer" for e in events)
    for e in events:
        if e["phase"] == "inner":
            assert not set(e["context_items"]) & set(e["validation_items"])


def test_exported_empty_training_model_abstains():
    data, _, _, _ = example_data(3)
    scheme = next(iter(data.schemes.values()))
    model = CQTree.fit(scheme, [], [], 1, 5, 23)
    assert replay_tree(export_tree(model, scheme), []) == (None, {}, None, [])


def test_peak_memory_is_measurable_without_hiding_pipeline_errors():
    assert _peak_memory() > 0
