"""Fixed-input matching, exact training costs, isolation and visible failures."""

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier
from tests.test_b1 import example_data

from argfallacy.aggregate.crossfit import B1CrossFitter
from argfallacy.aggregate.sparse_tree import (
    analytic_tree,
    constant_mask,
    encode,
    fit_cart,
    objective,
    replay,
    sample_weights,
)
from argfallacy.eval import gosdt_diagnostic as diagnostic
from argfallacy.eval.gosdt_diagnostic import CART, GOSDT, configuration, group_fit, select_candidate
from argfallacy.schemes import SchemeError


def test_boundary_encoding_and_training_only_constant_mask():
    values = np.array([[0.25, 0.10], [0.50, 0.50], [0.75, 1], [1, 0]])
    x = encode(values)
    assert x.tolist() == [
        [False, False, False, False, False],
        [True, False, False, True, False],
        [True, True, False, True, True],
        [True, True, True, False, False],
    ]
    mask = constant_mask(x[:1])
    assert not mask.any()
    assert x[1:, mask].shape == (3, 0)
    with pytest.raises(ValueError, match="finite"):
        encode(np.array([[np.nan, 0]]))


def test_native_cost_orientation_and_exact_root_are_not_failure_fallbacks():
    y = np.array(["a"] * 8 + ["b"] * 4 + ["c"] * 2)
    cost = objective(y, 2)
    c = np.array(cost["matrix"])
    w = sample_weights(y, cost)
    loss = sum(c[0, cost["classes"].index(t)] for t in y)
    assert loss == pytest.approx(w[y != "a"].sum() / len(y), abs=1e-12)
    assert not np.isclose(loss, sum(c[cost["classes"].index(t), 0] for t in y))
    tiny = objective(np.array(["a", "b"]), 2)
    tree, status = analytic_tree(np.array([[0], [1]], bool), np.array(["a", "b"]), tiny)
    assert status == "analytic_root" and tree["verdict"] == "a"
    single = fit_cart(np.ones((3, 0), bool), ["z"] * 3, 2, 5, 23)
    assert single["status"] == "single_class" and single["leaves"] == 1
    assert single["objective"] == single["cost"]["lambda"]
    empty = fit_cart(np.empty((0, 2), bool), [], 2, 2, 23)
    assert not empty["valid"] and empty["tree"]["verdict"] is None


def test_cart_pruning_matches_exhaustive_subtrees_and_replay():
    rng = np.random.default_rng(23)
    x = rng.integers(0, 2, (64, 5), dtype=np.int8)
    y = np.array(["a", "b", "c", "a"] * 16)
    for coefficient in (2, 5):
        evaluation_x = rng.integers(0, 2, (19, 5), dtype=np.int8)
        result = fit_cart(x, y, 3, coefficient, 23, evaluation_x=evaluation_x)
        assert result["evaluation_replay_verified"]
        assert result["evaluation_predictions"] == replay(result["tree"], evaluation_x)[0]
        cost = result["cost"]
        c = np.array(cost["matrix"])
        native = (
            DecisionTreeClassifier(max_depth=3, random_state=23)
            .fit(x, y, sample_weight=sample_weights(y, cost))
            .tree_
        )

        def possibilities(node, ids, cost=cost, c=c, native=native):
            counts = [np.sum(y[ids] == cls) for cls in cost["classes"]]
            candidates = [(float(min(c @ counts)) + cost["lambda"], 1)]
            if native.children_left[node] != -1:
                mask = x[ids, native.feature[node]] <= native.threshold[node]
                for a, b in itertools.product(
                    possibilities(native.children_left[node], ids[mask]),
                    possibilities(native.children_right[node], ids[~mask]),
                ):
                    candidates.append((a[0] + b[0], a[1] + b[1]))
            return candidates

        best = min(possibilities(0, np.arange(len(y))))
        assert result["objective"] == pytest.approx(best[0], abs=1e-12)
        assert result["leaves"] == best[1]
        pred, paths = replay(result["tree"], x)
        assert len(pred) == len(paths) == len(y)
        assert result["depth"] <= 3


def test_cache_release_retains_fitted_b1_and_predictions():
    data, labels, _, _ = example_data(8)
    source = B1CrossFitter(data, labels, 23)
    ids = sorted(data.items.item_id)[:6]
    a = source.fit(ids).p.copy()
    count = len(source.choices)
    model = source.models[tuple(ids)]
    source.release_predictions()
    b = source.fit(ids).p
    assert np.array_equal(a, b)
    assert source.models[tuple(ids)] is model and len(source.choices) == count


def test_resume_preserves_saved_upstream_models_and_choices(tmp_path):
    import pickle

    from argfallacy.eval.b3_diagnostic import json_write

    data, labels, _, _ = example_data(8)
    source = B1CrossFitter(data, labels, 23)
    ids = sorted(data.items.item_id)[:6]
    expected = source.fit(ids).p.copy()
    source.fit(ids[:-1])
    with (tmp_path / "upstream_fold0.pkl").open("wb") as file:
        pickle.dump(source.models, file)
    json_write(tmp_path / "upstream_choices_fold0.json", source.choices)
    restored = B1CrossFitter(data, labels, 23)
    diagnostic.restore_upstream(restored, tmp_path, 0)
    assert set(restored.models) == set(source.models)
    assert np.array_equal(restored.fit(ids).p, expected)
    assert len(restored.choices) == len(source.choices)


def test_nested_group_heldout_labels_do_not_change_its_choice_or_prediction(tmp_path):
    data, labels, _, _ = example_data(18)
    scheme = next(iter(data.schemes.values()))
    held = sorted(data.items.item_id)[:3]
    train = sorted(set(data.items.item_id) - set(held))
    calls = []

    def solver(key, x, y, depth, coefficient, evaluation_x):
        calls.append((key, x.copy(), np.asarray(y).copy()))
        result = fit_cart(x, y, depth, coefficient, 23, evaluation_x=evaluation_x)
        result["status"] = "synthetic_solver"
        return result

    source = B1CrossFitter(data, labels, 23)
    original = group_fit(source, train, held, scheme, 0, tmp_path, solver, 23)
    changed_labels = labels.copy()
    changed_labels.loc[changed_labels.item_id.isin(held), "target"] = 1
    changed_items = data.items.copy()
    changed_items.loc[changed_items.item_id.isin(held), "gold_fallacy"] = sorted(scheme.terminals)[
        0
    ]
    mutated = group_fit(
        B1CrossFitter(replace(data, items=changed_items), changed_labels, 23),
        train,
        held,
        scheme,
        0,
        tmp_path,
        solver,
        23,
    )
    assert original["complete"] and mutated["complete"]
    for method in (GOSDT, CART):
        original["results"][method]["model"].pop("elapsed_seconds")
        mutated["results"][method]["model"].pop("elapsed_seconds")
        assert original["results"][method] == mutated["results"][method]
    assert original["results"][GOSDT]["input_sha256"] == original["results"][CART]["input_sha256"]
    for record in original["fits"]:
        assert not set(record["b1_context"]) & (set(record["validation_items"]) | set(held))
    assert len(original["candidates"][GOSDT]) == 6
    assert calls
    from argfallacy.eval.b3_diagnostic import json_write

    json_write(tmp_path / "0_fixture.json", original)
    diagnostic.fit_reports(tmp_path, tmp_path)
    report = pd.read_csv(tmp_path / "fit_statuses.csv")
    assert set(report.configured_depth) == {2, 3, 4}
    assert set(report.coefficient) == {2, 5}
    assert len(pd.read_csv(tmp_path / "selected_configs.csv")) == 2


def test_failed_inner_candidate_is_not_selected_or_scored_on_successes_only(tmp_path):
    data, labels, _, _ = example_data(18)
    ids = sorted(data.items.item_id)
    source = B1CrossFitter(data, labels, 23)

    def fail(*args, **kwargs):
        return {"valid": False, "status": "synthetic_no_incumbent"}

    result = group_fit(
        source, ids[3:], ids[:3], next(iter(data.schemes.values())), 0, tmp_path, fail, 23
    )
    assert not result["complete"] and result["reason"] == "incomplete_inner_grid"
    assert any(not r["complete"] and r["macro_f1"] is None for r in result["candidates"][GOSDT])


def test_resource_gate_stops_before_reading_experiment_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostic, "_load", lambda *a: pytest.fail("must not load benchmark"))

    def insufficient(*args):
        raise MemoryError("synthetic low memory")

    monkeypatch.setattr(diagnostic, "require_memory", insufficient)
    config = configuration("configs/gosdt_schemes2_diagnostic.yaml")
    result = diagnostic.preflight(config, "unused", "unused", tmp_path)
    assert not result["ready"] and "low memory" in result["reason"]
    with pytest.raises(MemoryError):
        diagnostic.run("configs/gosdt_schemes2_diagnostic.yaml", "unused", "unused")


def test_ties_prefer_real_simplicity_before_grid_order():
    base = {"macro_f1": 0.5, "total_leaves": 4, "actual_depth": 2, "depth": 2, "coefficient": 2}
    assert select_candidate([base, {**base, "coefficient": 5}])["coefficient"] == 5
    assert select_candidate([base, {**base, "total_leaves": 3, "depth": 4}])["depth"] == 4


def test_primary_scoring_names_are_not_historical_b3():
    items = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "gold_fallacy": ["weak_analogy"] * 2,
            "gold_scheme": ["analogy"] * 2,
            "source": ["logic"] * 2,
            "origin": ["fallacy"] * 2,
            "reachable_gold": [True, False],
        }
    )
    predictions = pd.DataFrame(
        [
            {"item_id": i, "system": s, "verdict": v}
            for s, values in [
                (GOSDT, ["weak_analogy", None]),
                (CART, ["weak_analogy", "weak_analogy"]),
            ]
            for i, v in zip(items.item_id, values, strict=True)
        ]
    )
    metrics, contrasts, _ = diagnostic.score_tables(
        items, predictions, 23, 30, primary_system=GOSDT, reference_system=CART
    )
    assert set(metrics[metrics.subset == "all"].n_items) == {2}
    assert contrasts[contrasts.role == "primary"].iloc[0].difference < 0


def test_configuration_rejects_unagreed_grid(tmp_path):
    from pathlib import Path

    text = Path("configs/gosdt_schemes2_diagnostic.yaml").read_text()
    file = tmp_path / "changed.yaml"
    file.write_text(text.replace("depths: [2, 3, 4]", "depths: [1, 2]"))
    with pytest.raises(SchemeError, match="frozen"):
        configuration(file)


def test_solver_checkpoint_cannot_reuse_different_inputs_or_code(tmp_path):
    from pathlib import Path

    from argfallacy.eval.b3_diagnostic import digest, json_write
    from argfallacy.eval.gosdt_process import SolverProcess

    process = SolverProcess("unused", "unused", tmp_path)
    folder = tmp_path / "saved"
    folder.mkdir()
    x, y = np.array([[0], [1]], bool), np.array(["a", "b"])
    evaluation_x = x[::-1]
    arrays = folder / "input.npz"
    np.savez_compressed(arrays, x=x, y=y, evaluation_x=evaluation_x)
    json_write(folder / "input_hash.json", {"sha256": digest(arrays)})
    package = Path(diagnostic.__file__).parent
    request = {
        "depth": 2,
        "coefficient": 2,
        "seconds": 60,
        "memory_bytes": 4_000_000_000,
        "grace_seconds": 5,
        "worker_sha256": digest(package / "gosdt_worker.py"),
        "adapter_sha256": digest(package.parent / "aggregate/sparse_tree.py"),
    }
    json_write(folder / "request.json", request)
    json_write(folder / "result.json", {"valid": True, "status": "invented_saved"})
    json_write(folder / "supervision.json", {"result_sha256": digest(folder / "result.json")})
    assert process("saved", x, y, 2, 2, evaluation_x)["status"] == "invented_saved"
    with pytest.raises(ValueError, match="evaluation arrays"):
        process("saved", x, y, 2, 2, x)
    json_write(folder / "request.json", {**request, "worker_sha256": "different"})
    with pytest.raises(ValueError, match="configuration"):
        process("saved", x, y, 2, 2, evaluation_x)
