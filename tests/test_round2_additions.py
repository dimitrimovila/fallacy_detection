"""The additions to round 2 after its results: class-balanced losses, the mass divided by
the class frequency, A3 on the combined answers, the macro F1 over the frequent classes
and the human CQ answers in place of the model's."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from argfallacy.aggregate.interface import decide
from argfallacy.paths import REPO_ROOT
from argfallacy.schemes import load_all, traverse
from test_b1 import example_data
from test_b2_b3 import two_schemes

# ------------------------------------------------------------- balanced losses


def test_balanced_weights_give_every_class_the_same_total_and_mean_one():
    from argfallacy.aggregate.combiners import balanced_weights

    weights = balanced_weights(["a", "a", "a", "b"])
    np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 2 / 3, 2])
    assert weights.mean() == pytest.approx(1)
    assert weights[:3].sum() == pytest.approx(weights[3:].sum())
    np.testing.assert_allclose(balanced_weights(["a", "b"]), [1, 1])


def _imbalanced(data):
    """In each scheme, three items of four take the gold of the first item."""
    items = data.items.reset_index(drop=True)
    for _, own in items.groupby("gold_scheme"):
        first = own.gold_fallacy.iloc[0]
        moved = [index for k, index in enumerate(own.index) if k % 4 != 1]
        items.loc[moved, "gold_fallacy"] = first
    return replace(data, items=items)


def test_the_balanced_gold_loss_weighs_the_items_and_keeps_an_exact_gradient():
    from argfallacy.aggregate.combiners import DiagramLoss, Stacking, balanced_weights

    data = _imbalanced(example_data(24, full_scheme=True)[0])
    model = Stacking.at_centre(data)
    plain = DiagramLoss(data, data.items.item_id)
    weighted = DiagramLoss(data, data.items.item_id, balanced=True)
    p = model.predict(data)
    gold = data.items.set_index("item_id").gold_fallacy
    weights = balanced_weights([gold[i] for i in weighted.ids])
    assert len(set(gold)) > 1 and not np.allclose(weights, 1)
    value = weighted.objective(model.coef, data.x)[0]
    assert value == pytest.approx(-(weights * weighted.log_mass(p)).mean())
    assert plain.objective(model.coef, data.x)[0] == pytest.approx(-plain.log_mass(p).mean())
    _, gradient = weighted.objective(model.coef, data.x)
    direction = np.random.default_rng(3).normal(size=len(gradient))
    numerical = (weighted.objective(model.coef + 1e-6 * direction, data.x)[0] - value) / 1e-6
    assert gradient @ direction == pytest.approx(numerical, abs=2e-5)


def test_the_balanced_pooled_recalibration_keeps_an_exact_gradient():
    from argfallacy.aggregate.combiners import balanced_weights
    from argfallacy.aggregate.crossfit import ReferenceInputs
    from argfallacy.aggregate.tree import PooledCalibration

    data = _imbalanced(two_schemes())
    inputs = ReferenceInputs(data, 23).fit(data.items.item_id)
    plain = PooledCalibration.loss(inputs, data.items.item_id)
    weighted = PooledCalibration.loss(inputs, data.items.item_id, balanced=True)
    labels = [weighted.labels[g] for g in weighted.gold]
    np.testing.assert_allclose(weighted.weights, balanced_weights(labels))
    assert plain.weights is None and not np.allclose(weighted.weights, 1)
    centre = np.zeros(1 + len(weighted.labels))
    centre[0] = 1
    for objective in (plain, weighted):
        value, gradient = objective(centre)
        direction = np.random.default_rng(4).normal(size=len(centre))
        numerical = (objective(centre + 1e-6 * direction)[0] - value) / 1e-6
        assert gradient @ direction == pytest.approx(numerical, abs=2e-5)
    model = PooledCalibration.fit(inputs, data.items.item_id, c=1, balanced=True)
    assert model.objective_value <= weighted(centre)[0] + 1e-8


# ------------------------------------------------- mass divided by class frequency


def test_the_mass_divided_by_the_class_frequency_favours_the_rare_class():
    from argfallacy.aggregate.decision import class_priors, prior_corrected

    schemes = load_all(REPO_ROOT / "schemes/v1")
    scheme = schemes["analogy"]
    common, rare = sorted(scheme.terminals)[:2]
    aggregations, gold, folds = {}, {}, {}
    for k in range(10):
        item = f"t{k}"
        aggregations[item] = decide({common: 1.0}, 0.0, "")
        gold[item], folds[item] = (rare if k == 0 else common), 1
    aggregations["x"] = decide({common: 0.5, rare: 0.4}, 0.1, "x")
    aggregations["empty"] = decide({}, 1.0, "none")
    gold["x"] = gold["empty"] = common
    folds["x"] = folds["empty"] = 0
    scheme_of = dict.fromkeys(aggregations, "analogy")
    priors = class_priors(scheme, gold, [f"t{k}" for k in range(10)])
    k = len(scheme.terminals)
    assert priors[common] == pytest.approx(10 / (10 + k))
    assert priors[rare] == pytest.approx(2 / (10 + k))
    assert sum(priors.values()) == pytest.approx(1)
    out = prior_corrected(aggregations, schemes, scheme_of, gold, folds)
    assert aggregations["x"].verdict == common and out["x"].verdict == rare
    assert out["x"].abstention == pytest.approx(0.1)
    assert sum(out["x"].distribution.values()) == pytest.approx(0.9)
    assert out["empty"] == aggregations["empty"]
    # the training items of fold 1 are only those of fold 0, here x and empty
    assert out["t0"].verdict == common


# --------------------------------------------------- A3 on the combined answers


def test_the_combined_rows_carry_the_combined_answer_to_the_aggregators_of_round_1():
    from argfallacy.aggregate import LOGPROB
    from argfallacy.aggregate.b1 import combined_rows, hard_codes
    from argfallacy.aggregate.checklist import a3_out_of_fold
    from argfallacy.aggregate.combiners import majority
    from argfallacy.aggregate.soft import OUTSIDE, answer_masses

    data = example_data(30, full_scheme=True)[0]
    p = majority(data)
    rows = combined_rows(data, p)
    codes = hard_codes(data, p)
    for i, pair in enumerate(data.pairs.itertuples(index=False)):
        scheme = data.schemes[pair.scheme]
        a, b = scheme.answer_space(pair.cq_id)
        row = rows[pair.item_id][pair.cq_id]
        masses = answer_masses(scheme, pair.cq_id, row, LOGPROB)
        inside = 1 - data.outside[i]
        assert masses[a] == pytest.approx(inside * p[i])
        assert masses[b] == pytest.approx(inside * (1 - p[i]))
        assert masses.get(OUTSIDE, 0.0) == pytest.approx(data.outside[i])
        assert row["answer"] == (a, b, "cannot_be_determined")[codes[i]]
    folds = {item: k % 5 for k, item in enumerate(data.items.item_id)}
    verdicts, choices = a3_out_of_fold(data.schemes, data.items, rows, folds, 7, LOGPROB)
    assert set(verdicts) == set(data.items.item_id)
    assert {c["fold"] for c in choices} == set(range(5))


# ------------------------------------------------------------------ substitution


def _substitution_case():
    from argfallacy.eval.evaluate import Bench, resample_indices

    schemes = load_all(REPO_ROOT / "schemes/v1")
    scheme = schemes["analogy"]
    ids = [f"i{k}" for k in range(8)]
    model = {cq: scheme.answer_space(cq)[0] for cq in scheme.cq_order}
    rows = {i: {cq: {"answer": a} for cq, a in model.items()} for i in ids}
    # the human disagrees on the first CQ of half of the items
    first = scheme.cq_order[0]
    human = {i: {**model, first: scheme.answer_space(first)[1]} if k < 4 else dict(model)
             for k, i in enumerate(ids)}
    labels = pd.DataFrame([
        {"item_id": i, "scheme": "analogy", "cq_id": cq, "answer": a,
         "target": int(a == scheme.answer_space(cq)[0])}
        for i in ids for cq, a in human[i].items()])
    gold = [traverse(scheme, human[i]).verdict for i in ids]
    bench = Bench(gold, resample_indices(1, len(ids), 60, 1))
    return schemes, scheme, ids, rows, labels, bench, model, human, first


def test_the_human_answers_in_place_of_the_model_s_by_cq_and_on_a_curve():
    from argfallacy.eval.substitution import substitution

    schemes, scheme, ids, rows, labels, bench, model, _, first = _substitution_case()
    scheme_of = dict.fromkeys(ids, "analogy")
    every = np.arange(len(bench.observed.classes))
    table, curve = substitution(schemes, scheme_of, ids, rows, labels, bench, every, seed=1,
                                fractions=(0.0, 0.5, 1.0), draws=6)
    table = table.set_index(["scheme", "cq_id"])
    before = [traverse(scheme, model).verdict] * len(ids)
    assert table.loc[("analogy", first), "disagreeing"] == 4
    assert table.loc[("analogy", first), "labelled"] == len(ids)
    for cq in scheme.cq_order[1:]:
        assert table.loc[("analogy", cq), "disagreeing"] == 0
        assert table.loc[("analogy", cq), "verdicts_changed"] == 0
        assert table.loc[("analogy", cq), "macro_f1_difference"] == 0
    whole = table.loc[("all", "all")]
    assert whole.right_to_wrong == 0
    assert whole.wrong_to_right == whole.verdicts_changed
    base = bench.observed.metrics(before)
    assert whole.accuracy_difference == pytest.approx(1 - base["accuracy"][0])
    curve = curve.set_index("share")
    assert curve.loc[0.0, "pairs_replaced"] == 0 and curve.loc[1.0, "pairs_replaced"] == 4
    assert curve.loc[0.0, "macro_f1"] == pytest.approx(base["macro_f1"][0])
    assert curve.loc[1.0, "accuracy"] == pytest.approx(1.0)
    assert curve.loc[1.0, "agreement"] == pytest.approx(1.0)
    labelled = len(labels)
    assert curve.loc[0.5, "agreement"] == pytest.approx((labelled - 2) / labelled)
    assert curve.loc[0.0, "agreement"] == pytest.approx((labelled - 4) / labelled)
    # the same seed draws the same subsets
    again = substitution(schemes, scheme_of, ids, rows, labels, bench, every, seed=1,
                         fractions=(0.0, 0.5, 1.0), draws=6)[1].set_index("share")
    pd.testing.assert_frame_equal(curve, again)


# ------------------------------------------------------------------ the command


@pytest.fixture(scope="module")
def every_class_frequent(tmp_path_factory):
    """Round 2 with every class counted as frequent, so the two macro F1 coincide."""
    from argfallacy.eval import round2
    from argfallacy.eval.evaluate import evaluate
    from test_round2 import write_runs

    root = tmp_path_factory.mktemp("additions")
    hand = write_runs(root, {"analogy": 10, "example": 5})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(round2, "FREQUENT", 1)
        out = evaluate(replace(hand["config"], resamples=100), hand["runs"], hand["out"],
                       hand["items"], hand["annotations"])
    return hand, out


def test_the_additions_are_written_marked_and_paired(every_class_frequent):
    _, out = every_class_frequent
    for name in ("frequent_classes.csv", "frequent_classes_differences.csv",
                 "substitution.csv", "substitution_curve.csv"):
        assert (out / name).is_file(), name
    verdicts = pd.read_csv(out / "verdicts.csv")
    added = {"A3.combined", "A2.prior", "B1.prior", "B3.pooled.balanced",
             "B1.stacking.gold_balanced.soft", "B1.stacking.gold_balanced.hard"}
    assert added | {"A3"} <= set(verdicts.system)
    main = pd.read_csv(out / "main_table.csv").set_index("system")
    assert all(main.loc[n, "description"].endswith("(added after the results)") for n in added)
    assert (main.loc[sorted(added), "role"] == "exploratory").all()
    assert main.loc["A3", "role"] == "reference"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["added_after_the_results"]) == added
    assert manifest["frequent_classes"]["minimum_gold_items"] == 1
    explored = pd.read_csv(out / "exploratory.csv")
    assert {("A3.combined", "B1.majority.none.hard"), ("A3.combined", "A3"),
            ("B3.pooled.balanced", "B3.pooled"), ("A2.prior", "A2"), ("B1.prior", "B1"),
            ("B1.stacking.gold_balanced.soft", "B1.stacking.gold.soft")} <= set(
        zip(explored.system, explored.reference, strict=True))
    # every class is frequent here: the two macro F1 are the same number
    frequent = pd.read_csv(out / "frequent_classes.csv").set_index("system")
    np.testing.assert_allclose(frequent.loc[main.index, "macro_f1_frequent"],
                               main.macro_f1, atol=1e-4)
    differences = pd.read_csv(out / "frequent_classes_differences.csv")
    paired = differences[differences.kind == "exploratory"].merge(
        explored, on=["system", "reference"])
    np.testing.assert_allclose(paired.macro_f1_frequent_difference,
                               paired.macro_f1_difference, atol=1e-4)
    np.testing.assert_allclose(paired.macro_f1_frequent_low, paired.macro_f1_low, atol=1e-4)
    assert set(differences[differences.kind == "deciding"].question) <= {
        "H1.1", "H1.2", "H2", "H3.1", "H3.2", "H4"}
    substituted = pd.read_csv(out / "substitution.csv")
    assert substituted.iloc[-1][["scheme", "cq_id"]].tolist() == ["all", "all"]
    curve = pd.read_csv(out / "substitution_curve.csv")
    assert curve.share.tolist() == [k / 10 for k in range(11)]
    assert curve.agreement.is_monotonic_increasing
    page = (out / "summary.html").read_text(encoding="utf-8")
    assert 'id="additions"' in page
