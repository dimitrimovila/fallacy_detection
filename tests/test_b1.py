"""CQ pooling, supervised fitting, held-out isolation, and item-wise resampling."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from argfallacy.aggregate import a4
from argfallacy.paths import REPO_ROOT
from argfallacy.schemes import SchemeError, load_all


def example_data(n=40, seed=34, full_scheme=False):
    from argfallacy.aggregate.b1_inputs import build_dataset

    original = load_all(REPO_ROOT / "schemes/v1")["analogy"]
    scheme = (
        original
        if full_scheme
        else replace(
            original,
            cqs={"CQ4": original.cqs["CQ4"]},
            cq_order=["CQ4"],
            start="CQ4",
            nodes={"CQ4": original.nodes["CQ4"]},
        )
    )
    models = {}
    for e in range(5):
        models[f"e{e}"] = {
            "samples": 5 if e in (2, 3) else 1,
            "reasoning": {"enable_thinking": e == 4},
        }
    rng = np.random.default_rng(seed)
    items, rows, labels = [], [], []
    for i in range(n):
        item, target = f"i{i:03}", i % 2
        gold = scheme.nodes["CQ4"]["yes" if target else "no"].target
        items.append({"item_id": item, "gold_scheme": scheme.scheme_id, "gold_fallacy": gold})
        for cq in scheme.cq_order:
            labels.append(
                {"item_id": item, "scheme": scheme.scheme_id, "cq_id": cq, "target": target}
            )
            for e in range(5):
                p = (0.85 if target else 0.15) if e == 0 else rng.uniform(0.1, 0.9)
                answer = "yes" if p >= 0.5 else "no"
                row = {
                    "item_id": item,
                    "scheme": scheme.scheme_id,
                    "cq_id": cq,
                    "stage": "stage2",
                    "scheme_condition": "gold",
                    "model": f"e{e}",
                    "answer": answer,
                    "p_verbal": p,
                    "p_logprob": p,
                    "p_logprob_yes": p,
                    "p_logprob_no": 1 - p,
                    "p_logprob_cannot_be_determined": 0.0,
                    "p_logprob_na": 0.0,
                }
                if e in (2, 3):
                    row.update(
                        {
                            "p_sample": p,
                            "p_sample_yes": p,
                            "p_sample_no": 1 - p,
                            "p_sample_cannot_be_determined": 0.0,
                            "p_sample_na": 0.0,
                        }
                    )
                rows.append(row)
    frame, summary = pd.DataFrame(items), pd.DataFrame(rows)
    routing = {cq: ("abstain" if cq in ("CQ2", "CQ3") else "no") for cq in scheme.cq_order}
    data = build_dataset(
        {scheme.scheme_id: scheme},
        frame,
        summary,
        {"models": models},
        "e0",
        {scheme.scheme_id: routing},
    )
    return data, pd.DataFrame(labels), summary, {"models": models}


def test_sources_features_and_missing_entries_follow_a4():
    from argfallacy.aggregate.b1_inputs import build_dataset
    from argfallacy.aggregate.combiners import WeightedPool

    data, _, summary, manifest = example_data(2)
    assert len(data.signals) == 12 and data.x.shape == (2, 25)
    summary.loc[summary.model == "e1", "answer"] = "invalid"
    summary.loc[summary.model == "e1", [c for c in summary if c.startswith("p_")]] = np.nan
    data = build_dataset(data.schemes, data.items, summary, manifest, "e0", data.routing)
    assert (data.observations[:, 1] == -1).all()
    e1_signals = [j for j, signal in enumerate(data.signals) if signal.entry == "e1"]
    assert np.isnan(data.splits[:, e1_signals]).all()
    assert (data.x[:, 1 + len(data.signals) + np.array(e1_signals)] == 1).all()
    absent = build_dataset(
        data.schemes, data.items, summary[summary.model != "e1"], manifest, "e0", data.routing
    )
    np.testing.assert_equal(data.outside, absent.outside)
    np.testing.assert_equal(data.x, absent.x)
    pool = WeightedPool.equal(data)
    np.testing.assert_allclose(pool.predict(data), pool.predict(absent), atol=1e-12)


def test_every_entry_omitted_has_whole_outside_mass_and_no_hard_verdict():
    from argfallacy.aggregate.b1 import aggregate_combined
    from argfallacy.aggregate.b1_inputs import build_dataset
    from argfallacy.aggregate.combiners import DawidSkene, Stacking, WeightedPool, majority

    data, _, summary, manifest = example_data(2)
    summary["answer"] = "invalid"
    summary[[c for c in summary if c.startswith("p_")]] = np.nan
    data = build_dataset(data.schemes, data.items, summary, manifest, "e0", data.routing)
    np.testing.assert_equal(data.outside, np.ones(2))
    for p in [
        majority(data),
        WeightedPool.equal(data).predict(data),
        Stacking.at_centre(data).predict(data),
        DawidSkene.fit(data).predict(data),
    ]:
        assert np.isfinite(p).all()
        predictions = aggregate_combined(data, p, "hard")
        assert all(r.verdict is None and r.abstention == 1 for r in predictions.values())


@pytest.mark.parametrize("omit", [False, True])
def test_lambda_zero_matches_a4_with_unequal_outside_masses(omit):
    from argfallacy.aggregate.b1_inputs import build_dataset
    from argfallacy.aggregate.combiners import WeightedPool

    data, _, summary, manifest = example_data(2)
    for e, outside in enumerate([0.8, 0.15, 0.3, 0.0, 0.0]):
        mask = summary.model == f"e{e}"
        for source in ("logprob", "sample"):
            summary.loc[mask, f"p_{source}_yes"] *= 1 - outside
            summary.loc[mask, f"p_{source}_no"] *= 1 - outside
            summary.loc[mask, f"p_{source}_cannot_be_determined"] = outside
    if omit:
        summary = summary[summary.model != "e1"]
    data = build_dataset(data.schemes, data.items, summary, manifest, "e0", data.routing)
    split = WeightedPool.equal(data).predict(data)
    from argfallacy.aggregate.b1 import aggregate_combined

    for item in data.items.item_id:
        selected = summary[summary.item_id == item]
        rows = {
            entry: own.set_index("cq_id").to_dict("index")
            for entry, own in selected.groupby("model")
        }
        scheme = data.schemes[data.items.set_index("item_id").loc[item, "gold_scheme"]]
        old = a4(scheme, rows, data.sources, data.routing[scheme.scheme_id])
        result = aggregate_combined(data, split, "soft")[item]
        assert result.distribution == pytest.approx(old.distribution, abs=1e-12)
        assert result.abstention == pytest.approx(old.abstention, abs=1e-12)


def test_centre_is_the_equal_logarithmic_pool_and_unanimity_where_applicable():
    from argfallacy.aggregate.combiners import DawidSkene, Stacking, WeightedPool, majority

    data, _, _, _ = example_data()
    model = Stacking.at_centre(data)
    primary = data.splits[:, data.primary_signals]
    expected = expit(logit(np.clip(primary, 0.005, 0.995)).mean(axis=1))
    np.testing.assert_allclose(model.predict(data), expected, atol=1e-12)
    for probability, observed in [(0.9, 0), (0.1, 1)]:
        coherent = replace(
            data,
            splits=np.full_like(data.splits, probability),
            base_splits=np.full_like(data.base_splits, probability),
            observations=np.full_like(data.observations, observed),
        )
        coherent = coherent.with_features()
        confusion = np.tile([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], (5, 1, 1))
        ds = DawidSkene(confusion, {q: np.array([0.5, 0.5]) for q in data.questions})
        for prediction in [
            majority(coherent),
            WeightedPool.equal(coherent).predict(coherent),
            Stacking.at_centre(coherent).predict(coherent),
            ds.predict(coherent),
        ]:
            assert ((prediction >= 0.5) == (probability > 0.5)).all()


def test_cq_stacking_learns_the_informative_entry():
    from argfallacy.aggregate.combiners import Stacking

    data, labels, _, _ = example_data(160)
    model = Stacking.fit_cq(data, labels, data.items.item_id, c=10.0)
    contributions = {
        entry: sum(
            abs(model.coef[1 + j]) for j, signal in enumerate(data.signals) if signal.entry == entry
        )
        for entry in data.entries
    }
    assert max(contributions, key=contributions.get) == "e0"


def test_gold_stacking_improves_log_mass_without_using_cq_labels():
    from argfallacy.aggregate.combiners import DiagramLoss, Stacking

    data, _, _, _ = example_data(80)
    centre = Stacking.at_centre(data)
    loss = DiagramLoss(data, data.items.item_id)
    fitted = Stacking.fit_gold(data, data.items.item_id, c=10.0)
    assert loss.log_mass(fitted.predict(data)).mean() > loss.log_mass(centre.predict(data)).mean()
    value, gradient = loss.objective(centre.coef, data.x)
    numerical = np.zeros_like(gradient)
    for j in range(len(gradient)):
        shifted = centre.coef.copy()
        shifted[j] += 1e-6
        numerical[j] = (loss.objective(shifted, data.x)[0] - value) / 1e-6
    np.testing.assert_allclose(gradient, numerical, atol=2e-5)


@pytest.mark.parametrize(
    "combiner,supervision", [("stacking", "cq"), ("stacking", "gold"), ("weighted", "cq")]
)
def test_own_cq_and_gold_labels_change_no_held_out_prediction(combiner, supervision):
    from argfallacy.aggregate.b1 import B1Config, b1_out_of_fold

    data, labels, _, _ = example_data(20)
    folds = {item: i % 5 for i, item in enumerate(data.items.item_id)}
    cfg = B1Config(combiner=combiner, supervision=supervision)
    first = b1_out_of_fold(data, labels, folds, 23, cfg)
    changed = labels.copy()
    changed.loc[changed.item_id == "i000", "target"] = (
        1 - changed.loc[changed.item_id == "i000", "target"]
    )
    other_items = data.items.copy()
    other_items.loc[other_items.item_id == "i000", "gold_fallacy"] = "weak_analogy"
    second = b1_out_of_fold(replace(data, items=other_items), changed, folds, 23, cfg)
    assert first.aggregations["i000"] == second.aggregations["i000"]
    a = first.combined[first.combined.item_id == "i000"].p_first.to_numpy()
    b = second.combined[second.combined.item_id == "i000"].p_first.to_numpy()
    np.testing.assert_equal(a, b)


def test_dawid_skene_recovers_known_confusions_and_is_deterministic():
    from argfallacy.aggregate.combiners import DawidSkene

    data, _, _, _ = example_data(2400)
    rng = np.random.default_rng(4)
    truth = rng.integers(0, 2, len(data.pairs))
    confusion = np.array(
        [
            [[a, 1 - a - 0.1, 0.1], [1 - b - 0.1, b, 0.1]]
            for a, b in [(0.8, 0.78), (0.73, 0.76), (0.72, 0.74), (0.76, 0.8), (0.82, 0.8)]
        ]
    )
    observations = np.column_stack(
        [np.array([rng.choice(3, p=confusion[e, z]) for z in truth]) for e in range(5)]
    )
    observations[::19, 2] = -1
    data = replace(data, observations=observations)
    one, two = DawidSkene.fit(data), DawidSkene.fit(data)
    np.testing.assert_equal(one.confusion, two.confusion)
    np.testing.assert_allclose(one.confusion, confusion, atol=0.055)
    assert np.mean((one.predict(data) >= 0.5) == (truth == 0)) > 0.85


def test_same_seed_same_outputs_and_no_label_access_without_supervision():
    from argfallacy.aggregate.b1 import B1Config, b1_out_of_fold, configurations

    data, labels, _, _ = example_data(20)
    folds = {item: i % 5 for i, item in enumerate(data.items.item_id)}
    for name in ("majority", "dawid_skene"):
        cfg = B1Config(combiner=name, supervision="none")
        first = b1_out_of_fold(data, None, folds, 9, cfg)
        second = b1_out_of_fold(data, labels.assign(target=1 - labels.target), folds, 9, cfg)
        pd.testing.assert_frame_equal(first.combined, second.combined)
        assert len(first.models) == 1
    cfg = B1Config()
    first, second = [b1_out_of_fold(data, labels, folds, 9, cfg) for _ in range(2)]
    pd.testing.assert_frame_equal(first.combined, second.combined)
    assert first.choices == second.choices
    names = {c.name for c in configurations()}
    assert "B1" in names and "B1.hard" in names and len(names) == 10


@pytest.mark.parametrize("other_scheme", ["ad_hominem", "none", "idk"])
def test_cq_labels_are_binary_and_only_from_the_gold_sheet(other_scheme):
    from argfallacy.eval.b1 import cq_labels

    schemes = load_all(REPO_ROOT / "schemes/v1")
    items = pd.DataFrame([{"item_id": "i0", "gold_scheme": "analogy"}])
    annotations = []
    for sid in ["analogy", "ad_hominem"]:
        annotations.append(
            {
                "item_id": "i0",
                "annotator": "dumitru",
                "sheet": sid,
                "field": "scheme",
                "value": sid,
                "row": 0,
            }
        )
        for j, cq in enumerate(schemes[sid].cq_order):
            annotations.append(
                {
                    "item_id": "i0",
                    "annotator": "dumitru",
                    "sheet": sid,
                    "field": cq,
                    "value": "yes" if j else "idk",
                    "row": 0,
                }
            )
    annotations.append(
        {
            "item_id": "i0",
            "annotator": "other",
            "sheet": "analogy",
            "field": "CQ2",
            "value": "no",
            "row": 0,
        }
    )
    labels = cq_labels(pd.DataFrame(annotations), items, schemes)
    assert set(labels.scheme) == {"analogy"}
    assert len(labels) == len(schemes["analogy"].cqs) - 1
    assert set(labels.target) == {1}
    judged_other = pd.DataFrame(annotations)
    judged_other.loc[
        (judged_other.sheet == "analogy") & (judged_other.field == "scheme"), "value"
    ] = other_scheme
    assert cq_labels(judged_other, items, schemes).empty


def test_frozen_cq_labels_have_2656_pairs_under_the_pool_a_rule():
    from argfallacy.annotation_samples import pool_a
    from argfallacy.annotations import load_annotations
    from argfallacy.eval.b1 import cq_labels, gold_scheme_answers
    from argfallacy.eval.prior import load_test_set

    items, annotations = load_test_set(), load_annotations()
    schemes = load_all(REPO_ROOT / "schemes/v1")
    labels = cq_labels(annotations, items, schemes)
    assert len(labels) == 2656
    assert set(gold_scheme_answers(annotations, items, schemes).item_id) == pool_a(
        items, annotations, schemes
    )


def test_item_bootstrap_keeps_all_cqs_together_and_uses_paired_differences():
    from argfallacy.eval.b1 import cq_evaluation

    data, labels, _, _ = example_data(2, full_scheme=True)
    outputs = data.pairs.copy()
    outputs["p_first"], outputs["answer"] = 0.9, "yes"
    outputs["system"], outputs["supervision"] = "B1", "cq"
    reference = outputs.assign(system="entry:e0", p_first=0.1, answer="no", supervision="none")
    # Whole items, with no resample containing both items.
    indices = np.array([[0, 0], [1, 1]])
    report = cq_evaluation(data, labels, pd.concat([outputs, reference]), indices)
    row = report.level[(report.level.system == "B1") & (report.level.scheme == "all")].iloc[0]
    assert row.agreement == pytest.approx(0.5)
    assert row.agreement_low == pytest.approx(0.025) and row.agreement_high == pytest.approx(0.975)
    diff = report.differences[
        (report.differences.system == "B1")
        & (report.differences.reference == "entry:e0")
        & (report.differences.scheme == "all")
    ].iloc[0]
    assert diff.agreement_difference == pytest.approx(0)
    assert diff.agreement_low == pytest.approx(-0.95) and diff.agreement_high == pytest.approx(0.95)


def test_duplicate_or_misaligned_model_rows_fail_with_context():
    from argfallacy.aggregate.b1_inputs import build_dataset

    data, _, summary, manifest = example_data(2)
    with pytest.raises(SchemeError, match="i000.*CQ4|CQ4.*i000"):
        build_dataset(
            data.schemes,
            data.items,
            pd.concat([summary, summary.iloc[:1]]),
            manifest,
            "e0",
            data.routing,
        )


def test_majority_uses_mean_outside_mass_and_reference_arc_ties():
    from argfallacy.aggregate.b1 import aggregate_combined, hard_codes
    from argfallacy.aggregate.combiners import majority, majority_codes

    data, _, _, _ = example_data(4)
    # Three outside hard votes do not defeat two arc votes when outside mass is low.
    observed = np.array([[0, 0, 2, 2, 2], [1, 0, 2, 2, 2], [2, 0, 1, 2, 2], [0, 0, 0, 0, 0]])
    data = replace(data, observations=observed, outside=np.array([0.2, 0.5, 0.2, 0.51]))
    p = majority(data)
    np.testing.assert_equal(p, [1, 0.5, 0.5, 1])
    np.testing.assert_equal(hard_codes(data, p), [0, 1, 0, 2])
    np.testing.assert_equal(majority_codes(data), [0, 1, 0, 2])
    assert aggregate_combined(data, p, "hard")["i003"].verdict is None


def test_unanimous_weighted_pool_for_arbitrary_nonnegative_weights_and_shrinkage():
    from argfallacy.aggregate.combiners import WeightedPool

    data, labels, _, _ = example_data(8)
    for probability in [0.1, 0.9]:
        data = replace(
            data,
            base_splits=np.full_like(data.base_splits, probability),
            observations=np.full_like(data.observations, int(probability < 0.5)),
        )
        for shrinkage in [0, 0.25, 0.5, 0.75, 1]:
            pool = WeightedPool.fit(data, labels, data.items.item_id, shrinkage)
            np.testing.assert_allclose(pool.predict(data), probability, atol=1e-12)
        for weight in [np.array([0, 0, 0, 0, 1]), np.array([0.1, 0.5, 0.1, 0.2, 0.1])]:
            pool = replace(pool, weights={q: weight for q in data.questions})
            np.testing.assert_allclose(pool.predict(data), probability, atol=1e-12)


def test_transfer_and_exploratory_exclusions_keep_the_original_fit():
    from argfallacy.aggregate.b1 import B1Config, b1_out_of_fold, training_labels, transfer_b1

    data, labels, _, _ = example_data(20)
    folds = {item: i % 5 for i, item in enumerate(data.items.item_id)}
    source = b1_out_of_fold(data, labels, folds, 9)
    transferred = transfer_b1(source, data)
    np.testing.assert_equal(
        source.combined.p_first.to_numpy(), transferred.combined.p_first.to_numpy()
    )
    assert transferred.config.name == "B1@2.0"
    config = B1Config(exclude_schemes=tuple(data.schemes))
    kept, excluded = training_labels(data, labels, data.items.item_id, config)
    assert kept.empty and excluded == set(data.questions)
    repeats = labels.assign(target=1 - labels.target)
    config = B1Config(unstable=True)
    assert training_labels(data, labels, data.items.item_id[:9], config, repeats)[1] == set()
    assert training_labels(data, labels, data.items.item_id[:10], config, repeats)[1] == set(
        data.questions
    )


def test_exploratory_registry_and_baselines():
    from argfallacy.aggregate.b1 import aggregate_combined, baseline_aggregations, configurations
    from argfallacy.aggregate.combiners import WeightedPool

    configs = configurations(excluded_label_schemes=("analogy",), include_unstable=True)
    assert len(configs) == 12
    assert any(c.exclude_schemes == ("analogy",) for c in configs)
    assert any(c.unstable for c in configs)
    revised = configurations(schemes_version="2.0")
    assert len(revised) == 6 and all(c.name.endswith("@2.0") for c in revised)
    assert not any(c.supervision == "cq" for c in revised)
    data, _, _, _ = example_data(2)
    baselines = baseline_aggregations(data)
    assert set(baselines) == {"A2:e0", "A4"}
    equal = aggregate_combined(data, WeightedPool.equal(data).predict(data), "soft")
    for item, prediction in baselines["A4"].items():
        assert prediction.distribution == pytest.approx(equal[item].distribution)


def test_human_path_reference_reports_coverage_on_both_denominators():
    from argfallacy.eval.b1 import human_path_reference

    data, _, _, _ = example_data(2, full_scheme=True)
    scheme = next(iter(data.schemes.values()))
    rows = [
        {
            "item_id": item,
            "annotator": "dumitru",
            "sheet": "sheet",
            "field": cq,
            "value": "yes" if item == "i000" else "idk",
            "row": 0,
        }
        for item in data.items.item_id
        for cq in scheme.cq_order
    ]
    rows.extend(
        {
            "item_id": item,
            "annotator": "dumitru",
            "sheet": "sheet",
            "row": 0,
            "field": "scheme",
            "value": scheme.scheme_id,
        }
        for item in data.items.item_id
    )
    report = human_path_reference(pd.DataFrame(rows), data.items, data.schemes)
    assert report["role"] == "descriptive reference"
    assert report["usable_items"] == 1 and report["annotated_items"] == 2
    assert report["coverage_test"] == report["coverage_annotated"] == 0.5


@pytest.mark.parametrize("folder", ["v1", "v2"])
def test_gold_mass_and_gradient_follow_all_burden_routes_and_diagram_paths(folder):
    from argfallacy.aggregate import load_burden, routes
    from argfallacy.aggregate.b1 import aggregate_combined
    from argfallacy.aggregate.b1_inputs import build_dataset
    from argfallacy.aggregate.combiners import DiagramLoss, Stacking

    schemes = load_all(REPO_ROOT / "schemes" / folder)
    burden = load_burden(folder, schemes)
    routing = {sid: routes(scheme, burden[sid]) for sid, scheme in schemes.items()}
    _, _, template, manifest = example_data(1)
    items, rows = [], []
    for sid, scheme in schemes.items():
        for j, terminal in enumerate(scheme.terminals):
            item = f"{sid}_{j}"
            items.append({"item_id": item, "gold_scheme": sid, "gold_fallacy": terminal})
            for cq in scheme.cq_order:
                for original in template.to_dict("records"):
                    row = {k: v for k, v in original.items() if not k.startswith("p_")}
                    row.update(item_id=item, scheme=sid, cq_id=cq)
                    a, b = scheme.answer_space(cq)
                    row.update(answer=a, p_verbal=0.7)
                    for source in ["logprob", "sample"]:
                        row.update(
                            {
                                f"p_{source}_{a}": 0.35,
                                f"p_{source}_{b}": 0.15,
                                f"p_{source}_cannot_be_determined": 0.4,
                                f"p_{source}_na": 0.1,
                            }
                        )
                    rows.append(row)
    data = build_dataset(schemes, pd.DataFrame(items), pd.DataFrame(rows), manifest, "e0", routing)
    model = Stacking.at_centre(data)
    loss = DiagramLoss(data, data.items.item_id)
    p = model.predict(data)
    predictions = aggregate_combined(data, p, "soft")
    gold = data.items.set_index("item_id").gold_fallacy
    expected = [np.log(predictions[i].distribution.get(gold[i], 0) + 1e-6) for i in loss.ids]
    np.testing.assert_allclose(loss.log_mass(p), expected, atol=1e-12)
    value, gradient = loss.objective(model.coef, data.x)
    direction = np.random.default_rng(5).normal(size=len(gradient))
    numerical = (loss.objective(model.coef + 1e-6 * direction, data.x)[0] - value) / 1e-6
    assert gradient @ direction == pytest.approx(numerical, abs=2e-5)
