"""Acceptance checks for saved-prediction analyses; all inputs are synthetic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from argfallacy.eval.analysis import AnalysisConfig, analyze
from argfallacy.eval.analysis_bootstrap import Bootstrap, score_population
from argfallacy.eval.canonical import score_fallacies
from argfallacy.eval.evaluate import resample_indices, scores
from argfallacy.eval.folds import assign_folds
from argfallacy.schemes import SchemeError

GOOD, WEAK, CHERRY = "good_argumentation", "weak_analogy", "cherry_picking"


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def saved(tmp_path):
    items = pd.DataFrame(
        {
            "item_id": [f"i{k:03}" for k in range(40)],
            "source": ["LOGIC"] * 24
            + ["NLAS"] * 8
            + ["EthiX"] * 4
            + ["Reddit"] * 3
            + ["ElecDebate"],
            "gold_scheme": ["analogy"] * 36 + ["example"] * 4,
            "gold_fallacy": [GOOD] * 24 + [WEAK] * 12 + [CHERRY] * 4,
            "in_test_set": True,
        }
    )
    items.to_csv(tmp_path / "items.csv", index=False)
    folds = assign_folds(items, 17)
    run = tmp_path / "parsed"
    run.mkdir()
    manifest = {
        "run_id": "parsed",
        "commit": "run-commit",
        "schemes_version": {"tag": "1.1"},
        "config": {"models": ["ref"], "samples": 1},
        "models": {"ref": {"samples": 1}},
    }
    dump(run / "manifest.json", manifest)
    summary = []
    for item in items.itertuples():
        for stage, cq, answer in [
            ("stage1", "", item.gold_scheme),
            ("zeroshot", "", item.gold_fallacy),
            ("stage2", "CQ1", "yes"),
        ]:
            summary.append(
                {
                    "item_id": item.item_id,
                    "stage": stage,
                    "scheme_condition": "gold",
                    "scheme": item.gold_scheme,
                    "cq_id": cq,
                    "model": "ref",
                    "answer": answer,
                    "p_label": 0.0 if item.item_id == "i000" else 0.7,
                    "p_label_lower_bound": item.item_id == "i000",
                    "n_samples_ok": 1,
                    "invalid_count": 0,
                }
            )
    # Only CQ1 exists here: remaining expected CQs must be counted as missing.
    pd.DataFrame(summary).to_csv(run / "summary.csv", index=False)
    answers = pd.DataFrame(summary).assign(sample_index=0, parse_ok=True, finish_reason="stop")
    answers.to_csv(run / "answers.csv", index=False)
    for name, round_number in [("first", 1), ("second", 2)]:
        d = tmp_path / name
        d.mkdir()
        m = {
            "name": name,
            "round": round_number,
            "seed": 17,
            "resamples": 73,
            "items": 40,
            "items_without_nlas": 32,
            "folds": {"k": 5, "assignment": folds},
            "commit": "source-commit",
            "runs": {"answers": "parsed", "zeroshot": ["parsed"]},
            "zeroshot_sources": {"ref": "parsed"},
            "schemes": {"tag": "1.1"},
            "reference_model": "ref",
            "m": 5 if round_number == 1 else 4,
            "primary_configurations": {"A2": "saved"}
            if round_number == 1
            else {"B1": "saved", "B2": "saved"},
            "cq_label_exclusion": ["example"],
        }
        dump(d / "manifest.json", m)
        systems = (
            ["A0:ref", "A2", "zeroshot:ref"]
            if round_number == 1
            else ["A0:ref", "A2", "zeroshot:ref", "B1", "B2", "partial"]
        )
        rows, main = [], []
        for system in systems:
            own = items if system != "partial" else items.iloc[:36]
            main.append({"system": system, "schemes": "1.1", "n_items": len(own)})
            for k, item in enumerate(own.itertuples()):
                verdict = item.gold_fallacy if system == "zeroshot:ref" or k % 3 else ""
                rows.append(
                    {
                        "system": system,
                        "item_id": item.item_id,
                        "fold": folds[item.item_id],
                        "verdict": verdict,
                        "confidence": "" if system == "zeroshot:ref" else 0.5,
                    }
                )
        pd.DataFrame(rows).to_csv(d / "verdicts.csv", index=False)
        pd.DataFrame(main).to_csv(d / "main_table.csv", index=False)
        decide = [{"system": "A2", "reference": "A0:ref", "level": 0.99}]
        if round_number == 2:
            decide = [
                {
                    "hypothesis": "H1.1",
                    "system": "B1",
                    "reference": "A0:ref",
                    "level": 0.9875,
                    "read_after": "",
                    "read": True,
                    "computed": True,
                    "improvement": False,
                },
                {
                    "hypothesis": "H1.2",
                    "system": "B1",
                    "reference": "zeroshot:ref",
                    "level": 0.9875,
                    "read_after": "H1.1",
                    "read": False,
                    "computed": True,
                    "improvement": "",
                },
            ]
        pd.DataFrame(decide).to_csv(d / "deciding.csv", index=False)
        pd.DataFrame(
            [
                {
                    "system": "partial" if round_number == 2 else "A2",
                    "reference": "A2" if round_number == 2 else "zeroshot:ref",
                }
            ]
        ).to_csv(d / "exploratory.csv", index=False)
    config = {
        "version": 1,
        "items": "items.csv",
        "output": "out",
        "seed": 17,
        "resamples": 73,
        "bootstrap_batch_size": 7,
        "evaluations": {
            "first": {"path": "first", "round": 1},
            "second": {
                "path": "second",
                "round": 2,
                "populations": {"partial": {"exclude_schemes": ["example"]}},
            },
        },
        "parsed_runs": {"parsed": "parsed"},
    }
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, config, items


def table(root, name):
    return pd.read_csv(root / f"{name}.csv")


def test_shared_draws_and_metrics_are_chunk_invariant():
    gold = [GOOD, WEAK, GOOD, CHERRY]
    predicted = [GOOD, None, "invalid", WEAK]
    mask = np.array([False, True, False, True])
    for generator in (1, 2):
        expected = resample_indices(7, 4, 79, generator)
        for batch in (1, 7, 79):
            boot = Bootstrap(7, 79, batch)
            assert np.array_equal(np.concatenate(list(boot.indices(4, generator))), expected)
            observed, draws = score_population(gold, predicted, mask, boot, generator)
            for i, chosen in enumerate(expected):
                chosen = chosen[mask[chosen]]
                if not len(chosen):
                    assert all(np.isnan(v[i]) for v in draws.values())
                else:
                    canonical = score_fallacies(
                        pd.DataFrame(
                            {
                                "item_id": range(len(chosen)),
                                "predicted": [
                                    predicted[k] if predicted[k] != "invalid" else None
                                    for k in chosen
                                ],
                            }
                        ),
                        pd.DataFrame(
                            {
                                "item_id": range(len(chosen)),
                                "gold_fallacy": [gold[k] for k in chosen],
                            }
                        ),
                    )
                    assert draws["macro_f1"][i] == pytest.approx(canonical.macro_f1)
            legacy = scores(gold, expected, predicted)
            full, full_draws = score_population(gold, predicted, np.ones(4, bool), boot, generator)
            assert full == legacy.observed
            for metric in legacy.resampled:
                np.testing.assert_equal(full_draws[metric], legacy.resampled[metric])
            assert (
                observed["good_recall"] == 0
            )  # canonical convention with undefined flag elsewhere


def test_outputs_denominators_pairing_gates_provenance_and_no_optional_reads(saved, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def no_fit(*args, **kwargs):
        raise AssertionError("saved-prediction analyses must never fit a system")

    monkeypatch.setattr(LogisticRegression, "fit", no_fit)
    path, config, items = saved
    config["direct_verdict"] = {"answers": None, "samples": None}
    config["dq"] = None
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    original = Path.open

    def guarded(self, *args, **kwargs):
        assert not self.name.startswith("forbidden")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    out = analyze(AnalysisConfig.load(path))
    main = table(out, "main_table")
    total = main[
        (main.evaluation == "second") & (main.system == "partial") & (main.subset == "total")
    ]
    assert total.n_items.tolist() == [36]
    origin = main[
        (main.evaluation == "first") & (main.system == "A2") & (main.subset_kind == "gold origin")
    ]
    assert sorted(origin.n_items.tolist()) == [12, 28]
    assert "ElecDebate" not in set(main.loc[main.subset_kind == "corpus", "subset"])
    assert main.loc[main.subset == "example", "small_scheme"].all()
    assert not main.loc[main.subset == "analogy", "small_scheme"].all()
    scheme = main[
        (main.evaluation == "second") & (main.system == "B1") & (main.subset == "example")
    ].iloc[0]
    assert scheme.good_recall == 0 and scheme.good_recall_undefined
    assert GOOD in json.loads(scheme.excluded_classes)
    comparison = table(out, "comparisons")
    partial = comparison[(comparison.system == "partial") & (comparison.subset == "total")]
    assert partial.n_items.tolist() == [36]
    gated = comparison[(comparison.hypothesis == "H1.2")]
    parent = gated[gated.subset == "total"].iloc[0]
    assert parent.level == 0.9875 and not parent.read and parent.kind == "deciding"
    assert (gated[gated.subset != "total"].level == 0.95).all()
    assert set(gated[gated.subset != "total"].kind) == {"exploratory"}
    per = table(out, "per_class")
    absent = per[
        (per.evaluation == "second")
        & (per.system == "B1")
        & (per.subset == "example")
        & (per["class"] == GOOD)
    ].iloc[0]
    assert absent.gold_support == 0 and absent.absent
    confidence = table(out, "confidence")
    zero = confidence[(confidence.system == "zeroshot:ref") & (confidence.item_id == "i000")]
    assert set(zero.confidence) == {0.0} and zero.p_label_lower_bound.all()
    assert set(zero.confidence_source) == {"summary.csv:p_label"}
    integrity = table(out, "integrity")
    assert "missing CQ rows" in set(integrity.measure)
    assert set(integrity.unit) >= {"item", "CQ row", "sample"}
    stage = table(out, "stage1")
    assert stage.n_items.tolist() == [40, 40]
    assert stage.accuracy.tolist() == [1.0, 1.0]
    assert set(stage.evaluation) == {"first", "second"}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sources"]["first"]["manifest"]["commit"] == "source-commit"
    assert manifest["bootstrap"]["item_ids"] == items.item_id.tolist()
    assert manifest["bootstrap"]["without_nlas_generator"] == 2
    assert manifest["inputs"] and all(len(i["sha256"]) == 64 for i in manifest["inputs"])
    assert manifest["parts"]["direct_verdict"]["status"] == "pending"
    assert manifest["parts"]["routing_curve"]["status"] == "pending"
    assert "replay" in manifest["parts"]["routing_curve"]["reason"]
    assert manifest["parts"]["collapsed"]["level"] == 3
    rare = table(out, "rare_classes")
    assert set(rare.system) == {"B1", "A0:ref", "A2", "zeroshot:ref"}
    assert all(GOOD not in json.loads(value) for value in rare.rare_classes)
    example = rare[
        (rare.system == "B1") & (rare.subset == "example") & (rare.origin == "total")
    ].iloc[0]
    assert json.loads(example.supported_rare_classes) == [CHERRY]
    assert example.mean_recall == pytest.approx(0.5)
    # A second batch size changes provenance only, never the tables or the pairing.
    config["bootstrap_batch_size"] = 1
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    other = analyze(AnalysisConfig.load(path), out_root=path.parent / "other")
    for name in ("main_table", "comparisons", "per_class", "rare_classes"):
        pd.testing.assert_frame_equal(table(out, name), table(other, name))


@pytest.mark.parametrize(
    "change, message",
    [
        ({"bootstrap_batch_size": 0}, "batch"),
        ({"bootstrap_batch_size": True}, "batch"),
        ({"resamples": 1.5}, "resamples"),
        ({"version": 2}, "version"),
        ({"runs": {}}, "unknown"),
        ({"direct_verdict": {"oops": "x"}}, "direct_verdict"),
    ],
)
def test_validated_independent_configuration(saved, change, message):
    path, config, _ = saved
    config.update(change)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SchemeError, match=message):
        AnalysisConfig.load(path)


@pytest.mark.parametrize("target", ["first", ".", "parsed", "items.csv"])
def test_no_output_overlaps_inputs(saved, target):
    path, config, _ = saved
    config["output"] = target
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SchemeError, match="overlap"):
        analyze(AnalysisConfig.load(path))


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("duplicate", "duplicate"),
        ("id", "item"),
        ("verdict", "label"),
        ("fold", "fold"),
        ("population", "population"),
        ("version", "version"),
        ("confidence", "confidence"),
        ("agreement", "verdict"),
    ],
)
def test_saved_contracts_reject_inconsistent_inputs_before_writing(saved, mutation, message):
    path, _, _ = saved
    root = path.parent
    file = root / "second" / "verdicts.csv"
    frame = pd.read_csv(file, keep_default_na=False)
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif mutation == "id":
        frame.loc[0, "item_id"] = "not-in-frozen-test"
    elif mutation == "verdict":
        frame.loc[0, "verdict"] = "invented-label"
    elif mutation == "fold":
        frame.loc[0, "fold"] = 99
    elif mutation == "population":
        frame = frame.iloc[1:]
    elif mutation == "version":
        main = pd.read_csv(root / "second" / "main_table.csv")
        main.loc[0, "schemes"] = 3.5
        main.to_csv(root / "second" / "main_table.csv", index=False)
    elif mutation == "confidence":
        frame.loc[0, "confidence"] = "2.0"
    else:
        frame.loc[frame.system == "zeroshot:ref", "verdict"] = WEAK
    frame.to_csv(file, index=False)
    with pytest.raises(SchemeError, match=message):
        analyze(AnalysisConfig.load(path))
    assert not (root / "out").exists()


def test_reserved_paths_default_null_and_paths_relative_to_configuration(saved, monkeypatch):
    path, _, _ = saved
    monkeypatch.chdir(path.parent.parent)
    config = AnalysisConfig.load(path)
    assert config.items == path.parent / "items.csv"
    assert config.direct_verdict == {"answers": None, "samples": None} and config.dq is None


def test_seed_and_resample_protocol_must_match_sources(saved):
    path, config, _ = saved
    config["seed"] += 1
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SchemeError, match="seed"):
        analyze(AnalysisConfig.load(path))


def test_common_item_pairing_uses_filtered_whole_test_draws(saved):
    from argfallacy.eval.analysis_inputs import Inputs, read_evaluation, scheme_registry
    from argfallacy.eval.analysis_tables import comparisons

    path, _, items = saved
    config = AnalysisConfig.load(path)
    inputs = Inputs()
    source = read_evaluation(
        "second", config.evaluations["second"], config, items, scheme_registry(inputs), inputs
    )
    result = comparisons([source], items, Bootstrap(17, 73, 7))
    row = result[
        (result.system == "partial") & (result.subset == "example") & (result.origin == "total")
    ].iloc[0]
    assert row.n_items == 0 and row.macro_f1_omitted == 73
    assert row.status == "undefined" and np.isnan(row.macro_f1_low)
    for subset, generator in [("total", 1), ("without NLAS", 2)]:
        row = result[
            (result.system == "partial") & (result.subset == subset) & (result.origin == "total")
        ].iloc[0]
        base = items if generator == 1 else items[items.source != "NLAS"].reset_index(drop=True)
        full = resample_indices(17, len(base), 73, generator)
        mask = base.gold_scheme != "example"
        a = source.systems["partial"].rows.verdict.reindex(base.item_id).to_numpy()
        b = source.systems["A2"].rows.verdict.reindex(base.item_id).to_numpy()
        _, sa = score_population(
            base.gold_fallacy.tolist(), a, mask, Bootstrap(17, 73, 73), generator
        )
        _, sb = score_population(
            base.gold_fallacy.tolist(), b, mask, Bootstrap(17, 73, 73), generator
        )
        assert full.shape == (73, len(base))
        low, high = np.percentile(sa["macro_f1"] - sb["macro_f1"], [2.5, 97.5])
        assert row.macro_f1_low == pytest.approx(low)
        assert row.macro_f1_high == pytest.approx(high)


def test_zero_confidence_is_distinct_from_missing_and_verdicts_are_preserved(saved):
    from argfallacy.eval.analysis_inputs import (
        Inputs,
        read_evaluation,
        read_run,
        recover_confidence,
        scheme_registry,
    )

    path, _, items = saved
    config = AnalysisConfig.load(path)
    csv = path.parent / "parsed" / "summary.csv"
    summary = pd.read_csv(csv, keep_default_na=False)
    summary.loc[(summary.stage == "zeroshot") & (summary.item_id == "i001"), "p_label"] = np.nan
    summary.to_csv(csv, index=False)
    inputs = Inputs()
    registry = scheme_registry(inputs)
    evaluation = read_evaluation(
        "first", config.evaluations["first"], config, items, registry, inputs
    )
    original = evaluation.systems["zeroshot:ref"].rows.verdict.copy()
    manifest, version, summary = read_run(
        config.parsed_runs["parsed"], "parsed", items, registry, inputs
    )
    recover_confidence([evaluation], "parsed", manifest, version, summary)
    rows = evaluation.systems["zeroshot:ref"].rows
    assert rows.loc["i000", "confidence"] == 0.0
    assert np.isnan(rows.loc["i001", "confidence"])
    assert rows.loc["i001", "confidence_source"] == "missing"
    pd.testing.assert_series_equal(rows.verdict, original)


def test_exact_routing_reuses_pooled_thresholds_in_each_origin(saved):
    import hashlib

    from argfallacy.eval.analysis_inputs import Inputs, read_evaluation, scheme_registry
    from argfallacy.eval.analysis_routing import routed, routing_tables

    path, config, items = saved
    # Deliberately different threshold scales by origin. The threshold selected
    # on all items must not be selected anew on either origin.
    records = pd.DataFrame(
        {
            "item_id": items.item_id,
            "fold": [assign_folds(items, 17)[i] for i in items.item_id],
            "p_d": 0.2,
            "p_z": [0.21] * 24 + [0.9] * 16,
            "diagram_verdict": [GOOD] * 40,
            "zero_verdict": [WEAK] * 40,
        }
    )
    primary, _ = routed(records, 0)
    saved_file = path.parent / "second" / "verdicts.csv"
    frame = pd.read_csv(saved_file, keep_default_na=False)
    frame.loc[frame.system == "B2", "verdict"] = primary
    frame.to_csv(saved_file, index=False)
    exact = path.parent / "routing_records.csv"
    records.to_csv(exact, index=False)
    dump(
        exact.with_suffix(".manifest.json"),
        {
            "system": "B2",
            "scheme_version": "1.1",
            "source_manifest_sha256": hashlib.sha256(
                (path.parent / "second" / "manifest.json").read_bytes()
            ).hexdigest(),
            "records_sha256": hashlib.sha256(exact.read_bytes()).hexdigest(),
        },
    )
    config["evaluations"]["second"]["routing_records"] = "routing_records.csv"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = AnalysisConfig.load(path)
    inputs = Inputs()
    evaluation = read_evaluation(
        "second", cfg.evaluations["second"], cfg, items, scheme_registry(inputs), inputs
    )
    table, status = routing_tables(cfg, [evaluation], items, Bootstrap(17, 73, 7), inputs)
    assert status["status"] == "computed"
    selected = table[table.requested_share == 0.6]
    assert len(selected) == 3 and len(set(selected.tau)) == 1
    assert selected.loc[selected.subset == "total", "explained_share"].iloc[0] == 0.6
    shares = selected.set_index("subset").explained_share
    assert shares["fallacy corpora"] == pytest.approx(24 / 28)
    assert shares["scheme corpora"] == 0
    assert selected.explained_replicates.tolist() == [73, 73, 73]


def test_bootstrap_workspace_never_exceeds_configured_batch(monkeypatch):
    from argfallacy.eval import analysis_bootstrap

    shapes = []
    original = analysis_bootstrap.weights

    def bounded(indices, n):
        shapes.append(indices.shape)
        assert indices.shape[0] <= 6
        return original(indices, n)

    monkeypatch.setattr(analysis_bootstrap, "weights", bounded)
    gold = [GOOD, WEAK] * 301
    score_population(gold, gold, np.ones(len(gold), bool), Bootstrap(17, 101, 6))
    assert sum(shape[0] for shape in shapes) == 101


def test_opt_in_routing_export_verifies_saved_predictions_and_preserves_sources(saved):
    import hashlib

    from argfallacy.aggregate.interface import Aggregation
    from argfallacy.aggregate.routing import B2Result, Side
    from argfallacy.eval.analysis_routing import export_routing_records
    from argfallacy.eval.routing import routing_curve

    path, _, items = saved
    ids = items.item_id.tolist()
    folds = assign_folds(items, 17)
    upstream = {i: Aggregation(GOOD, 0.5, {}, 0, "diagram") for i in ids}
    zero = Side("zero-shot", {i: Aggregation(WEAK, 0.7, {}, 0, "zero") for i in ids})
    result = B2Result(
        {},
        pd.DataFrame({"item_id": ids, "p_d": 0.2, "p_z": 0.723456789012345}),
        {},
        [],
        upstream,
        (zero,),
        folds,
        "B2",
    )
    csv = path.parent / "second" / "verdicts.csv"
    saved_rows = pd.read_csv(csv, keep_default_na=False)
    saved_rows.loc[saved_rows.system == "B2", "verdict"] = WEAK
    saved_rows.to_csv(csv, index=False)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in csv.parent.iterdir()}
    plain = routing_curve(result, items, seed=17, resamples=13)
    exported = routing_curve(
        result,
        items,
        seed=17,
        resamples=13,
        export_dir=path.parent / "exact",
        source_evaluation=csv.parent,
    )
    pd.testing.assert_frame_equal(plain, exported)
    records = pd.read_csv(
        path.parent / "exact" / "routing_records.csv", float_precision="round_trip"
    )
    assert records.p_z.iloc[0] == 0.723456789012345
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in before.items())
    saved_rows.loc[saved_rows.system == "B2", "verdict"] = GOOD
    saved_rows.to_csv(csv, index=False)
    with pytest.raises(SchemeError, match="verdict"):
        export_routing_records(result, items, path.parent / "rejected", csv.parent)
    assert not (path.parent / "rejected").exists()


def test_stage_one_scores_all_frozen_items_and_excludes_non_test_rows(saved):
    from argfallacy.eval.analysis_diagnostics import stage_one

    _, _, items = saved
    summary = pd.DataFrame(
        {
            "item_id": ["i000", "outside"],
            "stage": "stage1",
            "scheme_condition": "gold",
            "model": "ref",
            "answer": ["analogy", "unrecognised-outside-test-answer"],
        }
    )
    row = stage_one("parsed", "1.1", {}, summary, items)[0]
    assert row["n_items"] == 40 and row["observed_rows"] == 1 and row["missing_rows"] == 39
    assert row["accuracy"] == 1 / 40


def test_input_hashes_detect_changes_between_read_and_write(tmp_path):
    from argfallacy.eval.analysis_inputs import Inputs

    path = tmp_path / "input.csv"
    path.write_text("item_id,value\ni1,1\n")
    inputs = Inputs()
    inputs.csv(path)
    path.write_text("item_id,value\ni1,2\n")
    with pytest.raises(SchemeError, match="input changed"):
        inputs.verify()


def test_confidence_serialization_tolerance_preserves_saved_values(saved):
    from argfallacy.eval.analysis_inputs import (
        Inputs,
        read_evaluation,
        read_run,
        recover_confidence,
        scheme_registry,
    )

    path, _, items = saved
    file = path.parent / "first" / "verdicts.csv"
    rows = pd.read_csv(file, keep_default_na=False)
    lower = np.nextafter(0.7, 0.0)
    rows.loc[(rows.system == "zeroshot:ref") & (rows.item_id == "i001"), "confidence"] = repr(
        float(lower)
    )
    rows.to_csv(file, index=False)
    config, inputs = AnalysisConfig.load(path), Inputs()
    registry = scheme_registry(inputs)
    source = read_evaluation("first", config.evaluations["first"], config, items, registry, inputs)
    manifest, version, summary = read_run(
        config.parsed_runs["parsed"], "parsed", items, registry, inputs
    )
    recover_confidence([source], "parsed", manifest, version, summary)
    normalized = source.systems["zeroshot:ref"].rows
    assert normalized.loc["i001", "confidence"] == lower
    assert normalized.loc["i001", "p_label"] == 0.7
    assert normalized.loc["i001", "confidence_source"] == "saved verdicts.csv"
    normalized.loc["i001", "confidence"] = 0.4
    with pytest.raises(SchemeError, match="confidence disagreement"):
        recover_confidence([source], "parsed", manifest, version, summary)
