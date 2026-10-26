"""Frozen CQ population contracts and item-bootstrap human agreement."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from argfallacy.eval.analysis import AnalysisConfig, analyze
from argfallacy.eval.analysis_bootstrap import Bootstrap, contribution_ratios
from argfallacy.eval.analysis_cq import (
    C1_POPULATION,
    P7_POPULATION,
    combined_predictions,
    cq_differences,
    cq_scores,
    cq_tables,
    entry_predictions,
    p7_predictions,
)
from argfallacy.eval.analysis_human import (
    agreement_bootstrap,
    historical_ratings,
    human_tables,
    reference_support,
)
from argfallacy.eval.analysis_inputs import Inputs, SavedEvaluation
from argfallacy.eval.c1 import cq_evaluation, cq_labels, entry_answers
from argfallacy.eval.cq import agreement, human_answers
from argfallacy.paths import REPO_ROOT
from argfallacy.schemes import SchemeError, load_all
from test_c1 import example_data

pytest_plugins = ["test_analysis"]

GOOD, WEAK = "good_argumentation", "weak_analogy"


def notes_for(items, schemes):
    notes = []
    for index, item in enumerate(items.itertuples()):
        scheme = schemes[item.gold_scheme]
        values = {
            "scheme": item.gold_scheme if index else "none",
            **{cq: ("yes" if index % 2 else "no") for cq in scheme.cqs},
        }
        for field, value in values.items():
            notes.append(
                {
                    "item_id": item.item_id,
                    "annotator": "dumitru",
                    "sheet": item.gold_scheme,
                    "row": index + 2,
                    "field": field,
                    "value": value,
                }
            )
        for person, value in [("enrico", GOOD), ("eleni", WEAK), ("dumitru", GOOD)]:
            notes.append(
                {
                    "item_id": item.item_id,
                    "annotator": person,
                    "sheet": item.gold_scheme,
                    "row": index + 2,
                    "field": "verdict",
                    "value": value,
                }
            )
        if index % 3:
            notes.append(
                {
                    "item_id": item.item_id,
                    "annotator": "enrico",
                    "sheet": item.gold_scheme,
                    "row": index + 2,
                    "field": "difficulty",
                    "value": str(index % 5 + 1),
                }
            )
    return pd.DataFrame(notes)


def test_item_clustering_is_identical_to_legacy_c1_and_individual_cqs_sum():
    data, labels, summary, manifest = example_data(24, full_scheme=True)
    items = data.items.assign(source=["LOGIC"] * 20 + ["NLAS"] * 4)
    binary = labels.assign(human=np.where(labels.target == 1, "yes", "no"))
    prediction = entry_answers(data).query("system == 'entry:e0'")
    predictions = entry_answers(data)
    boot = Bootstrap(21, 57, 7)
    draws = np.concatenate(list(boot.indices(len(items))))
    old = cq_evaluation(data, labels, predictions, indices=draws)
    expected = old.level[(old.level.system == "entry:e0") & (old.level.scheme == "all")].iloc[0]
    for batch in (1, 7, 57):
        records = cq_scores(prediction, binary, items, Bootstrap(21, 57, batch))
        pool = next(r for r in records if r["origin"] == "total" and r["scheme"] == "all")
        for metric in ("agreement", "brier"):
            assert pool[metric] == pytest.approx(expected[metric])
            assert pool[f"{metric}_low"] == pytest.approx(expected[f"{metric}_low"])
            assert pool[f"{metric}_high"] == pytest.approx(expected[f"{metric}_high"])
        cqs = [r for r in records if r["origin"] == "total" and r["cq_id"] != "all"]
        assert sum(r["n_cq_pairs"] for r in cqs) == pool["n_cq_pairs"]
        assert pool["n_items"] == 24 and pool["n_cq_pairs"] == 24 * len(data.schemes["analogy"].cqs)
        small = next(r for r in records if r["origin"] == "scheme corpora" and r["scheme"] == "all")
        assert small["annotated_items"] == 4 and small["hard_agreement_withheld"]
        assert np.isnan(small["agreement"]) and np.isnan(small["agreement_low"])
    # A million CQ contributions on one item remain an item problem, never B by pairs.
    positions = np.zeros(20000, int)
    observed, spread = contribution_ratios(positions, np.ones(20000), np.ones(20000), 24, boot)
    assert observed == 1
    assert set(spread[np.isfinite(spread)]) == {1}


def test_population_difference_outside_answers_orientation_and_common_probability_pairs():
    data, _, summary, manifest = example_data(24, full_scheme=True)
    items = data.items.assign(source="LOGIC")
    notes = notes_for(items, data.schemes)
    binary = cq_labels(notes, items, data.schemes).rename(columns={"answer": "human"})
    broad = human_answers(notes, items, data.schemes)
    broad["target"] = broad.human.eq("yes").astype(float)
    assert binary.item_id.nunique() == 23 and broad.item_id.nunique() == 24
    summary["p_label_lower_bound"] = "False"
    summary.loc[summary.model == "e0", "answer"] = "cannot_be_determined"
    summary.loc[summary.model == "e0", "p_logprob_yes"] = 0.2
    summary.loc[summary.model == "e0", "p_logprob_no"] = 0.3
    summary.loc[summary.model == "e0", "p_logprob_cannot_be_determined"] = 0.5
    frame = entry_predictions(summary, items, data.schemes, "e0", manifest["models"]["e0"])
    assert np.allclose(frame.p_first, 0.4)  # C1 conditional split on arcs
    p7 = p7_predictions(frame, data.schemes, "p_logprob")
    assert np.allclose(p7.p_first, 0.2)  # P7's unconditional yes probability
    row = cq_scores(frame, binary, items, Bootstrap(1, 31, 7))[0]
    assert row["population"] == C1_POPULATION and row["agreement"] == 0
    p7row = cq_scores(p7, broad, items, Bootstrap(1, 31, 7), P7_POPULATION)[0]
    assert p7row["annotated_items"] == 24
    verbal = p7_predictions(frame, data.schemes, "p_verbal")
    np.testing.assert_equal(verbal.p_first.to_numpy(), frame.p_verbal.to_numpy())
    left = frame.copy()
    right = frame.iloc[::2].copy()
    left.loc[left.index[0], "p_first"] = np.nan
    difference = cq_differences(left, right, binary, items, Bootstrap(1, 31, 7))[0]
    common = right.merge(binary, on=["item_id", "scheme", "cq_id"])
    assert difference["n_cq_pairs"] == len(common)
    assert difference["brier_n"] == len(common)  # first item has no binary labels
    assert difference["brier_difference"] == 0


def test_saved_combined_hard_ties_outside_and_rewritten_version(tmp_path):
    schemes = load_all()
    items = pd.DataFrame({"item_id": ["a", "b"], "gold_scheme": "analogy"})
    summary = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "scheme": "analogy",
            "cq_id": "CQ1",
            "model": "ref",
            "stage": "stage2",
            "scheme_condition": "gold",
            "answer": "no",
        }
    )
    combined = summary[["item_id", "scheme", "cq_id"]].assign(
        p_first=0.5, outside=[0.5, 0.50001], fold=0, supervision="cq"
    )
    got = combined_predictions(combined, summary, "ref", items, schemes, {"a": 0, "b": 0})
    assert got.answer.tolist() == ["no", "cannot_be_determined"]
    mixed = combined.copy()
    mixed["fold"] = [0.0, ""]
    mixed["supervision"] = ["cq", "none"]
    mixed.to_csv(tmp_path / "combined.csv", index=False)
    read = pd.read_csv(tmp_path / "combined.csv", keep_default_na=False)
    restored = combined_predictions(read, summary, "ref", items, schemes, {"a": 0, "b": 0})
    assert restored.answer.tolist() == got.answer.tolist()
    combined.loc[0, "fold"] = 1
    with pytest.raises(SchemeError, match="fold"):
        combined_predictions(combined, summary, "ref", items, schemes, {"a": 0, "b": 0})


def test_ssoft_counts_only_observed_paths_and_never_uses_frozen_11_targets(tmp_path):
    items = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "gold_scheme": "analogy",
            "gold_fallacy": GOOD,
            "source": ["LOGIC", "NLAS"],
        }
    )
    run = tmp_path / "ss"
    run.mkdir()
    manifest = {
        "schemes_version": {"tag": "2.0"},
        "models": {"ref": {"samples": 1, "reasoning": {"enable_thinking": False}}},
        "config": {"stages": ["ssoft"], "history": True},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    columns = ["item_id", "model", "stage", "scheme_condition", "scheme", "cq_id", "answer"]
    pd.DataFrame(columns=columns).to_csv(run / "summary.csv", index=False)
    rows = pd.DataFrame(
        {
            "item_id": "a",
            "model": "ref",
            "stage": "ssoft",
            "scheme_condition": "gold",
            "scheme": "analogy",
            "cq_id": ["CQ1", "CQ3", "CQ3"],
            "path": ["", "CQ1=yes", "CQ1=no"],
            "answer": "yes",
            "p_logprob_yes": 0.8,
            "p_logprob_no": 0.2,
            "p_logprob_cannot_be_determined": 0.0,
            "p_logprob_na": 0.0,
        }
    )
    rows.to_csv(run / "ssoft.csv", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "items": "items.csv",
                "seed": 17,
                "resamples": 23,
                "evaluations": {"second": {"path": ".", "round": 2}},
                "parsed_runs": {"ss": "ss"},
            }
        )
    )
    evaluation = SavedEvaluation(
        "second", 2, tmp_path, {"runs": {"ssoft": ["ss"]}, "reference_model": "ref"}, {}
    )
    tables, status = cq_tables(
        AnalysisConfig.load(config_path),
        [evaluation],
        items,
        {"1.1": REPO_ROOT / "schemes/v1", "2.0": REPO_ROOT / "schemes/v2"},
        notes_for(items, load_all()),
        Bootstrap(17, 23, 3),
        Inputs(),
    )
    level = tables["cq_level.csv"]
    pool = level[
        (level.origin == "total") & (level.scheme == "all") & (level.probability == "p_logprob")
    ].iloc[0]
    assert pool.n_items == 1 and pool.n_cq_pairs == 2 and pool.n_path_occurrences == 3
    assert set(level.scheme_version) == {"2.0"}
    assert level.annotated_items.eq(0).all() and level.brier.isna().all()
    assert "CQ2" not in set(level.cq_id)  # unvisited, not a missing prediction or error
    assert "pending" in status["second"]["rewritten_cq_targets"]


def test_human_nominal_alpha_matches_explicit_item_draws_and_preserves_missing():
    units = [["yes", "yes", "no"], ["no", "no", None], ["other", None, None], [None] * 3]
    mask = np.array([True, True, True, False])
    expected = None
    for batch in (1, 9, 63):
        boot = Bootstrap(7, 63, batch)
        observed, spread = agreement_bootstrap(units, mask, boot)
        assert observed == agreement(units[:3])
        for k, draw in enumerate(np.concatenate(list(boot.indices(4)))):
            explicit = agreement([units[i] for i in draw if mask[i]])
            for metric in ("alpha", "raw_agreement"):
                np.testing.assert_allclose(spread[metric][k], explicit[metric], equal_nan=True)
        if expected is not None:
            for metric in spread:
                np.testing.assert_equal(spread[metric], expected[metric])
        expected = spread


def test_historical_sheets_are_distinct_from_gold_scheme_paths():
    schemes = load_all()
    items = pd.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "gold_scheme": "analogy",
            "source": ["LOGIC", "NLAS", "LOGIC"],
            "gold_fallacy": GOOD,
        }
    )
    notes = notes_for(items, schemes)
    # A historical judgement may exist even when its gold-scheme path is ineligible.
    extra = notes.query("item_id == 'b' and annotator == 'dumitru' and field == 'verdict'").copy()
    extra["sheet"], extra["value"] = "another sheet", WEAK
    other = pd.DataFrame(
        [
            {
                "item_id": "b",
                "annotator": "dumitru",
                "sheet": "another sheet",
                "row": 3,
                "field": field,
                "value": value,
            }
            for field, value in {
                "scheme": "expert_opinion",
                **{cq: "no" for cq in schemes["expert_opinion"].cqs},
            }.items()
        ]
    )
    notes = pd.concat([notes, extra, other, notes.iloc[[0]]], ignore_index=True)
    ratings, conflicts, _ = historical_ratings(notes, items, schemes)
    assert conflicts.loc["b", "dumitru"] and pd.isna(ratings.loc["b", "dumitru"])
    tables, status = human_tables(notes, [], items, schemes, Bootstrap(7, 31, 3))
    path = tables["human_path.csv"]
    gold = path[(path.reference == "gold") & (path.origin == "total")].iloc[0]
    assert gold.annotated_items == 2 and gold.test_items == 3
    assert gold.n_classes == 1 and json.loads(gold.supported_classes) == [GOOD]
    assert gold.good_reference_support == gold.usable_items
    assert not gold.good_recall_undefined
    zero = reference_support([WEAK], [None], np.ones(1, bool))
    assert zero["good_reference_support"] == 0 and zero["good_prediction_count"] == 0
    assert zero["good_recall_undefined"] and zero["good_precision_undefined"]
    historical = path[(path.reference == "historical:dumitru") & (path.origin == "total")].iloc[0]
    assert historical.n_items < gold.usable_items
    assert status["second_human_cq"] and status["blind_repeats"]
    agreement_rows = tables["human_agreement.csv"]
    pending = agreement_rows[agreement_rows.status == "pending"]
    assert pending.alpha.isna().all() and len(pending) == 6
    conflicting = notes[(notes.item_id == "c") & (notes.field == "CQ1")].copy()
    conflicting["value"] = "no"  # c was no already: identical repeats are harmless
    human_answers(pd.concat([notes, conflicting]), items, schemes)
    conflicting["value"] = "yes"
    with pytest.raises(SchemeError, match="Conflicting human CQ"):
        human_answers(pd.concat([notes, conflicting]), items, schemes)


def test_complete_offline_command_writes_new_tables_and_never_opens_reserved_inputs(
    saved, monkeypatch
):
    path, config, items = saved
    annotations = notes_for(items, load_all())
    annotations.to_csv(path.parent / "frozen.csv", index=False)
    config["annotations"] = "frozen.csv"
    config["direct_verdict"] = {
        "answers": "forbidden_final.csv",
        "samples": "forbidden_samples.csv",
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    original = Path.open

    def guarded(self, *args, **kwargs):
        assert not self.name.startswith("forbidden")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    out = analyze(AnalysisConfig.load(path))
    for name in (
        "cq_level",
        "cq_level_differences",
        "human_agreement",
        "human_path",
        "abstention",
        "abstention_curves",
        "abstention_thresholds",
        "abstention_differences",
        "abstention_composition",
        "difficulty",
    ):
        assert (out / f"{name}.csv").is_file()
    record = json.loads((out / "manifest.json").read_text())
    assert record["parts"]["historical_human"]["status"] == "computed"
    assert record["parts"]["cq_populations"]["first"]["p7_annotated_items"] == 40
    assert record["parts"]["cq_populations"]["first"]["binary_items"] == 39
    cq = pd.read_csv(out / "cq_level.csv")
    assert {"CQ1", "CQ2", "CQ3", "CQ4"} <= set(cq.cq_id)
    thresholds = pd.read_csv(out / "abstention_thresholds.csv")
    assert thresholds.threshold.isin([np.inf, 0.5, 0.7, 0.0]).all()
    assert record["parts"]["direct_verdict"]["status"] == "pending"
    assert all(not Path(r["path"]).name.startswith("forbidden") for r in record["inputs"])
    config["bootstrap_batch_size"] = 1
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    again = analyze(AnalysisConfig.load(path), path.parent / "batch_one")
    for file in out.glob("*.csv"):
        assert file.read_bytes() == (again / file.name).read_bytes(), file.name
