"""Saved D+Q and exact conformal adapters, using synthetic saved tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from argfallacy.aggregate import a0
from argfallacy.aggregate.interface import decide
from argfallacy.client.run import schemes_hash
from argfallacy.eval.analysis import analyze
from argfallacy.eval.analysis_bootstrap import Bootstrap
from argfallacy.eval.analysis_conformal import conformal_tables, export_conformal
from argfallacy.eval.analysis_dq import DQ_SCHEMAS, integrate_dq
from argfallacy.eval.analysis_inputs import (
    AnalysisConfig,
    Inputs,
    read_evaluation,
    read_run,
    recover_confidence,
    scheme_registry,
)
from argfallacy.eval.analysis_tables import comparisons
from argfallacy.eval.evaluate import Scorer
from argfallacy.eval.uncertainty import conformal_sets, conformal_summary
from argfallacy.paths import SCHEMES_ROOT
from argfallacy.schemes import SchemeError, load_all
from test_analysis import saved as saved_fixture

saved = saved_fixture

GOOD = "good_argumentation"


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def dq_saved(saved):
    path, raw, items = saved
    d = path.parent / "second"
    manifest = json.loads((d / "manifest.json").read_text())
    sources, runs, predictions, main = {}, {}, [], []
    for role, version, stage, base in [
        ("dq", "1.1", "dq", "DQ"),
        ("dq_schemes_2", "2.0", "dq", "DQ"),
        ("zeroshot_schemes_2", "2.0", "zeroshot", "zeroshot"),
        ("zeroshot_rerun", "1.1", "zeroshot", "zeroshot.rerun"),
    ]:
        run_id = "synthetic_" + role
        folder = "v1" if version == "1.1" else "v2"
        schemes = load_all(SCHEMES_ROOT / folder)
        directory = path.parent / run_id
        directory.mkdir()
        raw["parsed_runs"][run_id] = run_id
        runs[role] = [run_id]
        suffix = "@2.0" if version == "2.0" else ""
        name = f"{base}:ref{suffix}"
        names = [name] + ([f"DQ.path:ref{suffix}", f"DQ.soft:ref{suffix}"] if stage == "dq" else [])
        rows = []
        for item in items.itertuples():
            common = dict(
                item_id=item.item_id,
                model="ref",
                stage=stage,
                scheme_condition="gold",
                scheme=item.gold_scheme,
            )
            rows.append(
                {
                    **common,
                    "cq_id": "",
                    "answer": GOOD,
                    "p_label": 0.834567891234,
                    "p_label_lower_bound": True,
                }
            )
            own = {cq: {"answer": "yes"} for cq in schemes[item.gold_scheme].cqs}
            if stage == "dq":
                rows.extend(
                    {
                        **common,
                        "cq_id": cq,
                        "answer": "yes",
                        "p_logprob_yes": 0.7,
                        "p_logprob_no": 0.3,
                        "p_logprob": 0.7,
                    }
                    for cq in own
                )
            hard = a0(schemes[item.gold_scheme], own)
            for n in names:
                predictions.append(
                    dict(
                        system=n,
                        item_id=item.item_id,
                        fold=manifest["folds"]["assignment"][item.item_id],
                        verdict=hard.verdict if n.startswith("DQ.path") else GOOD,
                        confidence=hard.confidence
                        if n.startswith("DQ.path")
                        else 0.634567891234
                        if n.startswith("DQ.soft")
                        else 0.834567891234,
                        abstention=0.0,
                        explanation="synthetic",
                    )
                )
        for n in names:
            sources[n] = run_id
            main.append(dict(system=n, schemes=version, role="comparison", n_items=len(items)))
        pd.DataFrame(rows).to_csv(directory / "summary.csv", index=False)
        write_json(
            directory / "manifest.json",
            dict(
                run_id=run_id,
                schemes_version=dict(
                    tag=version, folder=folder, content_sha256=schemes_hash(SCHEMES_ROOT / folder)
                ),
                prompt_versions=[stage + "_v1"],
                models={"ref": {"samples": 1}},
                config={"stages": [stage], "models": ["ref"]},
            ),
        )
    sources["C2.dq"] = dict(
        answers="parsed", fit_answers="parsed", zeroshot="parsed", dq="synthetic_dq"
    )
    main.append(dict(system="C2.dq", schemes="1.1", role="exploratory", n_items=len(items)))
    predictions.extend(
        dict(
            system="C2.dq",
            item_id=i,
            fold=manifest["folds"]["assignment"][i],
            verdict=GOOD,
            confidence=0.4123456789,
            abstention=0.0,
            explanation="routing",
        )
        for i in items.item_id
    )
    manifest["dq"] = dict(
        runs=runs,
        systems=sources,
        items=len(items),
        folds=manifest["folds"]["assignment"],
        family=dict(name="H_DQ", m=1, level=0.95),
    )
    write_json(d / "manifest.json", manifest)
    for name, columns in DQ_SCHEMAS.items():
        pd.DataFrame(columns=columns).to_csv(d / name, index=False)
    pd.DataFrame(predictions).to_csv(d / "dq_verdicts.csv", index=False)
    prediction_frame = pd.DataFrame(predictions)
    for row in main:
        chosen = prediction_frame[prediction_frame.system == row["system"]].verdict.tolist()
        point = Scorer(items.gold_fallacy.tolist(), np.ones((1, len(items)))).metrics(chosen)
        row["description"] = "synthetic saved system"
        for metric, value in point.items():
            for suffix in ("", "_low", "_high"):
                row[metric + suffix] = value[0]
    pd.DataFrame(main).to_csv(d / "dq_main_table.csv", index=False, float_format="%.4f")
    pd.DataFrame(
        [
            dict(
                hypothesis="H_DQ",
                component="DQ",
                system="DQ:ref@2.0",
                reference="zeroshot:ref@2.0",
                level=0.95,
                computed=True,
                outcome="no measurable difference",
                reason="",
                reading="not applicable",
                secondary_difference=0.0,
                improvement=False,
                macro_f1_difference=0.0,
                macro_f1_low=-0.1,
                macro_f1_high=0.1,
                accuracy_difference=0.0,
                accuracy_low=-0.1,
                accuracy_high=0.1,
                balanced_accuracy_difference=0.0,
                balanced_accuracy_low=-0.1,
                balanced_accuracy_high=0.1,
            )
        ]
    ).to_csv(d / "dq_deciding.csv", index=False, float_format="%.4f")
    raw["dq"] = "second"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path, raw, items


def load_saved(fixture):
    path, raw, items = fixture
    config, inputs = AnalysisConfig.load(path), Inputs()
    registry = scheme_registry(inputs)
    evaluations = [
        read_evaluation(n, s, config, items, registry, inputs)
        for n, s in config.evaluations.items()
    ]
    return config, items, registry, inputs, evaluations


def test_dq_merges_only_own_systems_and_exact_confidence_sources(dq_saved):
    config, items, registry, inputs, evaluations = load_saved(dq_saved)
    original = set(evaluations[1].systems)
    imported, status = integrate_dq(config, evaluations, items, registry, inputs)
    assert status["status"] == "computed" and len(imported) == 14
    second = evaluations[1]
    assert set(second.systems) == original | set(second.manifest["dq"]["systems"])
    for run_id, path in config.parsed_runs.items():
        m, version, summary = read_run(path, run_id, items, registry, inputs)
        recover_confidence(evaluations, run_id, m, version, summary)
    declared = second.systems["DQ:ref@2.0"].rows
    soft = second.systems["DQ.soft:ref@2.0"].rows
    assert declared.p_label_lower_bound.astype(bool).all()
    assert declared.confidence.eq(0.834567891234).all()
    assert soft.confidence.eq(0.634567891234).all() and soft.p_label.isna().all()
    assert second.systems["C2.dq"].rows.confidence.eq(0.4123456789).all()
    assert second.systems["C2.dq"].rows.p_label.isna().all()
    assert second.systems["DQ.path:ref@2.0"].rows.confidence.isna().all()
    assert (
        second.systems["zeroshot.rerun:ref"]
        .rows.confidence_run.eq("synthetic_zeroshot_rerun")
        .all()
    )
    assert "source_key" in imported["dq_deciding.csv"]
    rows = comparisons(evaluations, items, Bootstrap(17, 7, 2))
    h = rows[(rows.hypothesis == "H_DQ") & (rows.subset_kind == "total")].iloc[0]
    assert h.level == 0.95 and h.family_m == 1 and h.family == "H_DQ"
    assert second.manifest["m"] == 4


@pytest.mark.parametrize(
    "damage,match",
    [
        ("duplicate", "duplicate"),
        ("fold", "fold"),
        ("version", "version"),
        ("confidence", "confidence"),
        ("mapping", "mapping"),
        ("schema", "columns"),
    ],
)
def test_dq_bad_saved_contract_fails(dq_saved, damage, match):
    path, raw, items = dq_saved
    d = path.parent / "second"
    file = d / "dq_verdicts.csv"
    rows = pd.read_csv(file, keep_default_na=False)
    if damage == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]])
    elif damage == "fold":
        rows.loc[0, "fold"] = 99
    elif damage == "confidence":
        rows.loc[0, "confidence"] = "1.1"
    elif damage == "schema":
        rows = rows.drop(columns="verdict")
    elif damage in ("version", "mapping"):
        m = json.loads((d / "manifest.json").read_text())
        m["dq"]["systems"]["DQ:ref@2.0"] = "synthetic_dq" if damage == "version" else "unregistered"
        write_json(d / "manifest.json", m)
    rows.to_csv(file, index=False)
    config, items, registry, inputs, evaluations = load_saved((path, raw, items))
    with pytest.raises(SchemeError, match=match):
        integrate_dq(config, evaluations, items, registry, inputs)


def test_null_dq_reads_no_dq_only_files(saved, monkeypatch):
    config, items, registry, inputs, evaluations = load_saved(saved)
    original = Path.open

    def guard(path, *args, **kwargs):
        assert not path.name.startswith("dq_")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    tables, status = integrate_dq(config, evaluations, items, registry, inputs)
    assert status["status"] == "pending" and all(f.empty for f in tables.values())


def conformal_source(saved):
    config, items, registry, inputs, evaluations = load_saved(saved)
    evaluation = evaluations[1]
    system = evaluation.systems["A2"]
    aggs = {
        i: decide({GOOD: 0.612345678901234, "weak_analogy": 0.187654321098766}, 0.2, "invented")
        for i in items.item_id
    }
    # Explicit zero-terminal-mass fallback, and a threshold near a decimal boundary.
    aggs[items.item_id.iloc[0]] = decide({}, 1.0, "invented zero mass")
    system.rows["verdict"] = [aggs[i].verdict for i in items.item_id]
    system.rows["confidence"] = [aggs[i].confidence for i in items.item_id]
    predictions = pd.read_csv(evaluation.path / "verdicts.csv", keep_default_na=False)
    own = predictions.system.eq("A2")
    predictions.loc[own, "verdict"] = [
        aggs[i].verdict or "" for i in predictions.loc[own, "item_id"]
    ]
    predictions["confidence"] = predictions.confidence.astype(object)
    predictions.loc[own, "confidence"] = [
        aggs[i].confidence if aggs[i].confidence is not None else ""
        for i in predictions.loc[own, "item_id"]
    ]
    predictions.to_csv(evaluation.path / "verdicts.csv", index=False)
    # Use analogy throughout so the fixture's terminal distributions are compatible.
    items = items.assign(gold_scheme="analogy", gold_fallacy=GOOD)
    terminals = {i: sorted(load_all()["analogy"].terminals) for i in items.item_id}
    gold = items.set_index("item_id").gold_fallacy.to_dict()
    folds = evaluation.manifest["folds"]["assignment"]
    sets, thresholds = conformal_sets(aggs, terminals, gold, folds)
    groups = {
        "scheme": items.set_index("item_id").gold_scheme.to_dict(),
        "gold origin": dict(
            zip(items.item_id, ["fallacy-corpus"] * 24 + ["scheme-corpus"] * 16, strict=True)
        ),
    }
    # Use the implementation's origin identifiers.
    from argfallacy.eval.round2 import ORIGINS

    groups["gold origin"] = dict(
        zip(items.item_id, items.source.str.lower().map(ORIGINS), strict=True)
    )
    pd.DataFrame(conformal_summary("A2", sets, gold, groups)).to_csv(
        evaluation.path / "conformal.csv", index=False, float_format="%.4f"
    )
    evaluation.manifest["conformal_alpha"] = 0.1
    write_json(evaluation.path / "manifest.json", evaluation.manifest)
    return config, items, registry, Inputs(), evaluations, aggs, sets, thresholds


def test_conformal_sets_exact_export_and_aggregate_reproduction(saved):
    config, items, registry, inputs, evaluations, aggs, sets, thresholds = conformal_source(saved)
    evaluation = evaluations[1]
    out = config.path.parent / "exact_export"
    export_conformal(evaluation, items, registry, {"A2": aggs}, out)
    config.evaluations["second"]["conformal_records"] = out
    tables, status = conformal_tables(config, evaluations, items, registry, None, inputs)
    rows = tables["conformal_items.csv"]
    assert status["second/A2"]["status"] == "computed"
    assert {r.item_id: tuple(json.loads(r.terminal_ids)) for r in rows.itertuples()} == sets
    assert rows.iloc[0]["size"] == len(load_all()["analogy"].terminals)
    assert any(q != float(f"{q:.4f}") for q in thresholds.values())
    assert (
        rows.threshold.iloc[0]
        == thresholds[evaluation.manifest["folds"]["assignment"][rows.item_id.iloc[0]]]
    )
    saved_aggregate = pd.read_csv(evaluation.path / "conformal.csv").sort_values(["kind", "subset"])
    aggregate = tables["conformal.csv"].sort_values(["kind", "subset"])
    for metric in ("coverage", "mean_size", "singleton_share"):
        np.testing.assert_allclose(aggregate[metric], saved_aggregate[metric], atol=0.00005)
    assert (
        json.loads((out / "manifest.json").read_text())["resource_measurements"][
            "peak_process_memory_bytes"
        ]
        > 0
    )


def test_rounded_aggregates_cannot_supply_sets_and_export_is_opt_in(saved):
    config, items, registry, inputs, evaluations, _, _, _ = conformal_source(saved)
    tables, status = conformal_tables(config, evaluations, items, registry, None, inputs)
    assert tables["conformal_items.csv"].empty and tables["conformal.csv"].empty
    assert status["second/A2"]["status"] == "pending"
    assert "export" in status["second/A2"]["reason"]
    assert not (config.path.parent / "exact_export").exists()


def test_dq_offline_command_imports_summaries_without_fitting(dq_saved, monkeypatch):
    import argfallacy.eval.dq as dq
    import argfallacy.eval.round2 as round2

    monkeypatch.setattr(dq, "evaluate_dq", lambda *a, **k: pytest.fail("evaluation"))
    monkeypatch.setattr(round2, "evaluate_round2", lambda *a, **k: pytest.fail("fitting"))
    path, _, _ = dq_saved
    out = analyze(AnalysisConfig.load(path))
    main = pd.read_csv(out / "main_table.csv")
    assert "DQ:ref@2.0" in set(main.system)
    assert (out / "dq_routing_curve.csv").is_file()
    assert (out / "human_direct.csv").is_file() and (out / "conformal_items.csv").is_file()
    record = json.loads((out / "manifest.json").read_text())
    assert record["parts"]["dq"]["status"] == "computed"
    assert record["parts"]["direct_verdict"]["status"] == "pending"


def test_saved_summary_values_and_rounding_are_imported_unchanged(dq_saved):
    config, items, registry, inputs, evaluations = load_saved(dq_saved)
    tables, _ = integrate_dq(config, evaluations, items, registry, inputs)
    for name, imported in tables.items():
        source = pd.read_csv(config.dq / name, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(imported[source.columns], source)
        assert imported.source_key.is_unique
        assert imported.source_sha256.eq(inputs.files[(config.dq / name).resolve()]).all()


def test_inconsistent_saved_main_summary_fails(dq_saved):
    path, _, _ = dq_saved
    file = path.parent / "second" / "dq_main_table.csv"
    frame = pd.read_csv(file)
    frame.loc[0, "accuracy"] = 0.1234
    frame.to_csv(file, index=False)
    config, items, registry, inputs, evaluations = load_saved(dq_saved)
    with pytest.raises(SchemeError, match="saved main summary"):
        integrate_dq(config, evaluations, items, registry, inputs)


def test_dq_missing_confidence_and_invalid_verdict_are_preserved(dq_saved):
    from argfallacy.parse import INVALID

    config, items, registry, inputs, evaluations = load_saved(dq_saved)
    integrate_dq(config, evaluations, items, registry, inputs)
    system = evaluations[1].systems["DQ:ref"]
    item = items.item_id.iloc[0]
    system.rows.loc[item, "verdict"] = None
    system.rows.loc[item, "confidence"] = np.nan
    m, v, summary = read_run(
        config.parsed_runs["synthetic_dq"], "synthetic_dq", items, registry, inputs
    )
    summary["p_label"] = summary.p_label.astype(object)
    chosen = (summary.item_id == item) & summary.cq_id.eq("")
    summary.loc[chosen, "answer"], summary.loc[chosen, "p_label"] = INVALID, ""
    recover_confidence(evaluations, "synthetic_dq", m, v, summary)
    assert pd.isna(system.rows.loc[item, "verdict"]) and pd.isna(
        system.rows.loc[item, "confidence"]
    )


def test_dq_cq_version_targets_and_ssoft_dedicated_observed_population(dq_saved):
    from argfallacy.eval.analysis_cq import cq_tables
    from test_analysis_cq_human import notes_for

    config, items, registry, inputs, evaluations = load_saved(dq_saved)
    integrate_dq(config, evaluations, items, registry, inputs)
    directory = config.path.parent / "observed_ssoft"
    directory.mkdir()
    columns = ["item_id", "model", "stage", "scheme_condition", "scheme", "cq_id", "answer"]
    pd.DataFrame(columns=columns).to_csv(directory / "summary.csv", index=False)
    write_json(
        directory / "manifest.json",
        dict(
            run_id="observed_ssoft",
            models={"ref": {"samples": 1}},
            schemes_version=dict(tag="2.0"),
            config={"stages": ["ssoft"], "history": True},
        ),
    )
    # Two histories of one observed CQ of one item: every other CQ is unvisited.
    pd.DataFrame(
        [
            dict(
                item_id=items.item_id.iloc[0],
                model="ref",
                stage="ssoft",
                scheme_condition="gold",
                scheme="analogy",
                cq_id="CQ1",
                path=p,
                answer="no",
                p_logprob_no=0.7,
                p_logprob_yes=0.3,
            )
            for p in ("first", "second")
        ]
    ).to_csv(directory / "ssoft.csv", index=False)
    config.parsed_runs["observed_ssoft"] = directory
    evaluations[1].manifest["runs"]["ssoft"] = ["observed_ssoft"]
    tables, _ = cq_tables(
        config,
        evaluations,
        items,
        registry,
        notes_for(items, load_all()),
        Bootstrap(17, 5, 2),
        inputs,
    )
    frame = tables["cq_level.csv"]
    revised = frame[frame.system.eq("DQ:ref@2.0")]
    assert revised.annotated_items.eq(0).all() and revised.brier.isna().all()
    original = frame[frame.system.eq("DQ:ref")]
    assert original.annotated_items.max() > 0
    nodes = frame[
        (frame.system == "ssoft:ref") & (frame.origin == "total") & (frame.scheme == "all")
    ]
    assert nodes.n_items.eq(1).all() and nodes.n_cq_pairs.eq(1).all()
    assert nodes.n_observations.eq(2).all()
    assert (directory / "ssoft.csv").resolve() in inputs.files


def test_conformal_saved_distributions_and_explicit_bad_export(saved):
    config, items, registry, inputs, evaluations, aggs, expected, _ = conformal_source(saved)
    system = evaluations[1].systems["A2"]
    system.rows["distribution"] = [json.dumps(aggs[i].distribution) for i in items.item_id]
    tables, _ = conformal_tables(config, evaluations, items, registry, None, inputs)
    assert {
        r.item_id: tuple(json.loads(r.terminal_ids))
        for r in tables["conformal_items.csv"].itertuples()
    } == expected
    directory = config.path.parent / "bad_export"
    directory.mkdir()
    write_json(directory / "manifest.json", {"format": "unsupported"})
    config.evaluations["second"]["conformal_records"] = directory
    with pytest.raises(SchemeError, match="unsupported"):
        conformal_tables(config, evaluations, items, registry, None, inputs)


def test_conformal_cannot_export_labels_instead_of_distributions(saved):
    config, items, registry, _, evaluations, aggs, _, _ = conformal_source(saved)
    wrong = {i: decide({GOOD: 1.0}, 0.0, "invented one-hot") for i in items.item_id}
    with pytest.raises(SchemeError, match="saved"):
        export_conformal(evaluations[1], items, registry, {"A2": wrong}, config.path.parent / "bad")
    assert not (config.path.parent / "bad").exists()
