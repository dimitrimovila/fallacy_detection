"""D+Q evaluation on hand-written runs, paired draws and protected round-two CSVs."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from argfallacy.aggregate import a0, a2_pruned, load_burden, routes
from argfallacy.client import schemes_hash
from argfallacy.eval.evaluate import evaluate, resample_indices, scores
from argfallacy.paths import REPO_ROOT
from argfallacy.prompts import render_zeroshot
from argfallacy.schemes import SchemeError, load_all
from test_round2 import REFERENCE, _cq_row, _models, _write, write_runs

RUN_KEYS = ("zeroshot_schemes_2", "zeroshot_rerun", "dq", "dq_schemes_2")
LEGACY_FILES = (
    "deciding.csv",
    "deciding_breakdowns.csv",
    "exploratory.csv",
    "main_table.csv",
    "verdicts.csv",
    "cq_combined.csv",
    "cq_level.csv",
    "cq_level_differences.csv",
    "human_path.csv",
    "routing_curve.csv",
    "ssoft_changes.csv",
    "abstention.csv",
    "abstention_curves.csv",
    "abstention_composition.csv",
    "conformal.csv",
    "round2_choices.csv",
)
DQ_FILES = (
    "dq_verdicts.csv",
    "dq_main_table.csv",
    "dq_deciding.csv",
    "dq_deciding_breakdowns.csv",
    "dq_exploratory.csv",
    "dq_consistency.csv",
    "dq_departures.csv",
    "dq_cq_level.csv",
    "dq_cq_changes.csv",
    "dq_replication.csv",
    "dq_abstention.csv",
    "dq_abstention_curves.csv",
)


@pytest.fixture(scope="module", params=["v1", "v2"])
def diagrams(request):
    folder = request.param
    schemes = load_all(REPO_ROOT / "schemes" / folder)
    routing = {s: routes(schemes[s], table) for s, table in load_burden(folder, schemes).items()}
    return folder, schemes, routing


def test_traversals_equal_the_existing_aggregators(diagrams):
    from argfallacy.eval.dq import dq_traversals

    _, schemes, routing = diagrams
    for scheme in schemes.values():
        rows = {
            cq: _cq_row(
                "i",
                scheme.scheme_id,
                cq,
                "m",
                scheme.answer_space(cq)[0],
                scheme.answer_space(cq),
                0.77,
            )
            for cq in scheme.cq_order
        }
        for source in ("logprob", "sample"):
            path, soft = dq_traversals(scheme, rows, routing[scheme.scheme_id], source)
            assert path == a0(scheme, rows)
            expected = a2_pruned(scheme, rows, routing[scheme.scheme_id], source, 0.02)
            assert soft.distribution == pytest.approx(expected.distribution, abs=1e-12)
            assert soft.abstention == pytest.approx(expected.abstention, abs=1e-12)
            assert soft.pruned == pytest.approx(expected.pruned, abs=1e-12)
        assert dq_traversals(scheme, rows, routing[scheme.scheme_id], "verbal")[1] is None


def test_consistency_and_first_departure(diagrams):
    from argfallacy.eval.dq import consistency_case

    _, schemes, _ = diagrams
    for scheme in schemes.values():
        rows = {cq: {"answer": scheme.answer_space(cq)[0]} for cq in scheme.cq_order}
        path = a0(scheme, rows)
        matched = consistency_case(scheme, rows, path.verdict)
        assert matched["consistent"]
        assert not matched["stopped"]
        assert matched["departure_cq"] is None
        rows[scheme.start] = {"answer": "cannot_be_determined"}
        stopped = consistency_case(scheme, rows, path.verdict)
        assert not stopped["consistent"] and stopped["stopped"]
        assert stopped["departure_cq"] == scheme.start
        assert stopped["departure_answer"] == "cannot_be_determined"
        assert stopped["departure_reason"] == "stop"
        assert not consistency_case(scheme, rows, None)["consistent"]

    scheme = schemes["analogy"]
    rows = {cq: {"answer": "yes"} for cq in scheme.cq_order}
    rows["CQ1"] = {"answer": "no"}
    rows["CQ2"] = {"answer": "no"}
    declared = scheme.nodes["CQ1.1"]["no"].target
    result = consistency_case(scheme, rows, declared)
    assert result["departure_cq"] == "CQ1.1"
    assert result["departure_answer"] == "yes"
    assert result["departure_reason"] == "arc"


def test_primary_interval_and_reading_rule():
    from argfallacy.eval.dq import dq_decision

    scheme = load_all()["analogy"]
    labels = sorted(scheme.terminals)
    gold = [labels[i % len(labels)] for i in range(30)]
    draws = resample_indices(3, len(gold), 500, 1)
    wrong = [None] * len(gold)
    values = {
        "DQ:m@2.0": scores(gold, draws, gold),
        "zeroshot:m@2.0": scores(gold, draws, wrong),
        "DQ:m": scores(gold, draws, gold),
        "zeroshot.rerun:m": scores(gold, draws, wrong),
    }
    row = dq_decision(values, "m")
    spread = (
        values["DQ:m@2.0"].resampled["macro_f1"] - values["zeroshot:m@2.0"].resampled["macro_f1"]
    )
    assert row["level"] == 0.95
    assert (row["macro_f1_low"], row["macro_f1_high"]) == tuple(np.percentile(spread, [2.5, 97.5]))
    assert row["reading"] == "robust"
    values["DQ:m"] = values["zeroshot.rerun:m"]
    assert dq_decision(values, "m")["reading"] == "possibly tuning"
    values["DQ:m@2.0"] = values["zeroshot:m@2.0"]
    row = dq_decision(values, "m")
    assert row["macro_f1_low"] == row["macro_f1_high"] == 0
    assert row["outcome"] == "no measurable difference"
    assert row["reading"] == "not applicable"
    row = dq_decision({}, "m")
    assert row["outcome"] == "missing" and row["reason"]


def test_cq_scores_pair_items_before_resampling(monkeypatch):
    from argfallacy.eval import c1 as cq_module
    from argfallacy.eval.dq import dq_cq_evaluation

    scheme = load_all()["analogy"]
    items = pd.DataFrame({"item_id": ["a", "b"], "gold_scheme": [scheme.scheme_id] * 2})
    keys = [("a", "CQ1"), ("a", "CQ1.1"), ("b", "CQ1")]
    labels = pd.DataFrame(
        [
            {"item_id": item, "scheme": scheme.scheme_id, "cq_id": cq, "target": target}
            for (item, cq), target in zip(keys, [1, 0, 1], strict=True)
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "item_id": item,
                "scheme": scheme.scheme_id,
                "cq_id": cq,
                "answer": answer,
                "p_first": p,
                "system": system,
                "supervision": "none",
            }
            for system, hard, probs in [
                ("DQ:m", ["yes", "yes", "no"], [0.8, 0.7, 0.2]),
                ("entry:m", ["yes", "no", "yes"], [0.6, 0.4, 0.9]),
            ]
            for (item, cq), answer, p in zip(keys, hard, probs, strict=True)
        ]
    )
    draws = np.array([[0, 0], [1, 1], [0, 1]])
    original = np.zeros

    def guarded(shape, *args, **kwargs):
        if isinstance(shape, tuple) and shape[0] == len(draws) + 1:
            assert len(shape) == 2 and shape[1] <= len(items)
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", guarded)
    monkeypatch.setattr(cq_module, "generators", lambda *a: pytest.fail("new bootstrap draws"))
    table = dq_cq_evaluation(items, {scheme.scheme_id: scheme}, labels, predictions, "m", draws)
    own = table[(table.kind == "level") & (table.system == "DQ:m") & (table.scheme == "all")].iloc[
        0
    ]
    assert own.agreement == pytest.approx(1 / 3)
    assert own.brier == pytest.approx(0.39)
    assert (own.brier_low, own.brier_high) == pytest.approx(
        np.quantile([0.265, 0.64, 0.39], [0.025, 0.975])
    )
    delta = table[(table.kind == "difference") & (table.scheme == "all")].iloc[0]
    assert delta.agreement_difference == pytest.approx(-2 / 3)
    assert delta.brier_difference == pytest.approx(0.28)
    predictions.loc[(predictions.system == "DQ:m") & (predictions.item_id == "b"), "p_first"] = (
        np.nan
    )
    missing = dq_cq_evaluation(items, {scheme.scheme_id: scheme}, labels, predictions, "m", draws)
    row = missing[
        (missing.kind == "level") & (missing.system == "DQ:m") & (missing.scheme == "all")
    ].iloc[0]
    assert row.agreement == pytest.approx(1 / 3)
    assert row.brier == pytest.approx(0.265)
    assert row.brier_n == 2


def add_dq_runs(hand):
    items = pd.read_csv(hand["items"]).query("in_test_set")
    for key, folder, stage, _prefix in [
        ("dq", "v1", "dq", "DQ"),
        ("dq_schemes_2", "v2", "dq", "DQ"),
        ("zeroshot_rerun", "v1", "zeroshot", "zeroshot.rerun"),
        ("zeroshot_schemes_2", "v2", "zeroshot", "zeroshot"),
    ]:
        schemes = load_all(REPO_ROOT / "schemes" / folder)
        rows = []
        for item in items.itertuples():
            scheme = schemes[item.gold_scheme]
            if stage == "dq":
                rows += [
                    dict(
                        _cq_row(
                            item.item_id,
                            scheme.scheme_id,
                            cq,
                            REFERENCE,
                            scheme.answer_space(cq)[0],
                            scheme.answer_space(cq),
                            0.8,
                        ),
                        stage="dq",
                        p_logprob=0.76,
                    )
                    for cq in scheme.cq_order
                ]
            rows.append(
                {
                    "item_id": item.item_id,
                    "stage": stage,
                    "scheme_condition": "gold",
                    "scheme": scheme.scheme_id,
                    "cq_id": None,
                    "model": REFERENCE,
                    "answer": sorted(scheme.terminals)[0],
                    "p_label": 0.8,
                    "p_verbal": 0.8,
                }
            )
        path = REPO_ROOT / "schemes" / folder
        _write(
            hand["runs"] / key,
            rows,
            {
                "run_id": key,
                "models": _models([REFERENCE]),
                "config": {"stages": [stage], "schemes": folder},
                "prompt_versions": [f"{stage}_v1"],
                "schemes_version": {"folder": folder, "content_sha256": schemes_hash(path)},
            },
        )
    return {key: [key] for key in RUN_KEYS}


@pytest.fixture(scope="module")
def evaluated_variants(tmp_path_factory):
    root = tmp_path_factory.mktemp("dq_round2")
    hand = write_runs(root, {"analogy": 10, "example": 5}, ssoft=True)
    config = replace(hand["config"], resamples=40)
    configs = [config, replace(config, runs={**config.runs, **dict.fromkeys(RUN_KEYS, [])})]
    configs.append(replace(config, runs={**config.runs, **add_dq_runs(hand)}))
    outputs = []
    for i, own in enumerate(configs):
        out = evaluate(own, hand["runs"], root / f"out{i}", hand["items"], hand["annotations"])
        outputs.append(out)
    return hand, configs, outputs


def test_round2_csvs_unchanged_and_new_outputs_are_isolated(evaluated_variants):
    hand, configs, outputs = evaluated_variants
    for name in LEGACY_FILES:
        assert len({(out / name).read_bytes() for out in outputs}) == 1, name
    for out in outputs:
        assert all((out / name).is_file() for name in DQ_FILES)
        assert not (out / "dq_routing_curve.csv").exists()
        assert not (out / "dq_choices.csv").exists()
        assert set(p.name for p in out.glob("*.csv")) == set(LEGACY_FILES) | set(DQ_FILES)
    absent = pd.read_csv(outputs[0] / "dq_deciding.csv").iloc[0]
    assert absent.outcome == "missing"
    manifest = json.loads((outputs[2] / "manifest.json").read_text())
    assert manifest["m"] == 4 and manifest["dq"]["family"]["m"] == 1
    assert manifest["dq"]["family"]["level"] == 0.95
    assert manifest["dq"]["runs"] == {k: [k] for k in RUN_KEYS}
    assert manifest["dq"]["folds"] == manifest["folds"]["assignment"]
    systems = pd.read_csv(outputs[2] / "dq_verdicts.csv")
    assert {
        f"DQ:{REFERENCE}",
        f"DQ.path:{REFERENCE}",
        f"DQ.soft:{REFERENCE}",
        f"DQ:{REFERENCE}@2.0",
        f"DQ.path:{REFERENCE}@2.0",
        f"DQ.soft:{REFERENCE}@2.0",
        f"zeroshot:{REFERENCE}@2.0",
        f"zeroshot.rerun:{REFERENCE}",
    } == set(systems.system)
    assert not systems.system.str.startswith("C2").any()
    assert list(systems.columns) == list(pd.read_csv(outputs[2] / "verdicts.csv", nrows=0).columns)
    assert list(pd.read_csv(outputs[2] / "dq_main_table.csv").columns) == list(
        pd.read_csv(outputs[2] / "main_table.csv", nrows=0).columns
    )
    page = (outputs[2] / "summary.html").read_text(encoding="utf-8")
    assert page.index('id="dq"') > page.index('id="sources"')
    for anchor in [
        "dq-primary",
        "dq-breakdowns",
        "dq-consistency",
        "dq-cq-level",
        "dq-exploratory",
        "dq-abstention",
        "dq-missing",
    ]:
        assert f'id="{anchor}"' in page
    first = configs[2]
    from argfallacy.eval.dq import evaluate_dq

    original = {name: (outputs[2] / name).read_bytes() for name in DQ_FILES}
    items = pd.read_csv(hand["items"]).query("in_test_set").sort_values("item_id")
    indices = resample_indices(first.seed, len(items), first.resamples, 1)
    no_nlas = items[items.source.str.lower() != "nlas"]
    independent = resample_indices(first.seed, len(no_nlas), first.resamples, 2)
    evaluate_dq(
        first,
        hand["runs"],
        outputs[2],
        items,
        manifest["folds"]["assignment"],
        indices,
        independent,
        labels=None,
        annotations_path=hand["annotations"],
    )
    assert all((outputs[2] / name).read_bytes() == data for name, data in original.items())


@pytest.mark.parametrize(
    "field,value",
    [("folder", "v1"), ("content_sha256", "wrong"), ("content_sha256", None), ("prompt", "dq_v9")],
)
def test_version_checks_refuse_mismatches(tmp_path, field, value):
    from argfallacy.eval.dq import read_dq_runs

    hand = write_runs(tmp_path, {"analogy": 5})
    runs = add_dq_runs(hand)
    p = hand["runs"] / "dq_schemes_2/manifest.json"
    manifest = json.loads(p.read_text())
    if field == "prompt":
        manifest["prompt_versions"] = [value]
    elif value is None:
        del manifest["schemes_version"][field]
    else:
        manifest["schemes_version"][field] = value
    p.write_text(json.dumps(manifest))
    items = pd.read_csv(hand["items"]).query("in_test_set")
    with pytest.raises(SchemeError):
        read_dq_runs(
            replace(hand["config"], runs={**hand["config"].runs, **runs}), hand["runs"], items
        )


def test_incomplete_run_excludes_only_its_systems(evaluated_variants, tmp_path):
    from shutil import copytree

    from argfallacy.eval.dq import evaluate_dq

    hand, configs, outputs = evaluated_variants
    runs = tmp_path / "runs"
    copytree(hand["runs"], runs)
    out = tmp_path / "out"
    copytree(outputs[2], out)
    rows = pd.read_csv(runs / "dq_schemes_2/summary.csv")
    missing_item = rows.item_id.iloc[0]
    rows[rows.item_id != missing_item].to_csv(runs / "dq_schemes_2/summary.csv", index=False)
    items = pd.read_csv(hand["items"]).query("in_test_set").sort_values("item_id")
    manifest = json.loads((out / "manifest.json").read_text())
    config = configs[2]
    indices = resample_indices(config.seed, len(items), config.resamples, 1)
    independent = resample_indices(config.seed, sum(items.source != "nlas"), config.resamples, 2)
    evaluate_dq(
        config,
        runs,
        out,
        items,
        manifest["folds"]["assignment"],
        indices,
        independent,
        labels=None,
        annotations_path=hand["annotations"],
    )
    systems = set(pd.read_csv(out / "dq_verdicts.csv").system)
    assert f"DQ:{REFERENCE}" in systems and f"DQ:{REFERENCE}@2.0" not in systems
    assert f"zeroshot:{REFERENCE}@2.0" in systems
    assert pd.read_csv(out / "dq_deciding.csv").iloc[0].outcome == "missing"
    missing = json.loads((out / "manifest.json").read_text())["dq"]["missing"]
    assert any(missing_item in str(reason) for reason in missing.values())


def test_prompt_replication_split_is_found_by_rendering(evaluated_variants):
    hand, _, outputs = evaluated_variants
    v1, v2 = load_all(), load_all(REPO_ROOT / "schemes/v2")
    items = pd.read_csv(hand["items"]).query("in_test_set")
    identical = sum(
        render_zeroshot(r._asdict(), v1[r.gold_scheme]).text
        == render_zeroshot(r._asdict(), v2[r.gold_scheme]).text
        for r in items.itertuples(index=False)
    )
    table = pd.read_csv(outputs[2] / "dq_replication.csv")
    assert table.loc[table.subset == "identical prompts", "n_items"].iloc[0] == identical
    table = pd.read_csv(outputs[2] / "dq_exploratory.csv")
    changed = table[table.subset == "different prompts"].iloc[0]
    assert changed.n_items == len(items) - identical


def test_round2_releases_its_objects_before_dq(tmp_path, monkeypatch):
    import weakref

    from argfallacy.eval import dq, html_report, round2

    class Reserved:
        pass

    observed = {}
    draws = np.zeros((3, 2), dtype=int)

    def legacy(*args):
        owner = Reserved()
        owner.cycle = owner
        observed["legacy"] = weakref.ref(owner)
        return tmp_path, pd.DataFrame(), {}, draws, draws, pd.DataFrame()

    def new(config, runs, out, items, folds, indices, without, **kwargs):
        assert observed["legacy"]() is None
        assert indices is draws and without is draws
        observed["called"] = True

    monkeypatch.setattr(round2, "_round2_tables", legacy)
    monkeypatch.setattr(dq, "evaluate_dq", new)
    monkeypatch.setattr(html_report, "write_summary", lambda p: p)
    assert round2.evaluate_round2(None, tmp_path) == tmp_path
    assert observed["called"]


@pytest.mark.parametrize("case", ["invalid", "regimes"])
def test_invalid_rows_and_other_entry_regimes(evaluated_variants, tmp_path, case):
    from shutil import copytree

    from argfallacy.eval.dq import evaluate_dq

    hand, configs, outputs = evaluated_variants
    runs, out = tmp_path / "runs", tmp_path / "out"
    copytree(hand["runs"], runs)
    copytree(outputs[2], out)
    path = runs / "dq_schemes_2/summary.csv"
    rows = pd.read_csv(path)
    if case == "invalid":
        rows["answer"] = "invalid"
        for column in rows.columns:
            if column.startswith("p_logprob") or column == "p_label":
                rows[column] = np.nan
    else:
        extra = []
        for entry in ("sampled", "reasoned"):
            own = rows.assign(model=entry)
            if entry == "sampled":
                schemes = load_all(REPO_ROOT / "schemes/v2")
                for p, row in own[own.cq_id.notna()].iterrows():
                    arcs = schemes[row.scheme].answer_space(row.cq_id)
                    own.loc[p, f"p_sample_{arcs[0]}"] = 0.0
                    own.loc[p, f"p_sample_{arcs[1]}"] = 1.0
                    own.loc[p, "p_sample_cannot_be_determined"] = 0.0
                    own.loc[p, "p_sample_na"] = 0.0
            extra.append(own)
        rows = pd.concat([rows, *extra], ignore_index=True)
        metadata_path = runs / "dq_schemes_2/manifest.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["models"].update(
            {
                "sampled": {"model_id": "sampled", "samples": 5, "reasoning": {}},
                "reasoned": {
                    "model_id": "reasoned",
                    "samples": 1,
                    "reasoning": {"reasoning_effort": "high"},
                },
            }
        )
        metadata_path.write_text(json.dumps(metadata))
    rows.to_csv(path, index=False)
    items = pd.read_csv(hand["items"]).query("in_test_set").sort_values("item_id")
    config = configs[2]
    metadata = json.loads((out / "manifest.json").read_text())
    evaluate_dq(
        config,
        runs,
        out,
        items,
        metadata["folds"]["assignment"],
        resample_indices(config.seed, len(items), config.resamples, 1),
        resample_indices(config.seed, sum(items.source != "nlas"), config.resamples, 2),
        labels=None,
        annotations_path=hand["annotations"],
    )
    main = pd.read_csv(out / "dq_main_table.csv").set_index("system")
    if case == "invalid":
        assert main.loc[f"DQ:{REFERENCE}@2.0", "accuracy"] == 0
        assert main.loc[f"DQ:{REFERENCE}@2.0", "coverage"] == 0
        assert pd.read_csv(out / "dq_deciding.csv").iloc[0].computed
        consistency = pd.read_csv(out / "dq_consistency.csv")
        own = consistency[
            (consistency.system == f"DQ:{REFERENCE}@2.0") & (consistency.kind == "all")
        ].iloc[0]
        assert own.consistent == 0 and own.stopped == 1
    else:
        assert "DQ.soft:reasoned@2.0" not in main.index
        assert {"DQ:reasoned@2.0", "DQ.path:reasoned@2.0", "DQ.soft:sampled@2.0"} <= set(main.index)
        assert main.loc["DQ:sampled@2.0", "role"] == "comparison"
        schemes = load_all(REPO_ROOT / "schemes/v2")
        routing = {s: routes(schemes[s], table) for s, table in load_burden("v2", schemes).items()}
        verdicts = pd.read_csv(out / "dq_verdicts.csv")
        for item, group in rows[(rows.model == "sampled") & rows.cq_id.notna()].groupby("item_id"):
            scheme = schemes[group.scheme.iloc[0]]
            expected = a2_pruned(
                scheme,
                group.set_index("cq_id").to_dict("index"),
                routing[scheme.scheme_id],
                "sample",
                0.02,
            )
            actual = verdicts[
                (verdicts.system == "DQ.soft:sampled@2.0") & (verdicts.item_id == item)
            ].iloc[0]
            assert actual.verdict == expected.verdict
            assert actual.confidence == pytest.approx(expected.confidence)
