"""The round 2 evaluation of spec 09: conformal sets, deciding comparisons, the command."""

from __future__ import annotations

import json
import math
from shutil import copytree

import numpy as np
import pandas as pd
import pytest

from argfallacy.aggregate.interface import decide
from argfallacy.eval.evaluate import EvaluationConfig, evaluate, resample_indices, scores
from argfallacy.eval.round2 import deciding_family, hypotheses
from argfallacy.eval.uncertainty import ALPHA, conformal_sets, conformal_threshold
from argfallacy.paths import REPO_ROOT
from argfallacy.schemes import SchemeError, load_all, traverse

GOOD = "good_argumentation"
GOLD = (["weak_analogy"] * 6 + [GOOD] * 5 + ["cherry_picking"] * 4 + ["false_premise"] * 3
        + ["appeal_to_extremes"] * 2)


# ------------------------------------------------------------- conformal sets


def test_the_conformal_threshold_is_the_order_statistic_of_the_calibration_scores():
    nine = [0.9, 0.1, 0.4, 0.3, 0.8, 0.2, 0.7, 0.6, 0.5]
    # ceil((9 + 1)(1 - 0.10)) = 9: the largest of nine scores
    assert conformal_threshold(nine, 0.10) == 0.9
    # ceil(10 * 0.5) = 5: the fifth smallest
    assert conformal_threshold(nine, 0.5) == 0.5
    # ceil((19 + 1)(1 - 0.10)) = 18: the 18th smallest of 0.00, 0.01, ..., 0.18
    nineteen = [k / 100 for k in range(19)][::-1]
    assert conformal_threshold(nineteen, 0.10) == 0.17
    # ceil(3 * 0.9) = 3 > 2 scores: no finite threshold, every terminal enters the set
    assert conformal_threshold([0.2, 0.3], 0.10) == math.inf
    # ties count once each
    assert conformal_threshold([0.4] * 9 + [0.1], 0.10) == 0.4


def test_conformal_sets_use_the_scores_of_the_other_folds():
    terminals = ("a", "b", "c")
    # fold 1: nine items whose gold has probability 0.1, 0.2, ..., 0.9 once renormalised
    # over the terminals; the abstention mass is removed before the score is computed
    aggregations, gold, folds = {}, {}, {}
    for k in range(1, 10):
        item = f"cal{k}"
        p = k / 10
        aggregations[item] = decide({"a": 0.5 * p, "b": 0.5 * (1 - p)}, 0.5, "")
        gold[item], folds[item] = "a", 1
    # fold 0: the scores of fold 1 are 0.9, 0.8, ..., 0.1, so q = 0.9 (n = 9, alpha = 0.10)
    aggregations["x"] = decide({"a": 0.05, "b": 0.85, "c": 0.10}, 0.0, "")
    aggregations["y"] = decide({"a": 0.3, "b": 0.1}, 0.6, "")
    aggregations["z"] = decide({}, 1.0, "no terminal has mass")
    for item in ("x", "y", "z"):
        gold[item], folds[item] = "a", 0
    sets, thresholds = conformal_sets(
        aggregations, dict.fromkeys(aggregations, terminals), gold, folds, alpha=ALPHA)
    assert thresholds[0] == pytest.approx(0.9)
    # x: scores 0.95, 0.15, 0.90; at most 0.9 keeps b and c
    assert sets["x"] == ("b", "c")
    # y: renormalised 0.75 and 0.25, c has no mass: scores 0.25, 0.75, 1.0
    assert sets["y"] == ("a", "b")
    # no mass on any terminal: every terminal
    assert sets["z"] == terminals
    # fold 1 is calibrated on the three items of fold 0: ceil(4 * 0.9) = 4 > 3, every terminal
    assert thresholds[1] == math.inf
    assert all(sets[f"cal{k}"] == terminals for k in range(1, 10))


# ------------------------------------------------------- deciding comparisons


def _results(predictions):
    indices = resample_indices(7, n=len(GOLD), b=1000, generator=1)
    return {name: scores(GOLD, indices, predicted) for name, predicted in predictions.items()}


RIGHT = list(GOLD)
HALF = [g if n % 2 else None for n, g in enumerate(GOLD)]
WRONG_HALF = [g if n % 2 else "post_hoc" for n, g in enumerate(GOLD)]


def test_the_deciding_intervals_follow_m():
    results = _results({"B1": RIGHT, "A0:q": HALF, "zeroshot:q": RIGHT, "B2": HALF, "B3": HALF,
                        "B1.hard": HALF})
    family = hypotheses("q")
    table = deciding_family(results, family, m=4).set_index("hypothesis")
    spread = results["B1"].resampled["macro_f1"] - results["A0:q"].resampled["macro_f1"]
    h11 = table.loc["H1.1"]
    assert h11["level"] == pytest.approx(1 - 0.05 / 4)
    assert (h11["macro_f1_low"], h11["macro_f1_high"]) == tuple(
        np.percentile(spread, [0.625, 99.375]))
    assert h11["improvement"] and h11["read"] and h11["outcome"] == "improvement"
    # what is reported beside decides nothing and stays at 95 per cent
    accuracy = results["B1"].resampled["accuracy"] - results["A0:q"].resampled["accuracy"]
    assert (h11["accuracy_low"], h11["accuracy_high"]) == tuple(
        np.percentile(accuracy, [2.5, 97.5]))

    two = deciding_family(results, family, m=2).set_index("hypothesis").loc["H1.1"]
    assert (two["macro_f1_low"], two["macro_f1_high"]) == tuple(
        np.percentile(spread, [1.25, 98.75]))

    # a component not computed keeps its row and does not lower m
    h4 = table.loc["H4"]
    assert not h4["computed"] and not h4["read"] and h4["outcome"].startswith("missing")
    assert h4["level"] == pytest.approx(1 - 0.05 / 4)
    assert np.allclose(table["level"], 1 - 0.05 / 4)


def test_the_second_comparison_is_read_only_after_the_first_passes():
    results = _results({"B1": RIGHT, "A0:q": HALF, "zeroshot:q": RIGHT, "B2": WRONG_HALF,
                        "B3": HALF, "B1.hard": HALF})
    table = deciding_family(results, hypotheses("q"), m=4).set_index("hypothesis")

    # H1.1 passes, so H1.2 is read: equal to the zero-shot is no measurable difference
    assert table.loc["H1.2", "read"]
    assert table.loc["H1.2", "outcome"] == "no measurable difference"
    assert not table.loc["H1.2", "improvement"]

    # H3.1 does not pass (B3 equals the diagram on the same answers): H3.2 is not read,
    # even though its interval is computed and reported
    assert not table.loc["H3.1", "improvement"]
    assert table.loc["H3.2", "computed"] and not table.loc["H3.2", "read"]
    assert table.loc["H3.2", "outcome"] == "not read: H3.1 does not pass"
    assert pd_isna(table.loc["H3.2", "improvement"])
    assert np.isfinite(table.loc["H3.2", "macro_f1_low"])

    # against the zero-shot, an interval below zero is the cost of the diagram
    assert table.loc["H2", "macro_f1_high"] < 0
    assert table.loc["H2", "outcome"] == "cost of the diagram"

    # with H1.1 failing, H1.2 is not read either
    failing = _results({"B1": HALF, "A0:q": HALF, "zeroshot:q": RIGHT})
    table = deciding_family(failing, hypotheses("q"), m=4).set_index("hypothesis")
    assert not table.loc["H1.2", "read"]
    assert table.loc["H1.2", "outcome"] == "not read: H1.1 does not pass"


def pd_isna(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


# ------------------------------------------------------------- the command

ENTRIES = ("m0", "m1", "m2", "m3", "m4")
REFERENCE = "m0"
OPTIONS = ("yes", "no", "positive", "negative", "cannot_be_determined", "na")
CORPORA = ("logic", "nlas", "ethix", "reddit")


def _models(entries):
    return {e: {"model_id": e, "samples": 1,
                "reasoning": {"chat_template_kwargs": {"enable_thinking": False}}}
            for e in entries}


def _cq_row(item, scheme, cq, model, answer, arcs, lean):
    """A row of summary.csv whose logprobs lean toward the hard answer."""
    row = {"item_id": item, "stage": "stage2", "scheme_condition": "gold", "scheme": scheme,
           "cq_id": cq, "model": model, "answer": answer, "justification": "x"}
    for option in OPTIONS:
        row[f"p_logprob_{option}"] = row[f"p_sample_{option}"] = np.nan
    outside = 0.6 if answer == "cannot_be_determined" else 0.05
    first = lean if answer == arcs[0] else 1 - lean
    row.update({f"p_logprob_{arcs[0]}": (1 - outside) * first,
                f"p_logprob_{arcs[1]}": (1 - outside) * (1 - first),
                "p_logprob_cannot_be_determined": 0.8 * outside, "p_logprob_na": 0.2 * outside,
                "p_verbal": first})
    return row


def _write(folder, rows, manifest, name="summary.csv"):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / name, index=False)
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_runs(root, counts, seed=3, ssoft=False):
    """Hand-made runs of every kind that round 2 reads, ``counts`` items per gold scheme.

    The reference entry answers as the true answers behind the gold, the others with
    noise.  With ``ssoft`` the S-soft runs repeat the answers of full run v3, so that
    the answers do not depend on the history.
    """
    from argfallacy.aggregate import expand, load_burden, routes
    from argfallacy.aggregate.ssoft import node_masses

    rng = np.random.default_rng(seed)
    v1, v2 = load_all(REPO_ROOT / "schemes/v1"), load_all(REPO_ROOT / "schemes/v2")
    items, notes, full, revised, zero, judge = [], [], [], [], [], []
    own3 = {}
    for scheme_id, n in counts.items():
        s1, s2 = v1[scheme_id], v2[scheme_id]
        terminals = sorted(s1.terminals)
        for k in range(n):
            item = f"{scheme_id[:6]}{k:010d}"
            truth = {cq: s1.answer_space(cq)[rng.integers(2)] for cq in s1.cq_order}
            truth2 = {cq: s2.answer_space(cq)[rng.integers(2)] for cq in s2.cq_order}
            verdict = traverse(s1, truth).verdict
            gold = (verdict if verdict and rng.random() < 0.8
                    else terminals[rng.integers(len(terminals))])
            items.append({"item_id": item, "text": "t", "source": CORPORA[k % 4],
                          "in_test_set": True, "eleni_id": None, "gold_scheme": scheme_id,
                          "gold_fallacy": gold})
            sheet = {"item_id": item, "annotator": "dumitru", "sheet": f"s {scheme_id}", "row": k}
            notes.append({**sheet, "field": "scheme", "value": scheme_id, "raw": scheme_id})
            for cq in s1.cq_order:
                value = truth[cq] if rng.random() < 0.9 else "idk"
                notes.append({**sheet, "field": cq, "value": value, "raw": value})
            for e, entry in enumerate(ENTRIES):
                for scheme, given, rows in ((s1, truth, full), (s2, truth2, revised)):
                    for cq in scheme.cq_order:
                        arcs = scheme.answer_space(cq)
                        answer = given[cq]
                        if e and rng.random() < 0.4:
                            answer = arcs[1 - arcs.index(answer)]
                        if rng.random() < 0.05:
                            answer = "cannot_be_determined"
                        rows.append(_cq_row(item, scheme_id, cq, entry, answer, arcs,
                                            rng.uniform(0.6, 0.95)))
                        if rows is revised and entry == REFERENCE:
                            own3.setdefault(item, {})[cq] = rows[-1]
            for rows, stage in ((zero, "zeroshot"), (judge, "judge")):
                rows.append({"item_id": item, "stage": stage, "scheme_condition": "gold",
                             "scheme": scheme_id, "cq_id": None, "model": REFERENCE,
                             "answer": gold if rng.random() < 0.6
                             else terminals[rng.integers(len(terminals))],
                             "p_label": rng.uniform(0.3, 1.0)})
    items.append({"item_id": "f" * 16, "text": "t", "source": "logic", "in_test_set": False,
                  "eleni_id": None, "gold_scheme": None, "gold_fallacy": None})
    pd.DataFrame(items).to_csv(root / "items.csv", index=False)
    pd.DataFrame(notes).to_csv(root / "annotations.csv", index=False)

    runs = root / "runs"
    for name, rows, folder, stages, models, prompts in (
            ("full_v2", full, "v1", ["stage1", "stage2"], ENTRIES, ["stage2_v2"]),
            ("full_v3", revised, "v2", ["stage2"], ENTRIES, ["stage2_v3"]),
            ("zs", zero, "v1", ["zeroshot"], [REFERENCE], ["zeroshot_v1"]),
            ("judge", judge, "v1", ["judge"], [REFERENCE], ["judge_v1"])):
        config = {"name": name, "models": list(models), "stages": stages,
                  "scheme_condition": "gold"}
        if name == "judge":
            config["judge"] = {"answers_from": "full_v2", "answers_model": REFERENCE,
                               "justifications": True}
        tag = "1.1" if folder == "v1" else "2.0"
        _write(runs / name, rows, {"run_id": name, "config": config, "models": _models(models),
                                   "prompt_versions": prompts,
                                   "schemes_version": {"folder": folder, "tag": tag}})
    keys = {"ssoft": [], "ssoft_nohistory": []}
    if ssoft:
        routing = {s: routes(v2[s], t) for s, t in load_burden("v2", v2).items()}
        history, control = [], []
        for rows in own3.values():
            scheme = v2[next(iter(rows.values()))["scheme"]]
            result = expand(scheme, lambda state, rows=rows, scheme=scheme: node_masses(
                scheme, state.cq_id, rows[state.cq_id]), routing[scheme.scheme_id])
            history += [{**rows[s.cq_id], "stage": "ssoft", "path": s.path, "depth": s.depth}
                        for s in result.visited]
            control += [{**row, "stage": "ssoft", "path": "", "depth": None}
                        for row in rows.values()]
        for name, rows, flag in (("ss", history, True), ("ssn", control, False)):
            _write(runs / name, rows, {
                "run_id": name, "models": _models([REFERENCE]),
                "config": {"name": name, "models": [REFERENCE], "stages": ["ssoft"],
                           "history": flag, "threshold": 0.02, "schemes": "v2"},
                "prompt_versions": ["ssoft_v1"],
                "schemes_version": {"folder": "v2", "tag": "2.0"}}, "ssoft.csv")
        keys = {"ssoft": ["ss"], "ssoft_nohistory": ["ssn"]}
    config = EvaluationConfig(
        name="round2", reference_model=REFERENCE, seed=20261019, resamples=200, folds=5, m=4,
        runs={"answers": "full_v2", "answers_schemes_2": "full_v3", "zeroshot": ["zs"],
              "judge": ["judge"], **keys},
        cq_label_exclusion=list(counts)[1:])
    return {"config": config, "runs": runs, "items": root / "items.csv",
            "annotations": root / "annotations.csv", "out": root / "evaluations",
            "own3": own3}


def test_the_configuration_of_round_2():
    from argfallacy.eval.round2 import is_round2
    from argfallacy.labels import load_scheme_vocabulary

    config = EvaluationConfig.load(REPO_ROOT / "configs" / "evaluation_round2.yaml")
    assert is_round2(config)
    assert set(config.runs) == {"answers", "answers_schemes_2", "zeroshot", "judge",
                                "ssoft", "ssoft_nohistory", "zeroshot_schemes_2",
                                "zeroshot_rerun", "dq", "dq_schemes_2"}
    assert {key: value for key, value in config.runs.items()
            if key not in ("ssoft", "ssoft_nohistory", "zeroshot_schemes_2", "zeroshot_rerun",
                            "dq", "dq_schemes_2")} == {
                           "answers": "20261005T193017Z_full_v2",
                           "answers_schemes_2": "20261019T151336Z_full_v3",
                           "zeroshot": ["20261009T221950Z_zeroshot_v1",
                                        "20261021T111606Z_zeroshot_k2"],
                           "judge": ["20261021T111110Z_judge_v1",
                                     "20261021T111110Z_judge_v1_answers_only"]}
    for key in ("ssoft", "ssoft_nohistory"):
        assert isinstance(config.runs[key], list)
        assert all(isinstance(run, str) and run for run in config.runs[key])
    assert {key: config.runs[key] for key in
            ("zeroshot_schemes_2", "zeroshot_rerun", "dq", "dq_schemes_2")} == {
        "zeroshot_schemes_2": ["20261026T101601Z_zeroshot_schemes_v2"],
        "zeroshot_rerun": ["20261026T101601Z_zeroshot_rerun"],
        "dq": ["20261026T101601Z_dq_full_v2"],
        "dq_schemes_2": ["20261026T101601Z_dq_full_v3"],
    }
    assert (config.name, config.reference_model, config.seed, config.resamples, config.folds,
            config.m) == ("round2", "qwen3_8_27b", 20261019, 10000, 5, 4)
    assert len(config.cq_label_exclusion) == 3
    assert set(config.cq_label_exclusion) <= set(load_scheme_vocabulary()["schemes"])


@pytest.fixture(scope="module")
def round2(tmp_path_factory):
    from argfallacy.eval.round2 import evaluate_round2

    root = tmp_path_factory.mktemp("round2")
    hand = write_runs(root, {"analogy": 10, "example": 5})
    out = evaluate_round2(hand["config"], hand["runs"], hand["out"], hand["items"],
                          hand["annotations"], conformal_export_dir=root / "exact_conformal")
    return hand, out


def test_opt_in_conformal_companion_uses_actual_b5_sets(round2):
    from types import SimpleNamespace

    from argfallacy.eval.analysis_conformal import conformal_tables
    from argfallacy.eval.analysis_inputs import Inputs, read_evaluation, scheme_registry

    hand, out = round2
    items = pd.read_csv(hand["items"]).sort_values("item_id").reset_index(drop=True)
    items = items[items.in_test_set].reset_index(drop=True)
    inputs = Inputs()
    registry = scheme_registry(inputs)
    source = dict(path=out, round=2, populations={},
                  conformal_records=out.parent.parent / "exact_conformal")
    evaluation = read_evaluation("round2", source, hand["config"], items, registry, inputs)
    tables, status = conformal_tables(SimpleNamespace(evaluations={"round2": source}),
        [evaluation], items, registry, None, inputs)
    saved = pd.read_csv(out / "conformal.csv")
    assert set(tables["conformal_items.csv"].system) == set(saved.system)
    assert tables["conformal_items.csv"].shape[0] == len(items) * saved.system.nunique()
    assert all(row["status"] == "computed" for row in status.values())


def test_without_ssoft_runs_b4_is_missing_and_the_rest_is_computed(round2):
    _, out = round2
    for name in ("manifest.json", "verdicts.csv", "cq_combined.csv", "round2_choices.csv",
                 "cq_level.csv", "routing_curve.csv", "conformal.csv", "deciding.csv",
                 "deciding_breakdowns.csv", "exploratory.csv", "abstention.csv",
                 "summary.html"):
        assert (out / name).is_file(), name
    verdicts = pd.read_csv(out / "verdicts.csv")
    systems = set(verdicts.system)
    assert {"B1", "B2", "B3", "B1.hard", "A2.pruned", "A0:m0", "A0:m0@2.0", "zeroshot:m0", "A2",
            "A4", "A5", "B1.majority.none.soft", "B1.dawid_skene.none.hard",
            "B1.weighted.cq.soft", "B1.stacking.gold.hard", "B1.annotation_exclusion",
            "B1@2.0", "B1.hard@2.0", "B1.majority.none.soft@2.0",
            "B1.stacking.gold.soft@2.0", "B2.A0", "B3.pooled", "A3.tree"} <= systems
    assert not {"B4", "B4.path", "B4.nohistory"} & systems
    assert not verdicts.duplicated(["system", "item_id"]).any()
    # the exploratory B1 without the CQ labels of example is scored on the other scheme
    excluded = verdicts[verdicts.system == "B1.annotation_exclusion"]
    assert len(excluded) == 10 and excluded.item_id.str.startswith("analog").all()

    decide = pd.read_csv(out / "deciding.csv").set_index("hypothesis")
    assert list(decide.index) == ["H1.1", "H1.2", "H2", "H3.1", "H3.2", "H4"]
    assert np.allclose(decide.level, 1 - 0.05 / 4)
    assert decide.loc[["H1.1", "H1.2", "H2", "H3.1", "H3.2"], "computed"].all()
    assert not decide.loc["H4", "computed"]
    assert decide.loc["H4", "outcome"] == "missing: no S-soft run (runs.ssoft is empty)"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["round"], manifest["m"]) == (2, 4)
    for name in ("B4", "B4.path", "B4.nohistory", "B4 against B4.nohistory",
                 "B4.nohistory against A2.pruned"):
        assert name in manifest["missing"], name

    explored = pd.read_csv(out / "exploratory.csv")
    assert {("B1", "A4"), ("B1", "A2"), ("B1.annotation_exclusion", "B1")} <= set(
        zip(explored.system, explored.reference, strict=True))
    assert explored.set_index("system").loc["B1.annotation_exclusion", "n_items"] == 10
    breakdown = pd.read_csv(out / "deciding_breakdowns.csv")
    assert set(breakdown.subset) == {"fallacy corpora", "scheme corpora", "analogy", "example",
                                     "without NLAS"}
    combined = pd.read_csv(out / "cq_combined.csv")
    assert not combined.duplicated(["system", "item_id", "cq_id"]).any()
    conformal = pd.read_csv(out / "conformal.csv")
    assert {"B1", "B3.pooled", "A2", "A4", "B1.stacking.gold.soft@2.0"} <= set(conformal.system)
    assert not set(conformal.system) & {"B1.hard", "B3", "B2"}
    abstained = set(pd.read_csv(out / "abstention.csv").system)
    assert {"B1", "B2", "B3", "A2", "A4", "zeroshot:m0", "A5"} <= abstained
    assert not abstained & {"B1.hard", "A0:m0", "majority"}
    page = (out / "summary.html").read_text(encoding="utf-8")
    for anchor in ("deciding", "cq-level", "supervision", "routing", "ssoft", "uncertainty"):
        assert f'id="{anchor}"' in page


def test_the_same_seed_gives_the_same_outputs(round2, tmp_path):
    hand, out = round2
    again = evaluate(hand["config"], hand["runs"], tmp_path, hand["items"], hand["annotations"])
    for path in sorted(out.glob("*.csv")):
        assert path.read_bytes() == (again / path.name).read_bytes(), path.name


def test_the_ssoft_runs_give_b4_and_its_control(tmp_path):
    from argfallacy.aggregate import load_burden, routes
    from argfallacy.eval.round2 import _ssoft_rows, ssoft_systems

    hand = write_runs(tmp_path, {"analogy": 4, "example": 3}, ssoft=True)
    folder = REPO_ROOT / "schemes" / "v2"
    schemes = load_all(folder)
    routing = {s: routes(schemes[s], t) for s, t in load_burden("v2", schemes).items()}
    items = pd.read_csv(hand["items"])
    items = items[items.in_test_set]
    scheme_of = dict(zip(items.item_id, items.gold_scheme, strict=True))
    history, source = _ssoft_rows(hand["runs"], ["ss"], True, folder, "m0", scheme_of)
    control, _ = _ssoft_rows(hand["runs"], ["ssn"], False, folder, "m0", scheme_of)
    found, changes, pending = ssoft_systems(schemes, routing, list(scheme_of), scheme_of,
                                            hand["own3"], history, control, source)
    # answers that do not depend on the history give A2.pruned, path by path
    for item in scheme_of:
        reference = found["A2.pruned"][item]
        for name in ("B4", "B4.nohistory"):
            assert found[name][item].distribution == pytest.approx(reference.distribution,
                                                                   abs=1e-12)
            assert found[name][item].pruned == pytest.approx(reference.pruned, abs=1e-12)
        assert found["B4.path"][item].verdict == reference.path_verdict
    assert pending == {"ssoft": [], "ssoft_nohistory": []}
    for item in scheme_of:
        assert found["B4"][item] == found["B4.nohistory"][item]
    assert changes.set_index("scheme").loc["all", "changed"] == 0
    # a run with history is not the control, and the reverse
    with pytest.raises(SchemeError, match="without history"):
        _ssoft_rows(hand["runs"], ["ss"], False, folder, "m0", scheme_of)
    with pytest.raises(SchemeError, match="with history"):
        _ssoft_rows(hand["runs"], ["ssn"], True, folder, "m0", scheme_of)


@pytest.fixture(scope="module")
def complete_ssoft_round2(tmp_path_factory):
    root = tmp_path_factory.mktemp("complete_ssoft")
    hand = write_runs(root, {"analogy": 10, "example": 5}, ssoft=True)
    out = evaluate(hand["config"], hand["runs"], hand["out"], hand["items"],
                   hand["annotations"])
    return hand, out


@pytest.mark.parametrize("incomplete", [("ss",), ("ssn",), ("ss", "ssn")])
def test_incomplete_ssoft_runs_exclude_only_their_systems(
        complete_ssoft_round2, tmp_path, incomplete):
    hand, complete = complete_ssoft_round2
    runs = tmp_path / "runs"
    copytree(hand["runs"], runs)
    missing_item = next(iter(hand["own3"]))
    for run in incomplete:
        path = runs / run / "ssoft.csv"
        rows = pd.read_csv(path)
        rows[rows.item_id != missing_item].to_csv(path, index=False)
    out = evaluate(hand["config"], runs, tmp_path / "evaluations", hand["items"],
                   hand["annotations"])
    excluded = ({"B4", "B4.path"} if "ss" in incomplete else set()) | (
        {"B4.nohistory"} if "ssn" in incomplete else set())
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["m"] == 4
    assert "ssoft_items_with_pending_mass" not in manifest
    for run, field, names in (
            ("ss", "ssoft_pending_items", ("B4", "B4.path")),
            ("ssn", "ssoft_nohistory_pending_items", ("B4.nohistory",))):
        expected = [missing_item] if run in incomplete else []
        assert manifest[field] == {"count": len(expected), "item_ids": expected}
        for name in names:
            if expected:
                assert manifest["missing"][name] == (
                    f"incomplete S-soft run: pending mass for items {missing_item}")
            else:
                assert name not in manifest["missing"]
    for filename in ("verdicts.csv", "main_table.csv", "abstention.csv", "conformal.csv"):
        before, after = pd.read_csv(complete / filename), pd.read_csv(out / filename)
        assert excluded.isdisjoint(after.system)
        pd.testing.assert_frame_equal(
            before[~before.system.isin(excluded)].reset_index(drop=True), after)
    deciding = pd.read_csv(out / "deciding.csv").set_index("hypothesis")
    assert bool(deciding.loc["H4", "computed"]) == ("ss" not in incomplete)
    assert deciding.loc["H4", "level"] == 1 - 0.05 / 4
    if "ss" in incomplete:
        assert "incomplete S-soft run" in deciding.loc["H4", "outcome"]
        assert missing_item in deciding.loc["H4", "outcome"]
        assert (out / "ssoft_changes.csv").read_text().strip() == "scheme,cq_id,changed"
    explored = pd.read_csv(out / "exploratory.csv")
    assert excluded.isdisjoint(explored.system) and excluded.isdisjoint(explored.reference)
    for a, b in (("B4", "B4.nohistory"), ("B4.nohistory", "A2.pruned")):
        if a in excluded or b in excluded:
            reason = manifest["missing"][f"{a} against {b}"]
            assert "incomplete S-soft run" in reason and missing_item in reason
    for filename in ("cq_combined.csv", "cq_level.csv", "cq_level_differences.csv",
                     "human_path.csv", "routing_curve.csv"):
        assert (out / filename).read_bytes() == (complete / filename).read_bytes(), filename
    page = (out / "summary.html").read_text(encoding="utf-8")
    for run, description in (("ss", "history"), ("ssn", "control without history")):
        message = f"Incomplete S-soft {description} run: 1 item with pending mass"
        assert (message in page) == (run in incomplete)


def test_a_zero_shot_summary_without_p_label_is_refused(tmp_path):
    hand = write_runs(tmp_path, {"analogy": 5})
    path = hand["runs"] / "zs" / "summary.csv"
    pd.read_csv(path).drop(columns="p_label").to_csv(path, index=False)
    with pytest.raises(SchemeError, match="p_label"):
        evaluate(hand["config"], hand["runs"], hand["out"], hand["items"], hand["annotations"])
