"""The evaluation of spec 08, first stage, on data written by hand."""

from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from argfallacy.annotations import ITEMS_FILE
from argfallacy.eval import assign_folds, load_test_set, score_fallacies
from argfallacy.eval.evaluate import (
    EvaluationConfig,
    Scorer,
    deciding,
    difference,
    evaluate,
    reference_of,
    resample_indices,
    scores,
    without_nlas,
)
from argfallacy.eval.run_report import RunData, traversals
from argfallacy.parse import ANSWERS_COLUMNS, summarise
from argfallacy.schemes import SchemeError, traverse

GOOD = "good_argumentation"


def _test_set() -> pd.DataFrame:
    schemes = ["analogy"] * 11 + ["example"] * 7 + ["ad_hominem"] * 5
    return pd.DataFrame({"item_id": [f"{n:016x}" for n in range(len(schemes))][::-1],
                         "gold_scheme": schemes})


def test_the_folds_are_drawn_from_the_seed_and_balanced():
    items = _test_set()
    folds = assign_folds(items, seed=20261019)
    assert folds == assign_folds(items.sample(frac=1, random_state=1), seed=20261019)
    assert folds != assign_folds(items, seed=1)
    assert set(folds) == set(items["item_id"]) and set(folds.values()) == set(range(5))

    sizes = Counter(folds.values())
    assert max(sizes.values()) - min(sizes.values()) <= 1
    for scheme, own in items.groupby("gold_scheme"):
        per_fold = Counter(folds[item] for item in own["item_id"])
        counts = [per_fold.get(fold, 0) for fold in range(5)]
        assert max(counts) - min(counts) <= 1, scheme


# ------------------------------------------------------------- resamples


GOLD = (["weak_analogy"] * 6 + [GOOD] * 5 + ["cherry_picking"] * 4 + ["false_premise"] * 3
        + ["appeal_to_extremes"] * 2)
PREDICTED = ([None, "invalid", "weak_analogy", "weak_analogy", GOOD, "post_hoc"]
             + [GOOD, GOOD, "weak_analogy", GOOD, None]
             + ["cherry_picking", GOOD, "cherry_picking", "hasty_generalization"]
             + ["false_premise", "false_premise", "cherry_picking"]
             + ["appeal_to_extremes", GOOD])


def test_the_resamples_are_drawn_once_from_the_seed():
    first = resample_indices(20261019, n=20, b=50, generator=1)
    assert first.shape == (50, 20) and first.min() >= 0 and first.max() < 20
    assert np.array_equal(first, resample_indices(20261019, n=20, b=50, generator=1))
    assert not np.array_equal(first, resample_indices(20261019, n=20, b=50, generator=2))

    # every system is scored on the same weights: identical systems, identical values
    scorer = Scorer(GOLD, first)
    a, b = scorer.metrics(PREDICTED), scorer.metrics(list(PREDICTED))
    for metric in a:
        assert np.array_equal(a[metric], b[metric])


def _canonical(gold, predicted):
    """The canonical scorer, which takes an invalid answer as no verdict."""
    ids = [f"i{n}" for n in range(len(gold))]
    predicted = [None if p == "invalid" else p for p in predicted]
    return score_fallacies(pd.DataFrame({"item_id": ids, "predicted": predicted}),
                           pd.DataFrame({"item_id": ids, "gold_fallacy": gold}), "fine")


def _check(values, reference, coverage):
    good = reference.classes.set_index("class").loc[GOOD]
    expected = {"accuracy": reference.accuracy, "macro_f1": reference.macro_f1,
                "balanced_accuracy": reference.macro_recall, "good_recall": good["recall"],
                "good_precision": good["precision"], "coverage": coverage}
    for metric, value in expected.items():
        assert values[metric] == pytest.approx(value, abs=1e-12), metric


def test_the_vectorized_metrics_equal_the_canonical_scorer():
    observed = Scorer(GOLD, np.ones((1, len(GOLD)))).metrics(PREDICTED)
    covered = sum(p not in (None, "invalid") for p in PREDICTED) / len(PREDICTED)
    _check({k: v[0] for k, v in observed.items()}, _canonical(GOLD, PREDICTED), covered)

    # a resample with repeated items counts each as often as it is drawn
    drawn = [0, 0, 0, 3, 4, 4, 7, 11, 11, 12, 15, 15, 15, 17, 18, 19, 19, 2, 9, 13]
    weights = np.bincount(drawn, minlength=len(GOLD)).reshape(1, -1)
    resampled = Scorer(GOLD, weights).metrics(PREDICTED)
    gold, predicted = [GOLD[i] for i in drawn], [PREDICTED[i] for i in drawn]
    covered = sum(p not in (None, "invalid") for p in predicted) / len(predicted)
    _check({k: v[0] for k, v in resampled.items()}, _canonical(gold, predicted), covered)


def test_the_deciding_comparison():
    gold = GOLD
    indices = resample_indices(7, n=len(gold), b=1000, generator=1)
    right = list(gold)
    half = [g if n % 2 else None for n, g in enumerate(gold)]

    same = deciding(scores(gold, indices, half), scores(gold, indices, half), m=5)
    assert (same["difference"], same["low"], same["high"]) == (0.0, 0.0, 0.0)
    assert not same["improvement"]

    better = deciding(scores(gold, indices, right), scores(gold, indices, half), m=5)
    assert better["improvement"] and better["low"] > 0

    # the level follows m: with m = 5 the interval runs from the 0.5 to the 99.5 percentile
    a, b = scores(gold, indices, right), scores(gold, indices, half)
    spread = a.resampled["macro_f1"] - b.resampled["macro_f1"]
    assert (better["low"], better["high"]) == tuple(np.percentile(spread, [0.5, 99.5]))
    wide = deciding(a, b, m=1)
    assert (wide["low"], wide["high"]) == tuple(np.percentile(spread, [2.5, 97.5]))
    assert difference(a, b, "macro_f1") == {k: wide[k] for k in ("difference", "low", "high")}


def test_the_reference_model_is_the_best_a0_ties_to_the_first_name():
    assert reference_of({"b": 0.40, "a": 0.40, "c": 0.30}) == "a"
    assert reference_of({"b": 0.41, "a": 0.40}) == "b"


def test_the_configuration_of_full_run_v2():
    from argfallacy.paths import REPO_ROOT

    config = EvaluationConfig.load(REPO_ROOT / "configs" / "evaluation_v2.yaml")
    assert config.runs == {"answers": "20261005T193017Z_full_v2",
                           "zeroshot": "20261009T221950Z_zeroshot_v1", "judge": []}
    assert (config.name, config.reference_model, config.seed, config.resamples, config.folds,
            config.m) == ("decision_v2", "qwen3_8_27b", 20261019, 10000, 5, 5)


def test_without_nlas_the_frozen_data_give_404_items():
    items = load_test_set(ITEMS_FILE)
    assert len(items) == 601
    assert len(without_nlas(items)) == 404


# ------------------------------------------------------------ end to end


SCHEMES_USED = ("analogy", "example", "expert_opinion")
MODELS = {
    "m1": {"model_id": "x", "samples": 1,
           "reasoning": {"chat_template_kwargs": {"enable_thinking": False}}},
    "m2": {"model_id": "y", "samples": 3, "reasoning": {"reasoning_effort": "low"}},
}


def _answer_row(item, scheme, cq, model, sample, answer, first, second):
    lean = 0.8 if answer == first else 0.2
    outside = 0.5 if answer == "cannot_be_determined" else 0.1
    row = dict.fromkeys(ANSWERS_COLUMNS)
    row.update({
        "run_id": "full", "item_id": item, "stage": "stage2", "scheme_condition": "gold",
        "scheme": scheme, "cq_id": cq, "sample_index": sample, "model": model,
        "answer": answer, "confidence": 70, "justification": "x", "parse_ok": True,
        "finish_reason": "stop", "logprob_partial": False,
        f"p_logprob_{first}": lean * (1 - outside),
        f"p_logprob_{second}": (1 - lean) * (1 - outside),
        "p_logprob_cannot_be_determined": outside / 2, "p_logprob_na": outside / 2,
    })
    return row


@pytest.fixture
def hand_runs(tmp_path, schemes):
    rng = random.Random(5)
    items, rows, zeroshot = [], [], []
    for scheme_id in SCHEMES_USED:
        scheme = schemes[scheme_id]
        terminals = sorted(scheme.terminals)
        for k in range(12):
            item = f"{scheme_id[:4]}{k:012d}"
            given = {cq: rng.choice(scheme.answer_space(cq)) for cq in scheme.cq_order}
            if k % 6 == 5:
                given[scheme.cq_order[0]] = "cannot_be_determined"
            verdict = traverse(scheme, given).verdict
            gold = verdict if verdict and k % 4 else rng.choice(terminals)
            items.append({"item_id": item, "text": "t", "source": "nlas" if k % 3 == 0 else "logic",
                          "in_test_set": True, "eleni_id": None, "gold_scheme": scheme_id,
                          "gold_fallacy": gold})
            for cq in scheme.cq_order:
                first, second = scheme.answer_space(cq)
                rows.append(_answer_row(item, scheme_id, cq, "m1", 0, given[cq], first, second))
                for sample in range(3):
                    answer = rng.choice([first, second])
                    rows.append(_answer_row(item, scheme_id, cq, "m2", sample, answer,
                                            first, second))
            zeroshot.append({"item_id": item, "stage": "zeroshot", "scheme_condition": "gold",
                             "scheme": scheme_id, "cq_id": None, "model": "m1",
                             "answer": gold if k % 2 else ("invalid" if k == 4 else terminals[0])})
    items.append({"item_id": "f" * 16, "text": "t", "source": "logic", "in_test_set": False,
                  "eleni_id": None, "gold_scheme": None, "gold_fallacy": None})
    answers = pd.DataFrame(rows, columns=ANSWERS_COLUMNS)
    summary = summarise(answers, schemes)

    runs = tmp_path / "runs"
    for run_id, frame, models, stages in (("full", summary, ["m1", "m2"], ["stage1", "stage2"]),
                                          ("zs", pd.DataFrame(zeroshot), ["m1", "m2"],
                                           ["zeroshot"])):
        (runs / run_id).mkdir(parents=True)
        frame.to_csv(runs / run_id / "summary.csv", index=False)
        manifest = {"run_id": run_id, "config": {"name": run_id, "models": models,
                                                 "stages": stages, "scheme_condition": "gold"},
                    "models": MODELS, "prompt_versions": ["v"],
                    "schemes_version": {"folder": "v1", "tag": "1.1"}}
        (runs / run_id / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    items_path = tmp_path / "items.csv"
    pd.DataFrame(items).to_csv(items_path, index=False)
    config = EvaluationConfig(name="hand", runs={"answers": "full", "zeroshot": "zs", "judge": []},
                              reference_model="m1", seed=20261019, resamples=200, folds=5, m=5)
    return {"runs": runs, "items": items_path, "config": config, "out": tmp_path / "evaluations",
            "answers": answers, "summary": summary}


def _evaluate(hand, **changes):
    config = hand["config"]
    for key, value in changes.items():
        setattr(config, key, value)
    return evaluate(config, runs_dir=hand["runs"], out_root=hand["out"], items_path=hand["items"])


def test_verdicts_have_one_row_per_system_and_item(hand_runs, schemes):
    out = _evaluate(hand_runs)
    verdicts = pd.read_csv(out / "verdicts.csv")
    items = 36
    systems = ["A0:m1", "A0:m2", "A1", "A2", "A3", "A4", "zeroshot:m1", "majority",
               "ceiling:m1", "ceiling:m2", "A2.stop", "A2.branch", "A2.charitable", "A2:m2"]
    assert sorted(verdicts["system"].unique()) == sorted(systems)
    assert len(verdicts) == items * len(systems)
    assert not verdicts.duplicated(["system", "item_id"]).any()
    assert list(verdicts.columns) == ["system", "item_id", "fold", "verdict", "confidence",
                                      "abstention", "explanation"]

    data = RunData(run_id="full", answers=hand_runs["answers"], summary=hand_runs["summary"],
                   manifest={"config": {"models": ["m1", "m2"]}}, items=pd.DataFrame(),
                   annotations=pd.DataFrame(), schemes=schemes, raw_path=None)
    diagram = traversals(data, "diagram").set_index(["entry", "item_id"])["verdict"]
    for row in verdicts[verdicts["system"].str.startswith("A0:")].itertuples():
        expected = diagram[(row.system.removeprefix("A0:"), row.item_id)]
        assert (None if pd.isna(row.verdict) else row.verdict) == (
            None if pd.isna(expected) else expected)

    for name in ("main_table.csv", "deciding.csv", "diagnostic.csv", "exploratory.csv",
                 "main_table_without_nlas.csv", "deciding_without_nlas.csv",
                 "a1_orders.csv", "a3_choices.csv", "summary.md"):
        assert (out / name).is_file(), name
    main = pd.read_csv(out / "main_table.csv")
    assert list(main["system"]) == ["A0:m1", "A0:m2", "A1", "A2", "A3", "A4", "zeroshot:m1",
                                    "majority", "ceiling:m1", "ceiling:m2"]
    assert set(main["n_items"]) == {36}
    assert set(pd.read_csv(out / "main_table_without_nlas.csv")["n_items"]) == {24}
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "zeroshot:m2" in summary and "A5" in summary


def test_the_manifest_of_the_evaluation(hand_runs):
    out = _evaluate(hand_runs)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for key in ("runs", "seed", "resamples", "folds", "reference_model",
                "primary_configurations", "m", "commit", "date"):
        assert key in manifest, key
    assert manifest["seed"] == 20261019 and manifest["resamples"] == 200 and manifest["m"] == 5
    assert manifest["reference_model"] == "m1"
    assert len(manifest["folds"]["assignment"]) == 36
    assert set(manifest["folds"]["assignment"].values()) == set(range(5))
    assert set(manifest["primary_configurations"]) == {"A1", "A2", "A3", "A4"}


def test_the_command_stops_on_another_reference_model(hand_runs):
    with pytest.raises(SchemeError, match="m1"):
        _evaluate(hand_runs, reference_model="m2")


def test_a_summary_of_the_minimal_parser_is_refused(hand_runs):
    path = hand_runs["runs"] / "full" / "summary.csv"
    frame = pd.read_csv(path)
    frame.drop(columns=[c for c in frame.columns if c.startswith("p_logprob_")]).to_csv(
        path, index=False)
    with pytest.raises(SchemeError, match="parse"):
        _evaluate(hand_runs)
