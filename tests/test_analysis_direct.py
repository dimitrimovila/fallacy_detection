"""New human inputs use invented app files only, never discovered annotations."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from argfallacy.eval.analysis_bootstrap import Bootstrap
from argfallacy.eval.analysis_direct import (
    direct_tables,
    disagreement_statistics,
    read_direct,
)
from argfallacy.eval.analysis_inputs import AnalysisConfig, Inputs, SavedEvaluation, SavedSystem
from argfallacy.schemes import SchemeError, load_all
from test_analysis import saved as saved_fixture

saved = saved_fixture

GOOD, WEAK = "good_argumentation", "weak_analogy"


@pytest.fixture
def human(tmp_path):
    items = pd.DataFrame(
        {
            "item_id": list("abcdef"),
            "gold_scheme": "analogy",
            "gold_fallacy": GOOD,
            "source": ["LOGIC"] * 3 + ["NLAS"] * 3,
        }
    )
    samples = pd.DataFrame(
        {
            "item_id": list("abcd"),
            "sample": "direct",
            "scheme": "analogy",
            "round": [1, 1, 2, 2],
            "position": [1, 2, 3, 4],
        }
    )
    # A CQ row has no direct-verdict position or round.
    samples.loc[len(samples)] = ["e", "cq_v2", "analogy", "", ""]
    answers = pd.DataFrame(
        [
            ["a", 1, GOOD, "2026-10-28T10:00:00+01:00", 0],
            ["a", 1, WEAK, "2026-10-28T09:00:00+01:00", 3],
            ["a", 1, WEAK, "2026-10-28T08:00:00Z", 3],
            ["b", 2, GOOD, "2026-10-28T10:00:00+01:00", 2],
        ],
        columns=["item_id", "position", "verdict", "answered_at", "revision"],
    )
    config = SimpleNamespace(
        direct_verdict={
            "answers": tmp_path / "answers.csv",
            "samples": tmp_path / "sample.csv",
        }
    )
    samples.to_csv(config.direct_verdict["samples"], index=False)
    answers.to_csv(config.direct_verdict["answers"], index=False)
    return config, items, samples, answers


def test_revisions_ties_and_explicit_final_are_equivalent(human):
    config, items, _, answers = human
    first, status = read_direct(config, items, load_all(), Inputs())
    assert first.loc["a", "verdict"] == WEAK
    assert first.loc["a", "answered_at"] == "2026-10-28T08:00:00Z"
    assert status["planned"] == 4 and status["answered"] == 2 and status["missing"] == 2
    assert status["complete_rounds"] == [1] and status["partial_rounds"] == [2]
    assert not status["planned_minimum_reached"]
    assert sum(x["planned"] for x in status["by_scheme_origin"]) == 4
    final = config.direct_verdict["answers"].with_name("explicit_final.csv")
    answers.iloc[[2, 3]].to_csv(final, index=False)
    config.direct_verdict["answers"] = final
    second, again = read_direct(config, items, load_all(), Inputs())
    pd.testing.assert_frame_equal(first, second)
    assert again["answered"] == status["answered"]


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda a: pd.concat([a, a.iloc[[0]].assign(verdict=WEAK)]), "conflicting"),
        (lambda a: a.assign(position=0), "position"),
        (lambda a: a.assign(position=2), "position"),
        (lambda a: a.assign(revision=-1), "revision"),
        (lambda a: a.assign(revision=1.5), "revision"),
        (lambda a: a.assign(answered_at="2026-10-28T10:00:00"), "time.*zone"),
        (
            lambda a: a.assign(
                verdict=next(
                    t
                    for s in load_all().values()
                    for t in s.terminals
                    if t not in load_all()["analogy"].terminals
                )
            ),
            "menu",
        ),
        (lambda a: a.assign(verdict="Good Argumentation"), "canonical"),
        (lambda a: a.assign(item_id="unknown"), "unknown"),
        (lambda a: a.assign(item_id="e"), "sample"),
    ],
)
def test_bad_rows_fail_including_conflicting_older_revision(human, change, match):
    config, items, _, answers = human
    change(answers).to_csv(config.direct_verdict["answers"], index=False)
    with pytest.raises(SchemeError, match=match):
        read_direct(config, items, load_all(), Inputs())


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("position", 1, "position"),
        ("scheme", "example", "scheme"),
        ("item_id", "unknown", "unknown"),
        ("round", 0, "round"),
    ],
)
def test_bad_sample_membership(human, column, value, match):
    config, items, samples, _ = human
    samples.loc[1, column] = value
    samples.to_csv(config.direct_verdict["samples"], index=False)
    with pytest.raises(SchemeError, match=match):
        read_direct(config, items, load_all(), Inputs())


def test_empty_partial_samples_and_null_no_read(human, monkeypatch):
    config, items, samples, answers = human
    answers.iloc[:0].to_csv(config.direct_verdict["answers"], index=False)
    observed, status = read_direct(config, items, load_all(), Inputs())
    assert observed.empty and status["answered"] == 0 and status["missing"] == 4
    samples.iloc[:0].to_csv(config.direct_verdict["samples"], index=False)
    observed, status = read_direct(config, items, load_all(), Inputs())
    assert status["planned"] == 0 and observed.empty
    config.direct_verdict = {"answers": None, "samples": None}
    schemes = load_all()
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("null human file read"))
    _, status = read_direct(config, items, schemes, Inputs())
    assert status["status"] == "pending"


def test_paired_paths_required_config_relative_and_output_guard(saved):
    path, raw, _ = saved
    raw["direct_verdict"] = {"answers": "answers.csv"}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SchemeError, match="both"):
        AnalysisConfig.load(path)
    raw["direct_verdict"]["samples"] = "sample.csv"
    path.write_text(yaml.safe_dump(raw))
    config = AnalysisConfig.load(path)
    assert config.direct_verdict["answers"] == path.parent / "answers.csv"
    with pytest.raises(SchemeError, match="overlap"):
        config.guard_output(path.parent)


def test_disagreement_ties_empty_and_global_ranking_are_batch_invariant():
    ids = list("abcdef")
    verdicts = [GOOD] * 6
    confidence = np.array([0.99, 0.9, 0.8, 0.4, 0.4, 0.1])
    human = np.array([None, None, False, True, False, None], object)
    expected = None
    for batch in (1, 7, 41):
        result = disagreement_statistics(
            ids,
            verdicts,
            confidence,
            np.ones(6, bool),
            human,
            Bootstrap(17, 41, batch),
            levels=(0.5,),
        )
        assert result[0]["auroc"] == 0.75  # half credit for the .4 tie
        assert result[0]["observed_items"] == 3
        kept = next(r for r in result if r.get("selection") == "kept")
        assert kept["observed_items"] == 1 and kept["disagreement_prevalence"] == 0
        if expected is not None:
            pd.testing.assert_frame_equal(pd.DataFrame(result), pd.DataFrame(expected))
        expected = result
    undefined = disagreement_statistics(
        ids, verdicts, confidence, np.ones(6, bool), np.array([False] * 6, object), Bootstrap(17, 9)
    )
    assert np.isnan(undefined[0]["auroc"]) and undefined[0]["auroc_omitted"] == 9


def test_direct_tables_keep_gold_and_second_reference_separate(human):
    config, items, _, _ = human
    observed, _ = read_direct(config, items, load_all(), Inputs())
    rows = pd.DataFrame({"verdict": GOOD, "confidence": 0.6}, index=items.item_id)
    system = SavedSystem("DQ:ref", "1.1", rows, np.ones(6, bool), True)
    evaluation = SavedEvaluation("saved", 2, Path("."), {}, {system.name: system})
    tables = direct_tables(observed, None, [evaluation], items, load_all(), Bootstrap(7, 13, 2))
    direct = tables["human_direct.csv"]
    assert {"gold", "human_direct@1.1"} <= set(direct.reference)
    assert "paired direct/path against gold" in set(direct.comparison)
    assert "association with observed disagreement" in set(
        tables["uncertainty_disagreement.csv"].interpretation
    )
    # No observed label is imputed onto c..f.
    attributes = tables["human_disagreement"]
    assert pd.isna(attributes[2:]).all() and attributes[0] and not attributes[1]
    assert json.loads(
        direct.loc[direct.reference == "human_direct@1.1", "supported_classes"].iloc[0]
    )


def test_human_composition_selects_globally_before_origin_and_observation(human):
    from argfallacy.eval.analysis_selective import selective_tables
    from argfallacy.eval.round2 import SCHEME_CORPORA
    from argfallacy.paths import SCHEMES_ROOT

    _, items, _, _ = human
    rows = pd.DataFrame(
        {"verdict": GOOD, "confidence": [0.99, 0.9, 0.8, 0.4, 0.4, 0.1]}, index=items.item_id
    )
    system = SavedSystem("DQ:ref", "1.1", rows, np.ones(6, bool), True)
    evaluation = SavedEvaluation("saved", 2, Path("."), {}, {system.name: system})
    tables = selective_tables(
        [evaluation],
        items,
        Bootstrap(7, 5),
        np.full(6, np.nan),
        {"1.1": SCHEMES_ROOT / "v1"},
        {"human disagreement": [None, None, None, None, True, None]},
    )
    frame = tables["abstention_composition.csv"]
    row = frame[
        (frame.origin == SCHEME_CORPORA)
        & (frame.kind == "human disagreement")
        & (frame.requested_coverage == 0.7)
    ].iloc[0]
    assert row.known_kept == 0 and row.known_rejected == 1
    assert row.missing_all == 2 and row.group_items == 1
    assert row.n_items == 3 and row.global_n_items == 6
