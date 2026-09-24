"""The dictionary against the prior runs.

These read a folder outside the repository, named by ``PRIOR_RUNS_DIR``.  No
path is written here, and the whole module skips when the variable is unset.
Nothing is written back: the runs are read-only evidence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from argfallacy.labels import (
    COMPAT_SPACE,
    INVALID,
    UNKNOWN,
    audit,
    collapse,
    compat_normalize,
    format_audit,
    invalid_outputs,
    invalid_share,
    normalize_label,
)

TOLERANCE = 0.0005
FALLACY_COLUMNS = ["gold_fallacy", "predicted_fallacy"]
SCHEME_COLUMNS = ["gold_scheme", "predicted_scheme"]


def _items(run: Path) -> pd.DataFrame:
    """One row per item: results.csv is long, one row per critical question."""
    frame = pd.read_csv(run / "results.csv")
    return frame.drop_duplicates(subset=["id"]) if "id" in frame.columns else frame


def _run_id(run: Path) -> str:
    return "/".join(run.parts[-2:])


# --------------------------------------------- no unknown label in the runs
def _assert_no_unknown(values, kind, vocabulary, where):
    table = audit(values, kind=kind, vocabulary=vocabulary)
    unknown = table[table["status"] == UNKNOWN]
    assert unknown.empty, f"{where}: strings the dictionary cannot translate\n{format_audit(table)}"
    return table


def test_fallacy_columns_have_no_unknown(prior_runs, vocab):
    for run in prior_runs:
        frame = _items(run)
        for column in FALLACY_COLUMNS:
            if column in frame.columns:
                _assert_no_unknown(frame[column], "fallacy", vocab, f"{_run_id(run)}:{column}")


def test_scheme_columns_have_no_unknown(prior_runs, scheme_vocab):
    seen = False
    for run in prior_runs:
        frame = _items(run)
        for column in SCHEME_COLUMNS:
            if column in frame.columns:
                seen = True
                where = f"{_run_id(run)}:{column}"
                _assert_no_unknown(frame[column], "scheme", scheme_vocab, where)
    assert seen, "no run carried a scheme column, so the check would pass vacuously"


# ------------------------------- canonical against compat normalization
def test_canonical_normalization_is_at_least_as_generous_as_compat(prior_runs, vocab):
    """The canonical path may score higher: it knows aliases the compat scorer did not.

    It must never score lower.  Where the two differ, the difference is a row the
    compat scorer got wrong, not a regression.
    """
    for run in prior_runs:
        frame = _items(run)
        canonical = (
            frame["predicted_fallacy"].map(lambda s: normalize_label(s, vocab).id)
            == frame["gold_fallacy"].map(lambda s: normalize_label(s, vocab).id)
        ).mean()
        compat = (
            frame["predicted_fallacy"].map(lambda s: compat_normalize(s, vocab))
            == frame["gold_fallacy"].map(lambda s: compat_normalize(s, vocab))
        ).mean()
        assert canonical >= compat - TOLERANCE, f"{_run_id(run)}: {canonical:.6f} < {compat:.6f}"


# --------------------------------------------- gold support per label space
def test_gold_support_gives_twenty_three_eighteen_nineteen(prior_runs, vocab):
    gold = set()
    for run in prior_runs:
        gold |= {normalize_label(v, vocab).id for v in _items(run)["gold_fallacy"]}
    assert gold <= set(vocab["terminals"]), "a gold label fell outside the terminals"
    assert len(gold) == 23
    assert len({collapse(g, "collapsed_symmetric", vocab) for g in gold}) == 18
    assert len({collapse(g, COMPAT_SPACE, vocab) for g in gold}) == 19


def test_nineteen_is_the_prior_collapsed_class_count(prior_runs):
    for run in prior_runs:
        metrics = pd.read_csv(run / "metrics.csv")
        row = metrics[metrics["task"] == "final_label (collapsed)"]
        if len(row):
            assert int(row["n_classes_gold"].iloc[0]) == 19


# ------------------------------------------- invalid outputs, counted per model
def test_invalid_outputs_are_counted_per_model(prior_runs, vocab):
    """Every run gets its own count: an unusable output is a fact about that model."""
    seen_any = False
    for run in prior_runs:
        frame = _items(run)
        table = audit(frame["predicted_fallacy"], vocabulary=vocab)
        invalid = invalid_outputs(table)
        assert invalid_share(table) < 0.05, f"{_run_id(run)}: too many unusable outputs to ignore"
        if not invalid.empty:
            seen_any = True
            assert set(invalid["id"]) == {INVALID}
    assert seen_any, "no run produced an invalid output, so the check would pass vacuously"


def test_no_gold_label_is_ever_invalid(prior_runs, vocab):
    """The gold column is annotation, not model output: it must never contain one."""
    for run in prior_runs:
        table = audit(_items(run)["gold_fallacy"], vocabulary=vocab)
        assert invalid_outputs(table).empty, f"{_run_id(run)}: an invalid string in the gold"


def test_invalid_predictions_are_wrong_in_every_space_and_never_dropped(prior_runs, vocab):
    for run in prior_runs:
        frame = _items(run)
        predicted = frame["predicted_fallacy"].map(lambda s: normalize_label(s, vocab).id)
        gold = frame["gold_fallacy"].map(lambda s: normalize_label(s, vocab).id)
        rows = predicted == INVALID
        if not rows.any():
            continue
        assert len(predicted) == len(frame), "an invalid output was dropped from the run"
        for space in ("fine", "collapsed_symmetric", COMPAT_SPACE):
            here = [collapse(i, space, vocab) for i in predicted[rows]]
            there = [collapse(i, space, vocab) for i in gold[rows]]
            assert all(a != b for a, b in zip(here, there, strict=True)), (
                f"{_run_id(run)}: an invalid output scored in {space}"
            )
