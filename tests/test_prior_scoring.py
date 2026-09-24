"""The scorer against the prior runs.

Compat mode must give back every number of the files those runs wrote; the
canonical mode must see the same 601 items as the experiments.  The folder is
named by ``PRIOR_RUNS_DIR`` and read only; the module skips without it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from argfallacy.eval import (
    compat_classwise,
    compat_metrics,
    compat_per_source,
    load_test_set,
    majority_by,
    predicted_scheme_class,
    prior_items,
    read_prior_runs,
    score_prior,
)
from argfallacy.labels import normalize_label

EXACT = 1e-9
DIAGNOSIS = 0.0005


@pytest.fixture(scope="module")
def runs(prior_dir):
    found = read_prior_runs(prior_dir)
    assert len(found) == 8
    return found


def _key(run) -> tuple[str, str]:
    return run.model, run.folder


# ---------------------------------------------------------------- compat mode
def test_compat_reproduces_every_row_of_metrics_csv(runs):
    checked = 0
    for run in runs:
        reference = pd.read_csv(run.path / "metrics.csv").set_index("task")
        ours = compat_metrics(run).set_index("task")
        assert sorted(ours.index) == sorted(reference.index), _key(run)
        for task, expected in reference.iterrows():
            got = ours.loc[task]
            where = f"{_key(run)} {task}"
            assert int(got["n_samples"]) == int(expected["n_samples"]), where
            assert int(got["n_classes_gold"]) == int(expected["n_classes_gold"]), where
            for column in ("accuracy", "macro_f1", "precision", "recall"):
                assert abs(float(got[column]) - float(expected[column])) <= EXACT, (
                    f"{where} {column}: {got[column]} != {expected[column]}"
                )
            checked += 1
    assert checked == 28


def test_compat_reproduces_every_row_of_classwise_metrics_csv(runs):
    checked = 0
    for run in runs:
        reference = pd.read_csv(run.path / "classwise_metrics.csv", dtype=str,
                                keep_default_na=False)
        ours = compat_classwise(run)
        for task, expected in reference.groupby("task"):
            got = ours[ours["task"] == task].set_index("class_label")
            expected = expected.set_index("class_label")
            assert set(got.index) == set(expected.index), f"{_key(run)} {task}"
            for label, row in expected.iterrows():
                for column in ("precision", "recall", "f1-score", "support"):
                    assert abs(float(got.loc[label, column]) - float(row[column])) <= EXACT, (
                        f"{_key(run)} {task} {label!r} {column}"
                    )
                checked += 1
        assert set(ours["task"]) == set(reference["task"]), _key(run)
    assert checked == 397


def test_compat_reproduces_every_row_of_per_dataset_micro_f1(runs, prior_dir):
    reference = pd.read_csv(prior_dir / "per_dataset_micro_f1.csv", dtype=str,
                            keep_default_na=False)
    ours = pd.concat([compat_per_source(run) for run in runs])
    keys = ["model", "pipeline_type", "source"]
    merged = reference.merge(ours, on=keys, how="left", suffixes=("", "_ours"),
                             validate="one_to_one")
    assert merged["n_ours"].notna().all(), merged[merged["n_ours"].isna()][keys]
    assert len(merged) == 48
    for row in merged.itertuples():
        assert int(row.n) == int(row.n_ours), (row.model, row.pipeline_type, row.source)
        assert abs(float(row.micro_f1) - float(row.micro_f1_ours)) <= EXACT


# ------------------------------------------------------ majority and ceiling
DIAGNOSIS_TABLE = {  # model: (majority per predicted scheme, information ceiling)
    "gpt-5": (0.557, 0.669),
    "qwen3.8_27b": (0.562, 0.662),
    "llama3.3_70B": (0.558, 0.671),
    "deepseek": (0.554, 0.662),
}


def test_majority_and_ceiling_reproduce_the_diagnosis(runs, vocab, scheme_vocab):
    pipeline = [run for run in runs if run.pipeline]
    assert sorted(run.model for run in pipeline) == sorted(DIAGNOSIS_TABLE)
    for run in pipeline:
        rows = run.rows
        assert len(rows) == 607
        gold = [normalize_label(g, vocab).id for g in rows["gold_fallacy"]]
        schemes = [predicted_scheme_class(s, scheme_vocab)[0] for s in rows["predicted_scheme"]]
        vectors = run.answer_vectors().loc[rows["id"]].tolist()

        def accuracy(keys, gold=gold):
            predicted = majority_by(keys, gold)
            return sum(p == g for p, g in zip(predicted, gold, strict=True)) / len(gold)

        majority, ceiling = DIAGNOSIS_TABLE[run.model]
        assert abs(accuracy(schemes) - majority) <= DIAGNOSIS, run.model
        assert abs(accuracy(list(zip(schemes, vectors, strict=True))) - ceiling) <= DIAGNOSIS


# ---------------------------------------------------------------- rescoring
TWICE = "469;495"
DROPPED = {"175", "194", "490", "495", "503", "531"}
MISMATCHES = {"a122bba5789304c0", "c8cdc8019c39370e", "cf3b4d6783d834d0"}


def test_every_run_is_rescored_on_the_601_items(runs):
    items = load_test_set()
    twice = items.loc[items["eleni_id"] == TWICE, "item_id"].item()
    for run in runs:
        table, dropped = prior_items(run, items)
        assert len(table) == 601 and table["item_id"].is_unique, _key(run)
        assert set(dropped) == DROPPED, _key(run)
        row = table.set_index("item_id").loc[twice]
        source = run.rows.set_index("id").loc["469"]
        assert row["predicted_fallacy"] == source["predicted_fallacy"], _key(run)


def test_score_prior_writes_its_files_and_lists_the_scheme_mismatches(prior_dir, tmp_path: Path):
    written = score_prior(prior_dir, tmp_path)
    expected = {"compat_metrics.csv", "compat_classwise.csv", "compat_per_source.csv",
                "metrics.csv", "classwise.csv", "baselines.csv", "report.md"}
    assert {path.name for path in written.values()} == expected
    assert all(path.parent == tmp_path and path.is_file() for path in written.values())

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    for item_id in MISMATCHES:
        assert item_id in report

    metrics = pd.read_csv(tmp_path / "metrics.csv")
    main = metrics[metrics["task"] == "final_label"]
    assert len(main) == 16 and (main["n_items"] == 601).all()
    assert set(metrics["space"]) == {"fine", "collapsed_symmetric", "schemes"}
