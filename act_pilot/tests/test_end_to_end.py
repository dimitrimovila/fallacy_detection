"""The whole pipeline on synthetic data with fake models; it must run in under a minute."""

import csv
import time

from act_pilot.cli import main
from act_pilot.config import load_config
from act_pilot.pipeline import BudgetExceeded, estimate, load_inputs, run_all
from conftest import make_config
from fakes import FakeAnswerer, FakeOptimizer


def _read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_end_to_end_with_fakes(tmp_path):
    started = time.monotonic()
    config = load_config(make_config(tmp_path, steps=3, seeds=(0, 1)))
    answerer, optimizer = FakeAnswerer(), FakeOptimizer()
    inputs = load_inputs(config)
    costs = {c.boundary: c for c in estimate(config, inputs)}

    evaluations = run_all(config, backends={"answerer": answerer, "optimizer": optimizer})

    out = config.output_dir
    for name in ("report.md", "cost_estimate.md", "calls.jsonl",
                 "content_validity_template.csv"):
        assert (out / name).exists(), name
    for bid in ("B1", "B2"):
        for name in (f"act/{bid}_act_candidates.csv", f"act/{bid}_act_selected.csv",
                     f"evaluation/{bid}_items.csv", f"evaluation/{bid}_questions.csv",
                     f"evaluation/{bid}_sources.csv"):
            assert (out / name).exists(), name
        selected = _read(out / f"act/{bid}_act_selected.csv")
        assert len(selected) == 5 * 2  # outer folds x seeds
        questions = {r["candidate_id"]: r for r in _read(out / f"evaluation/{bid}_questions.csv")}
        expected = {"original", "manual_1", "manual_2", "manual_3", "act_seed0", "act_seed1",
                    "control_length", "control_named_person", "control_formal",
                    "control_question", "tfidf_cart", "tfidf_rulefit", "source_only"}
        assert set(questions) == expected
        # Without content_validity.csv no candidate can be accepted.
        assert all(questions[c]["acceptance"] in ("pending", "rejected")
                   for c in expected if c.startswith(("manual", "act")))
        items = _read(out / f"evaluation/{bid}_items.csv")
        n_items = costs[bid].n_items
        assert len(items) == n_items * len(expected)

    # The calls stay within the estimate made before the run.
    answer_calls = len(answerer.calls)
    assert answer_calls <= sum(c.answers_with_cache for c in costs.values())
    rewrites = sum(c["kind"] == "rewrite" for c in optimizer.calls)
    critiques = sum(c["kind"] == "critique" for c in optimizer.calls)
    assert critiques == sum(c.optimizer_min for c in costs.values()) // 2
    assert critiques + rewrites <= sum(c.optimizer_max for c in costs.values())
    assert all(c["temperature"] == 0.0 for c in answerer.calls)

    report = (out / "report.md").read_text(encoding="utf-8")
    for heading in ("## Calls", "## B1:", "## B2:", "### Questions", "### Baselines",
                    "### Acceptance", "### Per source", "### ACT runs"):
        assert heading in report
    assert chr(0x2014) not in report and chr(0x2013) not in report
    b1 = next(e for e in evaluations if e.boundary.id == "B1")
    assert b1.evaluable_sources == ["NLAS"]

    # A second run is answered entirely from the cache.
    answerer2, optimizer2 = FakeAnswerer(), FakeOptimizer()
    run_all(config, backends={"answerer": answerer2, "optimizer": optimizer2},
            boundary_ids=["B2"])
    assert answerer2.calls == [] and optimizer2.calls == []
    assert time.monotonic() - started < 60


def test_budget_guard_stops_before_any_call(tmp_path):
    config = load_config(make_config(tmp_path))
    answerer, optimizer = FakeAnswerer(), FakeOptimizer()
    try:
        run_all(config, backends={"answerer": answerer, "optimizer": optimizer}, max_calls=10)
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("the budget guard did not stop the run")
    assert answerer.calls == [] and optimizer.calls == []


def test_cli_estimate_makes_no_call(tmp_path, capsys):
    path = make_config(tmp_path)
    assert main(["estimate", "--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert "| B1 |" in out and "| B2 |" in out
    assert main(["run-act", "--config", str(path), "--dry-run"]) == 0
    assert main(["run-act", "--config", str(path), "--max-calls", "5"]) == 2
