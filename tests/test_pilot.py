"""The pilot report on a run written by hand.

Two expert opinion items, one entry per reasoning condition, five samples per
question.  Every number checked here is known in advance, so a change in how the
report counts shows up as a failed assertion, not as a drift in the real run.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from argfallacy.eval import pilot
from argfallacy.parse import ANSWERS_COLUMNS, summarise

CQS = ["CQ1", "CQ2", "CQ3", "CQ3.1", "CQ4", "CQ4.1"]
ENTRIES = ["m", "m_think"]


def _row(item, cq, sample, entry, answer, p_yes=None):
    row = dict.fromkeys(ANSWERS_COLUMNS)
    p_yes = (1.0 if answer == "yes" else 0.0) if p_yes is None else p_yes
    row.update({
        "run_id": "r", "item_id": item, "stage": "stage2", "scheme_condition": "gold",
        "scheme": "expert_opinion", "cq_id": cq, "sample_index": sample, "model": entry,
        "model_id": "x", "prompt_version": "stage2_v1", "cache_key": f"{item}{cq}{sample}{entry}",
        "from_cache": False, "answer": answer, "confidence": 90.0, "justification": "",
        "p_logprob_yes": p_yes, "p_logprob_no": 1.0 - p_yes,
        "p_logprob_cannot_be_determined": 0.0, "p_logprob_na": 0.0,
        "logprob_partial": False, "parse_ok": True, "finish_reason": "stop",
        "latency_s": 2.0 if entry == "m" else 20.0,
    })
    return row


def _answers() -> pd.DataFrame:
    """Item a: a good argument on every path.  Item b: CQ1 cannot be determined.

    Both entries answer item b's CQ1 with cannot_be_determined in all five samples;
    ``m`` also in sample 1 of item a.  On CQ1 that is 6 calls of 10 for ``m`` (above
    the threshold) and exactly 5 of 10 for ``m_think`` (not above it).
    """
    good = {"CQ1": "yes", "CQ2": "no", "CQ3": "yes", "CQ3.1": "yes", "CQ4": "no",
            "CQ4.1": "yes"}
    rows = []
    for entry in ENTRIES:
        for sample in range(5):
            for cq in CQS:
                answer_a = good[cq]
                if cq == "CQ1" and entry == "m" and sample == 1:
                    answer_a = "cannot_be_determined"
                rows.append(_row("a", cq, sample, entry, answer_a))
                answer = "cannot_be_determined" if cq == "CQ1" else good[cq]
                # at sample 0 the logprobs lean to no on CQ1: the logprob rule goes that way
                p_yes = 0.3 if (cq == "CQ1" and sample == 0) else None
                rows.append(_row("b", cq, sample, entry, answer, p_yes))
        for item in ("a", "b"):
            stage1 = dict.fromkeys(ANSWERS_COLUMNS)
            stage1.update({"run_id": "r", "item_id": item, "stage": "stage1",
                           "scheme_condition": "gold", "sample_index": 0, "model": entry,
                           "answer": "expert_opinion", "confidence": 90.0, "parse_ok": True,
                           "finish_reason": "stop", "latency_s": 1.0,
                           "logprob_partial": False})
            rows.append(stage1)
    return pd.DataFrame(rows, columns=ANSWERS_COLUMNS)


@pytest.fixture
def data(schemes) -> pilot.PilotData:
    answers = _answers()
    items = pd.DataFrame({
        "item_id": ["a", "b"], "text": ["", ""], "source": ["logic", "logic"],
        "in_test_set": [True, True], "eleni_id": [None, None],
        "gold_scheme": ["expert_opinion", "expert_opinion"],
        "gold_fallacy": ["good_argumentation", "irrelevant_authority"],
    })
    manifest = {
        "config": {"models": ENTRIES, "items": {"seed": 1, "min_per_scheme": 1},
                   "scheme_condition": "gold", "generation": {"max_tokens": 512}},
        "models": {e: {"model_id": "x", "max_tokens": 512, "max_concurrency": 1, "samples": 5,
                       "temperature": 1.0, "top_p": 1.0, "top_k": 0}
                   for e in ENTRIES},
        "calls_planned": len(answers), "calls_executed": len(answers), "calls_failed": 0,
        "calls_from_cache": 0, "started_at": "2026-09-27T00:00:00",
        "finished_at": "2026-09-27T01:00:00", "prompt_versions": ["stage2_v1"],
        "schemes_version": {"tag": "1.1"},
    }
    annotations = pd.DataFrame(columns=["item_id", "annotator", "sheet", "row", "field",
                                        "value", "raw"])
    return pilot.PilotData(run_id="r", answers=answers, summary=summarise(answers, schemes),
                           manifest=manifest, items=items, annotations=annotations,
                           schemes=schemes, raw_path=None)


def test_the_stopping_rule_lists_only_what_is_above_threshold(data):
    shares = pilot.cq_shares(data)
    assert shares.loc[("expert_opinion", "CQ1", "m_think"), "cannot_be_determined"] == 0.5
    flagged = pilot.stopping_rule(shares).reset_index()
    assert list(zip(flagged["cq_id"], flagged["model"], strict=True)) == [("CQ1", "m")]
    assert flagged["reason"].iloc[0] == "cannot_be_determined 60%"


def test_the_three_rules_on_an_answer_outside_the_arcs(data):
    by_rule = {rule: pilot.traversals(data, rule).set_index(["entry", "item_id"])
               for rule in pilot.RULES}
    diagram = by_rule["diagram"].loc[("m", "b")]
    assert pd.isna(diagram["verdict"]) and diagram["stopped_at"] == "CQ1"
    assert diagram["stop_answer"] == "cannot_be_determined"
    # toward_good: CQ1 yes continues toward good_argumentation
    assert by_rule["toward_good"].loc[("m", "b"), "verdict"] == "good_argumentation"
    # logprob: p(yes) 0.3 at CQ1, so the arc no, to irrelevant_authority
    assert by_rule["logprob"].loc[("m", "b"), "verdict"] == "irrelevant_authority"
    assert by_rule["diagram"].loc[("m", "a"), "verdict"] == "good_argumentation"


def test_the_verdicts_follow_the_majority_not_sample_zero(data, schemes):
    """Sample 0 of item b says cannot_be_determined on CQ1, the other four say yes."""
    answers = data.answers
    four = ((answers["model"] == "m") & (answers["item_id"] == "b") & (answers["cq_id"] == "CQ1")
            & (answers["sample_index"] > 0))
    answers.loc[four, "answer"] = "yes"
    data.summary = summarise(answers, schemes)
    diagram = pilot.traversals(data, "diagram").set_index(["entry", "item_id"])
    assert diagram.loc[("m", "b"), "verdict"] == "good_argumentation"


def test_toward_good_has_no_arc_where_no_path_reaches_good_argumentation(schemes):
    ad_hominem = schemes["ad_hominem"]
    assert all(pilot.toward_good_arc(ad_hominem, node) is None for node in ad_hominem.nodes)
    assert pilot.toward_good_arc(schemes["expert_opinion"], "CQ4") == "no"
    assert pilot.toward_good_arc(schemes["expert_opinion"], "CQ1") == "yes"


def test_no_verdict_is_wrong_and_lowers_coverage(data):
    scores = pilot.verdict_scores(data).set_index(["rule", "entry"])
    assert scores.loc[("diagram", "m"), "coverage"] == 0.5
    assert scores.loc[("diagram", "m"), "accuracy_fine"] == 0.5
    assert scores.loc[("logprob", "m"), "accuracy_fine"] == 1.0


def test_the_tokens_of_a_reasoning_field_are_counted_when_usage_says_zero(tmp_path):
    lines = [
        {"model": "k", "raw_response": {
            "usage": {"prompt_tokens": 500, "completion_tokens": 120},
            "raw": {"choices": [{"message": {"content": "{}", "reasoning_content": "abcd"}}]}}},
        {"model": "q", "raw_response": {
            "usage": {"prompt_tokens": 500, "completion_tokens": 100,
                      "completion_tokens_details": {"reasoning_tokens": 80}},
            "raw": {"choices": [{"message": {"content": "{}"}}]}}},
        # a call that failed is no answer: it must not pull the means down
        {"model": "q", "raw_response": None, "error": "timeout"},
    ]
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    tokens = pilot.token_counts(path)
    assert tokens.loc["k", "reasoning_tokens"] == 0 and tokens.loc["k", "reasoning_chars"] == 4
    assert tokens.loc["q", "answer_tokens"] == 20 and tokens.loc["q", "prompt_tokens"] == 500


def test_the_output_tokens_and_the_layout_of_the_json(tmp_path):
    def line(completion, finish, content, stage="stage2"):
        return {"model": "g", "stage": stage, "raw_response": {
            "usage": {"prompt_tokens": 10, "completion_tokens": completion},
            "finish_reason": finish, "content": content}}

    lines = [line(100, "stop", '{"answer": "yes", "confidence": 9}'),
             line(300, "stop", '{\n  "answer": "no"}'),
             line(32768, "length", "We need to"),
             line(50, "stop", '{"scheme": "analogy"}', stage="stage1")]
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    records = pilot.raw_records(path)

    out = pilot.output_tokens(records).loc["g"]
    assert out["calls"] == 4 and out["median"] == 200 and out["max"] == 32768
    assert out["length"] == 1
    share = pilot.layout_share(records).loc["g"]
    assert share["stage2"] == pytest.approx(1 / 3) and share["stage1"] == 1.0


def test_the_report_puts_the_flagged_questions_first(data):
    text = pilot.render(data)
    first, _, rest = text.partition("## 2. Integrity")
    assert "## 1. Stopping rule" in first
    assert "| expert_opinion | CQ1 | m | cannot_be_determined 60% |" in first
    assert "| expert_opinion | CQ1 | m_think |" not in first
    assert "Tokens: no `raw.jsonl`" in rest
