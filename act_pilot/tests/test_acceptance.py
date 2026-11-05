import pytest

from act_pilot.acceptance import (
    SourceComparison,
    assess,
    load_content_validity,
    write_validity_template,
)
from act_pilot.metrics import PairedDifference
from conftest import HOMINEM

Q = "Is the attack made before the person has spoken?"


def _assess(diff, cand=0.8, base=0.7, sources=(), scores=None):
    return assess(
        boundary=HOMINEM,
        questions=[Q],
        difference=diff,
        candidate_ba=cand,
        best_baseline="tfidf_cart",
        best_baseline_ba=base,
        sources=list(sources),
        validity_scores=scores,
        min_content_validity=2,
    )


GOOD = PairedDifference(0.10, 0.02, 0.18, 5000, 0.95)
ZERO = PairedDifference(0.05, -0.01, 0.11, 5000, 0.95)


def test_pending_without_validity_file():
    result = _assess(GOOD)
    assert result.status == "pending"
    assert result.criteria["i"].status == "pass"
    assert result.criteria["ii"].status == "pass"
    assert result.criteria["iii"].status == "pending"


def test_accepted_with_scores():
    scores = {(HOMINEM, Q.lower()): [3, 2]}
    assert _assess(GOOD, scores=scores).status == "accepted"


def test_rejected_by_interval_even_if_validity_missing():
    result = _assess(ZERO)
    assert result.status == "rejected" and result.criteria["i"].status == "fail"


def test_rejected_by_baseline_globally_or_in_a_source():
    assert _assess(GOOD, cand=0.65).criteria["ii"].status == "fail"
    within = [SourceComparison("NLAS", 0.6, "tfidf_rulefit", 0.7)]
    result = _assess(GOOD, sources=within)
    assert result.status == "rejected"
    assert "NLAS" in result.criteria["ii"].detail


def test_low_score_rejects():
    scores = {(HOMINEM, Q.lower()): [3, 1]}
    result = _assess(GOOD, scores=scores)
    assert result.status == "rejected" and result.criteria["iii"].status == "fail"


def test_validity_file_round_trip(tmp_path):
    template = tmp_path / "t.csv"
    write_validity_template(template, [(HOMINEM, "manual_1", Q), (HOMINEM, "act_seed0", Q.upper())])
    lines = template.read_text().splitlines()
    assert len(lines) == 2  # the duplicate question (case aside) is written once
    filled = tmp_path / "cv.csv"
    filled.write_text(
        "boundary,question,rater,content_validity\n"
        f"ad_hominem_poisoning_vs_abusive,\"{Q}\",r1,3\n"
        f"ad_hominem_poisoning_vs_abusive,\"  {Q.upper()} \",r2,2\n"
        "ad_hominem_poisoning_vs_abusive,Another question?,r1,\n"
    )
    scores = load_content_validity(filled)
    assert scores[(HOMINEM, Q.lower())] == [3, 2]
    assert scores[(HOMINEM, "another question?")] == []
    assert load_content_validity(tmp_path / "missing.csv") is None
    bad = tmp_path / "bad.csv"
    bad.write_text("boundary,question,content_validity\n" f"{HOMINEM},q?,5\n")
    with pytest.raises(ValueError):
        load_content_validity(bad)
