import pytest

from act_pilot.config import load_config, load_manual_rewrites
from act_pilot.constraints import check_question, document_frequency
from act_pilot.prompts import (
    GroupSummary,
    ShownExample,
    critique_text,
    parse_rewrite,
    rewrite_text,
)
from conftest import MANUAL, PILOT_CONFIG


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ACT_PILOT_DATA", "/data")
    return load_config(PILOT_CONFIG)


def _check(boundary, question, **kwargs):
    return check_question(
        question,
        seed_question=boundary.seed_question,
        banned_terms=boundary.banned_terms,
        max_logical_ops=2,
        max_chars=300,
        **kwargs,
    )


def test_seed_and_manual_questions_pass(config):
    manual = load_manual_rewrites(MANUAL, config.boundaries)
    for bid, boundary in config.boundaries.items():
        assert _check(boundary, boundary.seed_question) == []
        for q in manual[bid]:
            assert _check(boundary, q) == [], q


def test_controls_are_yes_no_questions(config):
    b1 = config.boundaries["B1"]
    for c in config.controls:
        assert _check(b1, c.question) == []


@pytest.mark.parametrize(
    "question, fragment",
    [
        ("Is this an ad populum argument?", "forbidden term"),
        ("Does the text commit a fallacy?", "forbidden term"),
        ("Why do many people accept A?", "yes/no question"),
        ("Is A popular and true and old or new?", "logical conjunctions"),
        ("Is A popular", "question mark"),
        ("Is A popular? Is it true?", "more than one question"),
    ],
)
def test_violations(config, question, fragment):
    assert any(fragment in v for v in _check(config.boundaries["B1"], question))


def test_example_specific_words_are_flagged(config):
    b1 = config.boundaries["B1"]
    corpus = ["Everyone drinks kombucha now."] + [f"Many people agree on topic {i}." for i in
                                                  range(40)]
    df = document_frequency(corpus)
    v = _check(b1, "Does the text mention kombucha as a reason?", example_texts=corpus[:1],
               corpus_df=df)
    assert any("kombucha" in x for x in v)
    # The same word is not flagged when it is frequent in the training texts.
    frequent = {"kombucha": 0.5}
    assert _check(b1, "Does the text mention kombucha as a reason?",
                  example_texts=corpus[:1], corpus_df=frequent) == []


def test_capitalised_names_are_flagged(config):
    v = _check(config.boundaries["B2"], "Is the attack on Trump made before the debate?")
    assert any("Trump" in x for x in v)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Question: Is A widely accepted for a stated reason?",
         "Is A widely accepted for a stated reason?"),
        ("Here it is.\n**Question:** \"Is the attack pre-emptive?\"", "Is the attack pre-emptive?"),
        ("Is the attack pre-emptive (yes/no)?", "Is the attack pre-emptive?"),
        ("<think>Question: draft?</think>Question: Final one?", "Final one?"),
        ("No question here.", None),
        ("", None),
    ],
)
def test_parse_rewrite(raw, expected):
    assert parse_rewrite(raw) == expected


def test_prompts_do_not_name_the_classes(config):
    b1 = config.boundaries["B1"]
    example = ShownExample("Everyone does it.", "no", "yes", "It is popular.")
    group = GroupSummary("answered yes", 3, 1, 2, 0.44, [example], [example])
    text = critique_text(b1, b1.seed_question, [group, group], weighted_gini=0.4,
                         balanced_accuracy=0.6, max_example_chars=100, show_reasoning=True)
    for name in (b1.positive, b1.negative, "ad populum", "good argumentation"):
        assert name not in text.lower()
    assert "Everyone does it." in text and "answerer's reasoning" in text
    rewrite = rewrite_text(b1, b1.seed_question, "A critique.", max_logical_ops=2,
                           previous_violations=["names a forbidden term"],
                           rejected_question="Is it ad populum?")
    assert "at most 2 logical conjunctions" in rewrite
    assert "previous proposal was rejected" in rewrite
