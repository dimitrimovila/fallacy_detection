import pytest

from act_pilot.act import Candidate, run_act, select_candidate
from act_pilot.config import ActConfig, Boundary, RoleConfig
from act_pilot.data import LabeledItem
from act_pilot.llm import CallLog, DiskCache, LLMClient
from fakes import FakeAnswerer, FakeOptimizer

BOUNDARY = Boundary(
    id="BX",
    scheme="popular_opinion",
    positive="good_argumentation",
    negative="ad_populum",
    seed_question="Is the popularity of A a good reason to believe A?",
    scheme_text="Many accept A. Therefore A.",
    question_function="It tests the ground of the presumption.",
    positive_description="acceptable",
    negative_description="popularity as proof",
    banned_terms=("ad populum", "fallacy"),
)


def _items(n_pos, n_neg, prefix):
    items = []
    for k in range(n_pos):
        items.append(LabeledItem(f"{prefix}p{k}", f"{prefix} positive text {k}", "S", "g", 1, 0))
    for k in range(n_neg):
        items.append(LabeledItem(f"{prefix}n{k}", f"{prefix} negative text {k}", "S", "a", 0, 0))
    return items


def _clients(tmp_path, answerer_backend, optimizer_backend):
    cache, log = DiskCache(tmp_path / "c"), CallLog(tmp_path / "l.jsonl")
    a = LLMClient(RoleConfig("answerer", "ans"), answerer_backend, cache, log, max_workers=1)
    o = LLMClient(RoleConfig("optimizer", "opt"), optimizer_backend, cache, log, max_workers=1)
    return a, o


def test_selection_uses_validation_not_training(tmp_path):
    train, val = _items(6, 6, "t"), _items(4, 4, "v")
    q_train = "Does the text rest on documented grounds rather than on supporters alone?"
    q_val = "Does the text rest on stated reasons rather than on supporters alone?"

    def rule(question, text):
        split, polarity, _, k = text.split()
        right = "yes" if polarity == "positive" else "no"
        wrong = "no" if right == "yes" else "yes"
        if question == q_train:  # perfect on training, half wrong on validation
            return right if split == "t" or int(k) % 2 == 0 else wrong
        if question == q_val:  # a quarter wrong on training, perfect on validation
            return right if split == "v" or int(k) % 4 != 0 else wrong
        return "yes"  # the seed question: balanced accuracy 0.5 everywhere

    optimizer = FakeOptimizer(scripted=[f"Question: {q_train}", f"Question: {q_val}"])
    a, o = _clients(tmp_path, FakeAnswerer(rule), optimizer)
    run = run_act(BOUNDARY, train, val, a, o, ActConfig(steps=2, examples_per_group=3), seed=0,
                  outer_fold=0)
    by_q = {c.question: c for c in run.candidates}
    assert by_q[q_train].train_ba > by_q[q_val].train_ba
    assert by_q[q_val].val_ba > by_q[q_train].val_ba
    assert run.selected_question == q_val
    assert sum(c.selected for c in run.candidates) == 1


def test_calls_match_the_loop(tmp_path):
    train, val = _items(10, 10, "t"), _items(3, 3, "v")
    answerer, optimizer = FakeAnswerer(), FakeOptimizer()
    a, o = _clients(tmp_path, answerer, optimizer)
    settings = ActConfig(steps=4, examples_per_group=2)
    run = run_act(BOUNDARY, train, val, a, o, settings, seed=1, outer_fold=0)
    assert len(run.candidates) == 5
    assert [c.origin for c in run.candidates] == ["seed"] + ["rewrite"] * 4
    assert len(answerer.calls) == 5 * (len(train) + len(val))
    assert [c["kind"] for c in optimizer.calls] == ["critique", "rewrite"] * 4
    # At most m correct and m wrong examples per group are shown.
    critique = optimizer.calls[0]["user"]
    assert critique.count("- [should be") <= 2 * 2 * 2


def test_rejected_rewrite_is_retried_then_recorded(tmp_path):
    train, val = _items(5, 5, "t"), _items(2, 2, "v")
    optimizer = FakeOptimizer(scripted=[
        "Question: Is this an ad populum argument?",
        "Question: Is this a fallacy?",
        "nothing useful",
    ])
    a, o = _clients(tmp_path, FakeAnswerer(), optimizer)
    settings = ActConfig(steps=2, examples_per_group=2, max_rewrite_attempts=3)
    run = run_act(BOUNDARY, train, val, a, o, settings, seed=0, outer_fold=0)
    rejected = [c for c in run.candidates if c.status == "rejected"]
    assert len(rejected) == 1 and rejected[0].rewrite_attempts == 3
    assert "no question found" in rejected[0].violations[0]
    second_attempt = [c["user"] for c in optimizer.calls if c["kind"] == "rewrite"][1]
    assert "previous proposal was rejected" in second_attempt
    assert run.candidates[-1].status == "evaluated"


def test_seeds_change_the_examples_shown(tmp_path):
    train, val = _items(30, 30, "t"), _items(4, 4, "v")
    shown = []
    for seed in (0, 1):
        optimizer = FakeOptimizer()
        a, o = _clients(tmp_path / str(seed), FakeAnswerer(), optimizer)
        run_act(BOUNDARY, train, val, a, o, ActConfig(steps=1, examples_per_group=3),
                seed=seed, outer_fold=0)
        shown.append(optimizer.calls[0]["user"])
    assert shown[0] != shown[1]


def test_select_candidate_ties_go_to_the_earliest():
    cands = [Candidate(0, "a", "seed", "evaluated", val_ba=0.7),
             Candidate(1, "b", "rewrite", "evaluated", val_ba=0.8),
             Candidate(2, "c", "rewrite", "evaluated", val_ba=0.8),
             Candidate(3, "d", "rewrite", "rejected")]
    assert select_candidate(cands).question == "b"
    with pytest.raises(ValueError):
        select_candidate([Candidate(0, "a", "seed", "rejected")])
