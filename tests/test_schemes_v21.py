"""The diagrams of version 2.1 (schemes/v3/) and of their variant 2.1b (schemes/v3b/).

What is checked: the diagrams are complete; the rule for an undetermined answer of each
CQ is the one written with the diagrams; the question with the answers ``a`` and ``b``
goes through prompt, parser and traversal like a yes/no one; the two wordings of the
transfer question of example are two questions with the same arcs, one per folder; the
questions asked are exactly those without an identical 2.0 question, and the answers of
the others are taken over from the 2.0 runs with their provenance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from argfallacy.aggregate import ABSTAIN, SPLIT, load_burden, routes
from argfallacy.aggregate.soft import OUTSIDE, traverse_soft
from argfallacy.client import RunConfig, execute, load_models, plan
from argfallacy.client.reuse import assemble, identical_questions, questions_to_ask
from argfallacy.parse import parse_answers, parse_run, summarise
from argfallacy.paths import REPO_ROOT, SCHEMES_ROOT
from argfallacy.prompts import render_stage2
from argfallacy.schemes import SchemeError, load_all, traverse
from fakes.llm import FakeBackend

FAKE_MODELS = load_models(Path(__file__).parent / "fakes" / "models.yaml")
V21, V21B = load_all(SCHEMES_ROOT / "v3"), load_all(SCHEMES_ROOT / "v3b")

# The rule for an undetermined or inapplicable answer, as written with the diagrams:
# the arc to follow, ABSTAIN for no verdict, SPLIT for half the weight on each arc.
RULES = {
    "example": {"CQ1": "no", "CQ2": "no", "CQ3": "no", "CQ4": "b", "CQ5": "no"},
    "analogy": {"CQ1": ABSTAIN, "CQ2": "no", "CQ3": ABSTAIN, "CQ4": "no"},
    "cause_to_effect": {cq: "no" for cq in ("CQ1", "CQ2", "CQ3", "CQ4", "CQ5")},
    "correlation_to_cause": {"CQ1": ABSTAIN, **{f"CQ{i}": "no" for i in range(2, 7)}},
    "ad_hominem": {f"CQ{i}": "no" for i in range(1, 6)},
    "expert_opinion": {"CQ1": ABSTAIN, "CQ2": "no", "CQ3": ABSTAIN, "CQ4": "yes",
                       "CQ5": SPLIT, "CQ6": ABSTAIN},
    "popular_opinion": {"CQ1": "no", "CQ2": "no", "CQ3": "no", "CQ4": "yes", "CQ5": "no",
                        "CQ6": "no"},
    "slippery_slope": {"CQ1": ABSTAIN, "CQ2": ABSTAIN, "CQ3": ABSTAIN},
}
ASKED = {"analogy": ["CQ2", "CQ4"], "cause_to_effect": ["CQ3", "CQ4"],
         "example": ["CQ2", "CQ3", "CQ4", "CQ5"], "expert_opinion": ["CQ2", "CQ6"],
         "popular_opinion": ["CQ1", "CQ2", "CQ3", "CQ4", "CQ5", "CQ6"],
         "slippery_slope": ["CQ3"]}


@pytest.mark.parametrize("folder", ["v3", "v3b"])
def test_every_diagram_is_complete_and_keeps_the_verdict_menu(folder, vocab):
    v20 = load_all(SCHEMES_ROOT / "v2")
    for scheme_id, scheme in load_all(SCHEMES_ROOT / folder).items():
        # the loader has already refused unreachable nodes, unasked CQs, dangling arcs,
        # cycles and arcs that do not cover the answer space
        assert scheme.terminals <= set(vocab["terminals"])
        assert scheme.terminals <= v20[scheme_id].terminals, scheme_id
        assert scheme.cq_order == [f"CQ{i}" for i in range(1, len(scheme.cq_order) + 1)] or (
            scheme_id == "example" and folder == "v3b")


@pytest.mark.parametrize("folder", ["v3", "v3b"])
def test_the_rule_for_undetermined_answers_is_the_one_of_the_diagrams(folder):
    schemes = load_all(SCHEMES_ROOT / folder)
    table = load_burden(folder, schemes)
    for scheme_id, scheme in schemes.items():
        expected = {("CQ5b" if cq == "CQ5" and folder == "v3b" and scheme_id == "example"
                     else cq): rule for cq, rule in RULES[scheme_id].items()}
        assert routes(scheme, table[scheme_id]) == expected, scheme_id
        for cq, rule in expected.items():
            assert rule in (ABSTAIN, SPLIT) or rule in scheme.nodes[cq]


def test_an_entry_that_names_its_arc_keeps_its_type_under_the_other_policies():
    table = load_burden("v3", V21)
    example = V21["example"]
    assert table["example"]["CQ4"] == "D"
    assert routes(example, table["example"], "branch")["CQ4"] == SPLIT
    assert routes(example, table["example"], "charitable")["CQ4"] == SPLIT
    assert routes(example, table["example"], "stop")["CQ4"] == ABSTAIN


def test_the_question_with_answers_a_and_b_is_rendered_with_its_letters():
    example = V21["example"]
    assert example.answer_space("CQ4") == ("a", "b")
    rendered = render_stage2({"item_id": "i", "text": "Some argument."}, example, "CQ4", "v4")
    assert rendered.json_schema["properties"]["answer"]["enum"] == [
        "a", "b", "cannot_be_determined", "na"]
    # the alternatives stay on their own lines, as written
    assert "generalization:\na) The argument selectively" in rendered.text
    assert "\nb) The cited examples are too few" in rendered.text
    assert "* `a`: for the argument in this text, the answer to the question is a" in (
        rendered.text)
    assert "no basis at all for answering `a` or `b`" in rendered.text


def test_a_and_b_lead_to_their_verdicts_and_undetermined_goes_to_b():
    example = V21["example"]
    path = {"CQ1": "no", "CQ2": "yes"}
    assert traverse(example, {**path, "CQ4": "a"}).verdict == "cherry_picking"
    assert traverse(example, {**path, "CQ4": "b"}).verdict == "hasty_generalization"
    assert traverse(example, {**path, "CQ4": "cannot_be_determined"}).verdict is None
    routing = routes(example, load_burden("v3", V21)["example"])
    masses = {cq: {"yes": 0.0, "no": 1.0} for cq in example.nodes}
    masses["CQ2"] = {"yes": 1.0, "no": 0.0}
    masses["CQ4"] = {"a": 0.25, "b": 0.25, OUTSIDE: 0.5}
    terminals, abstained, _ = traverse_soft(example, masses, routing)
    assert abstained == 0.0
    assert terminals["cherry_picking"] == pytest.approx(0.25)
    assert terminals["hasty_generalization"] == pytest.approx(0.75)


def test_the_two_wordings_of_the_transfer_question_take_the_same_place():
    main, variant = V21["example"], V21B["example"]
    assert "CQ5" in main.cqs and "CQ5b" not in main.cqs
    assert "CQ5b" in variant.cqs and "CQ5" not in variant.cqs
    assert main.nodes["CQ3"]["no"].target == "CQ5"
    assert variant.nodes["CQ3"]["no"].target == "CQ5b"
    arcs = {a: (e.target, e.is_terminal) for a, e in main.nodes["CQ5"].items()}
    assert arcs == {a: (e.target, e.is_terminal) for a, e in variant.nodes["CQ5b"].items()}
    assert arcs == {"yes": ("weak_analogy", True), "no": ("good_argumentation", True)}
    assert main.cqs["CQ5"].text.startswith("Does the argument infer that a distinct target")
    assert variant.cqs["CQ5b"].text == (
        "Is the property from the example generalized to a target case that is too "
        "different?")
    same = identical_questions("v3b", "v3")
    assert {(s, cq) for s, scheme in V21B.items() for cq in scheme.cq_order} - {
        (s, cq) for s, cqs in same.items() for cq in cqs} == {("example", "CQ5b")}


def test_the_runs_ask_exactly_the_questions_without_an_identical_earlier_one():
    asked = {s: cqs for s, cqs in questions_to_ask("v3", "v2").items() if cqs}
    assert asked == ASKED
    assert RunConfig.load(REPO_ROOT / "configs/full_v4.yaml").cqs == ASKED
    assert RunConfig.load(REPO_ROOT / "configs/full_v4_transfer2.yaml").cqs == {
        "example": ["CQ5b"]}
    # the same question under another id is reused; a new definition is a new question
    same = identical_questions("v3", "v2")
    assert same["ad_hominem"]["CQ1"] == "CQ4"
    assert same["expert_opinion"]["CQ5"] == "CQ4.1"
    assert "CQ6" not in same["expert_opinion"]   # 2.0 CQ5.1 with definitions added
    assert "CQ3" not in same["slippery_slope"]   # 2.0 CQ3 without its definitions


def _items():
    return pd.DataFrame([
        {"item_id": "e1", "gold_scheme": "example", "text": "An argument from example."},
        {"item_id": "e2", "gold_scheme": "example", "text": "Another one."},
        {"item_id": "s1", "gold_scheme": "slippery_slope", "text": "A slope."},
    ])


def _config(name, folder, prompts, cqs=None):
    data = {"name": name, "models": ["fake"], "stages": ["stage2"],
            "scheme_condition": "gold", "schemes": folder, "prompt_version": prompts,
            **({"cqs": cqs} if cqs is not None else {})}
    return RunConfig(raw=data, **data)


def test_only_the_listed_questions_are_planned():
    config = _config("t", "v3", "v4", {"example": ["CQ4"]})
    calls = plan(config, _items().to_dict("records"), FAKE_MODELS)
    assert sorted({(c.item_id, c.cq_id) for c in calls}) == [("e1", "CQ4"), ("e2", "CQ4")]
    with pytest.raises(SchemeError, match="example CQ9"):
        plan(_config("t", "v3", "v4", {"example": ["CQ9"]}), [], FAKE_MODELS)
    with pytest.raises(SchemeError, match="prompt_version"):
        _config("t", "v3", "v3")


def test_new_and_reused_answers_assemble_into_the_two_variants(tmp_path):
    items = _items()
    records = items.to_dict("records")
    runs = {
        "old": _config("old", "v2", "v3"),
        "new": _config("new", "v3", "v4", questions_to_ask("v3", "v2")),
        "transfer2": _config("transfer2", "v3b", "v4", {"example": ["CQ5b"]}),
    }
    for run_id, config in runs.items():
        execute(config, records, run_id=run_id, runs_dir=tmp_path, backend=FakeBackend(),
                models=FAKE_MODELS, sleep=lambda _: None)
        parse_run(tmp_path / run_id)

    answers = pd.read_csv(tmp_path / "new" / "answers.csv")
    letters = answers[answers["cq_id"] == "CQ4"]
    assert set(letters["item_id"]) == {"e1", "e2"}
    assert letters["parse_ok"].all() and letters["answer"].isin(["a", "b"]).any()
    assert letters["answer"].isin(["a", "b", "cannot_be_determined", "na"]).all()
    assert (letters["p_logprob_a"] + letters["p_logprob_b"]).between(0, 1.0001).all()
    summary = pd.read_csv(tmp_path / "new" / "summary.csv")
    row = summary[summary["cq_id"] == "CQ4"].iloc[0]
    assert row["p_logprob"] == row["p_logprob_a"]

    main = assemble("main", "v3", ["new", "old"], items, tmp_path)
    table = pd.read_csv(tmp_path / "main" / "summary.csv", dtype=str)
    assert len(table) == 2 * 5 + 3
    assert main["answers_by_cq"]["example CQ1"] == {"old": "CQ1"}
    assert main["answers_by_cq"]["example CQ4"] == {"new": "CQ4"}
    assert main["answers_by_cq"]["slippery_slope CQ3"] == {"new": "CQ3"}
    reused = table[table["source_run_id"] == "old"]
    assert set(zip(reused["scheme"], reused["cq_id"], strict=True)) == {
        ("example", "CQ1"), ("slippery_slope", "CQ1"), ("slippery_slope", "CQ2")}
    assert set(table.loc[table["source_run_id"] == "new", "source_prompt_version"]) == {
        "stage2_v4"}
    assert set(reused["source_prompt_version"]) == {"stage2_v3"}

    variant = assemble("variant", "v3b", ["transfer2", "new", "old"], items, tmp_path)
    assert variant["answers_by_cq"]["example CQ5b"] == {"transfer2": "CQ5b"}
    assert variant["answers_by_cq"]["example CQ4"] == {"new": "CQ4"}
    table = pd.read_csv(tmp_path / "variant" / "summary.csv", dtype=str)
    assert "CQ5" not in set(table.loc[table["scheme"] == "example", "cq_id"])

    with pytest.raises(SchemeError, match="already exists"):
        assemble("main", "v3", ["new", "old"], items, tmp_path)
    with pytest.raises(SchemeError, match="no answer in the runs given for: example CQ1"):
        assemble("incomplete", "v3", ["new"], items, tmp_path)


def test_a_parse_without_letters_keeps_the_earlier_columns(schemes):
    frame = parse_answers([], schemes)
    assert not [c for c in frame.columns if c in ("p_logprob_a", "p_logprob_b")]
    assert not [c for c in summarise(frame, schemes).columns if c.endswith("_a")]
