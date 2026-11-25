"""The diagrams of version 2.1 (schemes/v4/).

What is checked: the diagrams are complete; the rule for an undetermined answer of each
CQ is the one written with the diagrams; the question with the answers ``a`` and ``b``
goes through prompt, parser and traversal like a yes/no one; a question that appears twice
is asked once; the questions without an identical 1.1 or 2.0 question are the 12 asked
anew, and the answers of the others are taken over from the earlier runs with their
provenance.
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
from argfallacy.paths import SCHEMES_ROOT
from argfallacy.prompts import render_stage2
from argfallacy.schemes import SchemeError, load_all, traverse
from argfallacy.schemes.loader import load_scheme
from fakes.llm import FakeBackend

FAKE_MODELS = load_models(Path(__file__).parent / "fakes" / "models.yaml")
V4 = load_all(SCHEMES_ROOT / "v4")

# The rule for an undetermined or inapplicable answer, as written with the diagrams:
# the arc to follow, ABSTAIN for no verdict, SPLIT for half the weight on each arc.
RULES = {
    "example": {"CQ1": "no", "CQ2": "no", "CQ3": "no", "CQ4": "b", "CQ5": "no"},
    "analogy": {"CQ1": ABSTAIN, "CQ2": "no", "CQ3": ABSTAIN, "CQ4": "no"},
    "cause_to_effect": {"CQ1": "no", "CQ2": "no", "CQ3": ABSTAIN, "CQ4": "no", "CQ5": "no"},
    "correlation_to_cause": {"CQ1": ABSTAIN, **{f"CQ{i}": "no" for i in range(2, 7)}},
    "ad_hominem": {f"CQ{i}": "no" for i in range(1, 6)},
    "expert_opinion": {"CQ1": ABSTAIN, "CQ2": "no", "CQ3": ABSTAIN, "CQ4": "yes",
                       "CQ5": SPLIT, "CQ6": ABSTAIN},
    "popular_opinion": {"CQ1": "yes", **{f"CQ{i}": "no" for i in range(2, 10)}},
    "slippery_slope": {"CQ1": ABSTAIN, "CQ2": ABSTAIN, "CQ3": ABSTAIN},
}
# the questions without an identical question in 1.1 or 2.0, asked to the five entries
NEW = {"analogy": ["CQ2", "CQ4"], "example": ["CQ2", "CQ3", "CQ4", "CQ5"],
       "expert_opinion": ["CQ2", "CQ6"], "popular_opinion": ["CQ1", "CQ2", "CQ5", "CQ6"]}


def test_every_diagram_is_complete_and_keeps_the_verdict_menu(vocab):
    v20 = load_all(SCHEMES_ROOT / "v2")
    for scheme_id, scheme in V4.items():
        # the loader has already refused unreachable nodes, unasked CQs, dangling arcs,
        # cycles and arcs that do not cover the answer space
        assert scheme.terminals <= set(vocab["terminals"])
        assert scheme.terminals <= v20[scheme_id].terminals, scheme_id
        assert scheme.cq_order == [f"CQ{i}" for i in range(1, len(scheme.cq_order) + 1)]
    popular = V4["popular_opinion"]
    assert {q: c.same_as for q, c in popular.cqs.items() if c.same_as} == {
        "CQ7": "CQ2", "CQ8": "CQ4", "CQ9": "CQ6"}
    assert popular.nodes["CQ1"]["yes"].target == "CQ2"
    assert popular.nodes["CQ5"]["no"].target == "CQ7"
    assert {a: e.target for a, e in popular.nodes["CQ6"].items()} == {
        "yes": "hasty_generalization", "no": "good_argumentation"}
    assert {a: e.target for a, e in popular.nodes["CQ9"].items()} == {
        "yes": "hasty_generalization", "no": "ad_populum"}


def test_the_rule_for_undetermined_answers_is_the_one_of_the_diagrams():
    table = load_burden("v4", V4)
    for scheme_id, scheme in V4.items():
        assert routes(scheme, table[scheme_id]) == RULES[scheme_id], scheme_id
        for cq, rule in RULES[scheme_id].items():
            assert rule in (ABSTAIN, SPLIT) or rule in scheme.nodes[cq]


def test_the_question_with_answers_a_and_b_is_an_exception_that_falls_back_to_b():
    table = load_burden("v4", V4)
    example = V4["example"]
    assert table["example"]["CQ4"] == "E"
    assert routes(example, table["example"])["CQ4"] == "b"
    assert routes(example, table["example"], "charitable")["CQ4"] == "b"
    assert routes(example, table["example"], "branch")["CQ4"] == SPLIT


def test_an_entry_that_names_its_arc_keeps_its_type_under_the_other_policies():
    table = load_burden("v4", V4)
    popular = V4["popular_opinion"]
    assert table["popular_opinion"]["CQ7"] == "E"
    assert routes(popular, table["popular_opinion"])["CQ7"] == "no"
    assert routes(popular, table["popular_opinion"], "charitable")["CQ7"] == "yes"
    assert routes(popular, table["popular_opinion"], "branch")["CQ7"] == SPLIT
    assert routes(popular, table["popular_opinion"], "stop")["CQ7"] == ABSTAIN


def test_the_question_with_answers_a_and_b_is_rendered_with_its_letters():
    example = V4["example"]
    assert example.answer_space("CQ4") == ("a", "b")
    rendered = render_stage2({"item_id": "i", "text": "Some argument."}, example, "CQ4", "v5")
    assert rendered.json_schema["properties"]["answer"]["enum"] == [
        "a", "b", "cannot_be_determined", "na"]
    # the alternatives stay on their own lines, as written
    assert "generalization:\na) The argument selectively" in rendered.text
    assert "\nb) The cited examples are too few" in rendered.text
    assert "* `a`: for the argument in this text, the answer to the question is a" in (
        rendered.text)
    assert "no basis at all for answering `a` or `b`" in rendered.text


def test_a_and_b_lead_to_their_verdicts_and_undetermined_goes_to_b():
    example = V4["example"]
    path = {"CQ1": "no", "CQ2": "yes"}
    assert traverse(example, {**path, "CQ4": "a"}).verdict == "cherry_picking"
    assert traverse(example, {**path, "CQ4": "b"}).verdict == "hasty_generalization"
    assert traverse(example, {**path, "CQ4": "cannot_be_determined"}).verdict is None
    routing = routes(example, load_burden("v4", V4)["example"])
    masses = {cq: {"yes": 0.0, "no": 1.0} for cq in example.nodes}
    masses["CQ2"] = {"yes": 1.0, "no": 0.0}
    masses["CQ4"] = {"a": 0.25, "b": 0.25, OUTSIDE: 0.5}
    terminals, abstained, _ = traverse_soft(example, masses, routing)
    assert abstained == 0.0
    assert terminals["cherry_picking"] == pytest.approx(0.25)
    assert terminals["hasty_generalization"] == pytest.approx(0.75)


def test_the_new_questions_are_those_without_an_identical_earlier_one():
    from_v1 = identical_questions("v4", "v1")
    new = {s: [q for q in cqs if q not in from_v1.get(s, {})]
           for s, cqs in questions_to_ask("v4", "v2").items()}
    assert {s: cqs for s, cqs in new.items() if cqs} == NEW
    assert sum(map(len, NEW.values())) == 12
    # the same question under another id is reused; a new definition is a new question
    same = identical_questions("v4", "v2")
    assert same["ad_hominem"]["CQ1"] == "CQ4"
    assert same["expert_opinion"]["CQ5"] == "CQ4.1"
    assert "CQ6" not in same["expert_opinion"]   # 2.0 CQ5.1 with definitions added
    assert same["slippery_slope"]["CQ3"] == "CQ3"  # same text and definitions as 2.0
    assert from_v1["popular_opinion"]["CQ8"] == "CQ1"


def test_a_repeated_question_must_be_the_same_question(tmp_path):
    source = (SCHEMES_ROOT / "v4" / "popular_opinion.yaml").read_text(encoding="utf-8")
    broken = source.replace("same_as: CQ2", "same_as: CQ1")
    (tmp_path / "popular_opinion.yaml").write_text(broken, encoding="utf-8")
    with pytest.raises(SchemeError, match="repeats CQ1 with another text"):
        load_scheme(tmp_path / "popular_opinion.yaml")


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
    config = _config("t", "v4", "v5", {"example": ["CQ4"]})
    calls = plan(config, _items().to_dict("records"), FAKE_MODELS)
    assert sorted({(c.item_id, c.cq_id) for c in calls}) == [("e1", "CQ4"), ("e2", "CQ4")]
    with pytest.raises(SchemeError, match="example CQ9"):
        plan(_config("t", "v4", "v5", {"example": ["CQ9"]}), [], FAKE_MODELS)
    with pytest.raises(SchemeError, match="prompt_version"):
        _config("t", "v4", "v3")


def test_new_and_reused_answers_assemble_into_one_table(tmp_path):
    items = _items()
    records = items.to_dict("records")
    runs = {"old": _config("old", "v2", "v3"),
            "new": _config("new", "v4", "v5", questions_to_ask("v4", "v2"))}
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

    main = assemble("main", "v4", ["new", "old"], items, tmp_path)
    table = pd.read_csv(tmp_path / "main" / "summary.csv", dtype=str)
    assert len(table) == 2 * 5 + 3
    assert main["answers_by_cq"]["example CQ1"] == {"old": "CQ1"}
    assert main["answers_by_cq"]["example CQ4"] == {"new": "CQ4"}
    assert main["answers_by_cq"]["slippery_slope CQ3"] == {"old": "CQ3"}
    reused = table[table["source_run_id"] == "old"]
    assert set(zip(reused["scheme"], reused["cq_id"], strict=True)) == {
        ("example", "CQ1"), ("slippery_slope", "CQ1"), ("slippery_slope", "CQ2"),
        ("slippery_slope", "CQ3")}
    assert set(table.loc[table["source_run_id"] == "new", "source_prompt_version"]) == {
        "stage2_v5"}
    assert set(reused["source_prompt_version"]) == {"stage2_v3"}

    with pytest.raises(SchemeError, match="already exists"):
        assemble("main", "v4", ["new", "old"], items, tmp_path)
    with pytest.raises(SchemeError, match="no answer in the runs given for: example CQ1"):
        assemble("incomplete", "v4", ["new"], items, tmp_path)


def test_a_parse_without_letters_keeps_the_earlier_columns(schemes):
    frame = parse_answers([], schemes)
    assert not [c for c in frame.columns if c in ("p_logprob_a", "p_logprob_b")]
    assert not [c for c in summarise(frame, schemes).columns if c.endswith("_a")]


def test_a_repeated_question_is_never_asked_and_takes_the_answers_of_the_first(tmp_path):
    items = pd.DataFrame([{"item_id": "p1", "gold_scheme": "popular_opinion",
                           "text": "Everybody thinks so."}])
    config = _config("table_source", "v4", "v5")
    asked = {c.cq_id for c in plan(config, items.to_dict("records"), FAKE_MODELS)}
    assert asked == {f"CQ{i}" for i in range(1, 7)}
    with pytest.raises(SchemeError, match="repeats CQ2"):
        plan(_config("table_source", "v4", "v5", {"popular_opinion": ["CQ7"]}),
             items.to_dict("records"), FAKE_MODELS)
    execute(config, items.to_dict("records"), run_id="r", runs_dir=tmp_path,
            backend=FakeBackend(), models=FAKE_MODELS, sleep=lambda _: None)
    parse_run(tmp_path / "r")
    done = assemble("table", "v4", ["r"], items, tmp_path)
    assert done["answers_by_cq"]["popular_opinion CQ7"] == {"r": "CQ2"}
    table = pd.read_csv(tmp_path / "table" / "summary.csv", dtype=str).set_index("cq_id")
    assert list(table.index) == sorted(table.index) and len(table) == 9
    for twin, first in {"CQ7": "CQ2", "CQ8": "CQ4", "CQ9": "CQ6"}.items():
        assert table.loc[twin, "answer"] == table.loc[first, "answer"]
