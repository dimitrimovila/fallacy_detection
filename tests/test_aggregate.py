"""The aggregators, on the diagrams of ``schemes/`` and on answers written by hand.

No run and no network: every answer is a row of ``summary.csv`` built here, and
every expected verdict comes from the diagram itself, through ``traverse`` or
``propagate``, or from a mass worked out by hand.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from argfallacy.aggregate import (
    ABSTAIN,
    LOGPROB,
    OUTSIDE,
    SAMPLE,
    SPLIT,
    VERBAL,
    a0,
    a2,
    a4,
    answer_masses,
    load_burden,
    mean_masses,
    pass_arc,
    routes,
    sources_of,
)
from argfallacy.eval import run_report
from argfallacy.parse import ANSWERS_COLUMNS, summarise
from argfallacy.schemes import SchemeError, propagate, traverse

GOOD = "good_argumentation"


def _soft(scheme, cq, p_first, outside=0.0, answer=None, column="p_logprob"):
    """A row whose masses give ``p_first`` to the first arc, ``outside`` off the arcs."""
    first, second = scheme.answer_space(cq)
    row = {"answer": answer or (first if p_first >= 0.5 else second)}
    row[f"{column}_{first}"] = p_first * (1 - outside)
    row[f"{column}_{second}"] = (1 - p_first) * (1 - outside)
    row[f"{column}_cannot_be_determined"] = outside / 2
    row[f"{column}_na"] = outside / 2
    return row


def _hard(answers: dict[str, str]) -> dict[str, dict]:
    """Rows with a hard answer and no masses: each answer takes a mass of 1."""
    return {cq: {"answer": answer} for cq, answer in answers.items()}


def _reaches_good(scheme, edge) -> bool:
    if edge.is_terminal:
        return edge.target == GOOD
    return any(_reaches_good(scheme, e) for e in scheme.nodes[edge.target].values())


# ------------------------------------------------------------- burden table


def test_the_burden_table_types_every_cq_once(schemes):
    table = load_burden("v1", schemes)
    assert set(table) == set(schemes)
    for scheme_id, scheme in schemes.items():
        assert set(table[scheme_id]) == set(scheme.nodes), scheme_id
    counts = Counter(kind for types in table.values() for kind in types.values())
    assert counts == {"P": 14, "E": 28, "G": 5, "D": 1}


def test_gates_and_two_fallacies_follow_the_diagrams(schemes):
    table = load_burden("v1", schemes)
    for scheme_id, scheme in schemes.items():
        for cq, edges in scheme.nodes.items():
            both_reach = all(_reaches_good(scheme, e) for e in edges.values())
            assert (table[scheme_id][cq] == "G") == both_reach, (scheme_id, cq)
            if table[scheme_id][cq] == "D":
                assert all(e.is_terminal and e.target != GOOD for e in edges.values())


def test_the_routes_of_the_burden_of_proof(schemes):
    table = load_burden("v1", schemes)
    gates = {("cause_to_effect", "CQ2"): "no", ("correlation_to_cause", "CQ2"): "no",
             ("example", "CQ2"): "no", ("expert_opinion", "CQ4"): "yes",
             ("slippery_slope", "CQ4"): "yes"}
    for scheme_id, scheme in schemes.items():
        routing = routes(scheme, table[scheme_id])
        for cq, kind in table[scheme_id].items():
            if kind == "P":
                assert routing[cq] == ABSTAIN
            elif kind == "D":
                assert routing[cq] == SPLIT
            elif kind == "G":
                assert routing[cq] == gates[(scheme_id, cq)]
            else:
                expected = "yes" if (scheme_id, cq) == ("correlation_to_cause", "CQ5") else "no"
                assert routing[cq] == expected, (scheme_id, cq)
    ends = {("ad_hominem", "CQ6"): "ad_hominem_abusive", ("analogy", "CQ1.1"): "weak_analogy",
            ("popular_opinion", "CQ3.3"): "ad_populum",
            ("correlation_to_cause", "CQ5"): GOOD}
    for (scheme_id, cq), terminal in ends.items():
        scheme = schemes[scheme_id]
        assert scheme.nodes[cq][routes(scheme, table[scheme_id])[cq]].target == terminal
    expert = routes(schemes["expert_opinion"], table["expert_opinion"])
    assert expert["CQ3.1"] == SPLIT


def _write(path, table):
    import yaml

    path.write_text(yaml.safe_dump(table), encoding="utf-8")


def test_a_table_that_breaks_the_rules_is_refused(tmp_path, schemes):
    good = load_burden("v1", schemes)

    missing = {k: dict(v) for k, v in good.items()}
    del missing["analogy"]["CQ4"]
    _write(tmp_path / "v1.yaml", missing)
    with pytest.raises(SchemeError, match=r"analogy CQ4"):
        load_burden("v1", schemes, root=tmp_path)

    extra = {k: dict(v) for k, v in good.items()}
    extra["analogy"]["CQ9"] = "E"
    _write(tmp_path / "v1.yaml", extra)
    with pytest.raises(SchemeError, match=r"analogy CQ9"):
        load_burden("v1", schemes, root=tmp_path)

    # a gate needs both arcs to reach good_argumentation; CQ1.1 of analogy reaches none
    gate = {k: dict(v) for k, v in good.items()}
    gate["analogy"]["CQ1.1"] = "G"
    _write(tmp_path / "v1.yaml", gate)
    with pytest.raises(SchemeError, match=r"analogy CQ1\.1"):
        load_burden("v1", schemes, root=tmp_path)

    # two fallacies need both arcs on a fallacy; CQ4 of analogy has good_argumentation
    split = {k: dict(v) for k, v in good.items()}
    split["analogy"]["CQ4"] = "D"
    _write(tmp_path / "v1.yaml", split)
    with pytest.raises(SchemeError, match=r"analogy CQ4"):
        load_burden("v1", schemes, root=tmp_path)


def test_a_version_without_a_table_stops_naming_its_folder(tmp_path, schemes):
    with pytest.raises(SchemeError, match=r"schemes/v2/"):
        load_burden("v2", schemes, root=tmp_path)


# ------------------------------------------------------------------- A2


def test_without_mass_outside_the_arcs_a2_is_propagate(schemes):
    table = load_burden("v1", schemes)
    for scheme_id, scheme in schemes.items():
        nodes = list(scheme.nodes)
        p = {cq: 0.1 + 0.8 * index / max(1, len(nodes) - 1) for index, cq in enumerate(nodes)}
        rows = {cq: _soft(scheme, cq, p[cq]) for cq in nodes}
        result = a2(scheme, rows, routes(scheme, table[scheme_id]))
        first = {cq: scheme.answer_space(cq) for cq in nodes}
        expected = propagate(scheme, {cq: {first[cq][0]: p[cq], first[cq][1]: 1 - p[cq]}
                                      for cq in nodes})
        assert set(result.distribution) <= set(expected)
        for terminal, mass in expected.items():
            assert result.distribution.get(terminal, 0.0) == pytest.approx(mass, abs=1e-12)
        assert result.abstention == pytest.approx(0.0, abs=1e-12)


def _outside_at(scheme, cq, answers):
    rows = _hard(answers)
    rows[cq] = {"answer": "cannot_be_determined"}
    return rows


def test_mass_outside_the_arcs_follows_the_type_of_the_question(schemes):
    table = load_burden("v1", schemes)

    def run(scheme_id, rows):
        scheme = schemes[scheme_id]
        result = a2(scheme, rows, routes(scheme, table[scheme_id]))
        assert sum(result.distribution.values()) + result.abstention == pytest.approx(1.0)
        return result

    # P, slippery slope CQ1: the argument cannot be accepted, and no defect is shown
    result = run("slippery_slope", _outside_at(schemes["slippery_slope"], "CQ1", {}))
    assert result.abstention == pytest.approx(1.0) and result.verdict is None

    # E, analogy CQ1: the differences are not shown, the arc toward good_argumentation
    answers = {"CQ1.1": "yes", "CQ2": "yes", "CQ3": "yes", "CQ4": "no"}
    result = run("analogy", _outside_at(schemes["analogy"], "CQ1", answers))
    assert result.distribution == {GOOD: pytest.approx(1.0)}

    # G, example CQ2: the follow-up questions decide
    answers = {"CQ1": "no", "CQ2.1": "no", "CQ2.2": "yes", "CQ2.3": "no", "CQ3": "no"}
    result = run("example", _outside_at(schemes["example"], "CQ2", answers))
    expected = traverse(schemes["example"], {**answers, "CQ2": "no"}).verdict
    assert expected != traverse(schemes["example"], {**answers, "CQ2": "yes"}).verdict
    assert result.distribution == {expected: pytest.approx(1.0)}

    # D, expert opinion CQ3.1: half and half, the tie to the first id in alphabetical order
    answers = {"CQ1": "yes", "CQ2": "no", "CQ3": "no"}
    result = run("expert_opinion", _outside_at(schemes["expert_opinion"], "CQ3.1", answers))
    assert result.distribution == {"strawman": pytest.approx(0.5),
                                   "false_attribution": pytest.approx(0.5)}
    assert result.verdict == "false_attribution" and result.confidence == pytest.approx(0.5)


def test_a_cq_without_masses_and_an_invalid_answer(schemes):
    scheme = schemes["analogy"]
    routing = routes(scheme, load_burden("v1", schemes)["analogy"])
    answers = {"CQ1": "no", "CQ2": "yes", "CQ3": "yes", "CQ4": "no"}
    result = a2(scheme, _hard(answers), routing)
    assert result.distribution == {traverse(scheme, answers).verdict: 1.0}
    # an invalid answer at CQ1, an exception: all its mass outside, sent toward good
    result = a2(scheme, {**_hard(answers), "CQ1": {"answer": "invalid"}}, routing)
    assert result.distribution == {GOOD: pytest.approx(1.0)}
    # a row with masses uses them, whatever its hard answer
    rows = {**_hard(answers), "CQ4": _soft(scheme, "CQ4", 0.3, outside=0.2, answer="invalid")}
    result = a2(scheme, rows, routing)
    assert result.distribution["cherry_picking"] == pytest.approx(0.3 * 0.8)
    assert result.distribution[GOOD] == pytest.approx(0.7 * 0.8 + 0.2)


def test_the_three_exploratory_policies(schemes):
    table = load_burden("v1", schemes)
    for scheme_id, scheme in schemes.items():
        assert set(routes(scheme, table[scheme_id], "stop").values()) == {ABSTAIN}
        assert set(routes(scheme, table[scheme_id], "branch").values()) == {SPLIT}
        charitable = routes(scheme, table[scheme_id], "charitable")
        for cq, route in charitable.items():
            expected = SPLIT if table[scheme_id][cq] == "D" else pass_arc(scheme, cq)
            assert route == expected, (scheme_id, cq)

    analogy = schemes["analogy"]
    answers = {"CQ1.1": "yes", "CQ2": "yes", "CQ3": "yes", "CQ4": "no"}
    rows = _outside_at(analogy, "CQ1", answers)
    stop = a2(analogy, rows, routes(analogy, table["analogy"], "stop"))
    assert stop.abstention == pytest.approx(1.0) and stop.verdict is None
    branch = a2(analogy, rows, routes(analogy, table["analogy"], "branch"))
    assert branch.distribution == {"appeal_to_extremes": pytest.approx(0.5),
                                   GOOD: pytest.approx(0.5)}
    # charitable: a gate takes the pass arc, not the follow-up questions
    example = schemes["example"]
    answers = {"CQ1": "no", "CQ2.1": "yes", "CQ3": "no"}
    charitable = a2(example, _outside_at(example, "CQ2", answers),
                    routes(example, table["example"], "charitable"))
    assert charitable.distribution == {
        traverse(example, {**answers, "CQ2": "yes"}).verdict: pytest.approx(1.0)}


def test_the_explanation_of_a2_names_the_most_probable_path(schemes):
    scheme = schemes["expert_opinion"]
    rows = {"CQ1": _soft(scheme, "CQ1", 0.9), "CQ2": _soft(scheme, "CQ2", 0.8),
            **_hard({"CQ3": "yes", "CQ3.1": "yes", "CQ4": "no", "CQ4.1": "yes"})}
    result = a2(scheme, rows, routes(scheme, load_burden("v1", schemes)["expert_opinion"]))
    assert result.verdict == "appeal_to_authority"
    assert result.confidence == pytest.approx(0.72)
    assert result.explanation.startswith("CQ1=yes 0.90 > CQ2=yes 0.80 > appeal_to_authority")
    assert "irrelevant_authority 0.10" in result.explanation


# ------------------------------------------------------------------- A4


def test_five_identical_entries_give_a2_of_one(schemes):
    table = load_burden("v1", schemes)
    for scheme_id, scheme in schemes.items():
        nodes = list(scheme.nodes)
        rows = {cq: _soft(scheme, cq, 0.2 + 0.6 * i / max(1, len(nodes) - 1), outside=0.1)
                for i, cq in enumerate(nodes)}
        routing = routes(scheme, table[scheme_id])
        entries = [f"e{n}" for n in range(5)]
        combined = a4(scheme, dict.fromkeys(entries, rows), dict.fromkeys(entries, LOGPROB),
                      routing)
        alone = a2(scheme, rows, routing)
        assert combined.verdict == alone.verdict
        for terminal, mass in alone.distribution.items():
            assert combined.distribution[terminal] == pytest.approx(mass, abs=1e-12)
        assert combined.abstention == pytest.approx(alone.abstention, abs=1e-12)


def test_the_source_of_each_entry_follows_its_regime(schemes):
    manifest = {"models": {
        "off": {"samples": 1, "reasoning": {"chat_template_kwargs": {"enable_thinking": False}}},
        "several": {"samples": 5, "reasoning": {"reasoning_effort": "low"}},
        "reasons": {"samples": 1, "reasoning": {"chat_template_kwargs": {
            "reasoning_effort": "high"}}},
    }}
    assert sources_of(manifest) == {"off": LOGPROB, "several": SAMPLE, "reasons": VERBAL}

    scheme = schemes["analogy"]
    row = {**_soft(scheme, "CQ1", 0.9, outside=0.1),
           **_soft(scheme, "CQ1", 0.25, column="p_sample"), "p_verbal": 0.7, "answer": "no"}
    assert answer_masses(scheme, "CQ1", row, LOGPROB) == pytest.approx(
        {"yes": 0.81, "no": 0.09, OUTSIDE: 0.1})
    assert answer_masses(scheme, "CQ1", row, SAMPLE) == pytest.approx(
        {"yes": 0.25, "no": 0.75, OUTSIDE: 0.0})
    assert answer_masses(scheme, "CQ1", row, VERBAL) == pytest.approx({"yes": 0.7, "no": 0.3})
    assert answer_masses(scheme, "CQ1", {"answer": "na", "p_verbal": None}, VERBAL) == {
        OUTSIDE: 1.0}
    assert answer_masses(scheme, "CQ1", {"answer": "yes"}, VERBAL) == {"yes": 1.0}
    assert answer_masses(scheme, "CQ1", {"answer": "invalid"}, VERBAL) is None
    # ad hominem CQ1: the stated confidence is that of positive, its first answer
    hominem = schemes["ad_hominem"]
    assert answer_masses(hominem, "CQ1", {"answer": "negative", "p_verbal": 0.2},
                         VERBAL) == pytest.approx({"positive": 0.2, "negative": 0.8})


def test_an_invalid_entry_without_masses_is_left_out_of_the_mean(schemes):
    scheme = schemes["analogy"]
    sources = {"a": LOGPROB, "b": LOGPROB, "c": VERBAL}
    rows = {
        "a": {"CQ1": _soft(scheme, "CQ1", 0.8)},
        "b": {"CQ1": {"answer": "invalid"}},
        "c": {"CQ1": {"answer": "yes", "p_verbal": 0.6}},
    }
    masses = mean_masses(scheme, rows, sources)
    assert masses["CQ1"] == pytest.approx({"yes": 0.7, "no": 0.3, OUTSIDE: 0.0})
    # every entry left out: the whole mass outside the arcs; a CQ with no row, the same
    rows = {"b": {"CQ1": {"answer": "invalid"}}}
    masses = mean_masses(scheme, rows, {"b": LOGPROB})
    assert masses["CQ1"] == {OUTSIDE: 1.0} and masses["CQ2"] == {OUTSIDE: 1.0}


def test_the_explanation_of_a4_gives_the_hard_answers_of_the_entries(schemes):
    scheme = schemes["analogy"]
    table = load_burden("v1", schemes)
    answers = {"CQ1": "no", "CQ2": "yes", "CQ3": "yes", "CQ4": "no"}
    rows = {"a": _hard(answers), "b": _hard(answers),
            "c": {**_hard(answers), "CQ1": {"answer": "yes"}}}
    result = a4(scheme, rows, dict.fromkeys(rows, LOGPROB), routes(scheme, table["analogy"]))
    assert result.explanation.startswith("CQ1=no 0.67 (a no, b no, c yes) > CQ2=yes 1.00")


# ------------------------------------------------------------------- A0


def _summary_rows(schemes):
    """Sample 0 of one entry, on items whose answers walk every kind of path."""
    rows, expected = [], {}
    for scheme_id, scheme in sorted(schemes.items()):
        patterns = {
            "first": {cq: scheme.answer_space(cq)[0] for cq in scheme.nodes},
            "second": {cq: scheme.answer_space(cq)[1] for cq in scheme.nodes},
            "cbd": {cq: "cannot_be_determined" for cq in scheme.nodes},
            "na_late": {**{cq: scheme.answer_space(cq)[1] for cq in scheme.nodes},
                        list(scheme.nodes)[-1]: "na"},
            "invalid": {cq: "invalid" for cq in scheme.nodes},
        }
        for name, answers in patterns.items():
            item = f"{scheme_id}-{name}"
            expected[item] = (scheme_id, answers)
            for cq, answer in answers.items():
                row = dict.fromkeys(ANSWERS_COLUMNS)
                row.update({"run_id": "r", "item_id": item, "stage": "stage2",
                            "scheme_condition": "gold", "scheme": scheme_id, "cq_id": cq,
                            "sample_index": 0, "model": "m", "answer": answer,
                            "confidence": 80, "parse_ok": answer != "invalid",
                            "finish_reason": "stop", "logprob_partial": False})
                rows.append(row)
    return pd.DataFrame(rows, columns=ANSWERS_COLUMNS), expected


def test_a0_is_the_diagram_rule_of_the_run_report(schemes):
    answers, expected = _summary_rows(schemes)
    summary = summarise(answers, schemes)
    data = run_report.RunData(run_id="r", answers=answers, summary=summary,
                              manifest={"config": {"models": ["m"]}}, items=pd.DataFrame(),
                              annotations=pd.DataFrame(), schemes=schemes, raw_path=None)
    diagram = run_report.traversals(data, "diagram").set_index("item_id")["verdict"]
    seen = set()
    for item, (scheme_id, _) in expected.items():
        own = summary[summary["item_id"] == item]
        rows = {row["cq_id"]: row for _, row in own.iterrows()}
        result = a0(schemes[scheme_id], rows)
        reference = diagram[item]
        assert result.verdict == (None if pd.isna(reference) else reference), item
        assert result.confidence is None
        seen.add(result.verdict is None)
    assert seen == {True, False}


def test_the_explanation_of_a0_is_the_path(schemes):
    scheme = schemes["expert_opinion"]
    result = a0(scheme, _hard({"CQ1": "yes", "CQ2": "yes"}))
    assert result.explanation == "CQ1=yes > CQ2=yes > appeal_to_authority"
    assert result.distribution == {"appeal_to_authority": 1.0} and result.abstention == 0.0
    stopped = a0(scheme, _hard({"CQ1": "yes", "CQ2": "na"}))
    assert stopped.verdict is None and stopped.abstention == 1.0
    assert stopped.explanation == "CQ1=yes > CQ2=na: no verdict"
