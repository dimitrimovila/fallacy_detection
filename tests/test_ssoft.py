"""Sequential CQ calls, path pruning, burden routing, and resumable raw records."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace

import pandas as pd
import pytest

from argfallacy.aggregate import a2, load_burden, routes
from argfallacy.aggregate.diagram import good_terminal, reaches_good
from argfallacy.client import (
    ResponseCache,
    RunConfig,
    execute,
    load_items,
    load_models,
    plan,
    read_raw,
    select_items,
)
from argfallacy.parse import parse_run
from argfallacy.paths import REPO_ROOT
from argfallacy.prompts import render_stage2
from argfallacy.schemes import load_all
from fakes.llm import FakeBackend

ITEM = {"item_id": "item01", "text": "An argument for the test.", "gold_scheme": "analogy"}
MODELS = load_models(REPO_ROOT / "tests/fakes/models.yaml")
INSTRUCTION = "Take the earlier answers as given."


def config(history=True, model="fake_single"):
    return RunConfig(
        name="sequential",
        models=[model],
        stages=["ssoft"],
        schemes="v2",
        prompt_version="v1",
        history=history,
        threshold=0.02,
    )


def soft_row(scheme, cq, first=0.5, outside=0.0):
    a, b = scheme.answer_space(cq)
    return {
        "answer": a,
        f"p_logprob_{a}": first * (1 - outside),
        f"p_logprob_{b}": (1 - first) * (1 - outside),
        "p_logprob_cannot_be_determined": outside,
        "p_logprob_na": 0.0,
    }


class FixedBackend(FakeBackend):
    """Half the mass on each drawn arc, independent of the history."""

    def create(self, **kwargs):
        payload = super().create(**kwargs)
        options = kwargs["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"]
        a, b = options[:2]
        body = json.dumps({"answer": a, "confidence": 80, "justification": "A fixed answer."})
        choice = payload["choices"][0]
        choice["message"]["content"] = body
        choice["logprobs"] = {
            "content": [
                {"token": '{"answer": "', "logprob": 0.0, "top_logprobs": []},
                {
                    "token": a,
                    "logprob": math.log(0.5),
                    "top_logprobs": [
                        {"token": a, "logprob": math.log(0.5)},
                        {"token": b, "logprob": math.log(0.5)},
                    ],
                },
                {"token": body[len('{"answer": "') + len(a) :], "logprob": 0.0, "top_logprobs": []},
            ]
        }
        return payload


def test_v2_burden_covers_every_cq_and_its_routes():
    schemes = load_all(REPO_ROOT / "schemes/v2")
    table = load_burden("v2", schemes)
    assert set(table) == set(schemes)
    assert Counter(t for own in table.values() for t in own.values()) == {
        "P": 13,
        "G": 2,
        "D": 1,
        "E": 27,
    }
    gates = {("example", "CQ2"): "no", ("expert_opinion", "CQ5"): "yes"}
    for sid, scheme in schemes.items():
        assert set(table[sid]) == set(scheme.cqs) == set(scheme.nodes)
        routing = routes(scheme, table[sid])
        for cq, kind in table[sid].items():
            assert (kind == "G") == all(reaches_good(scheme, cq).values())
            if kind == "D":
                assert all(
                    e.is_terminal and e.target != good_terminal() for e in scheme.nodes[cq].values()
                )
                assert {e.target for e in scheme.nodes[cq].values()} == {
                    "strawman",
                    "false_attribution",
                }
            expected = {"P": "abstain", "D": "split", "E": "no"}.get(kind)
            assert routing[cq] == (gates[(sid, cq)] if kind == "G" else expected)
    for sid, cq, terminal in [
        ("analogy", "CQ4", good_terminal()),
        ("correlation_to_cause", "CQ6", good_terminal()),
        ("example", "CQ3", good_terminal()),
        ("popular_opinion", "CQ3", good_terminal()),
        ("ad_hominem", "CQ6", "ad_hominem_abusive"),
        ("analogy", "CQ1.1", "weak_analogy"),
        ("popular_opinion", "CQ1.2", "ad_populum"),
    ]:
        assert schemes[sid].nodes[cq]["no"].target == terminal


@pytest.mark.parametrize("folder", ["v1", "v2"])
def test_pruned_and_sequential_traversal_share_the_same_expansion(folder):
    from argfallacy.aggregate.soft import answer_masses
    from argfallacy.aggregate.ssoft import a2_pruned, expand

    schemes = load_all(REPO_ROOT / "schemes" / folder)
    table = load_burden(folder, schemes)
    for sid, scheme in schemes.items():
        rows = {cq: soft_row(scheme, cq, 0.37, 0.21) for cq in scheme.cq_order}
        routing = routes(scheme, table[sid])
        result = expand(
            scheme,
            lambda state, scheme=scheme, rows=rows: answer_masses(
                scheme, state.cq_id, rows[state.cq_id], "logprob"
            ),
            routing,
            threshold=0.02,
        )
        reference = a2_pruned(scheme, rows, routing, threshold=0.02)
        assert result.distribution == pytest.approx(reference.distribution, abs=1e-12)
        assert result.abstention == pytest.approx(reference.abstention, abs=1e-12)
        assert result.pruned == pytest.approx(reference.pruned, abs=1e-12)
        assert sum(
            result.distribution.values()
        ) + result.abstention + result.pruned == pytest.approx(1)
        assert not result.frontier
        unpruned = a2_pruned(scheme, rows, routing, threshold=0)
        old = a2(scheme, rows, routing)
        assert unpruned.distribution == pytest.approx(old.distribution, abs=1e-12)
        assert unpruned.abstention == pytest.approx(old.abstention, abs=1e-12)


def test_expansion_keeps_distinct_histories_and_prunes_only_node_arcs():
    from argfallacy.aggregate.ssoft import expand

    scheme = load_all(REPO_ROOT / "schemes/v2")["analogy"]
    schemes = load_all(REPO_ROOT / "schemes/v2")
    routing = routes(scheme, load_burden("v2", schemes)["analogy"])
    # A flat provider asks six times: CQ1.1 is reached on two paths.
    states = []

    def provide(state):
        states.append(state)
        return {a: 0.5 for a in scheme.answer_space(state.cq_id)}

    result = expand(scheme, provide, routing)
    assert len(states) == 6
    shared = [s for s in states if s.cq_id == "CQ1.1"]
    assert len(shared) == 2 and shared[0].path != shared[1].path
    assert all(s.mass >= 0.02 for s in states)
    assert result.pruned == 0
    states.clear()
    pruned = expand(scheme, provide, routing, threshold=0.26)
    assert pruned.pruned == pytest.approx(0.5)
    assert all(s.mass >= 0.26 for s in states)
    assert sum(pruned.distribution.values()) + pruned.pruned == pytest.approx(1)


def test_ssoft_prompt_adds_the_list_of_cqs_and_only_the_earlier_answers_that_exist():
    from argfallacy.client.cache import cache_key
    from argfallacy.client.call import Request
    from argfallacy.prompts import render_ssoft

    schemes = load_all(REPO_ROOT / "schemes/v2")
    for scheme in schemes.values():
        cq = scheme.start
        rendered = render_ssoft(ITEM, scheme, cq)
        assert rendered == render_ssoft(ITEM, scheme, cq)
        # at the first node there is no earlier answer: same prompt as without history
        assert rendered == render_ssoft(ITEM, scheme, cq, history=False)
        assert "Earlier questions" not in rendered.text
        assert INSTRUCTION not in rendered.text
        listed = "\n\n".join(f"**{n}.** {scheme.cqs[n].text}" for n in scheme.cq_order)
        block = (
            "## All the critical questions of this scheme\n\n"
            "These are all the questions asked about this argument, one at a time. Answer only\n"
            "the question indicated below.\n\n" + listed + "\n\n"
        )
        reference = render_stage2(ITEM, scheme, cq, "v3")
        expected = reference.text.replace(
            "## The critical question\n", block + "## The critical question you must answer\n", 1
        )
        assert rendered.text == expected
        assert rendered.answer_options == reference.answer_options
        assert {k: v for k, v in rendered.json_schema.items() if k != "title"} == {
            k: v for k, v in reference.json_schema.items() if k != "title"
        }
        assert rendered.prompt_version == "ssoft_v1"
    scheme = schemes["analogy"]
    first = render_ssoft(ITEM, scheme, "CQ1.1", (("CQ1", "yes"),))
    second = render_ssoft(ITEM, scheme, "CQ1.1", (("CQ1", "no"), ("CQ2", "no")))
    assert f"**CQ1.** {scheme.cqs['CQ1'].text}\nAnswer: yes." in first.text
    request = Request(model_id="fake", prompt=first.text, prompt_version="ssoft_v1", sample_index=0)
    assert cache_key(request) != cache_key(replace(request, prompt=second.text))
    # with earlier answers: the section and the instruction, and nothing else, are added
    without = render_ssoft(ITEM, scheme, "CQ1.1", history=False).text
    section = (
        "## Earlier questions on this argument, and their answers\n\n"
        f"**CQ1.** {scheme.cqs['CQ1'].text}\nAnswer: yes.\n\n"
    )
    heading = "## The critical question you must answer"
    assert first.text == without.replace(heading, section + heading, 1).replace(
        "Answer this question only", INSTRUCTION + "\n\nAnswer this question only", 1
    )


def test_ssoft_configs_plan_first_nodes_or_every_cq_and_one_sample():
    for name, count in [("ssoft_full_v3", 601), ("ssoft_nohistory_full_v3", 3162)]:
        cfg = RunConfig.load(REPO_ROOT / "configs" / f"{name}.yaml")
        calls = plan(cfg, select_items(cfg, load_items()))
        assert len(calls) == count
        assert {c.stage for c in calls} == {"ssoft"}
        assert {c.sample_index for c in calls} == {0}
        assert all(c.path == "" for c in calls)


def test_probability_transformation_precedes_routing_and_pruning():
    from argfallacy.aggregate.ssoft import expand

    schemes = load_all(REPO_ROOT / "schemes/v2")
    scheme = schemes["analogy"]
    routing = routes(scheme, load_burden("v2", schemes)["analogy"])
    seen = []

    def provide(state):
        seen.append(state)
        return {"yes": 0.5, "no": 0.5}

    def transformed(cq, masses):
        return {"yes": 0.99, "no": 0.01} if cq == scheme.start else masses

    result = expand(scheme, provide, routing, transform=transformed)
    assert result.pruned == pytest.approx(0.01)
    assert len(seen) == 2


@pytest.mark.parametrize("history", [True, False])
def test_fake_run_resumes_and_parses_paths_without_merging_them(tmp_path, history):
    from argfallacy.aggregate.ssoft import a2_pruned, ssoft

    schemes = load_all(REPO_ROOT / "schemes/v2")
    cfg = config(history)
    backend = FixedBackend(fail_after=2)
    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        execute(
            cfg,
            [ITEM],
            "run",
            tmp_path,
            backend,
            cache,
            MODELS,
            schemes,
            max_attempts=1,
            sleep=lambda _: None,
        )
        backend.fail_after = None
        manifest = execute(
            cfg,
            [ITEM],
            "run",
            tmp_path,
            backend,
            cache,
            MODELS,
            schemes,
            max_attempts=1,
            sleep=lambda _: None,
        )
        calls = backend.calls
        execute(cfg, [ITEM], "run", tmp_path, backend, cache, MODELS, schemes)
        assert backend.calls == calls
    lines = list(read_raw(tmp_path / "run"))
    successful = [row for row in lines if not row["error"]]
    expected = 6 if history else 5
    assert len(successful) == expected
    assert len({row["cache_key"] for row in successful}) == expected
    assert len(backend.prompts_seen) == expected
    assert manifest["calls_failed"] == 0 and manifest["calls_planned"] == expected
    assert manifest["models"]["fake_single"]["samples"] == 1
    written = parse_run(tmp_path / "run")
    assert "ssoft.csv" in written
    parsed = pd.read_csv(written["ssoft.csv"], keep_default_na=False)
    assert len(parsed) == expected
    assert not pd.read_csv(written["summary.csv"]).shape[0]
    assert not parsed["invalid_count"].any()
    if history:
        repeated = parsed[parsed["cq_id"] == "CQ1.1"]
        assert len(repeated) == 2 and repeated["path"].nunique() == 2
        assert set(repeated["depth"]) == {1, 2}
    else:
        assert set(parsed["path"]) == {""} and set(parsed["depth"]) == {""}
    scheme = schemes[ITEM["gold_scheme"]]
    routing = routes(scheme, load_burden("v2", schemes)[scheme.scheme_id])
    rows = parsed.to_dict("records")
    result = ssoft(scheme, rows, routing, history=history)
    fixed = {cq: soft_row(scheme, cq) for cq in scheme.cq_order}
    reference = a2_pruned(scheme, fixed, routing)
    assert result.distribution == pytest.approx(reference.distribution, abs=1e-12)
    assert result.pruned == pytest.approx(reference.pruned, abs=1e-12)


def test_ssoft_uses_verbal_confidence_oriented_on_the_first_arc():
    from argfallacy.aggregate.ssoft import node_masses

    scheme = load_all(REPO_ROOT / "schemes/v2")["analogy"]
    row = {"answer": "no", "p_verbal": 0.2}
    assert node_masses(scheme, "CQ1", row, "verbal") == pytest.approx({"yes": 0.2, "no": 0.8})


def test_a_single_ssoft_sample_keeps_the_producers_sampling_parameters():
    calls = plan(config(model="fake"), [ITEM], MODELS)
    assert len(calls) == 1 and calls[0].sample_index == 0
    assert calls[0].temperature == MODELS["fake"]["temperature"]
    assert calls[0].top_p == MODELS["fake"]["top_p"]
    assert calls[0].top_k == MODELS["fake"]["top_k"]


def test_pruning_includes_the_boundary_and_accounts_for_pending_mass():
    from argfallacy.aggregate.ssoft import expand

    schemes = load_all(REPO_ROOT / "schemes/v2")
    scheme = schemes["analogy"]
    routing = routes(scheme, load_burden("v2", schemes)["analogy"])

    def provide(state):
        return {"yes": 0.02, "no": 0.98} if state.cq_id == scheme.start else None

    result = expand(scheme, provide, routing)
    assert sorted(state.mass for state in result.frontier) == [0.02, 0.98]
    assert result.pending == pytest.approx(1) and result.pruned == 0
    assert result.verdict is None


def test_ssoft_resume_refuses_a_changed_expansion_policy(tmp_path):
    from argfallacy.schemes import SchemeError

    schemes = load_all(REPO_ROOT / "schemes/v2")
    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        execute(config(), [ITEM], "run", tmp_path, FixedBackend(), cache, MODELS, schemes)
        with pytest.raises(SchemeError, match="resume|configuration|policy"):
            execute(
                replace(config(), threshold=0.1),
                [ITEM],
                "run",
                tmp_path,
                FixedBackend(),
                cache,
                MODELS,
                schemes,
            )
