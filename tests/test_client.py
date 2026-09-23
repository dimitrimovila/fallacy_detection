"""The client, the plan, the manifest and the parser, all against the fake backend.

No test here touches the network.  The fake counts its
calls, which is the only way to check the two tests
that are about *not* calling: the cache and the resume.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argfallacy.client import (
    RawResponse,
    Request,
    ResponseCache,
    RunConfig,
    ask,
    build_manifest,
    cache_key,
    execute,
    load_models,
    plan,
    read_raw,
    select_items,
)
from argfallacy.paths import REPO_ROOT
from argfallacy.prompts import STAGE1, STAGE2, answer_schema
from argfallacy.schemes import SchemeError, load_all
from fakes.llm import FakeBackend

MODEL = "fake"
OTHER_MODEL = "fake"
FAKE_MODELS = load_models(Path(__file__).parent / "fakes" / "models.yaml")
"""The models of tests/fakes/models.yaml. serving/models.yaml holds real models only."""


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def cache(tmp_path) -> ResponseCache:
    with ResponseCache(tmp_path / "cache.sqlite") as store:
        yield store


@pytest.fixture
def items(schemes):
    """Ten items with a known gold scheme, cycling through the eight schemes."""
    names = sorted(schemes)
    return [
        {
            "item_id": f"item{index:02d}",
            "text": f"Argument number {index} about something.",
            "gold_scheme": names[index % len(names)],
        }
        for index in range(10)
    ]


def config_for(stages, samples=5, condition="gold", name="test") -> RunConfig:
    data = {
        "name": name, "models": [MODEL], "stages": list(stages),
        "scheme_condition": condition, "samples": samples, "prompt_version": "v1",
        "items": {}, "generation": {"temperature_sample0": 0.0, "temperature_rest": 0.7},
    }
    return RunConfig(raw=data, **{k: v for k, v in data.items()})


def request_for(prompt="ask me", **overrides) -> Request:
    base = {
        "model_id": "fake-model", "prompt": prompt, "prompt_version": "stage2_v1",
        "sample_index": 0, "temperature": 0.0, "max_tokens": 64,
        "json_schema": None, "logprobs": True,
    }
    base.update(overrides)
    return Request(**base)


def test_the_same_question_is_asked_once(backend, cache):
    request = request_for()
    key = cache_key(request)

    first = ask(request, backend)
    cache.put(key, request, first)
    assert backend.calls == 1

    assert cache.get(key) is not None
    assert backend.calls == 1, "a cached answer must not reach the backend"


@pytest.mark.parametrize("field,value", [
    ("prompt_version", "stage2_v2"),
    ("sample_index", 1),
    ("temperature", 0.7),
    ("max_tokens", 128),
    ("model_id", "other-model"),
    ("prompt", "a different question"),
    ("revision", "another-commit"),
    ("tier", 2),
    ("reasoning", {"reasoning_effort": "low"}),
])
def test_changing_anything_that_decides_the_answer_changes_the_key(field, value):
    assert cache_key(request_for()) != cache_key(request_for(**{field: value}))


def test_the_key_is_stable_across_processes():
    assert cache_key(request_for()) == cache_key(request_for())


def test_a_failed_call_is_not_cached(cache):
    request = request_for()
    failure = RawResponse(error="Timeout: took too long")
    assert cache.put(cache_key(request), request, failure) is False
    assert cache.get(cache_key(request)) is None


def test_a_new_prompt_version_leaves_the_old_answers_alone(backend, cache):
    old = request_for(prompt_version="stage2_v1")
    cache.put(cache_key(old), old, ask(old, backend))
    new = request_for(prompt_version="stage2_v2")
    assert cache.get(cache_key(old)) is not None
    assert cache.get(cache_key(new)) is None


def test_an_interrupted_run_resumes_without_duplicates(tmp_path, items, schemes, cache):
    config = config_for([STAGE1], samples=2)
    models = FAKE_MODELS
    expected = len(plan(config, items[:3], models, schemes))

    crashing = FakeBackend(fail_after=4)
    execute(config, items[:3], run_id="r1", runs_dir=tmp_path, backend=crashing,
            cache=cache, models=models, schemes=schemes, max_attempts=1, sleep=lambda _: None)
    partial = read_raw(tmp_path / "r1")
    assert 0 < len([r for r in partial if not r["error"]]) < expected

    healthy = FakeBackend()
    manifest = execute(config, items[:3], run_id="r1", runs_dir=tmp_path, backend=healthy,
                       cache=cache, models=models, schemes=schemes, sleep=lambda _: None)

    rows = read_raw(tmp_path / "r1")
    keys = [r["cache_key"] for r in rows if not r["error"]]
    assert len(keys) == len(set(keys)), "raw.jsonl has duplicate calls"
    assert set(keys) == {c.key() for c in plan(config, items[:3], models, schemes)}
    assert healthy.calls < expected, "successful calls were made again"
    assert manifest["calls_failed"] == 0


def test_a_failed_call_is_retried_on_the_next_run(tmp_path, items, schemes, cache):
    config = config_for([STAGE1], samples=1)
    models = FAKE_MODELS
    execute(config, items[:2], run_id="r", runs_dir=tmp_path, backend=FakeBackend(fail_after=0),
            cache=cache, models=models, schemes=schemes, max_attempts=1, sleep=lambda _: None)
    assert cache.count() == 0

    healthy = FakeBackend()
    manifest = execute(config, items[:2], run_id="r", runs_dir=tmp_path, backend=healthy,
                       cache=cache, models=models, schemes=schemes, sleep=lambda _: None)
    assert manifest["calls_executed"] == 2
    assert healthy.calls == 2


def test_the_plan_counts_stage_one_plus_the_cqs_of_each_scheme(items, schemes):
    config = config_for([STAGE1, STAGE2], samples=5)
    models = FAKE_MODELS
    calls = plan(config, items, models, schemes)

    expected = 5 * len(items)  # stage one, five samples each
    for item in items:
        expected += 5 * len(schemes[item["gold_scheme"]].cqs)
    assert len(calls) == expected * len(config.models)


def test_the_plan_is_in_a_deterministic_order(items, schemes):
    config = config_for([STAGE1, STAGE2], samples=2)
    models = FAKE_MODELS
    def shape(calls):
        return [(c.model, c.item_id, c.cq_id, c.sample_index) for c in calls]

    first = shape(plan(config, items, models, schemes))
    second = shape(plan(config, items, models, schemes))
    assert first == second
    assert first == sorted(first, key=lambda t: (t[0], first.index(t)))


def test_the_plan_makes_no_call(items, schemes, backend):
    plan(config_for([STAGE1, STAGE2]), items, FAKE_MODELS, schemes)
    assert backend.calls == 0


def test_sample_zero_is_greedy_and_the_others_are_not(items, schemes):
    config = config_for([STAGE1], samples=3)
    calls = plan(config, items[:1], FAKE_MODELS, schemes)
    assert calls[0].temperature == 0.0
    assert all(c.temperature == 0.7 for c in calls[1:])


def test_the_manifest_has_every_declared_field(items, schemes):
    config = config_for([STAGE1, STAGE2], samples=2)
    models = FAKE_MODELS
    manifest = build_manifest("run-1", config, plan(config, items, models, schemes), models)

    for field in ("run_id", "config", "models", "prompt_versions", "schemes_version",
                  "generation", "n_samples", "scheme_condition",
                  "started_at", "finished_at", "calls_planned", "calls_executed",
                  "calls_from_cache", "calls_failed"):
        assert field in manifest, field
    assert manifest["schemes_version"]["tag"] and manifest["schemes_version"]["content_sha256"]
    entry = manifest["models"][MODEL]
    assert entry["model_id"] and entry["display_name"]
    assert "logprobs_available" in entry


def test_the_manifest_is_written_and_updated(tmp_path, items, schemes, cache):
    config = config_for([STAGE1], samples=1)
    execute(config, items[:2], run_id="m", runs_dir=tmp_path, backend=FakeBackend(),
            cache=cache, models=FAKE_MODELS, schemes=schemes, sleep=lambda _: None)
    manifest = json.loads((tmp_path / "m" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["calls_planned"] == 2
    assert manifest["calls_executed"] == 2
    assert manifest["started_at"] and manifest["finished_at"]


def _raw_line(**overrides):
    line = {
        "run_id": "r", "item_id": "i1", "stage": STAGE2, "scheme_condition": "gold",
        "scheme": "analogy", "cq_id": "CQ1", "sample_index": 0, "model": MODEL,
        "model_id": "fake-model", "prompt_version": "stage2_v1", "params": {},
        "json_schema": answer_schema(load_all()["analogy"], "CQ1"),
        "cache_key": "k", "from_cache": False, "latency_s": 0.1, "error": None,
        "raw_response": {
            "content": '{"answer": "yes", "confidence": 80, "justification": "x"}',
            "finish_reason": "stop",
            "logprobs": {"content": [
                {"token": '{"', "logprob": -0.01, "top_logprobs": []},
                {"token": "answer", "logprob": -0.01, "top_logprobs": []},
                {"token": '": "', "logprob": -0.01, "top_logprobs": []},
                {"token": "yes", "logprob": -0.2, "top_logprobs": [
                    {"token": "yes", "logprob": -0.2231435513},   # 0.8
                    {"token": "no", "logprob": -2.3025850930},    # 0.1
                    {"token": "cannot", "logprob": -2.9957322736},  # 0.05
                    {"token": "na", "logprob": -2.9957322736},      # 0.05
                    {"token": "  ", "logprob": -9.0},
                ]},
                {"token": '", "confidence": 80, "justification": "x"}',
                 "logprob": -0.01, "top_logprobs": []},
            ]},
        },
    }
    for key, value in overrides.items():
        if key in ("content", "logprobs", "finish_reason"):
            line["raw_response"][key] = value
        else:
            line[key] = value
    return line


def test_the_parser_reads_a_good_answer(schemes):
    from argfallacy.parse import parse_answers

    row = parse_answers([_raw_line()], schemes).iloc[0]
    assert row["answer"] == "yes"
    assert row["confidence"] == 80
    assert row["parse_ok"]
    assert row["p_logprob_yes"] == pytest.approx(0.8, abs=1e-6)
    assert row["p_logprob_no"] == pytest.approx(0.1, abs=1e-6)
    assert not row["logprob_partial"]


@pytest.mark.parametrize("content", [
    '{"answer": "maybe", "confidence": 80, "justification": "x"}',
    '{"answer": "yes", "confidence": 150, "justification": "x"}',
    '{"answer": "yes", "confidence": 80}',
    '{"answer": "yes", "confidence": 80, "justification": "x", "extra": 1}',
])
def test_an_answer_outside_its_json_schema_is_invalid_and_kept(schemes, content):
    from argfallacy.parse import parse_answers

    answers = parse_answers([_raw_line(content=content)], schemes)
    assert len(answers) == 1
    assert answers.iloc[0]["answer"] == "invalid"
    assert not answers.iloc[0]["parse_ok"]


def test_the_summary_carries_the_three_probabilities(schemes):
    from argfallacy.parse import parse_answers, summarise

    lines = [_raw_line(sample_index=0)]
    lines += [_raw_line(sample_index=n, content='{"answer": "yes", "confidence": 60, '
                                                '"justification": "x"}') for n in (1, 2)]
    lines += [_raw_line(sample_index=n, content='{"answer": "no", "confidence": 60, '
                                                '"justification": "x"}') for n in (3, 4)]
    summary = summarise(parse_answers(lines, schemes), schemes).iloc[0]

    assert summary["answer"] == "yes"
    assert summary["p_logprob"] == pytest.approx(0.8, abs=1e-6)
    assert summary["p_verbal"] == pytest.approx(0.8, abs=1e-9)
    assert summary["p_sample"] == pytest.approx(3 / 5)
    assert summary["n_samples_ok"] == 5
    assert not summary["idk"]


def _one_request(model_name: str, items, schemes) -> Request:
    config = config_for([STAGE1], samples=1)
    config.models = [model_name]
    return plan(config, items[:1], FAKE_MODELS, schemes)[0].request()


def test_an_unpinned_revision_refuses_to_run(items, schemes, backend):
    """The guard reads the revision of the models file, not the model id."""
    request = _one_request("fake_unpinned", items, schemes)
    assert request.revision == "PLACEHOLDER"
    with pytest.raises(SchemeError, match="no pinned revision"):
        ask(request, backend)
    assert backend.calls == 0


def test_the_max_tokens_of_an_entry_wins_over_the_configuration(items, schemes):
    """Acceptance 8 of spec 04: `max_tokens` of a models.yaml entry, and its key."""
    config = config_for([STAGE1], samples=1)
    assert config.max_tokens() == 512

    config.models = ["fake"]
    plain = plan(config, items[:1], FAKE_MODELS, schemes)[0]
    config.models = ["fake_long"]
    own = plan(config, items[:1], FAKE_MODELS, schemes)[0]

    assert plain.max_tokens == 512
    assert own.max_tokens == 8192
    assert own.request().max_tokens == 8192
    assert own.key() != plain.key(), "max_tokens must reach the cache key"


def test_the_manifest_declares_the_max_tokens_actually_used(items, schemes):
    config = config_for([STAGE1], samples=1)
    config.models = ["fake", "fake_long"]
    calls = plan(config, items[:1], FAKE_MODELS, schemes)
    manifest = build_manifest("run-mt", config, calls, FAKE_MODELS)

    assert manifest["generation"]["max_tokens"] == 512
    assert manifest["models"]["fake"]["max_tokens"] == 512
    assert manifest["models"]["fake_long"]["max_tokens"] == 8192


def test_the_pilot_plans_the_calls_the_spec_declares(schemes):
    """50 items over four entries: 6960 calls, 250 of stage one and 1490 of stage two."""
    from argfallacy.client import load_items

    config = RunConfig.load(REPO_ROOT / "configs" / "pilot.yaml")
    chosen = select_items(config, load_items())
    calls = plan(config, chosen, load_models(), schemes)

    assert len(config.models) == 4
    assert len(calls) == 6960
    for model in config.models:
        mine = [c for c in calls if c.model == model]
        assert len([c for c in mine if c.stage == STAGE1]) == 250, model
        assert len([c for c in mine if c.stage == STAGE2]) == 1490, model


def test_the_pilot_entries_differ_only_in_reasoning_and_room(schemes):
    """The two conditions must be the same weights, and must not share a cache entry."""
    models = load_models()
    for off, on in (("qwen3_8_27b", "qwen3_8_27b_think"),
                    ("gemma4_31b", "gemma4_31b_think")):
        assert models[off]["model_id"] == models[on]["model_id"]
        assert models[off]["revision"] == models[on]["revision"]
        assert models[off]["reasoning"]["chat_template_kwargs"]["enable_thinking"] is False
        assert models[on]["reasoning"]["chat_template_kwargs"]["enable_thinking"] is True
        assert "max_tokens" not in models[off]
        assert models[on]["max_tokens"] == 8192


def test_item_selection_is_reproducible(schemes):
    from argfallacy.client import load_items

    config = RunConfig.load(REPO_ROOT / "configs" / "pilot.yaml")
    frame = load_items()
    first = [r["item_id"] for r in select_items(config, frame)]
    second = [r["item_id"] for r in select_items(config, frame)]
    assert first == second
    assert len(first) == len(set(first)) == config.items["n"]


def _reasoning_response(schemes):
    from argfallacy.prompts import render_stage2

    rendered = render_stage2({"item_id": "r", "text": "Some argument."},
                             schemes["analogy"], "CQ1")
    response = ask(Request(model_id="fake-model", prompt=rendered.text,
                           prompt_version=rendered.prompt_version, sample_index=0,
                           json_schema=rendered.json_schema, revision="x"),
                   FakeBackend(reasoning=True))
    return rendered, response


def test_the_logprob_is_read_on_the_answer_token_not_in_the_reasoning(schemes):
    from argfallacy.parse import answer_position, logprob_masses, parse_content
    from fakes.llm import REAL_WEIGHT, first_token

    rendered, response = _reasoning_response(schemes)
    tokens = response.logprobs["content"]
    answer = parse_content(response.content)["answer"]
    position = answer_position(response.logprobs, rendered.answer_options, "answer")

    assert tokens[position]["token"] == first_token(answer)
    assert "</think>" in "".join(t["token"] for t in tokens[:position]), (
        "the token read must come after the reasoning"
    )
    decoy = next(i for i, t in enumerate(tokens) if t["top_logprobs"])
    assert decoy < position, "the fake must put a decoy before the real answer"

    masses, partial = logprob_masses(response.logprobs, rendered.answer_options, "answer")
    assert masses[answer] == pytest.approx(REAL_WEIGHT, abs=1e-6)
    assert not partial
