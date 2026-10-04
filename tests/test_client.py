"""The client, the plan, the manifest and the parser, all against the fake backend.

No test here touches the network.  The fake counts its
calls, which is the only way to check the two tests
that are about *not* calling: the cache and the resume.
"""

from __future__ import annotations

import copy
import json
import shlex
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from argfallacy.client import (
    RAW_NAME,
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
from argfallacy.client.call import dumps
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


def config_for(stages, condition="gold", name="test") -> RunConfig:
    data = {
        "name": name, "models": [MODEL], "stages": list(stages),
        "scheme_condition": condition, "prompt_version": "v2",
        "items": {}, "generation": {},
    }
    return RunConfig(raw=data, **{k: v for k, v in data.items()})


def with_samples(samples: int) -> dict:
    """The fake models, with the entry ``fake`` taking ``samples`` samples."""
    models = copy.deepcopy(FAKE_MODELS)
    models[MODEL]["samples"] = samples
    return models


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
    ("top_p", 0.95),
    ("top_k", 20),
    ("max_tokens", 128),
    ("model_id", "other-model"),
    ("prompt", "a different question"),
    ("revision", "another-commit"),
    ("tier", 2),
    ("reasoning", {"reasoning_effort": "low"}),
    ("serve_args", ("--generation-config", "vllm")),
])
def test_changing_anything_that_decides_the_answer_changes_the_key(field, value):
    assert cache_key(request_for()) != cache_key(request_for(**{field: value}))


def test_the_three_sampling_parameters_always_go_out(backend):
    """Nothing is left to the defaults of the server, which a model can change."""
    ask(request_for(temperature=1.0, top_p=0.95, top_k=0), backend)
    sent = backend.last_kwargs
    assert sent["temperature"] == 1.0 and sent["top_p"] == 0.95
    assert sent["extra_body"]["top_k"] == 0


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
    config = config_for([STAGE1])
    models = with_samples(2)
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
    config = config_for([STAGE1])
    models = with_samples(1)
    execute(config, items[:2], run_id="r", runs_dir=tmp_path, backend=FakeBackend(fail_after=0),
            cache=cache, models=models, schemes=schemes, max_attempts=1, sleep=lambda _: None)
    assert cache.count() == 0

    healthy = FakeBackend()
    manifest = execute(config, items[:2], run_id="r", runs_dir=tmp_path, backend=healthy,
                       cache=cache, models=models, schemes=schemes, sleep=lambda _: None)
    assert manifest["calls_executed"] == 2
    assert healthy.calls == 2


def test_raw_jsonl_is_read_one_line_at_a_time(tmp_path):
    """A run can leave gigabytes of ``raw.jsonl``: it is never read whole.

    The first row comes back before the second line is decoded, and the line
    separators that ``ensure_ascii=False`` leaves inside a string do not cut
    the line in two.
    """
    first = {"cache_key": "k", "raw_response": {"content": "yes no\x85idk"}}
    (tmp_path / RAW_NAME).write_text(
        dumps(first) + "\n" + "not json\n", encoding="utf-8", newline="\n"
    )
    rows = read_raw(tmp_path)
    assert next(rows) == first
    with pytest.raises(json.JSONDecodeError):
        next(rows)


def test_the_logprobs_are_stored_once_and_without_bytes(tmp_path, items, schemes, cache):
    """``raw.jsonl`` and the cache drop the copy inside ``raw`` and every ``bytes``.

    The response in memory stays whole, and the key and ``top_logprobs`` do not move.
    """
    whole = ask(request_for(), FakeBackend())
    assert whole.raw["choices"][0]["logprobs"] == whole.logprobs
    assert '"bytes"' in json.dumps(whole.logprobs)

    config = config_for([STAGE1, STAGE2])
    models = with_samples(1)
    execute(config, items[:2], run_id="s", runs_dir=tmp_path, backend=FakeBackend(),
            cache=cache, models=models, schemes=schemes, sleep=lambda _: None)
    lines = list(read_raw(tmp_path / "s"))
    cached = [json.loads(row[0]) for row in
              cache.connection.execute("SELECT response FROM responses").fetchall()]
    assert len(cached) == len(lines) == len(plan(config, items[:2], models, schemes))
    for stored in [line["raw_response"] for line in lines] + cached:
        assert "logprobs" not in stored["raw"]["choices"][0]
        assert '"bytes"' not in json.dumps(stored)
        assert any(entry["top_logprobs"] for entry in stored["logprobs"]["content"])
    assert {line["params"]["top_logprobs"] for line in lines} == {20}


def test_a_response_stored_whole_still_reads_and_parses_the_same(cache, schemes):
    """What was written before the reduction, copy and ``bytes`` included, stays usable."""
    from argfallacy.parse import parse_row

    request = request_for(json_schema=answer_schema(schemes["analogy"], "CQ1"))
    whole = ask(request, FakeBackend(reasoning=True))
    key = cache_key(request)
    cache.connection.execute(
        "INSERT INTO responses (cache_key, model_id, prompt_version, sample_index, params, "
        "response) VALUES (?, ?, ?, ?, ?, ?)",
        (key, request.model_id, request.prompt_version, request.sample_index, "{}",
         json.dumps(whole.to_dict(), ensure_ascii=False)),
    )
    assert cache.get(key) == whole

    line = _raw_line()
    before = parse_row({**line, "raw_response": whole.to_dict()}, schemes)
    after = parse_row({**line, "raw_response": whole.to_stored()}, schemes)
    assert before == after
    assert before["parse_ok"] and before["p_logprob_yes"] is not None


def test_the_plan_counts_stage_one_plus_the_cqs_of_each_scheme(items, schemes):
    config = config_for([STAGE1, STAGE2])
    models = FAKE_MODELS
    calls = plan(config, items, models, schemes)

    expected = 5 * len(items)  # stage one, five samples each
    for item in items:
        expected += 5 * len(schemes[item["gold_scheme"]].cqs)
    assert len(calls) == expected * len(config.models)


def test_the_plan_is_in_a_deterministic_order(items, schemes):
    config = config_for([STAGE1, STAGE2])
    models = with_samples(2)
    def shape(calls):
        return [(c.model, c.item_id, c.cq_id, c.sample_index) for c in calls]

    first = shape(plan(config, items, models, schemes))
    second = shape(plan(config, items, models, schemes))
    assert first == second
    assert first == sorted(first, key=lambda t: (t[0], first.index(t)))


def test_the_plan_makes_no_call(items, schemes, backend):
    plan(config_for([STAGE1, STAGE2]), items, FAKE_MODELS, schemes)
    assert backend.calls == 0


def test_each_entry_takes_its_own_samples_drawn_the_same_way(items, schemes):
    """One greedy sample for one entry, five at temperature 1 for the other, seed = index."""
    config = config_for([STAGE1, STAGE2])
    config.models = ["fake_single", "fake"]
    calls = plan(config, items[:1], FAKE_MODELS, schemes)
    questions = 1 + len(schemes[items[0]["gold_scheme"]].cqs)

    single = [c.request() for c in calls if c.model == "fake_single"]
    assert len(single) == questions
    assert {(r.sample_index, r.seed, r.temperature, r.top_p, r.top_k) for r in single} == {
        (0, 0, 0.0, 1.0, 0)}
    assert single[0].serve_args == ("--generation-config", "vllm")

    five = [c.request() for c in calls if c.model == "fake"]
    assert len(five) == 5 * questions
    assert {r.temperature for r in five} == {1.0}
    assert [r.seed for r in five[:5]] == [r.sample_index for r in five[:5]] == [0, 1, 2, 3, 4]


def test_an_entry_without_its_sampling_cannot_be_planned(items, schemes):
    config = config_for([STAGE1])
    models = with_samples(1)
    del models[MODEL]["top_k"]
    with pytest.raises(SchemeError, match=r"'fake' has no \['top_k'\]"):
        plan(config, items[:1], models, schemes)


def test_the_flags_of_the_server_reach_the_cache_key(items, schemes):
    """Two servers launched with different flags do not share an answer."""
    config = config_for([STAGE1])
    models = with_samples(1)
    before = plan(config, items[:1], models, schemes)[0].key()
    models[MODEL]["serve_args"] = ["--generation-config", "vllm"]
    assert plan(config, items[:1], models, schemes)[0].key() != before


def test_the_manifest_has_every_declared_field(items, schemes):
    config = config_for([STAGE1, STAGE2])
    models = FAKE_MODELS
    manifest = build_manifest("run-1", config, plan(config, items, models, schemes), models)

    for field in ("run_id", "config", "models", "prompt_versions", "schemes_version",
                  "generation", "scheme_condition",
                  "started_at", "finished_at", "calls_planned", "calls_executed",
                  "calls_from_cache", "calls_failed", "calls_per_model"):
        assert field in manifest, field
    assert manifest["schemes_version"]["tag"] and manifest["schemes_version"]["content_sha256"]
    entry = manifest["models"][MODEL]
    assert entry["model_id"] and entry["display_name"]
    assert "logprobs_available" in entry
    for field in ("serve_args", "samples", "temperature", "top_p", "top_k"):
        assert field in entry, field


def test_the_manifest_is_written_and_updated(tmp_path, items, schemes, cache):
    config = config_for([STAGE1])
    execute(config, items[:2], run_id="m", runs_dir=tmp_path, backend=FakeBackend(),
            cache=cache, models=with_samples(1), schemes=schemes, sleep=lambda _: None)
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
    config = config_for([STAGE1])
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
    config = config_for([STAGE1])
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
    config = config_for([STAGE1])
    config.models = ["fake", "fake_long"]
    calls = plan(config, items[:1], FAKE_MODELS, schemes)
    manifest = build_manifest("run-mt", config, calls, FAKE_MODELS)

    assert manifest["generation"]["max_tokens"] == 512
    assert manifest["models"]["fake"]["max_tokens"] == 512
    assert manifest["models"]["fake_long"]["max_tokens"] == 8192


PILOT = REPO_ROOT / "configs" / "pilot2.yaml"


def test_the_pilot_plans_the_calls_the_spec_declares(schemes):
    """50 items: 50 questions of stage one and 298 of stage two, times the samples.

    One sample for the two entries with reasoning off, five for the two that reason.
    """
    from argfallacy.client import load_items

    config = RunConfig.load(PILOT)
    chosen = select_items(config, load_items())
    calls = plan(config, chosen, load_models(), schemes)

    assert config.models == ["qwen3_8_27b", "gemma4_31b", "gpt_oss_20b", "k2_horizon_32b"]
    assert len(calls) == 2 * 348 + 2 * 1740
    for model in config.models:
        samples = 1 if model in ("qwen3_8_27b", "gemma4_31b") else 5
        mine = [c for c in calls if c.model == model]
        assert len([c for c in mine if c.stage == STAGE1]) == 50 * samples, model
        assert len([c for c in mine if c.stage == STAGE2]) == 298 * samples, model


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

    config = RunConfig.load(PILOT)
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


def test_one_entry_at_a_time_writes_the_run_of_the_whole_configuration(tmp_path, schemes):
    """The pilot job serves one entry per launch and calls `execute --model` for each.

    Four invocations with the same run id must leave the same `raw.jsonl` and the
    same manifest, timestamps aside, as one invocation over the whole configuration.
    """
    from argfallacy.client import load_items

    config = RunConfig.load(PILOT)
    chosen = select_items(config, load_items())[:2]
    models = load_models()

    def run(where: Path, invocations) -> list[dict]:
        manifests = []
        with ResponseCache(where / "cache.sqlite") as store:
            for only in invocations:
                manifests.append(execute(
                    config, chosen, run_id="pilot", runs_dir=where, backend=FakeBackend(),
                    cache=store, models=models, schemes=schemes, only=only,
                    sleep=lambda _: None,
                ))
        return manifests

    whole = run(tmp_path / "whole", [None])
    split = run(tmp_path / "split", [[entry] for entry in config.models])

    whole_keys = [r["cache_key"] for r in read_raw(tmp_path / "whole" / "pilot")]
    split_keys = [r["cache_key"] for r in read_raw(tmp_path / "split" / "pilot")]
    assert split_keys == whole_keys

    on_disk = json.loads((tmp_path / "split" / "pilot" / "manifest.json").read_text("utf-8"))
    assert on_disk == split[-1]
    assert split[-1]["started_at"] == split[0]["started_at"], "started_at is the first one's"

    def untimed(manifest: dict) -> dict:
        return {k: v for k, v in manifest.items() if k not in ("started_at", "finished_at")}

    assert untimed(split[-1]) == untimed(whole[-1])
    planned = len(plan(config, chosen, models, schemes))
    assert split[0]["calls_planned"] == planned, "every invocation describes the whole run"
    assert split[-1]["calls_executed"] == planned
    for entry in config.models:
        counts = split[-1]["calls_per_model"][entry]
        assert counts["executed"] == counts["planned"] > 0, entry
        assert counts["failed"] == 0


def test_an_entry_not_in_the_configuration_stops_plan_and_execute(tmp_path, items, schemes,
                                                                    backend, cache):
    config = config_for([STAGE1])
    # in the models file, but not among the models of the configuration
    unknown = ["fake_long"]

    with pytest.raises(SchemeError, match=r"fake_long.*\['fake'\]"):
        plan(config, items[:1], FAKE_MODELS, schemes, only=unknown)
    with pytest.raises(SchemeError, match=r"fake_long.*\['fake'\]"):
        execute(config, items[:1], run_id="x", runs_dir=tmp_path, backend=backend, cache=cache,
                models=FAKE_MODELS, schemes=schemes, only=unknown, sleep=lambda _: None)
    assert backend.calls == 0
    assert not (tmp_path / "x").exists()


def test_the_command_line_lists_the_entries_of_the_configuration(capsys):
    from argfallacy.cli import main

    with pytest.raises(SystemExit) as stop:
        main(["run", "plan", str(PILOT), "--model", "nope"])
    message = str(stop.value)
    assert "nope" in message
    for entry in ("qwen3_8_27b", "gemma4_31b", "gpt_oss_20b", "k2_horizon_32b"):
        assert entry in message
    assert "calls" not in capsys.readouterr().out, "nothing was planned"


def _serving_script(name: str):
    import importlib.util

    path = REPO_ROOT / "serving" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flag(args: list[str], name: str) -> str | None:
    return args[args.index(name) + 1] if name in args else None


@pytest.mark.parametrize("entry,parser,room", [
    ("qwen3_8_27b", None, "8192"),
    ("gemma4_31b", None, "8192"),
    ("qwen3_8_27b_think", "qwen3", "16384"),
    ("gemma4_31b_think", "gemma4", "16384"),
])
def test_the_serve_command_of_each_pilot_entry(entry, parser, room):
    """Reasoning off without a parser; reasoning on with the model's parser and room."""
    spec = load_models()[entry]
    args = _serving_script("serve_command").command(entry)

    assert args[:3] == [spec["model_id"], "--revision", spec["revision"]]
    assert _flag(args, "--reasoning-parser") == parser
    assert _flag(args, "--max-model-len") == room
    assert args.count("--max-model-len") == 1


def test_the_serve_command_refuses_an_unpinned_revision(capsys, monkeypatch):
    serve_command = _serving_script("serve_command")
    fakes = Path(__file__).parent / "fakes" / "models.yaml"
    monkeypatch.setattr(serve_command, "command",
                        partial(serve_command.command, models_file=fakes))
    assert "PLACEHOLDER" in load_models(fakes)["fake_unpinned"]["revision"]

    assert serve_command.main(["fake_unpinned"]) != 0
    printed = capsys.readouterr()
    assert printed.out == "", "a job reading the command must get nothing to launch"
    assert "no pinned revision" in printed.err


def test_the_smoke_test_checks_the_reasoning_of_the_entries_that_reason():
    """``enable_thinking`` where the entry sets it, ``is_reasoning`` where it does not.

    Every entry is a reasoning model, the two main entries of the pilot included:
    checking ``is_reasoning`` alone would fail them for not reasoning.
    """
    thinking_on = _serving_script("smoke_test").thinking_on
    models = load_models()
    for entry in ("qwen3_8_27b_think", "gemma4_31b_think", "k2_horizon_32b",
                  "gpt_oss_20b", "gpt_oss_120b"):
        assert thinking_on(models[entry]), entry
    for entry in ("qwen3_8_27b", "gemma4_31b"):
        assert models[entry]["is_reasoning"] and not thinking_on(models[entry]), entry


def _server_like(prompt_tokens: int | None = None, reasoning: str | None = None):
    """The fake backend with ``usage`` and the message changed as a real server changes them.

    ``usage`` counts no reasoning tokens, as vLLM did for K2 Horizon.
    """
    fake = FakeBackend()

    def create(**kwargs):
        payload = fake.create(**kwargs)
        payload["usage"]["completion_tokens_details"] = {"reasoning_tokens": 0}
        if prompt_tokens is not None:
            payload["usage"]["prompt_tokens"] = prompt_tokens
        if reasoning is not None:
            payload["choices"][0]["message"]["reasoning_content"] = reasoning
        return payload

    return SimpleNamespace(create=create)


def test_the_smoke_test_reads_the_reasoning_in_the_message(monkeypatch, capsys):
    """``usage`` at zero and the thinking in ``reasoning_content``: there was reasoning."""
    smoke_test = _serving_script("smoke_test")
    thinking = "The claim rests on a doctorate, not on expertise in tariffs."
    monkeypatch.setattr(smoke_test, "build_backend", lambda *_: _server_like(reasoning=thinking))
    assert smoke_test.main(["--model", "fake", "--fake-reasoning"]) == 0
    printed = capsys.readouterr().out
    assert f"Thinking  : {len(thinking)} characters, from the reasoning_content" in printed

    monkeypatch.setattr(smoke_test, "build_backend", lambda *_: _server_like())
    assert smoke_test.main(["--model", "fake", "--fake-reasoning"]) != 0, "no reasoning anywhere"


def test_the_smoke_test_fails_on_a_prompt_the_server_did_not_read(monkeypatch):
    """Ten prompt tokens for a prompt of thousands of characters, as K2 Horizon read it."""
    smoke_test = _serving_script("smoke_test")
    monkeypatch.setattr(smoke_test, "build_backend", lambda *_: _server_like(prompt_tokens=10))
    assert smoke_test.main(["--model", "fake"]) != 0

    monkeypatch.setattr(smoke_test, "build_backend", lambda *_: _server_like())
    assert smoke_test.main(["--model", "fake"]) == 0


@pytest.mark.parametrize("entry", ["qwen3_8_27b", "gemma4_31b", "gpt_oss_20b", "k2_horizon_32b"])
def test_the_servers_of_the_pilot_compact_the_json_with_their_own_sampling(entry):
    """xgrammar without free whitespace, vLLM's neutral defaults, and the JSON intact.

    The job reads the line of ``serve_command.py`` back as shell words, so the
    JSON of ``--structured-outputs-config`` must come back as one argument.
    """
    args = _serving_script("serve_command").command(entry)
    assert shlex.split(shlex.join(args)) == args
    assert json.loads(_flag(args, "--structured-outputs-config")) == {
        "backend": "xgrammar", "disable_any_whitespace": True}
    assert _flag(args, "--generation-config") == "vllm"
    spec = load_models()[entry]
    if spec.get("max_tokens", 512) > 512:
        room = int(_flag(args, "--max-model-len"))
        assert room % 1024 == 0 and room > spec["max_tokens"]


def _summary_of(schemes, answers):
    """The summary row of one question, from (answer, confidence, finish_reason) per sample."""
    from argfallacy.parse import parse_answers, summarise

    lines = []
    for index, (answer, confidence, finish) in enumerate(answers):
        content = json.dumps({"answer": answer, "confidence": confidence, "justification": "x"})
        lines.append(_raw_line(sample_index=index, content=content, finish_reason=finish))
    return summarise(parse_answers(lines, schemes), schemes).iloc[0]


def test_one_sample_is_the_hard_answer_and_has_no_frequency(schemes):
    summary = _summary_of(schemes, [("no", 70, "stop")])
    assert summary["answer"] == "no"
    assert summary["p_verbal"] == pytest.approx(0.3)
    assert summary["p_logprob"] == pytest.approx(0.8, abs=1e-6)
    assert pd.isna(summary["p_sample"])


def test_the_majority_counts_only_the_valid_samples(schemes):
    """Two cut-off yes do not outvote two no: they are not valid samples."""
    summary = _summary_of(schemes, [("yes", 90, "length"), ("yes", 90, "length"),
                                    ("no", 60, "stop"), ("no", 60, "stop"),
                                    ("yes", 90, "stop")])
    assert summary["answer"] == "no"
    assert summary["n_samples_ok"] == 3
    assert summary["p_sample"] == pytest.approx(1 / 3)


def test_a_tie_goes_to_the_higher_confidence_then_to_the_lower_index(schemes):
    by_confidence = _summary_of(schemes, [("yes", 60, "stop"), ("no", 90, "stop"),
                                          ("yes", 70, "stop"), ("no", 80, "stop")])
    assert by_confidence["answer"] == "no"
    by_index = _summary_of(schemes, [("no", 80, "stop"), ("yes", 70, "stop"),
                                     ("yes", 90, "stop"), ("no", 80, "stop")])
    assert by_index["answer"] == "no"
    nothing = _summary_of(schemes, [("yes", 80, "length"), ("no", 80, "length")])
    assert nothing["answer"] == "invalid"


def _spaced(fake_create):
    """A server that lays the JSON out with a newline and an indent."""
    def create(**kwargs):
        payload = fake_create(**kwargs)
        choice = payload["choices"][0]
        choice["message"]["content"] = choice["message"]["content"].replace("{", "{\n  ", 1)
        first = (choice["logprobs"] or {}).get("content") or []
        if first:
            first[0]["token"] = first[0]["token"].replace("{", "{\n  ", 1)
        return payload

    return SimpleNamespace(create=create)


def test_the_smoke_test_fails_when_the_json_is_not_compact(monkeypatch, capsys):
    """Three calls at temperature 1 must all begin with ``{"answer": "``."""
    smoke_test = _serving_script("smoke_test")
    monkeypatch.setattr(smoke_test, "build_backend", lambda *_: FakeBackend())
    assert smoke_test.main(["--model", "fake"]) == 0
    assert capsys.readouterr().out.count("Layout") == 3

    monkeypatch.setattr(smoke_test, "build_backend",
                        lambda *_: _spaced(FakeBackend().create))
    assert smoke_test.main(["--model", "fake"]) != 0
