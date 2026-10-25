"""D+Q prompts, independent calls, per-key probabilities and format gates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import replace

import pandas as pd
import pytest

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
from argfallacy.parse import (
    ANSWERS_COLUMNS,
    SUMMARY_COLUMNS,
    answer_position,
    parse_answers,
    parse_run,
    summarise,
)
from argfallacy.paths import REPO_ROOT
from argfallacy.prompts import answer_options, load_answer_meanings, render_zeroshot, verdicts
from argfallacy.prompts.render import listed_options
from argfallacy.schemes import SchemeError, load_all
from fakes.llm import FakeBackend

MODELS = load_models(REPO_ROOT / "tests/fakes/models.yaml")
ITEM = {"item_id": "test01", "text": "An argument for a format test.", "gold_scheme": "analogy"}


@pytest.fixture(scope="module", params=["v1", "v2"])
def versioned(request):
    folder = request.param
    return (
        folder,
        load_all(REPO_ROOT / "schemes" / folder),
        load_answer_meanings("v2" if folder == "v1" else "v3"),
    )


def config(folder="v1", model="fake_single"):
    return RunConfig(
        name="dq_test", models=[model], stages=["dq"], schemes=folder, prompt_version="v1"
    )


def test_prompt_and_schema(versioned, vocab):
    from argfallacy.prompts import render_dq

    _, schemes, definitions = versioned
    for scheme in schemes.values():
        rendered = render_dq(ITEM, scheme, definitions)
        assert rendered == render_dq(ITEM, scheme, definitions)
        direct = render_zeroshot(ITEM, scheme)
        assert (
            rendered.text.split("## The scheme:")[1].split("## The critical questions")[0]
            == direct.text.split("## The scheme:")[1].split("## The verdicts")[0]
        )
        assert (
            rendered.text.split("## The verdicts you may give")[1].split("## What to do")[0]
            == direct.text.split("## The verdicts you may give")[1].split("## What to do")[0]
        )
        positions = []
        for cq in scheme.cq_order:
            positions.append(rendered.text.index(f"**{cq}.** {scheme.cqs[cq].text}"))
            block = rendered.text.split(f"**{cq}.** ")[1].split("\n\n")[0]
            options = listed_options(scheme, cq, definitions)
            assert "Options: " + ", ".join(f"`{a}`" for a, _ in options) + "." in block
            own = definitions.get("per_cq", {}).get(scheme.scheme_id, {}).get(cq)
            for answer, meaning in options:
                assert (f"* `{answer}`: {meaning}" in block) == (own is not None)
        assert positions == sorted(positions)
        for meaning in definitions["default"].values():
            assert meaning not in rendered.text
        assert rendered.text.count("* `cannot_be_determined`:") == 1
        assert rendered.text.count("* `na`:") == 1
        assert definitions["na"] in rendered.text
        assert "diagram" not in rendered.text.lower()
        for terminal in vocab["terminals"]:
            assert f"`{terminal}`" not in rendered.text
        schema = rendered.json_schema
        assert list(schema["properties"]) == [
            *scheme.cq_order,
            "verdict",
            "confidence",
            "justification",
        ]
        assert schema["required"] == list(schema["properties"])
        assert schema["additionalProperties"] is False
        for cq in scheme.cq_order:
            assert schema["properties"][cq]["enum"] == list(answer_options(scheme, cq))
        assert schema["properties"]["verdict"]["enum"] == [v.shown for v in verdicts(scheme)]


def test_pairing_rules():
    for stage in ["stage1", "stage2"]:
        with pytest.raises(SchemeError):
            RunConfig(
                name="bad",
                models=["fake_single"],
                stages=[stage],
                schemes="v2",
                prompt_version="v2",
            )
        with pytest.raises(SchemeError):
            RunConfig(
                name="bad",
                models=["fake_single"],
                stages=[stage],
                schemes="v1",
                prompt_version="v3",
            )
    for stage in ["dq", "zeroshot"]:
        for folder in ["v1", "v2"]:
            RunConfig(
                name="ok",
                models=["fake_single"],
                stages=[stage],
                schemes=folder,
                prompt_version="v1",
            )
    with pytest.raises(SchemeError):
        RunConfig(
            name="mixed", models=["fake_single"], stages=["dq", "zeroshot"], prompt_version="v1"
        )
    with pytest.raises(SchemeError):
        RunConfig(
            name="bad", models=["fake_single"], stages=["ssoft"], schemes="v1", prompt_version="v1"
        )


def test_zero_shot_split_and_configurations(vocab, monkeypatch):
    from argfallacy.prompts import render as rendering

    # Share the vocabulary during this full-population check; its contents are unchanged.
    monkeypatch.setattr(rendering, "load_vocabulary", lambda **kw: vocab)
    items = load_items()
    configurations = {}
    for name, expected in [
        ("dq_full_v2", 601),
        ("dq_full_v3", 601),
        ("dq_pilot_v2", 50),
        ("dq_pilot_v3", 50),
        ("zeroshot_schemes_v2", 601),
    ]:
        cfg = RunConfig.load(REPO_ROOT / "configs" / f"{name}.yaml")
        selected = select_items(cfg, items)
        calls = plan(cfg, selected, only=["qwen3_8_27b"])
        assert len(calls) == expected
        if name.startswith("dq"):
            assert all(c.stage == "dq" and not c.cq_id for c in calls)
        configurations[name] = (cfg, selected, calls)
    pilot_items = select_items(RunConfig.load(REPO_ROOT / "configs/pilot2.yaml"), items)
    for version in ["v2", "v3"]:
        assert configurations[f"dq_pilot_{version}"][1] == pilot_items
        full = {c.item_id: c.key() for c in configurations[f"dq_full_{version}"][2]}
        assert all(c.key() == full[c.item_id] for c in configurations[f"dq_pilot_{version}"][2])
    original = RunConfig.load(REPO_ROOT / "configs/zeroshot_v1.yaml")
    old = plan(original, select_items(original, items), only=["qwen3_8_27b"])
    new = configurations["zeroshot_schemes_v2"][2]
    equal = [(a.prompt == b.prompt, a.key() == b.key()) for a, b in zip(old, new, strict=True)]
    assert all(a == b for a, b in equal)
    assert Counter(a for a, _ in equal) == {True: 409, False: 192}
    v1, v2 = load_all(), load_all(REPO_ROOT / "schemes/v2")
    unchanged = {
        s for s in v1 if render_zeroshot(ITEM, v1[s]).text == render_zeroshot(ITEM, v2[s]).text
    }
    assert all(
        same == (call.scheme in unchanged) for call, (same, _) in zip(old, equal, strict=True)
    )


def test_resume_samples_and_pilot_cache(tmp_path, versioned):
    folder, schemes, _ = versioned
    cfg = config(folder)
    items = [dict(ITEM, item_id=f"test{i}", text=f"Argument {i}.") for i in range(4)]
    backend = FakeBackend(fail_after=2)
    models = copy.deepcopy(MODELS)
    models["fake_single"]["max_concurrency"] = 1
    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        execute(cfg, items, "resume", tmp_path, backend, cache, models, schemes)
        backend.fail_after = None
        execute(cfg, items, "resume", tmp_path, backend, cache, models, schemes)
        rows = list(read_raw(tmp_path / "resume"))
        success = [r for r in rows if not r["error"]]
        assert len(success) == len({r["cache_key"] for r in success}) == 4
        before = backend.calls
        execute(cfg, items, "resume", tmp_path, backend, cache, models, schemes)
        assert backend.calls == before
    from argfallacy.format_report import check_format

    parse_run(tmp_path / "resume", schemes)
    checks, _ = check_format(tmp_path / "resume")
    assert checks["failed"].eq(0).all()
    with ResponseCache(tmp_path / "pilot-cache.sqlite") as cache:
        execute(cfg, items[:2], "pilot", tmp_path, backend, cache, models, schemes)
        before = backend.calls
        execute(cfg, items, "full", tmp_path, backend, cache, models, schemes)
        assert backend.calls - before == 2
        assert sum(r["from_cache"] for r in read_raw(tmp_path / "full")) == 2
    calls = plan(config(folder, "fake"), items, MODELS, schemes)
    assert len(calls) == len(items) * 5
    assert {c.sample_index for c in calls} == set(range(5))
    manifest = json.loads((tmp_path / "resume/manifest.json").read_text())
    source = REPO_ROOT / "prompts" / f"stage2_{'v2' if folder == 'v1' else 'v3'}_answers.yaml"
    assert manifest["answer_definitions"] == {
        "file": source.relative_to(REPO_ROOT).as_posix(),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def test_fifty_pilot_calls_reused_by_full(tmp_path, monkeypatch, vocab):
    from argfallacy.client import run
    from argfallacy.prompts import render as rendering

    monkeypatch.setattr(rendering, "load_vocabulary", lambda **kw: vocab)
    monkeypatch.setattr(run, "load_vocabulary", lambda **kw: vocab)
    pilot = RunConfig.load(REPO_ROOT / "configs/dq_pilot_v3.yaml")
    full = RunConfig.load(REPO_ROOT / "configs/dq_full_v3.yaml")
    models = {name: MODELS["fake_single"] for name in full.models}
    items = load_items()
    backend = FakeBackend()
    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        execute(pilot, select_items(pilot, items), "pilot", tmp_path, backend, cache, models)
        assert backend.calls == 50
        execute(
            full,
            select_items(full, items),
            "full",
            tmp_path,
            backend,
            cache,
            models,
            only=["qwen3_8_27b"],
        )
        assert backend.calls == 601
        assert sum(r["from_cache"] for r in read_raw(tmp_path / "full")) == 50


def test_fake_run_expands_every_scheme(tmp_path, versioned):
    folder, schemes, _ = versioned
    items = [dict(ITEM, item_id=f"item{i}", gold_scheme=sid) for i, sid in enumerate(schemes)]
    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        execute(config(folder), items, "all", tmp_path, FakeBackend(), cache, MODELS, schemes)
    parse_run(tmp_path / "all", schemes)
    answers = pd.read_csv(tmp_path / "all/answers.csv", keep_default_na=False)
    assert answers["parse_ok"].all()
    for sid, scheme in schemes.items():
        own = answers[answers["scheme"] == sid].set_index("cq_id")
        assert len(own) == len(scheme.cq_order) + 1
        for cq in scheme.cq_order:
            columns = [f"p_logprob_{a}" for a in answer_options(scheme, cq)]
            assert sum(float(own.loc[cq, col]) for col in columns) == pytest.approx(1)


def hand_line(scheme, order=None, sample=0):
    from argfallacy.prompts import render_dq

    schema = render_dq(ITEM, scheme, load_answer_meanings("v2")).json_schema
    order = order or scheme.cq_order
    values = {cq: answer_options(scheme, cq)[sample % 2] for cq in order}
    values.update(
        verdict=verdicts(scheme)[sample % len(verdicts(scheme))].shown,
        confidence=75,
        justification="The stated reason.",
    )
    body = json.dumps(values)
    tokens = []
    for index, (key, value) in enumerate(values.items()):
        head = ("{" if index == 0 else ", ") + json.dumps(key) + ": "
        if key in scheme.cqs or key == "verdict":
            tokens.append({"token": head + '"', "logprob": 0, "top_logprobs": []})
            options = schema["properties"][key]["enum"]
            mass = 0.51 + 0.04 * index
            weights = {a: mass if a == value else (1 - mass) / (len(options) - 1) for a in options}
            tokens.append(
                {
                    "token": value,
                    "logprob": math.log(mass),
                    "top_logprobs": [
                        {"token": a, "logprob": math.log(w)} for a, w in weights.items()
                    ],
                }
            )
            tokens.append({"token": '"', "logprob": 0, "top_logprobs": []})
        else:
            tokens.append({"token": head + json.dumps(value), "logprob": 0, "top_logprobs": []})
    tokens.append({"token": "}", "logprob": 0, "top_logprobs": []})
    assert "".join(t["token"] for t in tokens) == body
    return {
        "run_id": "hand",
        "item_id": "test01",
        "stage": "dq",
        "scheme_condition": "gold",
        "scheme": scheme.scheme_id,
        "cq_id": "",
        "sample_index": sample,
        "model": "fake",
        "model_id": "fake-model",
        "prompt_version": "dq_v1",
        "json_schema": schema,
        "cache_key": f"key{sample}",
        "from_cache": False,
        "error": None,
        "raw_response": {"content": body, "finish_reason": "stop", "logprobs": {"content": tokens}},
    }


@pytest.fixture
def prefix_scheme(schemes):
    base = schemes[ITEM["gold_scheme"]]
    cq = next(iter(base.cqs.values()))
    keys = ["CQ1", "CQ1.1", "CQ2", "CQ2.1"]
    return replace(base, cq_order=keys, cqs={k: replace(cq, id=k) for k in keys})


@pytest.mark.parametrize(
    "order", [None, ["CQ1.1", "CQ2.1", "CQ1", "CQ2"], ["CQ2", "CQ1", "CQ2.1", "CQ1.1"]]
)
def test_parser_exact_keys_and_positions(prefix_scheme, order):
    scheme = prefix_scheme
    line = hand_line(scheme, order)
    answers = parse_answers([line], {scheme.scheme_id: scheme}).set_index("cq_id")
    assert len(answers) == 5
    assert answers["parse_ok"].all()
    assert answers["keys_in_order"].eq(order is None).all()
    positions = {}
    for index, key in enumerate(order or scheme.cq_order):
        positions[key] = answer_position(
            line["raw_response"]["logprobs"], answer_options(scheme, key), key
        )
        assert answers.loc[key, "p_logprob_yes"] == pytest.approx(0.51 + index * 0.04)
        assert pd.isna(answers.loc[key, "confidence"])
        assert pd.isna(answers.loc[key, "justification"])
    assert len(set(positions.values())) == 4
    assert answers.loc["", "p_label"] == pytest.approx(0.67)
    assert answers.loc["", "answer"] == verdicts(scheme)[0].terminal
    summary = summarise(answers.reset_index(), {scheme.scheme_id: scheme})
    assert summary.loc[summary.cq_id != "", "p_verbal"].isna().all()


def test_invalid_object_invalidates_all_rows(prefix_scheme):
    line = hand_line(prefix_scheme)
    content = json.loads(line["raw_response"]["content"])
    content["CQ1"] = "outside the enumeration"
    line["raw_response"]["content"] = json.dumps(content)
    answers = parse_answers([line], {prefix_scheme.scheme_id: prefix_scheme})
    assert answers["answer"].eq("invalid").all()
    assert not answers["parse_ok"].any()


def test_majorities_are_per_cq_and_verdict(prefix_scheme):
    scheme = prefix_scheme
    lines = [hand_line(scheme, sample=i % 2) for i in range(5)]
    for i, line in enumerate(lines):
        line["sample_index"] = i
        value = json.loads(line["raw_response"]["content"])
        value["CQ1"] = "no" if i < 3 else "yes"
        line["raw_response"]["content"] = json.dumps(value)
    summary = summarise(
        parse_answers(lines, {scheme.scheme_id: scheme}), {scheme.scheme_id: scheme}
    ).set_index("cq_id")
    assert summary.loc["CQ1", "answer"] == "no"
    assert summary.loc["CQ2", "answer"] == "yes"
    assert summary.loc["", "answer"] == verdicts(scheme)[0].terminal
    assert summary.loc["CQ1", "p_sample_no"] == pytest.approx(0.6)
    assert summary.loc["", "p_sample"] == pytest.approx(0.6)


def test_existing_tables_keep_their_columns(schemes):
    scheme = schemes[ITEM["gold_scheme"]]
    line = hand_line(scheme)
    line["stage"] = "zeroshot"
    rendered = render_zeroshot(ITEM, scheme)
    line["json_schema"] = rendered.json_schema
    value = json.loads(line["raw_response"]["content"])
    line["raw_response"]["content"] = json.dumps(
        {"answer": value["verdict"], "confidence": 75, "justification": "x"}
    )
    answers = parse_answers([line], schemes)
    assert list(answers.columns) == ANSWERS_COLUMNS
    assert list(summarise(answers, schemes).columns) == SUMMARY_COLUMNS


@pytest.mark.parametrize("stage", ["dq", "zeroshot"])
@pytest.mark.parametrize(
    "fault", [None, "error", "missing", "invalid", "length", "order", "cq_masses", "label"]
)
def test_format_gate(tmp_path, monkeypatch, stage, fault):
    from argfallacy.cli import main
    from argfallacy.client import run

    cfg = (
        config()
        if stage == "dq"
        else RunConfig(
            name="direct", models=["fake_single"], stages=["zeroshot"], prompt_version="v1"
        )
    )
    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        execute(cfg, [ITEM], "clean", tmp_path, FakeBackend(reasoning=True), cache, MODELS)
    raw_path = tmp_path / "clean/raw.jsonl"
    rows = list(read_raw(raw_path))
    line = rows[0]
    if fault == "error":
        line["error"] = "a failed call"
        line["raw_response"] = {}
    elif fault == "invalid":
        line["raw_response"]["content"] = "{}"
    elif fault == "length":
        line["raw_response"]["finish_reason"] = "length"
    elif fault == "order":
        value = json.loads(line["raw_response"]["content"])
        line["raw_response"]["content"] = json.dumps(dict(reversed(list(value.items()))))
    elif fault in ["cq_masses", "label"]:
        key = "verdict" if stage == "dq" else "answer"
        if fault == "cq_masses" and stage == "dq":
            key = next(iter(line["json_schema"]["properties"]))
        logprobs = line["raw_response"]["logprobs"]
        enum = line["json_schema"]["properties"][key]["enum"]
        position = answer_position(logprobs, enum, key)
        logprobs["content"][position]["token"] = "unrecognised"
    if fault == "missing":
        rows = []
    raw_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    parse_run(tmp_path / "clean")
    monkeypatch.setenv("ARGFALLACY_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(run, "load_items", lambda *a, **kw: pytest.fail("format reads gold"))
    assert (main(["report", "format", "clean", "--model", "fake_single"]) == 0) == (fault is None)


def test_fake_returns_each_property_and_alternatives(prefix_scheme):
    schema = hand_line(prefix_scheme)["json_schema"]
    backend = FakeBackend(reasoning=True)
    response = backend.create(
        model="fake",
        messages=[{"content": "test"}],
        logprobs=True,
        response_format={"json_schema": {"schema": schema}},
    )
    choice = response["choices"][0]
    content = json.loads(choice["message"]["content"])
    assert list(content) == list(schema["properties"])
    for key, prop in schema["properties"].items():
        if "enum" in prop:
            position = answer_position(choice["logprobs"], prop["enum"], key)
            assert position is not None
            assert choice["logprobs"]["content"][position]["top_logprobs"]


def test_format_entry_selection_and_no_logprobs(tmp_path, monkeypatch):
    from argfallacy.cli import main

    models = copy.deepcopy(MODELS)
    models["fake_single"]["supports_logprobs"] = False
    cfg = replace(config(), models=["fake_single", "fake"])
    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        execute(
            cfg, [ITEM], "partial", tmp_path, FakeBackend(), cache, models, only=["fake_single"]
        )
    parse_run(tmp_path / "partial")
    monkeypatch.setenv("ARGFALLACY_RUNS_DIR", str(tmp_path))
    assert main(["report", "format", "partial", "--model", "fake_single"]) == 0
    assert main(["report", "format", "partial"]) != 0
    assert main(["report", "format", "partial", "--model", "unknown"]) != 0
