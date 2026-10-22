"""A5, the LLM judge: its prompt, the stage ``judge`` of the client, and how its lines parse.

The judge prompt is the zero-shot prompt plus the answers to the critical questions,
so the tests check that nothing else differs.  The answers come from the
``summary.csv`` of a run, written here by hand.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from argfallacy.client import (
    RunConfig,
    execute,
    load_items,
    load_models,
    plan,
    read_raw,
    select_items,
)
from argfallacy.paths import REPO_ROOT
from argfallacy.prompts import JUDGE, render_judge, render_zeroshot, verdict_schema, verdicts
from argfallacy.schemes import SchemeError, load_all
from fakes.llm import FakeBackend

ITEM = {"item_id": "0123456789abcdef", "text": "Everyone I know buys it, so it must be good."}
FAKE_MODELS = load_models(Path(__file__).parent / "fakes" / "models.yaml")
SECTION = "## Answers to the critical questions"
SENTENCE = ("Weigh the answers to the critical questions against the text: they are evidence, "
            "not a verdict, and you may disagree with any of them.")


def _answers(scheme, invalid=None):
    out = {cq: (scheme.answer_space(cq)[0], f"Because of {cq}.") for cq in scheme.cq_order}
    if invalid:
        out[invalid] = ("invalid", None)
    return out


def test_rendering_is_deterministic(schemes):
    for scheme in schemes.values():
        first = render_judge(ITEM, scheme, _answers(scheme))
        second = render_judge(ITEM, scheme, _answers(scheme))
        assert first.text == second.text and first.json_schema == second.json_schema
        assert "{{" not in first.text
        assert first.stage == JUDGE and first.prompt_version == "judge_v1"


def test_the_prompt_lists_every_cq_word_for_word_with_its_answer(schemes):
    for scheme in schemes.values():
        last = scheme.cq_order[-1]
        answers = _answers(scheme, invalid=last)
        text = render_judge(ITEM, scheme, answers).text
        for cq in scheme.cq_order:
            assert f"**{cq}.** {scheme.cqs[cq].text}\n" in text, (scheme.scheme_id, cq)
            if cq != last:
                assert f"Answer: `{answers[cq][0]}`. Because of {cq}." in text
        assert text.count("Answer: none usable.") == 1
        assert text.index("**CQ1.**") < text.index(f"**{last}.**")

        bare = render_judge(ITEM, scheme, answers, justifications=False).text
        assert "Because of" not in bare
        for cq in scheme.cq_order[:-1]:
            assert f"Answer: `{answers[cq][0]}`.\n" in bare


def test_the_only_difference_from_the_zero_shot_is_the_answers(schemes):
    for scheme in schemes.values():
        judge = render_judge(ITEM, scheme, _answers(scheme))
        zeroshot = render_zeroshot(ITEM, scheme)
        head, rest = judge.text.split(SECTION)
        tail = rest[rest.index("## The verdicts you may give"):]
        assert (head + tail).replace(SENTENCE + "\n\n", "") == zeroshot.text
        assert tail.index(SENTENCE) < tail.index("Choose the single verdict")
        assert judge.answer_options == zeroshot.answer_options
        schema, reference = judge.json_schema, zeroshot.json_schema
        assert schema["title"] == f"judge_v1_{scheme.scheme_id}"
        assert {k: v for k, v in schema.items() if k not in ("title", "description")} == {
            k: v for k, v in reference.items() if k not in ("title", "description")}


def _source_run(runs: Path, run_id: str, items, schemes, model="qwen3_8_27b", drop=None,
                folder="v1"):
    """The summary.csv of a run with an answer and a justification to every CQ."""
    rows = []
    for item in items:
        scheme = schemes[item["gold_scheme"]]
        for cq in scheme.cq_order:
            if (item["item_id"], cq) == drop:
                continue
            rows.append({"item_id": item["item_id"], "stage": "stage2",
                         "scheme_condition": "gold", "scheme": scheme.scheme_id, "cq_id": cq,
                         "model": model, "answer": scheme.answer_space(cq)[1],
                         "justification": f"Seen in {item['item_id']}."})
    (runs / run_id).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(runs / run_id / "summary.csv", index=False)
    (runs / run_id / "manifest.json").write_text(
        json.dumps({"schemes_version": {"folder": folder}}), encoding="utf-8")


def _config(justifications=True, model="fake") -> RunConfig:
    data = {"name": "judge", "models": [model], "stages": [JUDGE], "scheme_condition": "gold",
            "prompt_version": "v1", "items": {}, "generation": {},
            "judge": {"answers_from": "src", "answers_model": model,
                      "justifications": justifications}}
    return RunConfig(raw=data, **data)


def _items(schemes):
    return [{"item_id": f"item{n:02d}", "text": f"Argument {n}.", "gold_scheme": s}
            for n, s in enumerate(sorted(schemes))]


def test_a_source_run_without_a_cq_of_an_item_stops_the_plan(tmp_path, schemes):
    items = _items(schemes)
    _source_run(tmp_path, "src", items, schemes, model="fake", drop=("item03", "CQ2"))
    with pytest.raises(SchemeError, match=r"item03.*CQ2"):
        plan(_config(), items, FAKE_MODELS, schemes, runs_dir=tmp_path)


def test_a_summary_without_justifications_stops_the_plan(tmp_path, schemes):
    items = _items(schemes)
    _source_run(tmp_path, "src", items, schemes, model="fake")
    path = tmp_path / "src" / "summary.csv"
    pd.read_csv(path).drop(columns="justification").to_csv(path, index=False)
    with pytest.raises(SchemeError, match="parse"):
        plan(_config(), items, FAKE_MODELS, schemes, runs_dir=tmp_path)


def test_the_judge_configurations_plan_601_calls(tmp_path, schemes):
    try:
        frame = load_items()
    except SchemeError:
        pytest.skip("no data/items.csv")
    models = load_models()
    keys = {}
    for name in ("judge_full_v2", "judge_full_v2_answers_only"):
        config = RunConfig.load(REPO_ROOT / "configs" / f"{name}.yaml")
        items = select_items(config, frame)
        _source_run(tmp_path, config.judge["answers_from"], items, schemes,
                    model=config.judge["answers_model"])
        calls = plan(config, items, models, schemes, runs_dir=tmp_path)
        assert len(calls) == 601
        assert {c.stage for c in calls} == {JUDGE} and {c.cq_id for c in calls} == {None}
        keys[name] = {c.key() for c in calls}
    assert not keys["judge_full_v2"] & keys["judge_full_v2_answers_only"]


@pytest.mark.parametrize("folder", ["v1", "v2"])
def test_a_judge_run_on_the_fake_backend_parses_to_terminals(tmp_path, folder):
    from argfallacy.client import ResponseCache
    from argfallacy.parse import parse_answers, summarise

    schemes = load_all(REPO_ROOT / "schemes" / folder)
    config = replace(_config(), schemes=folder)
    items = _items(schemes)
    _source_run(tmp_path, "src", items, schemes, model="fake", folder=folder)
    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        execute(config, items, run_id="judged", runs_dir=tmp_path, backend=FakeBackend(),
                cache=cache, models=FAKE_MODELS, schemes=schemes, sleep=lambda _: None)
    lines = list(read_raw(tmp_path / "judged"))
    assert len(lines) == len(items) * int(FAKE_MODELS["fake"]["samples"])
    assert {line["stage"] for line in lines} == {JUDGE}
    assert all(not line["cq_id"] for line in lines)
    answers = parse_answers(lines, schemes)
    assert answers["parse_ok"].all()
    for row in answers.itertuples():
        assert row.answer in schemes[row.scheme].terminals
        assert row.p_label is not None and not math.isnan(row.p_label)
    summary = summarise(answers, schemes)
    assert summary["p_label"].notna().all() and summary["p_logprob"].isna().all()


@pytest.mark.parametrize("justifications", [True, False])
def test_the_full_v3_judge_uses_the_revised_cqs_and_the_same_template(tmp_path, justifications):
    schemes = load_all(REPO_ROOT / "schemes" / "v2")
    name = "judge_full_v3" + ("" if justifications else "_answers_only")
    config = RunConfig.load(REPO_ROOT / "configs" / f"{name}.yaml")
    items = select_items(config, load_items())
    _source_run(tmp_path, config.judge["answers_from"], items, schemes,
                model=config.judge["answers_model"], folder="v2")
    calls = plan(config, items, runs_dir=tmp_path)
    assert len(calls) == 601
    assert {c.prompt_version for c in calls} == {"judge_v1"}
    for call in calls:
        scheme = schemes[call.scheme]
        for cq in scheme.cq_order:
            assert f"**{cq}.** {scheme.cqs[cq].text}\n" in call.prompt
        assert ("Seen in" in call.prompt) is justifications
    # The prompt remains the zero-shot prompt with only the answers and judge instruction added.
    scheme = schemes[items[0]["gold_scheme"]]
    judged = render_judge(items[0], scheme, _answers(scheme))
    head, rest = judged.text.split(SECTION)
    tail = rest[rest.index("## The verdicts you may give"):]
    assert (head + tail).replace(SENTENCE + "\n\n", "") == render_zeroshot(items[0], scheme).text


def test_a_judge_rejects_answers_from_other_schemes_even_when_cq_ids_match(tmp_path, schemes):
    config = replace(_config(), schemes="v2")
    items = _items(schemes)
    _source_run(tmp_path, "src", items, schemes, model="fake")
    with pytest.raises(SchemeError, match="schemes.*v1.*v2"):
        plan(config, items, FAKE_MODELS, schemes, runs_dir=tmp_path)


def test_a_judge_rejects_a_source_with_different_scheme_contents(tmp_path, schemes):
    items = _items(schemes)
    _source_run(tmp_path, "src", items, schemes, model="fake")
    path = tmp_path / "src" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schemes_version"]["content_sha256"] = "different"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SchemeError, match="different scheme contents"):
        plan(_config(), items, FAKE_MODELS, schemes, runs_dir=tmp_path)


def test_a_judge_line_parses_to_a_terminal_id_with_its_p_label(schemes):
    from argfallacy.parse import parse_answers

    scheme = schemes["analogy"]
    shown = {v.terminal: v.shown for v in verdicts(scheme)}["weak_analogy"]
    content = json.dumps({"answer": shown, "confidence": 70, "justification": "x"})
    tokens = [('{"', -0.01, []), ("answer", -0.01, []), ('": "', -0.5, []),
              (shown, -0.2, [(shown, -0.2), ("False", -1.9)]), ('"', -0.01, []),
              (', "confidence": 70, "justification": "x"}', -0.4, [])]
    logprobs = {"content": [
        {"token": t, "logprob": lp, "top_logprobs": [{"token": a, "logprob": b} for a, b in alts]}
        for t, lp, alts in tokens]}
    line = {"run_id": "r", "item_id": "i1", "stage": JUDGE, "scheme_condition": "gold",
            "scheme": "analogy", "cq_id": "", "sample_index": 0, "model": "fake",
            "model_id": "x", "prompt_version": "judge_v1", "params": {},
            "json_schema": verdict_schema(scheme, stage=JUDGE), "cache_key": "k",
            "from_cache": False, "latency_s": 0.1, "error": None,
            "raw_response": {"content": content, "finish_reason": "stop", "logprobs": logprobs}}
    row = parse_answers([line], schemes).iloc[0]
    assert row["answer"] == "weak_analogy"
    assert row["p_label"] == pytest.approx(math.exp(-0.2))
