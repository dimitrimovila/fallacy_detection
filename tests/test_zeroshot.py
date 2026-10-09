"""The zero-shot condition: its prompt, its verdicts, and how the parser reads them back.

The prompt shows names and definitions from ``labels/fallacies.yaml``; the model answers
with one of the names, and the row of ``answers.csv`` must carry the terminal id it
stands for, and the probability of the whole name as the model generated it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from argfallacy.client import RunConfig, execute, load_models, plan, read_raw
from argfallacy.paths import REPO_ROOT
from argfallacy.prompts import ZEROSHOT, render_zeroshot, verdict_schema, verdicts
from argfallacy.schemes import SchemeError
from fakes.llm import REAL_WEIGHT, FakeBackend

ITEM = {"item_id": "0123456789abcdef", "text": "Everyone I know buys it, so it must be good."}
FAKE_MODELS = load_models(Path(__file__).parent / "fakes" / "models.yaml")


def _listed(prompt: str) -> list[str]:
    """The names the prompt lists under its verdicts, in order."""
    block = prompt.split("## The verdicts you may give")[1].split("## What to do")[0]
    return [line.split("`")[1] for line in block.splitlines() if line.startswith("* `")]


def _zeroshot_line(scheme, shown: str, logprobs=None) -> dict:
    content = json.dumps({"answer": shown, "confidence": 70, "justification": "x"})
    return {
        "run_id": "r", "item_id": "i1", "stage": ZEROSHOT, "scheme_condition": "gold",
        "scheme": scheme.scheme_id, "cq_id": "", "sample_index": 0, "model": "fake",
        "model_id": "fake-model", "prompt_version": "zeroshot_v1", "params": {},
        "json_schema": verdict_schema(scheme), "cache_key": "k", "from_cache": False,
        "latency_s": 0.1, "error": None,
        "raw_response": {"content": content, "finish_reason": "stop", "logprobs": logprobs},
    }


def test_rendering_is_deterministic(schemes):
    for scheme in schemes.values():
        first, second = render_zeroshot(ITEM, scheme), render_zeroshot(ITEM, scheme)
        assert first.text == second.text
        assert first.json_schema == second.json_schema
        assert "{{" not in first.text


def test_the_admitted_verdicts_are_the_terminals_of_the_scheme(schemes, vocab):
    terminals = vocab["terminals"]
    for scheme in schemes.values():
        rendered = render_zeroshot(ITEM, scheme)
        shown = {v.shown: v.terminal for v in verdicts(scheme)}
        enum = rendered.json_schema["properties"]["answer"]["enum"]
        assert {shown[name] for name in enum} == scheme.terminals, scheme.scheme_id
        assert _listed(rendered.text) == enum == list(rendered.answer_options)
        fallacies = [n for n in enum if terminals[shown[n]]["type"] != "good"]
        assert fallacies == sorted(fallacies, key=str.casefold)
        if scheme.has_good_argumentation:
            assert terminals[shown[enum[-1]]]["type"] == "good"
        for verdict in verdicts(scheme):
            assert verdict.definition in rendered.text


def test_ad_hominem_has_no_good_argumentation(schemes, vocab):
    rendered = render_zeroshot(ITEM, schemes["ad_hominem"])
    good = vocab["terminals"]["good_argumentation"]["label"]
    assert good not in rendered.json_schema["properties"]["answer"]["enum"]
    assert good not in _listed(rendered.text)
    assert {"Abusive", "Circumstantial", "Guilt by Association", "Tu Quoque"} <= set(
        rendered.answer_options)


def test_the_names_shown_go_back_to_the_ids(schemes):
    from argfallacy.parse import parse_answers

    for scheme in schemes.values():
        listed = verdicts(scheme)
        answers = parse_answers([_zeroshot_line(scheme, v.shown) for v in listed], schemes)
        assert list(answers["answer"]) == [v.terminal for v in listed], scheme.scheme_id
        assert answers["parse_ok"].all()

    wrong = parse_answers([_zeroshot_line(schemes["ad_hominem"], "Ad Hominem (Abusive)")],
                          schemes)
    assert wrong.iloc[0]["answer"] == "invalid"


def test_the_probability_of_the_chosen_label_spans_all_its_tokens(schemes):
    from argfallacy.parse import parse_answers

    tokens = [('{"', -0.01), ("answer", -0.01), ('": "', -0.5), ("We", -0.1), ("ak", -0.2),
              (' Analogy"', -0.3), (', "confidence": 70, "justification": "x"}', -0.4)]
    logprobs = {"content": [{"token": t, "logprob": lp, "top_logprobs": []} for t, lp in tokens]}
    row = parse_answers([_zeroshot_line(schemes["analogy"], "Weak Analogy", logprobs)],
                        schemes).iloc[0]
    assert row["answer"] == "weak_analogy"
    assert row["p_label"] == pytest.approx(math.exp(-0.6))


def test_the_probability_of_the_chosen_label_on_the_fake_backend(tmp_path, schemes):
    from argfallacy.client import ResponseCache
    from argfallacy.parse import parse_answers, summarise

    data = {"name": "zs", "models": ["fake_single", "fake"], "stages": [ZEROSHOT],
            "scheme_condition": "gold", "prompt_version": "v1", "items": {}, "generation": {}}
    config = RunConfig(raw=data, **data)
    items = [{"item_id": f"item{n}", "text": f"Argument {n}.", "gold_scheme": s}
             for n, s in enumerate(sorted(schemes))]
    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        execute(config, items, run_id="zs", runs_dir=tmp_path, backend=FakeBackend(),
                cache=cache, models=FAKE_MODELS, schemes=schemes, sleep=lambda _: None)

    answers = parse_answers(read_raw(tmp_path / "zs"), schemes)
    assert len(answers) == len(items) * (1 + 5)
    assert answers["parse_ok"].all()
    for row in answers.itertuples():
        assert row.answer in schemes[row.scheme].terminals
        assert row.p_label == pytest.approx(REAL_WEIGHT)
    summary = summarise(answers, schemes)
    assert len(summary) == len(items) * 2
    assert summary[["p_logprob", "p_verbal", "p_sample"]].isna().all().all()


def test_the_zeroshot_configurations_plan_one_prompt_per_item(schemes):
    from argfallacy.client import load_items, select_items

    try:
        frame = load_items()
    except SchemeError:
        pytest.skip("no data/items.csv")
    models = load_models()
    for name, items_expected in (("zeroshot_pilot", 50), ("zeroshot_v1", 601)):
        config = RunConfig.load(REPO_ROOT / "configs" / f"{name}.yaml")
        items = select_items(config, frame)
        assert len(items) == items_expected
        calls = plan(config, items, models, schemes)
        samples = sum(int(models[m]["samples"]) for m in config.models)
        assert len(calls) == items_expected * samples
        assert {c.stage for c in calls} == {ZEROSHOT}
