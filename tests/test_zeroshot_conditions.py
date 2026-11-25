"""The zero-shot prompt without a scheme, the zero-shot prompt on the predicted scheme, and
the stage one without `none` that predicts it.

Without a scheme the model sees one menu with every terminal of the diagrams, and the
probability of the label it chose must still be read correctly where many labels share a
beginning. With the predicted scheme, an item whose prediction is the gold scheme gets no
call: the call it would get must be, byte for byte, the call of the gold run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from argfallacy.client import ResponseCache, RunConfig, execute, load_models, plan, read_raw
from argfallacy.client.run import load_scheme_predictions
from argfallacy.labels import scheme_ids
from argfallacy.parse import parse_answers, parse_run, summarise
from argfallacy.parse.tables import label_probability
from argfallacy.paths import REPO_ROOT, SCHEMES_ROOT
from argfallacy.prompts import (
    STAGE1,
    ZEROSHOT,
    all_verdicts,
    load_template,
    render_stage1,
    render_zeroshot,
    render_zeroshot_all,
    verdicts,
)
from argfallacy.schemes import SchemeError, load_all
from fakes.llm import SPLIT_REST, SPLIT_SHORT, SPLIT_WHOLE, FakeBackend

FAKE_MODELS = load_models(Path(__file__).parent / "fakes" / "models.yaml")
ITEM = {"item_id": "0123456789abcdef", "text": "Everyone I know buys it, so it must be good."}
V2 = load_all(SCHEMES_ROOT / "v2")


def config(**data) -> RunConfig:
    data = {"name": "zs", "models": ["fake_single"], "stages": [ZEROSHOT],
            "prompt_version": "v1", "items": {}, "generation": {}, **data}
    return RunConfig(raw=data, **data)


# ------------------------------------------------------------------ the single menu
def test_the_single_menu_lists_every_terminal_once(vocab):
    menu = all_verdicts(V2)
    terminals = set().union(*(s.terminals for s in V2.values()))
    assert [v.terminal for v in menu].count("good_argumentation") == 1
    assert {v.terminal for v in menu} == terminals and len(menu) == len(terminals) == 24
    assert menu[-1].terminal == "good_argumentation"
    names = [v.shown for v in menu[:-1]]
    assert names == sorted(names, key=str.casefold)
    # the label, never the display label: the ad hominem variants keep their prefix
    assert [v.shown for v in menu] == [vocab["terminals"][v.terminal]["label"] for v in menu]
    assert "Ad Hominem (Abusive)" in names and "Abusive" not in names
    # the definition of a terminal is the one every scheme menu gives it
    for scheme in V2.values():
        for verdict in verdicts(scheme):
            assert next(v for v in menu if v.terminal == verdict.terminal).definition \
                == verdict.definition


def test_the_prompt_without_a_scheme_keeps_the_rest_of_the_prompt_word_for_word():
    scheme_template, all_template = load_template(ZEROSHOT, "v1"), load_template(ZEROSHOT, "all_v1")
    # with one sentence that no longer points at a scheme
    old, new = ("the argument that follows the\nscheme.", "the main argument the text\nmakes.")
    assert all_template.split("## Text", 1)[1] == \
        scheme_template.split("## Text", 1)[1].replace(old, new)
    assert old not in all_template
    assert "{{scheme_name}}" not in all_template and "{{variables}}" not in all_template

    rendered = render_zeroshot_all(ITEM, V2)
    menu = all_verdicts(V2)
    assert rendered.scheme_id is None and rendered.prompt_version == "zeroshot_all_v1"
    assert rendered.json_schema["properties"]["answer"]["enum"] == [v.shown for v in menu]
    assert list(rendered.answer_options) == [v.shown for v in menu]
    assert "## The scheme" not in rendered.text and "{{" not in rendered.text
    for verdict in menu:
        assert f"* `{verdict.shown}`: {verdict.definition}" in rendered.text


# ------------------------------------------------------------------ p_label on the menu
HEAD = [('{"', 0.0, []), ("answer", 0.0, []), ('": "', 0.0, [])]
TAIL = [('", "confidence": 70, "justification": "x"}', 0.0, [])]


def tokens_of(label: str) -> list[str]:
    """The first two characters, then one token per word: labels that share their first
    two characters ("Ad", "Ap", "Fa", "Po") or whole words share tokens."""
    words = label[2:].split(" ")
    return [label[:2], words[0], *(" " + w for w in words[1:])] if words[0] else \
        [label[:2], *(" " + w for w in words[1:])]


def trie(labels: list[str]) -> dict:
    """Prefix tree over the token sequences, every child with a probability; a node with
    one child gives it probability 1, as the grammar of the JSON schema does."""
    root: dict = {}
    for label in labels:
        node = root
        for token in tokens_of(label):
            node = node.setdefault(token, {})
    def weigh(node: dict) -> None:
        children = sorted(node)
        total = sum(range(1, len(children) + 1))
        node["__p__"] = {c: (i + 1) / total for i, c in enumerate(children)}
        for c in children:
            weigh(node[c])
    weigh(root)
    return root


def trace(root: dict, label: str, extra: dict | None = None) -> tuple[dict, float]:
    """The logprob block of a model that writes ``label`` with the tree's probabilities,
    and the probability of the label under the tree. ``extra`` adds alternatives, by
    position, with their probability."""
    content, node, mass = [], root, 1.0
    for position, token in enumerate(tokens_of(label)):
        probabilities = node["__p__"]
        alternatives = [(t, math.log(p)) for t, p in probabilities.items()]
        alternatives += [(t, math.log(p)) for t, p in (extra or {}).get(position, [])]
        content.append((token, math.log(probabilities[token]), alternatives))
        mass *= probabilities[token]
        node = node[token]
    tokens = HEAD + content + TAIL
    return {"content": [{"token": t, "logprob": lp,
                         "top_logprobs": [{"token": a, "logprob": b} for a, b in alts]}
                        for t, lp, alts in tokens]}, mass


def test_p_label_on_the_whole_menu_is_the_probability_of_each_label():
    names = [v.shown for v in all_verdicts(V2)]
    root = trie(names)
    total = 0.0
    for name in names:
        logprobs, mass = trace(root, name)
        p, bound = label_probability(logprobs, names, name)
        assert p == pytest.approx(mass), name
        assert not bound, name
        total += p
    # every label gets its own share and nothing else: the shares make the whole mass
    assert total == pytest.approx(1.0)
    shared = {name[:2] for name in names if sum(n[:2] == name[:2] for n in names) > 1}
    assert {"Ad", "Ap", "Fa", "Po"} <= shared


def test_p_label_sums_two_ways_of_writing_the_rest_of_a_shared_label():
    names = [v.shown for v in all_verdicts(V2)]
    root = trie(names)
    # after "Ad Hominem (" the four variants part; "Ab" is another way to start Abusive
    logprobs, mass = trace(root, "Ad Hominem (Abusive)", extra={2: [(" (Ab", 0.05)]})
    p, bound = label_probability(logprobs, names, "Ad Hominem (Abusive)")
    parting = root["Ad"][" Hominem"]["__p__"][" (Abusive)"]
    assert p == pytest.approx(mass / parting * (parting + 0.05))
    assert not bound
    # an alternative that writes a shared beginning another way is a path not followed
    logprobs, mass = trace(root, "Appeal to Nature", extra={0: [("Appeal", 0.1)]})
    p, bound = label_probability(logprobs, names, "Appeal to Nature")
    assert p == pytest.approx(mass) and bound


def zeroshot_all_line(shown: str, logprobs=None) -> dict:
    rendered = render_zeroshot_all(ITEM, V2)
    content = json.dumps({"answer": shown, "confidence": 70, "justification": "x"})
    return {"run_id": "r", "item_id": "i1", "stage": ZEROSHOT, "scheme_condition": "none",
            "scheme": None, "cq_id": "", "sample_index": 0, "model": "fake",
            "model_id": "fake-model", "prompt_version": rendered.prompt_version, "params": {},
            "json_schema": rendered.json_schema, "cache_key": "k", "from_cache": False,
            "latency_s": 0.1, "error": None,
            "raw_response": {"content": content, "finish_reason": "stop",
                             "logprobs": logprobs}}


def test_the_parser_reads_the_single_menu_back_to_the_ids():
    menu = all_verdicts(V2)
    answers = parse_answers([zeroshot_all_line(v.shown) for v in menu], V2)
    assert list(answers["answer"]) == [v.terminal for v in menu]
    assert answers["parse_ok"].all()
    # the name a scheme menu shows for an ad hominem variant is not in the single menu
    assert parse_answers([zeroshot_all_line("Abusive")], V2).iloc[0]["answer"] == "invalid"


def test_the_condition_without_a_scheme_on_the_fake_backend(tmp_path):
    cfg = config(name="zs_all", models=["fake_single", "fake"], scheme_condition="none",
                 prompt_version="all_v1", schemes="v2")
    items = [{"item_id": f"item{n}", "text": f"Argument {n}.", "gold_scheme": s}
             for n, s in enumerate(sorted(V2) * 3)]
    calls = plan(cfg, items, FAKE_MODELS, V2)
    assert len(calls) == len(items) * (1 + 5)
    assert {c.scheme for c in calls} == {None}
    assert {c.prompt_version for c in calls} == {"zeroshot_all_v1"}
    # the prompt does not depend on the scheme of the item
    assert len({c.prompt for c in calls if c.item_id == "item0"}) == 1
    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        manifest = execute(cfg, items, run_id="zs_all", runs_dir=tmp_path,
                           backend=FakeBackend(split_labels=True), cache=cache,
                           models=FAKE_MODELS, schemes=V2, sleep=lambda _: None)
    assert manifest["scheme_condition"] == "none" and manifest["calls_failed"] == 0
    written = parse_run(tmp_path / "zs_all", V2)
    answers = pd.read_csv(written["answers.csv"], dtype=str, keep_default_na=False)
    assert answers["parse_ok"].eq("True").all() and answers["scheme"].eq("").all()
    shown = {v.terminal: v.shown for v in all_verdicts(V2)}
    for row in answers.itertuples():
        label = shown[row.answer]
        alone = [n for n in shown.values() if n.startswith(label[:2])] == [label]
        expected = SPLIT_SHORT + SPLIT_WHOLE if alone else SPLIT_SHORT * SPLIT_REST
        assert float(row.p_label) == pytest.approx(expected), label
    summary = summarise(parse_answers(read_raw(tmp_path / "zs_all"), V2), V2)
    assert len(summary) == len(items) * 2 and summary["p_label"].notna().all()


def test_the_condition_without_a_scheme_goes_only_with_its_prompt():
    with pytest.raises(SchemeError, match="none"):
        config(scheme_condition="none", prompt_version="v1")
    with pytest.raises(SchemeError, match="none"):
        config(scheme_condition="gold", prompt_version="all_v1")
    with pytest.raises(SchemeError, match="scheme_condition"):
        config(scheme_condition="absent")
    with pytest.raises(SchemeError, match="predictions_from"):
        config(scheme_condition="predicted", predictions_from="r")
    with pytest.raises(SchemeError, match="predictions_from"):
        config(scheme_condition="gold", predictions_from="r", gold_from="g")


# ------------------------------------------------------------------ predicted scheme
ITEMS = [{"item_id": f"item{n}", "text": f"Argument {n}.", "gold_scheme": s}
         for n, s in enumerate(["analogy", "example", "expert_opinion", "popular_opinion",
                                "slippery_slope", "ad_hominem"])]
STAGE_ONE = {"item0": "analogy", "item1": "analogy", "item2": "none",
             "item3": "popular_opinion", "item4": "invalid", "item5": "expert_opinion"}


def stage_one_run(runs: Path, answers: dict, planned: int | None = None) -> None:
    run = runs / "s1"
    run.mkdir(parents=True)
    rows = [{"item_id": "other", "stage": "stage1", "scheme_condition": "gold", "scheme": "",
             "cq_id": "", "model": "fake_single", "answer": "analogy"}]
    rows += [{"item_id": i, "stage": "stage1", "scheme_condition": "gold", "scheme": "",
              "cq_id": "", "model": "fake_single", "answer": a} for i, a in answers.items()]
    pd.DataFrame(rows).to_csv(run / "summary.csv", index=False)
    done = len(rows) if planned is None else planned
    (run / "manifest.json").write_text(json.dumps({
        "config": {"stages": ["stage1"]},
        "calls_per_model": {"fake_single": {"planned": len(rows), "executed": done,
                                            "from_cache": 0, "failed": 0}}}))


def gold_run(runs: Path) -> None:
    gold = config(name="gold", scheme_condition="gold")
    with ResponseCache(runs / "gold_cache.sqlite") as cache:
        execute(gold, ITEMS, run_id="gold", runs_dir=runs, backend=FakeBackend(),
                cache=cache, models=FAKE_MODELS, sleep=lambda _: None)
    parse_run(runs / "gold")


def predicted() -> RunConfig:
    return config(name="pred", scheme_condition="predicted", predictions_from="s1",
                  gold_from="gold")


def test_the_predicted_scheme_plans_a_call_only_where_it_differs_from_the_gold(tmp_path):
    stage_one_run(tmp_path, STAGE_ONE)
    gold_run(tmp_path)
    calls = plan(predicted(), ITEMS, FAKE_MODELS, runs_dir=tmp_path)
    # item0 and item3 are right, item2 none, item4 invalid: no call for any of them
    assert [(c.item_id, c.scheme) for c in calls] == [
        ("item1", "analogy"), ("item5", "expert_opinion")]
    schemes = load_all()
    assert calls[0].prompt == render_zeroshot(ITEMS[1], schemes["analogy"]).text

    found = load_scheme_predictions(predicted(), ITEMS, ["fake_single"], tmp_path)
    record = found.record()
    assert record["outcomes"]["fake_single"] == {
        "different": 2, "none_or_invalid": 2, "same_as_gold": 2}
    source = pd.read_csv(tmp_path / "s1" / "summary.csv", dtype=str)
    for item, row in record["rows"]["fake_single"].items():
        assert source.loc[row["summary_row"], ["item_id", "answer"]].tolist() == [
            item, row["answer"]]

    with ResponseCache(tmp_path / "pred_cache.sqlite") as cache:
        manifest = execute(predicted(), ITEMS, run_id="pred", runs_dir=tmp_path,
                           backend=FakeBackend(), cache=cache, models=FAKE_MODELS,
                           sleep=lambda _: None)
    assert manifest["calls_planned"] == 2
    assert manifest["predictions"] == record
    assert set(manifest["gold_from"]) == {"run_id", "manifest.json", "answers.csv"}
    lines = list(read_raw(tmp_path / "pred"))
    assert [(r["item_id"], r["scheme"], r["scheme_condition"]) for r in lines] == [
        ("item1", "analogy", "predicted"), ("item5", "expert_opinion", "predicted")]


def test_a_reused_call_that_is_not_the_gold_call_stops_the_plan(tmp_path):
    stage_one_run(tmp_path, STAGE_ONE)
    gold_run(tmp_path)
    path = tmp_path / "gold" / "answers.csv"
    answers = pd.read_csv(path, dtype=str, keep_default_na=False)
    answers.loc[answers.item_id == "item3", "cache_key"] = "0" * 64
    answers.to_csv(path, index=False)
    with pytest.raises(SchemeError, match="1 of the 2 calls .* item3"):
        plan(predicted(), ITEMS, FAKE_MODELS, runs_dir=tmp_path)


def test_a_gold_run_with_another_prompt_stops_the_plan(tmp_path):
    stage_one_run(tmp_path, STAGE_ONE)
    gold = config(name="gold", scheme_condition="gold")
    with ResponseCache(tmp_path / "gold_cache.sqlite") as cache:
        execute(gold, [dict(i, text=i["text"] + "!") for i in ITEMS], run_id="gold",
                runs_dir=tmp_path, backend=FakeBackend(), cache=cache, models=FAKE_MODELS,
                sleep=lambda _: None)
    parse_run(tmp_path / "gold")
    with pytest.raises(SchemeError, match="2 of the 2 calls"):
        plan(predicted(), ITEMS, FAKE_MODELS, runs_dir=tmp_path)


def test_an_incomplete_stage_one_stops_the_plan(tmp_path):
    stage_one_run(tmp_path, STAGE_ONE, planned=3)
    gold_run(tmp_path)
    with pytest.raises(SchemeError, match="not all answered"):
        plan(predicted(), ITEMS, FAKE_MODELS, runs_dir=tmp_path)


def test_a_missing_or_unknown_stage_one_answer_stops_the_plan(tmp_path):
    stage_one_run(tmp_path, {k: v for k, v in STAGE_ONE.items() if k != "item5"})
    gold_run(tmp_path)
    with pytest.raises(SchemeError, match="item5"):
        plan(predicted(), ITEMS, FAKE_MODELS, runs_dir=tmp_path)

    unknown = tmp_path / "unknown"
    stage_one_run(unknown, {**STAGE_ONE, "item5": "expert"})
    gold_run(unknown)
    with pytest.raises(SchemeError, match="expert"):
        plan(predicted(), ITEMS, FAKE_MODELS, runs_dir=unknown)


def test_the_configurations_of_the_two_conditions(tmp_path):
    from argfallacy.client import load_items, select_items

    try:
        frame = load_items()
    except SchemeError:
        pytest.skip("no data/items.csv")
    cfg = RunConfig.load(REPO_ROOT / "configs" / "zeroshot_all_v2.yaml")
    items = select_items(cfg, frame)
    calls = plan(cfg, items, load_models())
    assert len(items) == len(calls) == 601
    assert {c.model for c in calls} == {"qwen3_8_27b"}
    assert {c.prompt_version for c in calls} == {"zeroshot_all_v1"}
    pred = RunConfig.load(REPO_ROOT / "configs" / "zeroshot_pred_v2.yaml")
    assert (pred.schemes, pred.prompt_version, pred.models) == ("v2", "v1", ["qwen3_8_27b"])
    gold = RunConfig.load(REPO_ROOT / "configs" / "zeroshot_schemes_v2.yaml")
    assert (gold.schemes, gold.prompt_version, gold.generation) == (
        pred.schemes, pred.prompt_version, pred.generation)
    none = RunConfig.load(REPO_ROOT / "configs" / "zeroshot_pred_none_v2.yaml")
    assert {k: v for k, v in none.raw.items() if k not in ("name", "predictions_from")} == {
        k: v for k, v in pred.raw.items() if k not in ("name", "predictions_from")}
    stage1 = RunConfig.load(REPO_ROOT / "configs" / "stage1_v3.yaml")
    assert pred.predictions_from.endswith(f"_{stage1.name}")
    assert none.predictions_from == "20261005T193017Z_full_v2"
    calls = plan(stage1, select_items(stage1, frame), load_models())
    assert len(calls) == 601 and {c.prompt_version for c in calls} == {"stage1_v3"}


# ------------------------------------------------------------ stage one without none
def stage1_line(scheme: str) -> dict:
    rendered = render_stage1(ITEM, version="v3")
    content = json.dumps({"scheme": scheme, "confidence": 70, "justification": "x"})
    return {"run_id": "r", "item_id": "i1", "stage": STAGE1, "scheme_condition": "gold",
            "scheme": None, "cq_id": "", "sample_index": 0, "model": "fake",
            "model_id": "fake-model", "prompt_version": rendered.prompt_version, "params": {},
            "json_schema": rendered.json_schema, "cache_key": "k", "from_cache": False,
            "latency_s": 0.1, "error": None,
            "raw_response": {"content": content, "finish_reason": "stop", "logprobs": None}}


def test_stage_one_v3_is_stage_one_v2_without_none():
    v2, v3 = load_template(STAGE1, "v2"), load_template(STAGE1, "v3")
    for old, new in [("scheme, if any, the following", "scheme the following"),
                     ("### none\n\nNone of the eight schemes above is present in the text.\n\n",
                      ""),
                     ("scheme identifiers, or `none`.", "scheme identifiers.")]:
        assert v2.count(old) == 1
        v2 = v2.replace(old, new)
    assert v2 == v3
    rendered = render_stage1(ITEM, version="v3")
    assert rendered.prompt_version == "stage1_v3" and "`none`" not in rendered.text
    assert rendered.json_schema["properties"]["scheme"]["enum"] == [
        s for s in scheme_ids() if s != "none"]
    # the same catalogue of the eight schemes as stage1_v2, on the 2.0 folder it goes with
    assert render_stage1(ITEM, V2, version="v3").text == rendered.text
    assert rendered.text.split("## The schemes")[1].split("## What to do")[0] in \
        render_stage1(ITEM, version="v2").text


def test_stage_one_v3_reads_none_as_invalid_and_its_masses_over_eight(tmp_path):
    answers = parse_answers([stage1_line("analogy"), stage1_line("none")])
    assert list(answers["answer"]) == ["analogy", "invalid"]
    cfg = config(name="s1v3", stages=[STAGE1], prompt_version="v3", schemes="v2")
    items = [{"item_id": f"item{n}", "text": f"Argument {n}.", "gold_scheme": "analogy"}
             for n in range(6)]
    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        execute(cfg, items, run_id="s1v3", runs_dir=tmp_path, backend=FakeBackend(),
                cache=cache, models=FAKE_MODELS, sleep=lambda _: None)
    written = parse_run(tmp_path / "s1v3")
    answers = pd.read_csv(written["answers.csv"])
    eight = [f"p_logprob_{s}" for s in scheme_ids() if s != "none"]
    with_masses = answers[answers[eight].notna().all(axis=1)]
    assert len(with_masses) and answers["p_logprob_none"].isna().all()
    assert (with_masses[eight].sum(axis=1) - 1).abs().max() < 1e-9
    assert answers["answer"].isin([s for s in scheme_ids() if s != "none"]).all()
