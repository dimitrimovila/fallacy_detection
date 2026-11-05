from act_pilot.act import run_act
from act_pilot.config import load_config
from act_pilot.costs import estimate_boundary
from act_pilot.llm import CallLog, DiskCache, LLMClient
from act_pilot.pipeline import load_inputs
from act_pilot.splits import fold_splits
from conftest import HOMINEM, POPULAR, make_config
from fakes import FakeAnswerer, FakeOptimizer


def test_estimate_formula(tmp_path):
    config = load_config(make_config(tmp_path, steps=4, seeds=(0, 1, 2)))
    inputs = load_inputs(config)
    items = inputs.items[POPULAR]
    cost = estimate_boundary(config, POPULAR, items, n_manual=3)
    splits = fold_splits(items, config.act)
    n_tv = sum(len(s.inner_train) + len(s.inner_val) for s in splits)
    assert cost.act_answers_without_cache == 3 * 5 * n_tv
    assert cost.act_answers_with_cache == 3 * 4 * n_tv
    assert cost.eval_answers_with_cache == (1 + 3 + 4) * len(items) + 3 * len(items)
    assert cost.optimizer_min == 5 * 3 * 4 * 2
    assert cost.optimizer_max == 5 * 3 * 4 * (1 + config.act.max_rewrite_attempts)
    assert [f["test"] for f in cost.per_fold] == [len(s.test) for s in splits]


def test_one_run_costs_what_the_estimate_says(tmp_path):
    config = load_config(make_config(tmp_path, steps=3, seeds=(0,)))
    items = load_inputs(config).items[HOMINEM]
    split = fold_splits(items, config.act)[0]
    answerer, optimizer = FakeAnswerer(), FakeOptimizer()
    cache, log = DiskCache(tmp_path / "c"), CallLog(tmp_path / "l.jsonl")
    a = LLMClient(config.llm.answerer, answerer, cache, log, max_workers=1)
    o = LLMClient(config.llm.optimizer, optimizer, cache, log, max_workers=1)
    run_act(config.boundaries[HOMINEM], split.inner_train, split.inner_val, a, o, config.act,
            seed=0, outer_fold=split.outer_fold)
    n_tv = len(split.inner_train) + len(split.inner_val)
    assert len(answerer.calls) == (config.act.steps + 1) * n_tv
    assert len(optimizer.calls) == 2 * config.act.steps
