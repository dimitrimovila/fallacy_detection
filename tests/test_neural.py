"""Scientific isolation, input semantics, learning capacity and exact replay."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
import yaml

from argfallacy.aggregate.neural_inputs import NeuralInputs, build_neural_inputs
from argfallacy.labels import INVALID, label_space
from argfallacy.prompts import CANNOT_BE_DETERMINED, NOT_APPLICABLE
from argfallacy.schemes import SchemeError, load_all

torch = pytest.importorskip("torch")
from argfallacy.aggregate.neural import (  # noqa: E402
    FittedNetwork,
    NetworkConfig,
    fit_network,
    neural_out_of_fold,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def encoded_fixture():
    schemes = load_all()
    sid = sorted(schemes)[0]
    scheme = schemes[sid]
    cq = scheme.cq_order[0]
    a, b = scheme.answer_space(cq)
    labels = label_space()[:2]
    items = pd.DataFrame(
        [{"item_id": str(i), "gold_scheme": sid, "gold_fallacy": labels[i % 2]} for i in range(6)]
    )
    answers = [a, b, CANNOT_BE_DETERMINED, NOT_APPLICABLE, INVALID]
    rows = [
        {
            "item_id": str(i),
            "scheme": sid,
            "cq_id": cq,
            "model": "m",
            "stage": "stage2",
            "scheme_condition": "gold",
            "answer": answer,
        }
        for i, answer in enumerate(answers)
    ]
    manifest = {"models": {"m": {"reasoning": {"enable_thinking": False}}}}
    return schemes, items, pd.DataFrame(rows), manifest, ("m",)


def test_encoding_preserves_answer_meanings_order_and_label_independence():
    args = encoded_fixture()
    data = build_neural_inputs(*args)
    assert len(np.unique(data.x, axis=0)) == 6
    assert data.audit["soft_available"] == 0
    items = args[1].sample(frac=1, random_state=5).assign(gold_fallacy="not read")
    shuffled = build_neural_inputs(args[0], items, args[2].iloc[::-1], *args[3:])
    np.testing.assert_array_equal(data.x, shuffled.x)
    assert data.names == shuffled.names
    # Changing CQ wording makes a checkpoint incompatible even if IDs stay the same.
    sid = sorted(args[0])[0]
    scheme = args[0][sid]
    cq = scheme.cq_order[0]
    changed = dict(args[0])
    changed[sid] = replace(scheme, cqs={**scheme.cqs, cq: replace(scheme.cqs[cq], text="changed")})
    assert build_neural_inputs(changed, *args[1:]).names != data.names


def test_encoding_soft_sources_and_invalid_data():
    schemes, items, rows, manifest, entries = encoded_fixture()
    sid, cq = rows.iloc[0][["scheme", "cq_id"]]
    options = (*schemes[sid].answer_space(cq), CANNOT_BE_DETERMINED, NOT_APPLICABLE)
    for a, p in zip(options, [0.1, 0.2, 0.3, 0.4], strict=True):
        rows[f"p_logprob_{a}"] = p
    data = build_neural_inputs(schemes, items, rows, manifest, entries)
    assert data.audit["soft_available"] == 5
    soft_index = next(
        i for i, n in enumerate(data.names) if n.endswith(f"/m/logprob/soft/{options[0]}")
    )
    np.testing.assert_allclose(
        data.x[:5, soft_index : soft_index + 4], np.tile([0.1, 0.2, 0.3, 0.4], (5, 1))
    )
    verbal = {"models": {"m": {"reasoning": {"enable_thinking": True}}}}
    rows["p_verbal"] = 0.2
    data = build_neural_inputs(schemes, items, rows, verbal, entries)
    np.testing.assert_allclose(data.x[1, soft_index : soft_index + 4], [0.2, 0.8, 0, 0])
    with pytest.raises(SchemeError, match="duplicate"):
        build_neural_inputs(schemes, items, pd.concat([rows, rows]), manifest, entries)
    rows.loc[0, f"p_logprob_{options[0]}"] = -1
    with pytest.raises(SchemeError, match="invalid probability"):
        build_neural_inputs(schemes, items, rows, manifest, entries)
    rows.loc[0, f"p_logprob_{options[0]}"] = np.nan
    with pytest.raises(SchemeError, match="incomplete"):
        build_neural_inputs(schemes, items, rows, manifest, entries)
    rows.loc[0, "cq_id"] = "unknown"
    with pytest.raises(SchemeError, match="unknown neural CQ"):
        build_neural_inputs(schemes, items, rows, manifest, entries)


def xor_inputs(repeats=8):
    x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32), (repeats, 1))
    classes = label_space()[:2]
    y = [classes[int(a != b)] for a, b in x]
    data = NeuralInputs(
        tuple(f"i{i:03}" for i in range(len(x))), ("s",) * len(x), ("q1", "q2"), x, {}
    )
    return data, y


def test_nonlinear_learning_repeatability_checkpoint_and_edge_cases(tmp_path):
    data, y = xor_inputs()
    config = NetworkConfig(hidden=16, weight_decay=0.001, epochs=500)
    first = fit_network(data.x, y, data.names, config, 17)
    assert [a.verdict for a in first.predict(data)] == y
    second = fit_network(data.x, y, data.names, config, 17)
    np.testing.assert_array_equal(first.probabilities(data.x), second.probabilities(data.x))
    path = tmp_path / "fit.pt"
    first.save(path)
    loaded = FittedNetwork.load(path)
    np.testing.assert_array_equal(first.probabilities(data.x), loaded.probabilities(data.x))
    with pytest.raises(SchemeError, match="schema mismatch"):
        loaded.predict(replace(data, names=("q2", "q1")))
    single = fit_network(data.x, [y[0]] * len(y), data.names, config, 17)
    assert all(a.verdict == y[0] and a.confidence == 1 for a in single.predict(data))
    with pytest.raises(SchemeError, match="empty"):
        fit_network(data.x[:0], [], data.names, config, 17)


def test_outer_test_labels_cannot_change_selection_or_probabilities():
    data, y = xor_inputs(4)
    folds = {item: i % 2 for i, item in enumerate(data.item_ids)}
    targets = dict(zip(data.item_ids, y, strict=True))
    candidates = [NetworkConfig(hidden=h, epochs=15) for h in (4, 8)]
    before, choices_before, models = neural_out_of_fold(data, targets, folds, candidates, 17, 29, 2)
    changed = {i: (label_space()[-1] if folds[i] == 0 else y) for i, y in targets.items()}
    after, choices_after, _ = neural_out_of_fold(data, changed, folds, candidates, 17, 29, 2)
    assert [c for c in choices_before if c["fold"] == 0] == [
        c for c in choices_after if c["fold"] == 0
    ]
    assert label_space()[-1] not in models[0].classes
    for item in data.item_ids:
        if folds[item] == 0:
            assert before[item].distribution == after[item].distribution


def test_isolated_runner_preflight_metrics_and_saved_replay(tmp_path):
    from argfallacy.eval.neural import prepare, run

    schemes, items, summary, manifest, entries = encoded_fixture()
    run_dir = tmp_path / "inputs"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary.to_csv(run_dir / "summary.csv", index=False)
    items["in_test_set"] = True
    items_path = run_dir / "items.csv"
    items.to_csv(items_path, index=False)
    config = {
        "run": str(run_dir),
        "items": str(items_path),
        "entries": list(entries),
        "fold_seed": 29,
        "folds": 2,
        "inner_folds": 2,
        "seeds": [17],
        "threads": 1,
        "candidates": [asdict(NetworkConfig(hidden=4, epochs=5))],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = tmp_path / "evaluation"
    assert run(config_path, out, check_only=True)["status"] == "preflight_only"
    assert not out.exists()
    run(config_path, out)
    assert json.loads((out / "manifest.json").read_text())["status"] == "complete"
    result = pd.read_csv(out / "verdicts.csv", dtype={"item_id": str})
    assert len(result) == len(items)
    assert pd.read_csv(out / "metrics.csv").macro_f1.notna().all()
    _, _, inputs, folds, _ = prepare(config_path)
    for fold in set(folds.values()):
        saved = FittedNetwork.load(out / "checkpoints" / f"seed_17_fold_{fold}.pt")
        replay = dict(zip(inputs.item_ids, saved.predict(inputs), strict=True))
        for row in result[result.fold == fold].itertuples(index=False):
            assert replay[row.item_id].distribution == json.loads(row.distribution)
    with pytest.raises(FileExistsError):
        run(config_path, out)
    with pytest.raises(SchemeError, match="explicit new"):
        run(config_path)
    items.loc[0, "gold_fallacy"] = "unknown-label"
    items.to_csv(items_path, index=False)
    with pytest.raises(SchemeError, match="unknown-label"):
        prepare(config_path)
