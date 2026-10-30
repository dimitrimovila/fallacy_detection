"""Offline end-to-end reproduction and report boundaries on invented inputs."""

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest
import yaml

from argfallacy.cli import main
from argfallacy.eval.analysis import AnalysisConfig, analyze
from argfallacy.eval.analysis_report import PREVIEW_ROWS, _table
from argfallacy.eval.html_report import write_summary
from argfallacy.schemes import SchemeError
from test_analysis import saved as saved_fixture
from test_analysis_cq_human import notes_for
from test_analysis_dq_conformal import dq_saved as dq_fixture

saved = saved_fixture
dq_saved = dq_fixture


def test_report_preview_preserves_full_csv_and_validates_hidden_rows(tmp_path):
    file = tmp_path / "example.csv"
    content = "value\n" + "\n".join(str(i) for i in range(PREVIEW_ROWS + 5)) + "\n"
    file.write_text(content, encoding="utf-8")
    contract = dict(
        columns=["value"],
        rows=PREVIEW_ROWS + 5,
        numeric_columns=["value"],
        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
    )
    manifest = dict(tables={file.name: contract}, parts={"example": {"status": "computed"}})
    html = _table(tmp_path, file.name, manifest, {})
    payload = html.split('<script type="application/json">')[1].split("</script>")[0]
    assert len(json.loads(payload)["rows"]) == PREVIEW_ROWS
    assert f'data-rows="{PREVIEW_ROWS + 5}"' in html
    assert "Search covers this preview only" in html
    assert 'href="example.csv"' in html
    assert file.read_text(encoding="utf-8") == content
    file.write_text(content.rsplit("\n", 2)[0] + "\ninvalid\n", encoding="utf-8")
    contract["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    with pytest.raises(SchemeError, match="invalid numeric"):
        _table(tmp_path, file.name, manifest, {})


def digests(folder):
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file() and "out" not in p.parts
    }


def invented_optional(path, raw, items, human=False, conformal=False):
    from argfallacy.aggregate.interface import decide
    from argfallacy.eval.analysis_conformal import calculate
    from argfallacy.eval.analysis_inputs import Inputs, read_evaluation, scheme_registry
    from argfallacy.schemes import load_all

    notes_for(items, load_all()).to_csv(path.parent / "frozen.csv", index=False)
    raw["annotations"] = "frozen.csv"
    if human:
        own = items.iloc[:27]
        pd.DataFrame(
            dict(
                item_id=own.item_id,
                sample="direct",
                scheme=own.gold_scheme,
                round=1,
                position=range(1, len(own) + 1),
            )
        ).to_csv(path.parent / "invented_sample.csv", index=False)
        pd.DataFrame(
            dict(
                item_id=own.item_id,
                position=range(1, len(own) + 1),
                verdict="good_argumentation",
                answered_at="2026-10-28T10:00:00Z",
                revision=2,
            )
        ).to_csv(path.parent / "invented_final.csv", index=False)
        raw["direct_verdict"] = dict(answers="invented_final.csv", samples="invented_sample.csv")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    if conformal:
        file = path.parent / "second" / "verdicts.csv"
        frame = pd.read_csv(file, keep_default_na=False)
        frame["confidence"] = frame.confidence.astype(object)
        aggs = {i: decide({"good_argumentation": 0.75}, 0.25, "invented") for i in items.item_id}
        own = frame.system.eq("A2")
        frame.loc[own, "verdict"] = "good_argumentation"
        frame.loc[own, "confidence"] = 0.75
        frame["distribution"] = ""
        frame.loc[own, "distribution"] = json.dumps({"good_argumentation": 0.75})
        frame.to_csv(file, index=False)
        config, inputs = AnalysisConfig.load(path), Inputs()
        registry = scheme_registry(inputs)
        e = read_evaluation("second", config.evaluations["second"], config, items, registry, inputs)
        e.manifest["conformal_alpha"] = 0.1
        (e.path / "manifest.json").write_text(json.dumps(e.manifest), encoding="utf-8")
        rows, summaries = calculate(e, e.systems["A2"], items, registry, aggs)
        pd.DataFrame(summaries).to_csv(e.path / "conformal.csv", index=False, float_format="%.4f")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.targets, self.files = set(), set(), set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "id":
                assert value not in self.ids
                self.ids.add(value)
            if key == "href":
                assert not value.startswith(("http:", "https:", "//"))
                if value.startswith("#"):
                    self.targets.add(value[1:])
                else:
                    self.files.add(value)


@pytest.mark.parametrize("mode", ["absent", "dq_only", "all"])
def test_offline_end_to_end(saved, dq_saved, mode, monkeypatch):
    fixture = saved if mode == "absent" else dq_saved
    path, raw, items = fixture
    # Fixtures share one root; the disabled case must explicitly opt out.
    if mode == "absent":
        raw["dq"] = None
    if mode == "all":
        invented_optional(path, raw, items, human=True, conformal=True)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    before = digests(path.parent)
    first = analyze(AnalysisConfig.load(path))
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["report_format"] == "thesis_analyses_v1"
    stats = {p.name: p.read_bytes() for p in first.glob("*.csv")}
    # Repeat and change the batch boundary: all saved scalar estimates and intervals agree.
    raw["bootstrap_batch_size"] = 19
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    second = analyze(AnalysisConfig.load(path), path.parent / "repeat")
    for name, content in stats.items():
        assert (second / name).read_bytes() == content, name
    again = analyze(AnalysisConfig.load(path), path.parent / "repeat")
    assert all((again / name).read_bytes() == content for name, content in stats.items())
    assert manifest["parameters"]["exploratory_augrc_pairs"] == [
        ["B2", "zeroshot:qwen3_8_27b"],
        ["B4", "A2.pruned"],
        ["B1", "A2"],
        ["A5", "zeroshot:qwen3_8_27b"],
    ]
    # No reader can reopen source runs, data, code, or external humans while rendering.
    original = Path.open

    def report_only(self, *args, **kwargs):
        assert self.resolve().parent == first.resolve(), self
        assert self.name in {*stats, "manifest.json", "summary.html"}, self
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", report_only)
        patch.setattr("argfallacy.eval.analysis.analyze", lambda *a, **k: pytest.fail("analyze"))
        assert main(["report", "summary", str(first), "--format", "analysis"]) == 0
        html = write_summary(first).read_text(encoding="utf-8")
        assert write_summary(first).read_text(encoding="utf-8") == html
        assert all((first / name).read_bytes() == content for name, content in stats.items())
    links = Links()
    links.feed(html)
    assert links.targets <= links.ids
    assert set(stats) <= links.files
    assert all((first / name).is_file() for name in links.files)
    for name, contract in manifest["tables"].items():
        assert f'data-file="{name}"' in html
        assert f'data-rows="{contract["rows"]}"' in html
    assert "Stage-one scheme recognition" in html
    assert "not predicted-scheme verdict evaluation" in html
    assert "No measurable difference is not equivalence" in html
    assert "gold support &lt;20" in html and "gold support ≥10" in html
    if mode != "absent":
        assert "DQ:ref@2.0" in html and "no measurable difference" in html
        assert "H_DQ" in html and "m=1" in html and "m=4" in html and "m=5" in html
        assert manifest["parts"]["dq"]["status"] == "computed"
    if mode == "all":
        assert manifest["parts"]["direct_verdict"]["status"] == "computed"
        assert manifest["parts"]["conformal_items"]["rows"] == len(items)
    else:
        assert manifest["parts"]["direct_verdict"]["status"] == "pending"
    after = digests(path.parent)
    assert all(after[name] == digest for name, digest in before.items() if name != path.name), (
        "protected sources changed"
    )


@pytest.mark.parametrize("damage", ["column", "numeric", "missing", "count", "empty"])
def test_malformed_computed_tables_fail_clearly(saved, damage):
    path, _, _ = saved
    out = analyze(AnalysisConfig.load(path))
    file = out / "main_table.csv"
    frame = pd.read_csv(file)
    if damage == "column":
        frame = frame.drop(columns="n_items")
    elif damage == "numeric":
        frame["accuracy"] = "broken"
    elif damage == "count":
        frame = frame.iloc[:-1]
    elif damage == "empty":
        frame = frame.iloc[:0]
    if damage == "missing":
        file.unlink()
    else:
        frame.to_csv(file, index=False)
    with pytest.raises(SchemeError, match="main_table.csv"):
        write_summary(out)


def test_pending_missing_optional_and_output_alias_guards(saved):
    path, _, _ = saved
    out = analyze(AnalysisConfig.load(path))
    (out / "human_direct.csv").unlink()
    assert "pending" in write_summary(out).read_text()
    # A report symlink/hard link must never overwrite a table or protected source.
    target = out / "summary.html"
    target.unlink()
    target.hardlink_to(path.parent / "first" / "verdicts.csv")
    with pytest.raises(SchemeError, match="overlap|shared file"):
        write_summary(out)
    with pytest.raises(SchemeError, match="optional|analysis"):
        write_summary(out, judge_audit="forbidden")


def test_analysis_scorer_reuses_encodings_without_retaining_batch_workspaces(monkeypatch):
    # The module is imported explicitly because the package also exports evaluate().
    import importlib

    from argfallacy.eval.analysis_bootstrap import Bootstrap, score_population

    module = importlib.import_module("argfallacy.eval.evaluate")
    original, reads = module.load_vocabulary, []

    def counted(*args, **kwargs):
        reads.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "load_vocabulary", counted)
    boot = Bootstrap(17, 31, 7)
    gold = ["good_argumentation", "weak_analogy"] * 3
    import numpy as np

    mask = np.ones(len(gold), bool)
    correct, a = score_population(gold, gold, mask, boot)
    missing, b = score_population(gold, [None] * len(gold), mask, boot)
    again, c = score_population(gold, gold, mask, boot)
    assert correct == again and correct["accuracy"] == 1 and missing["accuracy"] == 0
    np.testing.assert_equal(a["macro_f1"], c["macro_f1"])
    assert np.all(b["accuracy"] == 0) and len(reads) == 1
    assert len(boot._scorers) == 1 and boot.scorer(gold).weights.shape == (0, len(gold))
    assert boot.scorer(gold).support.shape[0] == 0


def test_saved_conformal_summaries_remain_available_without_exact_sets(saved):
    path, raw, items = saved
    invented_optional(path, raw, items, conformal=True)
    file = path.parent / "second" / "verdicts.csv"
    frame = pd.read_csv(file, keep_default_na=False).drop(columns="distribution")
    frame.to_csv(file, index=False)
    out = analyze(AnalysisConfig.load(path))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["parts"]["source_conformal"]["status"] == "computed"
    assert manifest["parts"]["conformal_items"]["status"] == "pending"
    html = write_summary(out).read_text(encoding="utf-8")
    assert "saved aggregate only" in html and "exact item exports independent" in html


def test_replay_rejects_saved_confidence_disagreement_before_export(saved, monkeypatch):
    from types import SimpleNamespace

    from argfallacy.aggregate.b1 import C_GRID

    for name in (
        "CUDA_VISIBLE_DEVICES",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "" if name == "CUDA_VISIBLE_DEVICES" else "1")
    from serving.postprocess import replay

    path, raw, items = saved
    invented_optional(path, raw, items)
    file = path.parent / "second" / "manifest.json"
    source = json.loads(file.read_text())
    source["grids"] = {"B1 stacking C": list(C_GRID), "B2 C": list(C_GRID)}
    file.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr("argfallacy.aggregate.b1_inputs.build_dataset", lambda *a: object())
    monkeypatch.setattr("argfallacy.aggregate.crossfit.B1CrossFitter", lambda *a: object())
    result = SimpleNamespace(
        aggregations={i: SimpleNamespace(verdict=None, confidence=0.999) for i in items.item_id}
    )
    monkeypatch.setattr("argfallacy.aggregate.routing.b2_out_of_fold", lambda *a: result)
    monkeypatch.setattr(
        "argfallacy.eval.analysis_routing.export_routing_records",
        lambda *a: pytest.fail("unverified replay must not export"),
    )
    out = path.parent / "replay"
    config = AnalysisConfig.load(path)
    ready = replay(config, "second", out, True)
    assert ready["resource_status"].startswith("fitted replay not profiled")
    with pytest.raises(SchemeError, match="confidence disagreement"):
        replay(config, "second", out, False)
    assert not out.exists()


def test_full_size_shared_draws_and_scalar_intervals_are_batch_invariant():
    import numpy as np

    from argfallacy.eval.analysis_bootstrap import Bootstrap, percentile, score_population

    rng = np.random.default_rng(20261019)
    labels = np.array(["good_argumentation", "weak_analogy", "cherry_picking"], object)
    gold = labels[rng.integers(0, len(labels), 601)].tolist()
    predictions = labels[rng.integers(0, len(labels), 601)].tolist()
    predictions[::17] = [None] * len(predictions[::17])
    for generator in (1, 2):
        previous = None
        for batch in (64, 128):
            boot = Bootstrap(20261019, 10000, batch)
            digest = hashlib.sha256()
            for indices in boot.indices(601, generator):
                digest.update(indices.tobytes())
            estimates = []
            # Full and origin-like views have varying intervals; the sparse view
            # additionally exercises omitted whole-test replicates.
            for mask in (np.ones(601, bool), np.arange(601) < 260, np.arange(601) == 5):
                point, spread = score_population(gold, predictions, mask, boot, generator)
                intervals = {
                    k: [percentile(v, level) for level in (0.95, 0.9875, 0.99)]
                    for k, v in spread.items()
                }
                estimates.append((point, intervals))
            current = digest.hexdigest(), estimates
            if previous is not None:
                assert current == previous
            previous = current


def test_fitted_replay_preserves_missing_numeric_signals_without_editing_csv(tmp_path):
    import numpy as np
    from serving.postprocess import fitting_summary

    from argfallacy.aggregate.b1_inputs import build_dataset
    from argfallacy.eval.analysis_inputs import Inputs
    from test_b1 import example_data

    original, _, summary, manifest = example_data(4, full_scheme=True)
    summary.loc[summary.model.eq("e1"), [c for c in summary if c.startswith("p_")]] = np.nan
    expected = build_dataset(
        original.schemes, original.items, summary, manifest, original.reference, original.routing
    )
    path = tmp_path / "summary.csv"
    summary.to_csv(path, index=False)
    before = path.read_bytes()
    inputs = Inputs()
    native = fitting_summary(tmp_path, inputs)
    observed = build_dataset(
        original.schemes, original.items, native, manifest, original.reference, original.routing
    )
    for name in ("splits", "base_splits", "arc_mass", "outside", "x"):
        np.testing.assert_allclose(
            getattr(observed, name), getattr(expected, name), atol=1e-14, rtol=0, equal_nan=True
        )
    assert np.isnan(observed.splits).any()
    assert path.read_bytes() == before
    inputs.verify()
