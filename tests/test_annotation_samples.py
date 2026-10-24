"""The samples of the human annotation and the sheet of the direct verdict.

They are drawn from the frozen data, so these tests run on ``data/`` as it is.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from argfallacy.annotation_samples import (
    SHEET_FIELDS,
    SampleConfig,
    draw,
    eleni_items,
    pool_a,
    write_samples,
)
from argfallacy.annotations import load_annotations
from argfallacy.eval.prior import load_test_set
from argfallacy.paths import REPO_ROOT
from argfallacy.prompts import render_zeroshot
from argfallacy.schemes import SchemeError

CONFIG = REPO_ROOT / "configs" / "annotation_samples.yaml"


@pytest.fixture(scope="module")
def config():
    return SampleConfig.load(CONFIG)


@pytest.fixture(scope="module")
def data():
    return load_test_set(), load_annotations()


@pytest.fixture(scope="module")
def drawn(config, data, schemes):
    return draw(config, *data, schemes)


@pytest.fixture(scope="module")
def written(config, tmp_path_factory):
    out = tmp_path_factory.mktemp("sheets")
    write_samples(config, out)
    return out


def _sheet(out: Path) -> list[dict]:
    return [json.loads(line) for line in
            (out / "direct_verdict_sheet.jsonl").read_text(encoding="utf-8").splitlines()]


def test_pool_a_and_the_items_with_a_fine_verdict_have_the_documented_sizes(data, schemes):
    items, annotations = data
    pool = pool_a(items, annotations, schemes)
    assert len(pool) == 476
    assert len(eleni_items(items, annotations, schemes) & pool) == 89


def test_the_samples_are_disjoint(drawn):
    direct, cq_v2 = set(drawn.direct.item_id), set(drawn.cq_v2.item_id)
    assert not direct & cq_v2
    assert len(direct) == len(drawn.direct) and len(cq_v2) == len(drawn.cq_v2)


def test_the_samples_have_the_configured_sizes(config, drawn, data, schemes):
    items, annotations = data
    expected = {**config.cq_v2_per_scheme, **config.never_annotated}
    assert Counter(drawn.cq_v2.scheme) == expected
    pool = pool_a(items, annotations, schemes)
    assert set(drawn.direct.item_id) == pool - set(drawn.cq_v2.item_id)
    gold = items.set_index("item_id").gold_scheme
    annotated = set(annotations.loc[annotations.annotator == "dumitru", "item_id"])
    for scheme in config.never_annotated:
        taken = drawn.cq_v2[drawn.cq_v2.scheme == scheme].item_id
        assert not set(taken) & annotated and (gold[taken] == scheme).all()


def test_the_items_with_a_fine_verdict_are_all_in_the_direct_sample(drawn, data, schemes):
    items, annotations = data
    fine = eleni_items(items, annotations, schemes) & pool_a(items, annotations, schemes)
    assert fine <= set(drawn.direct.item_id)
    assert not fine & set(drawn.cq_v2.item_id)


def test_a_short_stratum_stops_the_draw(config, data, schemes):
    short = SampleConfig(**{**config.__dict__, "cq_v2_per_scheme":
                            {**config.cq_v2_per_scheme, "correlation_to_cause": 500}})
    with pytest.raises(SchemeError, match="correlation_to_cause"):
        draw(short, *data, schemes)


def test_the_sheet_has_only_the_allowed_fields(written):
    for row in _sheet(written):
        assert list(row) == list(SHEET_FIELDS)
        for verdict in row["verdicts"]:
            assert list(verdict) == ["id", "label", "definition"]


def test_the_sheet_carries_the_text_and_the_scheme_unchanged(written, data, schemes):
    items = data[0].set_index("item_id")
    for row in _sheet(written):
        assert row["text"] == items.text[row["item_id"]]
        assert row["scheme_name"] == schemes[items.gold_scheme[row["item_id"]]].name


def test_every_menu_is_the_one_of_the_zero_shot_prompt(written, data, schemes):
    items = data[0].set_index("item_id")
    for row in _sheet(written):
        scheme = schemes[items.gold_scheme[row["item_id"]]]
        prompt = render_zeroshot({"item_id": row["item_id"], "text": "x"}, scheme).text
        block = prompt.split("## The verdicts you may give")[1].split("## What to do")[0]
        listed = [line for line in block.strip().splitlines() if line.startswith("* `")]
        assert [f"* `{v['label']}`: {v['definition']}" for v in row["verdicts"]] == listed
        assert {v["id"] for v in row["verdicts"]} == set(scheme.terminals)


def test_positions_run_from_one_and_follow_the_round_rule(config, written, data):
    rows = _sheet(written)
    assert [r["position"] for r in rows] == list(range(1, len(rows) + 1))
    table = pd.read_csv(written / "samples.csv", dtype=str, keep_default_na=False)
    direct = table[table["sample"] == "direct"].astype({"round": int, "position": int})
    assert direct.sort_values("position").item_id.tolist() == [r["item_id"] for r in rows]
    assert direct.groupby("round").position.agg(lambda p: p.max() - p.min() + 1).equals(
        direct.groupby("round").size())
    totals = direct.scheme.value_counts()
    size = config.round_size
    for number, group in direct.groupby("round"):
        counts = group.scheme.value_counts()
        expected = {s: min(size, n - size * (number - 1)) for s, n in totals.items()
                    if n > size * (number - 1)}
        assert counts.to_dict() == expected
    assert direct["round"].max() == -(-totals.max() // size)


def test_the_same_seed_gives_the_same_bytes(config, written, tmp_path):
    write_samples(config, tmp_path)
    for name in ("direct_verdict_sheet.jsonl", "samples.csv", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (written / name).read_bytes()


def test_another_seed_gives_another_sheet(config, written, tmp_path):
    write_samples(SampleConfig(**{**config.__dict__, "seed": config.seed + 1}), tmp_path)
    assert ((tmp_path / "direct_verdict_sheet.jsonl").read_bytes()
            != (written / "direct_verdict_sheet.jsonl").read_bytes())


def test_the_manifest_hashes_the_files_written(written):
    import hashlib

    manifest = json.loads((written / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((written / name).read_bytes()).hexdigest() == digest


def test_a_folder_inside_the_repository_is_refused(config):
    target = REPO_ROOT / "scratch" / "annotation_sheets_refused"
    with pytest.raises(SchemeError, match="inside the repository"):
        write_samples(config, target)
    assert not target.exists()
