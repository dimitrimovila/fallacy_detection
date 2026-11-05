import csv
import re
from collections import Counter

import pytest
import yaml

from act_pilot.config import ConfigError, load_config, load_manual_rewrites
from act_pilot.data import (
    DataError,
    LabeledItem,
    boundary_items,
    inner_split,
    load_folds,
    load_items,
)
from conftest import HOMINEM, PILOT_CONFIG, POPULAR, make_config


def test_pilot_config_loads(monkeypatch, tmp_path):
    # an absolute folder on every system ("/data/pilot" has no drive on Windows)
    monkeypatch.setenv("ACT_PILOT_DATA", str(tmp_path))
    config = load_config(PILOT_CONFIG)
    assert config.items == (tmp_path / "items.csv").resolve()
    assert config.folds == (tmp_path / "folds.csv").resolve()
    assert set(config.boundaries) == {POPULAR, HOMINEM}
    b1, b2 = config.boundaries[POPULAR], config.boundaries[HOMINEM]
    assert (b1.scheme, b1.positive, b1.negative) == (
        "popular_opinion", "good_argumentation", "ad_populum")
    assert (b2.scheme, b2.positive, b2.negative) == (
        "ad_hominem", "poisoning_the_well", "ad_hominem_abusive")
    # the seed questions are read from the scheme files of version 2.0
    assert (b1.seed_cq, b2.seed_cq) == ("CQ1", "CQ6")
    assert b1.schemes_version == b2.schemes_version == "2.0"
    assert b1.seed_question.startswith("Is the reference group in a position to know whether A")
    assert b1.seed_question.endswith("a reasonable basis for accepting A?")
    assert b2.seed_question.startswith("Is the attack made before the other person has presented")
    assert b2.seed_question.endswith("to make the audience distrust in advance anything they say?")
    assert config.act.steps == 10 and config.act.examples_per_group == 20
    assert config.act.seeds == (0, 1, 2) and config.llm.temperature == 0.0
    assert len(config.controls) == 4
    assert config.evaluation.undetermined_policy == "negative"
    assert config.evaluation.sensitivity_policies == ("positive",)
    manual = load_manual_rewrites(config.manual_rewrites, config.boundaries)
    assert len(manual[POPULAR]) == 3 and len(manual[HOMINEM]) == 3


def test_unset_variable_is_an_error(monkeypatch):
    monkeypatch.delenv("ACT_PILOT_DATA", raising=False)
    with pytest.raises(ConfigError, match="unset environment variable"):
        load_config(PILOT_CONFIG)


def test_unknown_key_is_refused(tmp_path):
    path = make_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["act"]["stpes"] = 3
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="stpes"):
        load_config(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(schemes_version="1.1"), "has version '2.0'"),
        (lambda raw: raw["boundaries"][HOMINEM].update(seed_cq="CQ9"),
         "no critical question 'CQ9'"),
        # CQ1 of ad_hominem is answered positive or negative
        (lambda raw: raw["boundaries"][HOMINEM].update(seed_cq="CQ1"),
         "not answered yes or no"),
        # the "yes" branch of CQ5 ends in ad_fidentia
        (lambda raw: raw["boundaries"][HOMINEM].update(seed_cq="CQ5"),
         "'yes' branch of ad_hominem CQ5"),
        # inverted polarity
        (lambda raw: raw["boundaries"][POPULAR].update(
            positive="ad_populum", negative="good_argumentation"), "'yes' branch"),
    ],
)
def test_seed_cq_is_checked_against_the_scheme_file(tmp_path, change, message):
    path = make_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    change(raw)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(path)


def test_nonzero_temperature_is_refused(tmp_path):
    path = make_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["llm"]["temperature"] = 0.7
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="temperature"):
        load_config(path)


@pytest.mark.parametrize("policies", [["negative"], ["positive", "positive"], ["both"]])
def test_sensitivity_policies_are_checked(tmp_path, policies):
    path = make_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["evaluation"]["sensitivity_policies"] = policies
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="sensitivity_policies"):
        load_config(path)


def test_items_with_column_mapping(tmp_path):
    path = tmp_path / "items.csv"
    with path.open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["item_id", "text", "gold_scheme", "gold_fallacy", "source"])
        w.writerow(["a", "Everyone does it.", "Popular_Opinion", "AD_POPULUM", "logic"])
    items = load_items(path, {"item_id": "item_id", "text": "text", "scheme": "gold_scheme",
                              "gold": "gold_fallacy", "source": "source"})
    assert items[0].scheme == "popular_opinion" and items[0].gold == "ad_populum"
    with pytest.raises(DataError, match="missing columns"):
        load_items(path, {"item_id": "item_id", "text": "text", "scheme": "scheme",
                          "gold": "gold", "source": "source"})


def test_folds_are_checked(tmp_path):
    path = tmp_path / "folds.csv"
    path.write_text("item_id,outer_fold\na,0\nb,7\n")
    with pytest.raises(DataError, match="outside"):
        load_folds(path, 5)


def test_boundary_items_and_missing_folds(tmp_path):
    config = load_config(make_config(tmp_path))
    items = load_items(config.items, config.columns)
    folds = load_folds(config.folds, 5)
    b1 = boundary_items(items, folds, config.boundaries[POPULAR])
    assert Counter(i.label for i in b1) == Counter({1: 22, 0: 24})
    assert all(i.gold in ("good_argumentation", "ad_populum") for i in b1)
    folds.pop(b1[0].item_id)
    with pytest.raises(DataError, match="no outer fold"):
        boundary_items(items, folds, config.boundaries[POPULAR])


def _items(n_pos, n_neg, sources=("S",)):
    out = []
    for k in range(n_pos + n_neg):
        out.append(LabeledItem(f"i{k:03d}", f"text {k}", sources[k % len(sources)], "g",
                               int(k < n_pos), 0))
    return out


def test_inner_split_is_stratified_and_deterministic():
    items = _items(40, 60, sources=("A", "B"))
    tr, va = inner_split(items, 0.2, seed=1)
    assert len(tr) + len(va) == 100 and not {i.item_id for i in tr} & {i.item_id for i in va}
    labels = Counter(i.label for i in va)
    assert labels[1] == 8 and labels[0] == 12
    assert inner_split(list(reversed(items)), 0.2, seed=1)[1] == va or {
        i.item_id for i in inner_split(list(reversed(items)), 0.2, seed=1)[1]
    } == {i.item_id for i in va}
    assert {i.item_id for i in inner_split(items, 0.2, seed=2)[1]} != {i.item_id for i in va}


def test_inner_split_keeps_both_classes_in_validation():
    items = _items(2, 30)
    _, va = inner_split(items, 0.2, seed=0)
    assert {i.label for i in va} == {0, 1}
