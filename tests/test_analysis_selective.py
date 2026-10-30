"""Item-draw reranking, attainable thresholds and partially observed attributes."""

import json

import numpy as np
import pandas as pd
import pytest

from argfallacy.aggregate.interface import Aggregation
from argfallacy.eval.analysis_bootstrap import Bootstrap, percentile
from argfallacy.eval.analysis_inputs import SavedEvaluation, SavedSystem
from argfallacy.eval.analysis_selective import (
    AUGRC_PAIRS,
    composition_rows,
    difficulty_correlation,
    difficulty_ratings,
    selective_tables,
    spearman,
    terminal_margin,
)
from argfallacy.eval.uncertainty import (
    _cut,
    abstention,
    effective_thresholds,
    saved_ranking,
    selective_bootstrap,
)
from argfallacy.paths import REPO_ROOT
from argfallacy.schemes import SchemeError, load_all

GOOD, WEAK = "good_argumentation", "weak_analogy"


def test_order_and_every_bootstrap_replicate_match_the_original_p11():
    ids = np.array(["d", "c", "b", "a", "e"])
    verdicts = np.array([None, GOOD, GOOD, WEAK, GOOD], object)
    confidence = np.array([1.0, np.nan, 0.7, 0.7, 0.0])
    wrong = np.array([True, False, False, True, True])
    assert ids[saved_ranking(ids, verdicts, confidence)].tolist() == ["a", "b", "e", "c", "d"]
    aggregations = {
        item: Aggregation(
            verdicts[k], None if np.isnan(confidence[k]) else confidence[k], {}, 0, ""
        )
        for k, item in enumerate(ids)
    }
    expected = None
    for batch in (1, 7, 83):
        boot = Bootstrap(17, 83, batch)
        draws = np.concatenate(list(boot.indices(len(ids))))
        legacy, curve, _ = abstention("saved", ids, wrong, aggregations, draws)
        observed, spread, order = selective_bootstrap(
            ids, wrong, verdicts, confidence, np.ones(5, bool), boot
        )
        for metric in observed:
            assert observed[metric] == pytest.approx(legacy[metric])
            low, high, _ = percentile(spread[metric])
            assert (low, high) == pytest.approx((legacy[f"{metric}_low"], legacy[f"{metric}_high"]))
        if expected is not None:
            for key in spread:
                np.testing.assert_equal(spread[key], expected[key])
        expected = spread
        # A partial population and an origin filter must be applied before reranking.
        mask = np.array([True, False, True, True, False])
        _, filtered, _ = selective_bootstrap(ids, wrong, verdicts, confidence, mask, boot)
        rank = {position: k for k, position in enumerate(order)}
        for replicate, draw in enumerate(draws):
            chosen = sorted(draw[mask[draw]], key=rank.get)
            errors = np.cumsum(wrong[chosen])
            n = len(chosen)
            assert (
                filtered["augrc"][replicate] == pytest.approx(errors.mean() / n)
                if n
                else np.isnan(filtered["augrc"][replicate])
            )
            if n:
                cut = _cut(0.9, n)
                assert filtered["selective_accuracy_90"][replicate] == pytest.approx(
                    1 - errors[cut - 1] / cut
                )
    assert curve.generalized_risk.iloc[-1] == 3 / 5
    assert _cut(0.95, 10) == 10  # half-up, not banker's rounding
    assert _cut(0.9, 5) == 5


def test_effective_thresholds_keep_ties_empty_and_missing_operational_values():
    ids = ["a", "b", "c", "d", "e", "f"]
    p = np.array([0.7000000000000001, 0.7, 0.7, np.nan, 0.0, 0.9])
    verdicts = np.array([GOOD, GOOD, WEAK, GOOD, GOOD, None], object)
    wrong = np.array([0, 0, 1, 0, 0, 1], bool)
    rows = effective_thresholds(ids, wrong, verdicts, p, np.ones(6, bool))
    assert [r["threshold"] for r in rows] == [np.inf, 0.9, p[0], 0.7, 0.0]
    assert [r["kept"] for r in rows] == [0, 0, 1, 3, 4]
    assert rows[0]["generalized_risk"] == 0
    assert np.isnan(rows[0]["selective_risk"])
    assert rows[-1]["coverage"] == 4 / 6
    assert rows[-1]["errors"] == 1
    assert rows[-1]["selective_risk"] == 0.25
    assert rows[-1]["generalized_risk"] == 1 / 6


def test_optional_composition_attributes_are_missing_not_negative():
    items = pd.DataFrame(
        {
            "gold_scheme": ["analogy"] * 5,
            "source": ["LOGIC"] * 5,
            "gold_fallacy": [GOOD, GOOD, WEAK, WEAK, WEAK],
        }
    )
    attrs = {"human disagreement": [True, pd.NA, False, None, True]}
    rows = pd.DataFrame(composition_rows(items, np.arange(5), attrs))
    at90 = rows[(rows.requested_coverage == 0.9) & (rows.kind == "human disagreement")]
    assert set(at90.actual_cut) == {5}
    assert set(at90.missing_all) == {2}
    assert set(at90.known_all) == {3}
    assert set(at90.missing_kept) == {2}
    at70 = rows[(rows.requested_coverage == 0.7) & (rows.kind == "human disagreement")]
    yes = at70[at70.value.eq(True)].iloc[0]
    assert yes["count"] == 1
    assert yes.share_rejected == 1
    assert yes.rejection_rate == 0.5
    no = at70[at70.value.eq(False)].iloc[0]
    assert no["count"] == 0 and no.group_items == 1
    assert set(at70.actual_cut) == {4}


def test_difficulty_validation_missingness_and_undefined_replicates():
    items = pd.DataFrame({"item_id": list("abcd")})
    notes = pd.DataFrame(
        {
            "item_id": ["a", "a", "b", "c", "d"],
            "annotator": "enrico",
            "field": "difficulty",
            "value": ["1", "1", "5", "", "3"],
        }
    )
    values = difficulty_ratings(notes, items)
    np.testing.assert_equal(values, [1, 5, np.nan, 3])
    for invalid in ("0", "6", "2.5", "low", "nan"):
        bad = notes.copy()
        bad.loc[0, "value"] = invalid
        with pytest.raises(SchemeError, match="difficulty"):
            difficulty_ratings(bad, items)
    bad = notes.copy()
    bad.loc[0, "value"] = "2"
    with pytest.raises(SchemeError, match="conflicting difficulty"):
        difficulty_ratings(bad, items)
    confidence = np.array([0.9, 0.1, 0.5, np.nan])
    boot = Bootstrap(9, 91, 7)
    row = difficulty_correlation(values, confidence, np.ones(4, bool), boot)
    assert row["rated_eligible_items"] == 2
    assert row["spearman"] == pytest.approx(1)
    assert 0 < row["spearman_omitted"] < 91
    for batch in (1, 91):
        assert (
            difficulty_correlation(values, confidence, np.ones(4, bool), Bootstrap(9, 91, batch))
            == row
        )
    assert np.isnan(spearman([1, 1], [0.1, 0.9]))
    assert np.isnan(spearman([1, 2], [0.5, 0.5]))
    assert np.isnan(spearman([1], [0.5]))
    constant = difficulty_correlation(np.ones(4), confidence, np.ones(4, bool), boot)
    assert constant["spearman_undefined"] and constant["spearman_omitted"] == 91


def test_exact_four_pairs_origins_partial_denominators_and_reference_points(tmp_path):
    n = 10
    items = pd.DataFrame(
        {
            "item_id": [str(k) for k in range(n)],
            "gold_scheme": "analogy",
            "source": ["LOGIC"] * 6 + ["NLAS"] * 4,
            "gold_fallacy": GOOD,
        }
    )
    systems = {}
    names = sorted({s for pair in AUGRC_PAIRS for s in pair}) + [
        "A0:ref",
        "B3",
        "B3.pooled",
        "B2.variant",
        "B4.nohistory",
        "A3",
        "A4",
        "A5.answers_only",
        "zeroshot:other",
    ]
    for index, name in enumerate(names):
        eligible = np.arange(n) < 8 if name == "B4" else np.ones(n, bool)
        rows = pd.DataFrame(
            {
                "verdict": [GOOD if (k + index) % 3 else WEAK for k in range(n)],
                "confidence": [np.nan] * n
                if name == "A0:ref"
                else np.linspace(0, 1, n)[:: (-1 if index % 2 else 1)],
            },
            index=items.item_id,
        )
        if name == "B4":
            rows.loc[~eligible, :] = np.nan
        systems[name] = SavedSystem(name, "1.1", rows, eligible, True)
    evaluation = SavedEvaluation("second", 2, tmp_path, {}, systems)
    tables = selective_tables(
        [evaluation],
        items,
        Bootstrap(3, 37, 4),
        np.full(n, np.nan),
        {"1.1": REPO_ROOT / "schemes/v1"},
    )
    differences = tables["abstention_differences.csv"]
    assert set(zip(differences.system, differences.reference, strict=True)) == set(AUGRC_PAIRS)
    assert len(differences) == 12
    assert set(differences.kind) == {"exploratory"}
    partial = differences[differences.system == "B4"].set_index("origin")
    assert partial.n_items.to_dict() == {"total": 8, "fallacy corpora": 6, "scheme corpora": 2}
    draws = np.concatenate(list(Bootstrap(3, 37, 4).indices(n)))
    left, right = systems["B4"], systems["A2.pruned"]
    common = left.eligible & right.eligible
    deltas = []
    for draw in draws:
        chosen = draw[common[draw]]
        values = []
        for side in (left, right):
            rank = saved_ranking(items.item_id, side.rows.verdict, side.rows.confidence)
            positions = {position: k for k, position in enumerate(rank)}
            selected = sorted(chosen, key=positions.get)
            errors = np.cumsum(side.rows.verdict.to_numpy()[selected] != GOOD)
            values.append(errors.mean() / len(selected) if len(selected) else np.nan)
        deltas.append(values[0] - values[1])
    low, high, used = percentile(deltas)
    assert partial.loc["total", "augrc_low"] == pytest.approx(low)
    assert partial.loc["total", "augrc_high"] == pytest.approx(high)
    assert partial.loc["total", "augrc_replicates"] == used
    main = tables["abstention.csv"]
    assert set(main.system) == set(names)
    assert not main[main.system == "A0:ref"].ranking_available.any()
    assert "A0:ref" not in set(tables["abstention_curves.csv"].system)
    assert main[main.system != "A0:ref"].margin_status.str.startswith("pending").all()
    assert set(tables["difficulty.csv"].status) == {"pending"}
    b4 = main[(main.system == "B4") & (main.origin == "total")].iloc[0]
    assert b4.actual_cut_95 == 8 and b4.n_items == 8


def test_margin_requires_genuine_terminal_distribution():
    items = pd.DataFrame({"item_id": ["a"], "gold_scheme": ["analogy"]})
    rows = pd.DataFrame({"verdict": [GOOD], "confidence": [0.8]}, index=items.item_id)
    system = SavedSystem("A2", "1.1", rows, np.ones(1, bool), True)
    schemes = load_all()
    assert terminal_margin(system, items, schemes) is None
    rows["distribution"] = json.dumps({GOOD: 0.8, WEAK: 0.2})
    assert terminal_margin(system, items, schemes) == pytest.approx([0.6])
    rows["distribution"] = json.dumps({GOOD: 0.6, WEAK: 0.4})
    with pytest.raises(SchemeError, match="confidence"):
        terminal_margin(system, items, schemes)
