from pathlib import Path

import pandas as pd
import pytest

from argfallacy.eval.reference_sensitivity import (
    APP,
    ORIGINAL,
    align_predictions,
    audit_population,
    score_pair,
    validate_references,
)
from argfallacy.eval.reference_sensitivity_sources import System
from argfallacy.labels import INVALID, SchemeError, label_space
from argfallacy.schemes import load_all, load_vocabulary


def test_references_are_joined_by_id_and_copied_gold_is_checked():
    vocab = load_vocabulary(check_schemes=False)
    historical, new, *_ = label_space("fine", vocab)
    scheme = next(iter(load_all()))
    ids = [f"item-{i:03d}" for i in range(601)]
    items = pd.DataFrame(
        {
            "item_id": ids,
            "in_test_set": "True",
            "gold_scheme": scheme,
            "gold_fallacy": historical,
            "eleni_id": ids,
        }
    )
    integrated = items.drop(columns=["in_test_set", "eleni_id"]).assign(
        integrated_verdict=historical,
        verdict_origin=ORIGINAL,
        original_verdict=historical,
        dumitru_direct="",
    )
    integrated.loc[:405, ["verdict_origin", "original_verdict"]] = [APP, ""]
    integrated.loc[:154, "integrated_verdict"] = new
    integrated.loc[:405, "dumitru_direct"] = integrated.loc[:405, "integrated_verdict"]
    integrated = integrated.sample(frac=1, random_state=0)
    result = validate_references(items, integrated, vocab)
    assert result.loc[result.label_changed, "item_id"].tolist() == ids[:155]

    integrated.loc[integrated.item_id.eq(ids[-1]), "gold_fallacy"] = new
    with pytest.raises(SchemeError, match=ids[-1]):
        validate_references(items, integrated, vocab)


def test_audit_cohort_is_the_items_with_absent_or_different_original_labels():
    vocab = load_vocabulary(check_schemes=False)
    historical, different, *_ = label_space("fine", vocab)
    ids = [f"item-{i:03d}" for i in range(601)]
    items = pd.DataFrame(
        {
            "item_id": ids,
            "gold_scheme": "example",
            "gold_fallacy": historical,
            "integrated_verdict": historical,
            "verdict_origin": [APP] * 406 + [ORIGINAL] * 195,
        }
    )
    audit = pd.DataFrame(
        {
            "item_id": ids,
            "gold_scheme": "example",
            "gold_fallacy": historical,
            "original_label_id": [different] * 3 + [""] * 402 + [historical] * 196,
            "status": ["specialized", "changed", "composite"]
            + ["absent"] * 402
            + ["unchanged"] * 196,
        }
    ).sample(frac=1, random_state=0)
    assert audit_population(items, audit) == set(ids[:405])
    items.loc[items.item_id.eq(ids[405]), "integrated_verdict"] = different
    with pytest.raises(SchemeError, match="changes an unchanged original label"):
        audit_population(items, audit)


def test_fixed_item_scores_count_abstentions_and_keep_out_of_menu_terminals():
    vocab = load_vocabulary(check_schemes=False)
    scheme = next(iter(load_all().values()))
    inside = sorted(scheme.terminals)[0]
    outside = next(label for label in label_space("fine", vocab) if label not in scheme.terminals)
    items = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d", "e"],
            "gold_scheme": scheme.scheme_id,
            "gold_fallacy": inside,
            "integrated_verdict": [outside, outside, inside, inside, outside],
            "verdict_origin": APP,
            "label_changed": [True, True, False, False, True],
        }
    )
    rows = pd.DataFrame(
        {"item_id": ["c", "b", "e", "a"], "predicted": ["", inside, INVALID, outside]}
    )
    system = System(
        "test",
        "test",
        "test",
        "1.1",
        "gold_scheme",
        "diagnostic",
        "fixed",
        "restricted",
        ("fine",),
        rows,
        Path("unused.csv"),
        "test",
    )
    aligned = align_predictions(
        system, "fine", items, vocab, {"1.1": {scheme.scheme_id: scheme.terminals}}
    )
    result, _ = score_pair(aligned, "fine", vocab)
    assert aligned.transition.tolist() == [
        "wrong_to_correct",
        "correct_to_wrong",
        "wrong_both",
        "outside_original_population",
        "wrong_both",
    ]
    assert result["accuracy_historical"] == result["accuracy_integrated"] == 0.25
    assert result["n_prediction_outside_diagram_menu"] == 1
    assert result["n_integrated_outside_diagram_menu"] == 3
    assert result["n_population"] == 5
    assert result["n_scored"] == 4
    assert result["n_no_verdict_or_abstention"] == result["n_invalid_or_nonclass"] == 1
    assert result["coverage"] == 0.5
