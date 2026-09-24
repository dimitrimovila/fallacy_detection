"""The canonical rules of the scorer, on small tables written by hand.

Each table is a handful of items with a known right answer, so a change in the
rules shows up as a wrong count rather than as a drift in a real run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from argfallacy.eval import score_fallacies
from argfallacy.labels import SchemeError


def _tables(rows):
    """rows: (item_id, gold fallacy, raw prediction) -> (predictions, gold)."""
    ids, gold, predicted = zip(*rows, strict=True)
    gold_frame = pd.DataFrame({"item_id": ids, "gold_fallacy": gold})
    return pd.DataFrame({"item_id": ids, "predicted": predicted}), gold_frame


def test_a_variant_spelling_counts_as_its_label(vocab):
    predictions, gold = _tables([("a", "strawman", "Straw Man"),
                                 ("b", "causal_reductionism", "Casual reductionism")])
    for space in ("fine", "collapsed_symmetric"):
        assert score_fallacies(predictions, gold, space, vocabulary=vocab).accuracy == 1.0


def test_invalid_and_missing_verdicts_are_wrong_in_every_space(vocab):
    predictions, gold = _tables([("a", "strawman", "Red Herring/Straw Man"),
                                 ("b", "strawman", None),
                                 ("c", "strawman", ""),
                                 ("d", "strawman", "Strawman")])
    for space in ("fine", "collapsed_symmetric"):
        score = score_fallacies(predictions, gold, space, vocabulary=vocab)
        assert score.n_items == 4
        assert score.accuracy == 0.25
        assert score.not_a_class == {"invalid": 1, "no_verdict": 2}


def test_a_family_prediction_is_wrong_in_fine_and_right_when_collapsed(vocab):
    predictions, gold = _tables([("a", "ad_hominem_abusive", "Ad Hominem"),
                                 ("b", "ad_hominem_tu_quoque", "Ad Hominem")])
    fine = score_fallacies(predictions, gold, "fine", vocabulary=vocab)
    assert fine.accuracy == 0.0
    assert fine.not_a_class == {"family": 2}
    collapsed = score_fallacies(predictions, gold, "collapsed_symmetric", vocabulary=vocab)
    assert collapsed.accuracy == 1.0
    assert collapsed.not_a_class == {}


@pytest.mark.parametrize("label", ["False Cause", "Ad Hominem", "Appeal to Emotion", "?", "YES"])
def test_a_gold_that_is_not_a_terminal_stops_the_scorer(vocab, label):
    predictions, gold = _tables([("a", "strawman", "Strawman"),
                                 ("item_with_vague_gold", label, "Strawman")])
    with pytest.raises(SchemeError, match="item_with_vague_gold"):
        score_fallacies(predictions, gold, vocabulary=vocab)


def test_macro_leaves_out_the_classes_without_gold_support(vocab):
    # Two classes with support.  One strawman item is predicted as a class with no
    # support (an error that stays in accuracy and in the recall of strawman), one
    # post hoc item gets an invalid output (it lowers only the recall of post hoc).
    predictions, gold = _tables([("a", "strawman", "Strawman"),
                                 ("b", "strawman", "Definist Fallacy"),
                                 ("c", "post_hoc", "Post Hoc"),
                                 ("d", "post_hoc", "Red Herring/Straw Man")])
    score = score_fallacies(predictions, gold, vocabulary=vocab)
    assert score.accuracy == 0.5
    assert score.n_classes_gold == 2
    assert "definist_fallacy" in score.left_out
    assert "strawman" not in score.left_out and "post_hoc" not in score.left_out
    assert len(score.left_out) == 23
    rows = score.classes.set_index("class")
    assert len(rows) == 25
    assert rows.loc["strawman", ["support", "precision", "recall"]].tolist() == [2, 1.0, 0.5]
    assert rows.loc["post_hoc", ["support", "precision", "recall"]].tolist() == [2, 1.0, 0.5]
    assert rows.loc["definist_fallacy", ["support", "precision"]].tolist() == [0, 0.0]
    assert score.macro_precision == 1.0
    assert score.macro_recall == 0.5
    assert score.macro_f1 == pytest.approx(2 / 3)


def test_a_subset_scores_only_its_items(vocab):
    predictions, gold = _tables([("a", "strawman", "Strawman"),
                                 ("b", "post_hoc", "Strawman")])
    assert score_fallacies(predictions, gold, items=["a"], vocabulary=vocab).accuracy == 1.0
    assert score_fallacies(predictions, gold, items=["b"], vocabulary=vocab).accuracy == 0.0
