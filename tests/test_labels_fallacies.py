"""The fallacy dictionary, the spaces, the constraints."""

from __future__ import annotations

import pytest

from argfallacy.labels import label_space, normalize_label
from argfallacy.schemes import (
    SchemeError,
    check_vocabulary_against_schemes,
    family_members,
    label_matches,
    resolve_for_scheme,
)


def test_invented_string_raises_and_names_itself(vocab):
    with pytest.raises(SchemeError) as excinfo:
        normalize_label("Argumentum ad Lasagnam", vocab)
    assert "Argumentum ad Lasagnam" in str(excinfo.value)


def test_false_cause_stays_coarse(vocab):
    label = normalize_label("false cause", vocab)
    assert label.kind == "coarse"
    assert label.id == "false_cause"
    assert len(label.accepts) == 3


def test_family_gold_gives_partial_credit_under_its_own_scheme(vocab, schemes):
    """Gold 'Ad Hominem' under ad_hominem: coarse, satisfied by any member."""
    scheme = schemes["ad_hominem"]
    gold = resolve_for_scheme("Ad Hominem", scheme, schemes, vocab)

    assert gold.label.kind == "family"
    assert gold.status == "coarse"
    assert gold.accepts == family_members("ad_hominem", vocab) & scheme.terminals
    assert len(gold.accepts) == 6
    assert all(label_matches(member, gold) for member in gold.accepts)


def test_the_credit_is_asymmetric(vocab, schemes):
    """Coarse gold gives partial credit; a prediction that names the family gives none."""
    scheme = schemes["ad_hominem"]

    coarse_gold = resolve_for_scheme("Ad Hominem", scheme, schemes, vocab)
    assert label_matches("ad_hominem_tu_quoque", coarse_gold), "gold grosso: credito parziale"

    exact_gold = resolve_for_scheme("Tu quoque", scheme, schemes, vocab)
    assert exact_gold.status == "exact"
    assert not label_matches("ad_hominem", exact_gold), "predizione vaga: nessun credito"


def test_vocabulary_matches_the_diagrams(vocab, schemes):
    check_vocabulary_against_schemes(vocab, schemes)


def test_terminal_count_is_twenty_five(vocab, schemes):
    graph_terminals = set().union(*(s.terminals for s in schemes.values()))
    assert len(vocab["terminals"]) == 25
    assert set(vocab["terminals"]) == graph_terminals


def test_space_sizes(vocab):
    assert len(label_space("fine", vocab)) == 25
    assert len(label_space("collapsed_symmetric", vocab)) == 20
    assert len(label_space("collapsed_compat", vocab)) == 21
