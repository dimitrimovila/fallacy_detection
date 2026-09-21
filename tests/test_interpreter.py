"""The generic interpreter: what traverse() does when an answer draws no arc.

na is never on a diagram's arcs (schemes/ stays frozen); this is the case the
interpreter was already built for, exercised here with na as the example.
"""

from __future__ import annotations

from argfallacy.schemes.interpreter import traverse


def test_traverse_on_na_stops_at_that_node(schemes):
    scheme = schemes["expert_opinion"]
    result = traverse(scheme, {"CQ1": "na"})
    assert result.status == "incomplete"
    assert result.stopped_at == "CQ1"
    assert result.verdicts == set()
    assert result.verdict is None


def test_traverse_on_na_past_the_first_node_stops_there(schemes):
    scheme = schemes["expert_opinion"]
    result = traverse(scheme, {"CQ1": "yes", "CQ2": "na"})
    assert result.status == "incomplete"
    assert result.stopped_at == "CQ2"
    assert result.visited_cqs == {"CQ1", "CQ2"}
