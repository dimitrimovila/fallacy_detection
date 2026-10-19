"""A1 and A3, the aggregators that learn from the gold, on answers written by hand.

The tables of blocks are those of the specification, written here by hand: they
are what the derivation must give, not a record of what it gave.
"""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from argfallacy.aggregate import (
    C_GRID,
    a0,
    a1_out_of_fold,
    a3_out_of_fold,
    choose_c,
    choose_order,
    derive_blocks,
    original_order,
    rebuild,
)
from argfallacy.paths import SCHEMES_ROOT
from argfallacy.schemes import SchemeError, load_all, traverse
from argfallacy.schemes.loader import CriticalQuestion, Edge, Scheme

GOOD = "good_argumentation"

# scheme: (blocks, each as (first node, other nodes); sub-chains; orders)
BLOCKS_V1 = {
    "ad_hominem": ([("CQ1",), ("CQ2",), ("CQ3",), ("CQ4",), ("CQ5",), ("CQ6",)], [], 720),
    "analogy": ([("CQ1", "CQ1.1"), ("CQ2", "CQ1.1"), ("CQ3",), ("CQ4",)], [], 24),
    "cause_to_effect": ([("CQ1",), ("CQ2", "CQ2.1", "CQ2.2"), ("CQ3",), ("CQ4",)],
                        [("CQ2.1", "CQ2.2")], 48),
    "correlation_to_cause": ([("CQ1",), ("CQ2", "CQ2.1", "CQ2.2"), ("CQ3",), ("CQ4",),
                              ("CQ5",)], [("CQ2.1", "CQ2.2")], 240),
    "example": ([("CQ1",), ("CQ2", "CQ2.1", "CQ2.2", "CQ2.3"), ("CQ3",)],
                [("CQ2.1", "CQ2.2", "CQ2.3")], 36),
    "expert_opinion": ([("CQ1",), ("CQ2",), ("CQ3", "CQ3.1"), ("CQ4", "CQ4.1")], [], 24),
    "popular_opinion": ([("CQ1",), ("CQ2",), ("CQ3", "CQ3.1", "CQ3.2", "CQ3.3")],
                        [("CQ3.1", "CQ3.2", "CQ3.3")], 36),
    "slippery_slope": ([("CQ1",), ("CQ2",), ("CQ3",), ("CQ4", "CQ4.1"), ("CQ5",)], [], 120),
}
BLOCKS_V2 = {
    "ad_hominem": ([("CQ1",), ("CQ2",), ("CQ3",), ("CQ4",), ("CQ5",), ("CQ6",)], [], 720),
    "analogy": ([("CQ1", "CQ1.1"), ("CQ2", "CQ1.1"), ("CQ3",), ("CQ4",)], [], 24),
    "cause_to_effect": ([("CQ1",), ("CQ2",), ("CQ3",), ("CQ4",), ("CQ5",)], [], 120),
    "correlation_to_cause": ([("CQ1",), ("CQ2",), ("CQ3",), ("CQ4",), ("CQ5",), ("CQ6",)],
                             [], 720),
    "example": ([("CQ1",), ("CQ2", "CQ2.1", "CQ2.2", "CQ2.3"), ("CQ3",)],
                [("CQ2.1", "CQ2.2", "CQ2.3")], 36),
    "expert_opinion": ([("CQ1",), ("CQ2",), ("CQ3",), ("CQ4", "CQ4.1"), ("CQ5", "CQ5.1")],
                       [], 120),
    "popular_opinion": ([("CQ1", "CQ1.1", "CQ1.2"), ("CQ2",), ("CQ3",)],
                        [("CQ1.1", "CQ1.2")], 12),
    "slippery_slope": ([("CQ1",), ("CQ2",), ("CQ3",)], [], 6),
}


def _arcs(scheme) -> dict:
    return {node: {a: (e.target, e.is_terminal) for a, e in arcs.items()}
            for node, arcs in scheme.nodes.items()}


@pytest.mark.parametrize("folder,table", [("v1", BLOCKS_V1), ("v2", BLOCKS_V2)])
def test_the_blocks_of_the_specification(folder, table):
    schemes = load_all(SCHEMES_ROOT / folder)
    assert set(schemes) == set(table)
    for scheme_id, (blocks, subchains, n_orders) in table.items():
        layout = derive_blocks(schemes[scheme_id])
        got = [(block[0], set(block[1:])) for block in layout.blocks]
        assert got == [(block[0], set(block[1:])) for block in blocks], scheme_id
        assert list(layout.subchains) == subchains, scheme_id
        assert layout.n_orders == n_orders, scheme_id
        end = "ad_hominem_abusive" if scheme_id == "ad_hominem" else GOOD
        assert layout.end == end


@pytest.mark.parametrize("folder", ["v1", "v2"])
def test_the_original_order_rebuilds_every_diagram(folder):
    for scheme in load_all(SCHEMES_ROOT / folder).values():
        layout = derive_blocks(scheme)
        again = rebuild(scheme, layout, original_order(layout))
        assert again.start == scheme.start
        assert _arcs(again) == _arcs(scheme), scheme.scheme_id


def _scheme(arcs: dict[str, dict[str, str]], scheme_id: str = "hand") -> Scheme:
    """A diagram written by hand; a target that is a key of ``arcs`` is a node."""
    nodes = {
        node: {answer: Edge(answer, target, target not in arcs) for answer, target in out.items()}
        for node, out in arcs.items()
    }
    cqs = {node: CriticalQuestion(node, f"Question {node}?", ("yes", "no")) for node in arcs}
    return Scheme(scheme_id=scheme_id, name=scheme_id, schema="", identification_question="",
                  variables={}, cqs=cqs, cq_order=list(arcs), start=next(iter(arcs)),
                  nodes=nodes)


def test_an_arc_to_a_later_node_of_the_main_path_stops_a1():
    scheme = _scheme({
        "CQ1": {"no": "CQ2", "yes": "CQ1.1"},
        "CQ1.1": {"yes": "cherry_picking", "no": "CQ1.2"},
        "CQ1.2": {"yes": "false_premise", "no": "CQ3"},
        "CQ2": {"yes": "weak_analogy", "no": "CQ3"},
        "CQ3": {"yes": "appeal_to_extremes", "no": GOOD},
    })
    with pytest.raises(SchemeError, match=r"CQ1\.2 no -> CQ3"):
        derive_blocks(scheme)


def test_another_order_moves_blocks_and_sub_chains(schemes):
    scheme = schemes["example"]
    layout = derive_blocks(scheme)
    order = next(o for o in layout.orders()
                 if o.blocks == (2, 0, 1) and o.chains == (("CQ2.3", "CQ2.1", "CQ2.2"),))
    moved = rebuild(scheme, layout, order)
    assert moved.start == "CQ3"
    assert moved.nodes["CQ3"]["no"].target == "CQ1"         # was good_argumentation
    assert moved.nodes["CQ1"]["no"].target == "CQ2"
    assert moved.nodes["CQ2"]["yes"].target == GOOD          # was CQ3, now last
    assert moved.nodes["CQ2"]["no"].target == "CQ2.3"        # enters the new first
    assert moved.nodes["CQ2.3"]["no"].target == "CQ2.1"
    assert moved.nodes["CQ2.1"]["no"].target == "CQ2.2"
    assert (moved.nodes["CQ2.2"]["no"].target, moved.nodes["CQ2.2"]["no"].is_terminal) == (
        GOOD, True)                                          # the end, after the move
    assert moved.nodes["CQ2.1"]["yes"].target == "cherry_picking"
    assert order.describe(layout) == "CQ3 > CQ1 > CQ2 [CQ2.3, CQ2.1, CQ2.2]"


def test_the_original_order_gives_the_verdicts_of_a0(schemes):
    for scheme in schemes.values():
        layout = derive_blocks(scheme)
        again = rebuild(scheme, layout, original_order(layout))
        nodes = list(scheme.nodes)
        for values in itertools.product(*[(*scheme.answer_space(cq), "na") for cq in nodes]):
            answers = dict(zip(nodes, values, strict=True))
            rows = {cq: {"answer": answer} for cq, answer in answers.items()}
            assert traverse(again, answers).verdict == a0(scheme, rows).verdict


CHAIN = {
    "CQ1": {"yes": "cherry_picking", "no": "CQ2"},
    "CQ2": {"yes": "false_premise", "no": "CQ3"},
    "CQ3": {"yes": "weak_analogy", "no": GOOD},
}


def test_the_order_chosen_puts_first_the_block_that_decides():
    scheme = _scheme(CHAIN)
    # CQ1 and CQ2 say yes on every item; only CQ3 tells the two classes apart
    answers = [{"CQ1": "yes", "CQ2": "yes", "CQ3": "yes"}] * 3
    answers += [{"CQ1": "yes", "CQ2": "yes", "CQ3": "no"}] * 3
    gold = ["weak_analogy"] * 3 + ["cherry_picking"] * 3
    layout = derive_blocks(scheme)
    chosen = choose_order(scheme, answers, gold)
    assert chosen.describe(layout) == "CQ3 > CQ1 > CQ2"


def test_a_tie_keeps_the_original_order():
    scheme = _scheme(CHAIN)
    answers = [{"CQ1": "no", "CQ2": "no", "CQ3": "no"}] * 4
    chosen = choose_order(scheme, answers, [GOOD, GOOD, "weak_analogy", GOOD])
    assert chosen == original_order(derive_blocks(scheme))


def _items(schemes, scheme_id, n, gold_of):
    return pd.DataFrame({"item_id": [f"{scheme_id}{i:02d}" for i in range(n)],
                         "gold_scheme": scheme_id,
                         "gold_fallacy": [gold_of(i) for i in range(n)]})


def test_a1_out_of_fold_trains_on_the_other_folds(schemes):
    """Items of one fold get the order chosen on the others, and each fold is recorded."""
    scheme = schemes["analogy"]
    items = _items(schemes, "analogy", 10,
                   lambda i: "cherry_picking" if i % 2 else "false_premise")
    rows = {}
    for i, item in enumerate(items["item_id"]):
        # CQ3 no (false premise) on the even items, CQ4 yes on every item
        rows[item] = {"CQ1": {"answer": "no"}, "CQ2": {"answer": "yes"},
                      "CQ3": {"answer": "yes" if i % 2 else "no"}, "CQ4": {"answer": "yes"},
                      "CQ1.1": {"answer": "no"}}
    folds = {item: i % 5 for i, item in enumerate(items["item_id"])}
    verdicts, chosen = a1_out_of_fold({"analogy": scheme}, items, rows, folds)
    assert set(verdicts) == set(items["item_id"])
    assert [record["fold"] for record in chosen] == [0, 1, 2, 3, 4]
    assert all(record["scheme"] == "analogy" for record in chosen)
    for item, gold in zip(items["item_id"], items["gold_fallacy"], strict=True):
        assert verdicts[item].verdict == gold
        assert verdicts[item].confidence is None
        assert verdicts[item].explanation.startswith("order CQ1 > CQ2 > CQ3 > CQ4: ")


# ------------------------------------------------------------------- A3


def _row(scheme, cq, p_first):
    first, second = scheme.answer_space(cq)
    return {"answer": first if p_first >= 0.5 else second,
            f"p_logprob_{first}": p_first, f"p_logprob_{second}": 1 - p_first,
            "p_logprob_cannot_be_determined": 0.0, "p_logprob_na": 0.0}


def _separable(schemes, n=20):
    """CQ4 of analogy separates cherry picking from good argumentation; the rest is noise."""
    scheme = schemes["analogy"]
    items = _items(schemes, "analogy", n, lambda i: "cherry_picking" if i % 2 else GOOD)
    rows = {}
    for i, item in enumerate(items["item_id"]):
        noise = (i * 7 % 10) / 10
        rows[item] = {cq: _row(scheme, cq, noise) for cq in scheme.cq_order}
        rows[item]["CQ4"] = _row(scheme, "CQ4", 0.9 if i % 2 else 0.1)
    folds = {item: i % 5 for i, item in enumerate(items["item_id"])}
    return scheme, items, rows, folds


def test_a3_learns_the_cq_that_separates_the_classes(schemes):
    scheme, items, rows, folds = _separable(schemes)
    verdicts, choices = a3_out_of_fold({"analogy": scheme}, items, rows, folds, seed=7)
    for item, gold in zip(items["item_id"], items["gold_fallacy"], strict=True):
        assert verdicts[item].verdict == gold, item
        assert set(verdicts[item].distribution) == {"cherry_picking", GOOD}
        assert sum(verdicts[item].distribution.values()) == pytest.approx(1.0)
        assert "CQ4 " in verdicts[item].explanation
    assert [c["fold"] for c in choices] == [0, 1, 2, 3, 4]
    assert all(c["C"] in C_GRID for c in choices)


def test_a3_with_one_class_in_training_predicts_it(schemes):
    scheme, items, rows, folds = _separable(schemes, n=10)
    items["gold_fallacy"] = GOOD
    verdicts, choices = a3_out_of_fold({"analogy": scheme}, items, rows, folds, seed=7)
    assert all(v.verdict == GOOD and v.confidence == 1.0 for v in verdicts.values())
    assert all(c["C"] is None for c in choices)


def test_a3_breaks_a_tie_in_the_inner_criterion_toward_the_smaller_c():
    # two classes apart on one feature each: every C gets every item right, a tie
    features = [[1.0, 0.0], [0.0, 1.0]] * 6
    gold = ["cherry_picking", GOOD] * 6
    assert choose_c(features, gold, seed=3) == min(C_GRID)


def test_a3_leaves_the_scheme_mismatch_items_out_of_training(schemes):
    scheme, items, rows, folds = _separable(schemes)
    # a gold that analogy cannot produce, on items that look like cherry picking
    items.loc[items.index % 4 == 1, "gold_fallacy"] = "post_hoc"
    verdicts, _ = a3_out_of_fold({"analogy": scheme}, items, rows, folds, seed=7)
    for verdict in verdicts.values():
        assert "post_hoc" not in verdict.distribution
    wrong = items[items["gold_fallacy"] == "post_hoc"]["item_id"]
    assert all(verdicts[item].verdict == "cherry_picking" for item in wrong)


# ------------------------------------------------------------- determinism


def test_a1_and_a3_give_the_same_verdicts_twice(schemes):
    scheme, items, rows, folds = _separable(schemes)
    hard = {item: {cq: {"answer": row["answer"]} for cq, row in own.items()}
            for item, own in rows.items()}
    runs = [a1_out_of_fold({"analogy": scheme}, items, hard, folds) for _ in range(2)]
    assert ({k: v.verdict for k, v in runs[0][0].items()}
            == {k: v.verdict for k, v in runs[1][0].items()})
    assert runs[0][1] == runs[1][1]
    runs = [a3_out_of_fold({"analogy": scheme}, items, rows, folds, seed=11) for _ in range(2)]
    assert ({k: v.distribution for k, v in runs[0][0].items()}
            == {k: v.distribution for k, v in runs[1][0].items()})
    assert runs[0][1] == runs[1][1]
