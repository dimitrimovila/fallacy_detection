"""A1: the diagram with its blocks, and the questions inside a block, in another order.

Only ad hominem is a chain of questions, so the order is defined on blocks derived
from the graph.  The main path starts at the first node and follows the pass arcs
to a terminal T; the block of a node of the main path is the node and what its
other arc reaches before the main path.  Inside a block, a sub-chain is a run of
at least two nodes, other than the first, each with one arc to a terminal and
one arc, with the same answer along the run, to the next.

The order is chosen out of fold: on the items of the other folds, the order with
the highest macro F1, then the highest accuracy, then the one closest to the
original, then the first by its node ids.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations, product
from math import factorial, prod
from typing import Any

import pandas as pd

from ..eval.metrics import accuracy
from ..schemes import Scheme, SchemeError, traverse
from .diagram import a0, pass_arc
from .interface import TIE, Aggregation, macro_f1


@dataclass(frozen=True)
class Order:
    """A permutation of the blocks, by their original index, and the new sequence of
    every sub-chain."""

    blocks: tuple[int, ...]
    chains: tuple[tuple[str, ...], ...]

    def describe(self, layout: Layout) -> str:
        """The blocks by their first node, each sub-chain in brackets after its block."""
        parts = []
        for index in self.blocks:
            block = layout.blocks[index]
            text = block[0]
            for chain, sequence in zip(layout.subchains, self.chains, strict=True):
                if chain[0] in block:
                    text += f" [{', '.join(sequence)}]"
            parts.append(text)
        return " > ".join(parts)

    def node_ids(self, layout: Layout) -> tuple[str, ...]:
        """The first nodes of the blocks in this order, then every sub-chain."""
        return (*(layout.blocks[i][0] for i in self.blocks), *(n for c in self.chains for n in c))


@dataclass(frozen=True)
class Layout:
    """The blocks of a diagram in the original order, its sub-chains and T."""

    blocks: tuple[tuple[str, ...], ...]
    subchains: tuple[tuple[str, ...], ...]
    links: tuple[str, ...]
    """The answer that links the nodes of each sub-chain."""
    end: str

    @property
    def main(self) -> tuple[str, ...]:
        return tuple(block[0] for block in self.blocks)

    @property
    def n_orders(self) -> int:
        return factorial(len(self.blocks)) * prod(factorial(len(c)) for c in self.subchains)

    def orders(self) -> Iterator[Order]:
        for blocks in permutations(range(len(self.blocks))):
            for chains in product(*(permutations(chain) for chain in self.subchains)):
                yield Order(blocks, tuple(chains))


def original_order(layout: Layout) -> Order:
    return Order(tuple(range(len(layout.blocks))), layout.subchains)


def derive_blocks(scheme: Scheme) -> Layout:
    """The blocks and sub-chains of a diagram; an arc that leaves its block stops A1."""
    passes = {node: pass_arc(scheme, node) for node in scheme.nodes}
    main, node = [], scheme.start
    while True:
        main.append(node)
        edge = scheme.nodes[node][passes[node]]
        if edge.is_terminal:
            end = edge.target
            break
        node = edge.target
    on_main = set(main)

    blocks = []
    for node in main:
        members = [node]
        frontier = [edge.target for answer, edge in scheme.nodes[node].items()
                    if answer != passes[node] and not edge.is_terminal
                    and edge.target not in on_main]
        while frontier:
            current = frontier.pop(0)
            if current in members:
                continue
            members.append(current)
            frontier += [edge.target for edge in scheme.nodes[current].values()
                         if not edge.is_terminal and edge.target not in on_main]
        blocks.append(tuple(members))

    for index, block in enumerate(blocks):
        following = main[index + 1] if index + 1 < len(main) else None
        for member in block:
            for answer, edge in scheme.nodes[member].items():
                if edge.is_terminal or edge.target in block or edge.target == following:
                    continue
                raise SchemeError(
                    f"{scheme.scheme_id}: the arc {member} {answer} -> {edge.target} leaves "
                    f"the block of {block[0]} for a node other than the next of the main "
                    f"path, so A1 cannot reorder this diagram")

    subchains, links = [], []
    for block in blocks:
        for chain, answer in _subchains(scheme, block[1:]):
            subchains.append(chain)
            links.append(answer)
    return Layout(tuple(blocks), tuple(subchains), tuple(links), end)


def _subchains(scheme: Scheme, inner: Sequence[str]) -> list[tuple[tuple[str, ...], str]]:
    """The longest runs of linked nodes among ``inner``, each with its linking answer."""

    def links(node: str, answer: str) -> bool:
        others = [edge for a, edge in scheme.nodes[node].items() if a != answer]
        return answer in scheme.nodes[node] and len(others) == 1 and others[0].is_terminal

    found = []
    for start in inner:
        for answer in scheme.nodes[start]:
            chain, current = [], start
            while current in inner and current not in chain and links(current, answer):
                chain.append(current)
                edge = scheme.nodes[current][answer]
                if edge.is_terminal:
                    break
                current = edge.target
            if len(chain) >= 2:
                found.append((tuple(chain), answer))
    return [(chain, answer) for chain, answer in found
            if not any(set(chain) < set(other) for other, _ in found)]


def rebuild(scheme: Scheme, layout: Layout, order: Order) -> Scheme:
    """The diagram with its blocks and sub-chains in ``order``.

    An arc that ended on the next node of the main path (on T for the last block)
    ends on the first node of the next block in the new order (on T for the new
    last block); the arc that entered a sub-chain enters its new first node, its
    arcs follow the new sequence, and the last one goes to the end of the sub-chain.
    Every other arc stays.
    """
    nodes = {node: dict(arcs) for node, arcs in scheme.nodes.items()}
    for chain, answer, sequence in zip(layout.subchains, layout.links, order.chains,
                                       strict=True):
        end = scheme.nodes[chain[-1]][answer]
        for node, arcs in scheme.nodes.items():
            for a, edge in arcs.items():
                if node not in chain and not edge.is_terminal and edge.target == chain[0]:
                    nodes[node][a] = dataclasses.replace(edge, target=sequence[0])
        for current, following in zip(sequence, sequence[1:], strict=False):
            nodes[current][answer] = dataclasses.replace(
                scheme.nodes[current][answer], target=following, is_terminal=False)
        nodes[sequence[-1]][answer] = end

    chained = {node: dict(arcs) for node, arcs in nodes.items()}
    last = len(layout.blocks) - 1
    position = {block: place for place, block in enumerate(order.blocks)}
    for index, block in enumerate(layout.blocks):
        place = position[index]
        target = (layout.blocks[order.blocks[place + 1]][0] if place < last else layout.end)
        for member in block:
            for a, edge in chained[member].items():
                to_next = (edge.target == layout.end and edge.is_terminal if index == last
                           else edge.target == layout.main[index + 1] and not edge.is_terminal)
                if to_next:
                    nodes[member][a] = dataclasses.replace(edge, target=target,
                                                           is_terminal=place == last)
    start = layout.blocks[order.blocks[0]][0]
    return dataclasses.replace(scheme, nodes=nodes, start=start)


def _inversions(sequence: Sequence[Any], original: Sequence[Any]) -> int:
    rank = {value: index for index, value in enumerate(original)}
    ranks = [rank[value] for value in sequence]
    return sum(1 for i in range(len(ranks)) for j in range(i + 1, len(ranks))
               if ranks[i] > ranks[j])


def _best(layout: Layout, candidates: Sequence[Order], verdicts: Sequence[Sequence[str | None]],
          gold: Sequence[str]) -> int:
    """The index of the order chosen among ``candidates``, from their verdicts on ``gold``."""
    f1 = [macro_f1(gold, own) for own in verdicts]
    top = max(f1)
    kept = [i for i in range(len(candidates)) if f1[i] >= top - TIE]
    right = {i: accuracy(list(gold), list(verdicts[i])) for i in kept}
    top = max(right.values())
    kept = [i for i in kept if right[i] >= top - TIE]
    blocks = tuple(range(len(layout.blocks)))

    def closeness(i: int) -> tuple[int, int, tuple[str, ...]]:
        order = candidates[i]
        chains = sum(_inversions(new, old)
                     for new, old in zip(order.chains, layout.subchains, strict=True))
        return _inversions(order.blocks, blocks), chains, order.node_ids(layout)

    return min(kept, key=closeness)


def choose_order(scheme: Scheme, answers: Sequence[Mapping[str, str]],
                 gold: Sequence[str], layout: Layout | None = None) -> Order:
    """The order with the best macro F1 of the hard answers on ``gold``, by the rule of A0."""
    layout = layout or derive_blocks(scheme)
    candidates = list(layout.orders())
    rebuilt = [rebuild(scheme, layout, order) for order in candidates]
    verdicts = [[traverse(r, given).verdict for given in answers] for r in rebuilt]
    return candidates[_best(layout, candidates, verdicts, gold)]


def a1_out_of_fold(
    schemes: Mapping[str, Scheme], items: pd.DataFrame,
    rows_by_item: Mapping[str, Mapping[str, Mapping[str, Any]]], folds: Mapping[str, int],
) -> tuple[dict[str, Aggregation], list[dict[str, Any]]]:
    """A1 on every item, with the order chosen on the items of its scheme in the other folds.

    ``items`` has ``item_id``, ``gold_scheme`` and ``gold_fallacy``.  Returns the
    aggregation of every item and, per fold and scheme, the order chosen.
    """
    out: dict[str, Aggregation] = {}
    chosen: list[dict[str, Any]] = []
    for scheme_id, own in sorted(items.groupby("gold_scheme"), key=lambda pair: pair[0]):
        scheme = schemes[scheme_id]
        layout = derive_blocks(scheme)
        own = own.sort_values("item_id")
        ids = own["item_id"].tolist()
        gold = dict(zip(ids, own["gold_fallacy"], strict=True))
        answers = {item: {cq: row["answer"] for cq, row in rows_by_item[item].items()}
                   for item in ids}
        candidates = list(layout.orders())
        rebuilt = [rebuild(scheme, layout, order) for order in candidates]
        verdicts = [[traverse(r, answers[item]).verdict for item in ids] for r in rebuilt]
        for fold in sorted(set(folds.values())):
            train = [k for k, item in enumerate(ids) if folds[item] != fold]
            best = _best(layout, candidates, [[own_v[k] for k in train] for own_v in verdicts],
                         [gold[ids[k]] for k in train]) if train else 0
            label = candidates[best].describe(layout)
            chosen.append({"fold": fold, "scheme": scheme_id, "order": label})
            for item in ids:
                if folds[item] == fold:
                    result = a0(rebuilt[best], rows_by_item[item])
                    out[item] = dataclasses.replace(result,
                                                    explanation=f"order {label}: "
                                                                f"{result.explanation}")
    chosen.sort(key=lambda record: (record["fold"], record["scheme"]))
    return out, chosen
