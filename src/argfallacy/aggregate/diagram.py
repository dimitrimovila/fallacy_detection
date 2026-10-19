"""A0, the diagram as drawn, and the arc of each node that keeps the argument standing."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any

from ..schemes import Scheme, SchemeError, traverse
from ..schemes.loader import Edge, load_vocabulary
from .interface import Aggregation


@cache
def good_terminal() -> str:
    """The one terminal of type ``good`` in the label dictionary."""
    terminals = load_vocabulary(check_schemes=False)["terminals"]
    (good,) = [tid for tid, entry in terminals.items() if entry.get("type") == "good"]
    return good


def distances_to_good(scheme: Scheme) -> dict[str, int | None]:
    """Arcs from every node to ``good_argumentation`` along the shortest path, or None."""
    memo: dict[str, int | None] = {}

    def distance(node: str) -> int | None:
        if node not in memo:
            steps = [arc_distance(edge, distance) for edge in scheme.nodes[node].values()]
            reachable = [step for step in steps if step is not None]
            memo[node] = min(reachable) if reachable else None
        return memo[node]

    return {node: distance(node) for node in scheme.nodes}


def arc_distance(edge: Edge, distance: Any) -> int | None:
    """Arcs after this one to ``good_argumentation``: 0 for an arc that ends on it.

    ``distance`` gives the distance of a node, as :func:`distances_to_good` does.
    """
    if edge.is_terminal:
        return 0 if edge.target == good_terminal() else None
    below = distance(edge.target)
    return None if below is None else below + 1


def reaches_good(scheme: Scheme, node: str) -> dict[str, bool]:
    """Whether each arc of ``node`` can still reach ``good_argumentation``."""
    distances = distances_to_good(scheme)
    return {answer: arc_distance(edge, distances.get) is not None
            for answer, edge in scheme.nodes[node].items()}


def pass_arc(scheme: Scheme, node: str) -> str:
    """The arc of ``node`` that keeps the argument standing.

    The arc with the shortest path to ``good_argumentation``, ties to the arc written
    first.  Where no arc reaches it, the only arc that leads to a node; where both
    arcs end on terminals, the arc ``no``.
    """
    distances = distances_to_good(scheme)
    edges = scheme.nodes[node]
    reach = {answer: arc_distance(edge, distances.get) for answer, edge in edges.items()}
    reach = {answer: steps for answer, steps in reach.items() if steps is not None}
    if reach:
        return min(reach, key=reach.get)
    to_nodes = [answer for answer, edge in edges.items() if not edge.is_terminal]
    if len(to_nodes) == 1:
        return to_nodes[0]
    if not to_nodes and "no" in edges:
        return "no"
    raise SchemeError(f"{scheme.scheme_id} {node}: no arc reaches good_argumentation and "
                      f"none is the only arc to a node or the arc no: {list(edges)}")


def a0(scheme: Scheme, rows: Mapping[str, Mapping[str, Any]]) -> Aggregation:
    """The diagram traversed with the hard answers of one entry.

    An answer outside the arcs, or a missing one, stops the traversal with no
    verdict.  The explanation is the path.
    """
    answers = {cq: row["answer"] for cq, row in rows.items() if row is not None}
    result = traverse(scheme, answers)
    steps = [f"{cq}={answer}" for cq, answer in result.paths[0]]
    if result.verdict is None:
        stop = result.stopped_at
        steps.append(f"{stop}={answers.get(stop, 'missing')}")
        return Aggregation(None, None, {}, 1.0, " > ".join(steps) + ": no verdict")
    return Aggregation(result.verdict, None, {result.verdict: 1.0}, 0.0,
                       " > ".join([*steps, result.verdict]))
