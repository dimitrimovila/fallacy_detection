"""A single generic interpreter that executes any scheme diagram.

Two ways of reading the same YAML, both limited to what the diagrams actually
contain:

``traverse``       hard answers -> verdict.  This is the rule-based baseline and
                   it mirrors the flowchart exactly: one answer per node, one
                   path, one terminal.
``propagate``      per-CQ answer distributions -> distribution over terminals.
                   It does not add arcs, it weights the ones already drawn.

The diagrams admit exactly the answers written on their arcs, ``yes`` and ``no``
everywhere except ``ad_hominem`` CQ1, which branches on ``positive`` and
``negative``.  Anything an annotation protocol might admit beyond those (an
"I don't know", an abstention, a tolerant default) is **not** in the diagrams and
is not implemented here.  Those belong to the revision proposals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from .loader import Scheme


@dataclass
class Traversal:
    """Outcome of running one item through one diagram."""

    verdicts: set[str]
    paths: list[list[tuple[str, str]]] = field(default_factory=list)
    status: Literal["resolved", "incomplete"] = "resolved"
    stopped_at: str | None = None
    visited_cqs: set[str] = field(default_factory=set)

    @property
    def verdict(self) -> str | None:
        """The single verdict, or None when the traversal did not reach one."""
        return next(iter(self.verdicts)) if len(self.verdicts) == 1 else None


def traverse(scheme: Scheme, answers: Mapping[str, str]) -> Traversal:
    """Walk the diagram with a dict of ``{cq_id: answer}``.

    ``answers`` may contain more CQs than the path visits — that is expected,
    since the annotation protocol records the full vector while the diagram
    exits early.  Answers off the path are simply never read.

    Two outcomes only, because the diagram is deterministic:

    ``resolved``    a terminal was reached; ``verdicts`` holds exactly one label.
    ``incomplete``  a node was reached whose answer is missing from ``answers``
                    or is not one of the arcs drawn at that node.  ``stopped_at``
                    names it and ``verdicts`` is empty.
    """
    path: list[tuple[str, str]] = []
    visited: set[str] = set()
    node = scheme.start

    while True:
        visited.add(node)
        raw = answers.get(node)
        answer = None if raw is None else str(raw).strip().lower()
        edge = scheme.nodes[node].get(answer) if answer is not None else None

        if edge is None:
            return Traversal(verdicts=set(), paths=[path], status="incomplete",
                             stopped_at=node, visited_cqs=visited)

        path = path + [(node, answer)]
        if edge.is_terminal:
            return Traversal(verdicts={edge.target}, paths=[path],
                             status="resolved", visited_cqs=visited)
        node = edge.target


def propagate(
    scheme: Scheme,
    probabilities: Mapping[str, Mapping[str, float]],
    default: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Probabilistic traversal: per-CQ answer distributions -> terminal distribution.

    ``probabilities`` maps ``cq_id -> {answer: p}``.  Any CQ without an entry
    falls back to ``default`` (uniform over the arcs of that node if not given).

    Assumption to declare in the thesis: CQ answers are treated as **independent**
    conditional on the path.  They are almost certainly not — CQ2.1/2.2/2.3 of
    *example* clearly co-vary — so this is a first-order approximation whose error
    is itself measurable against the empirical joint distribution of the annotations.
    """
    memo: dict[str, dict[str, float]] = {}

    def dist(node: str) -> dict[str, float]:
        if node in memo:
            return memo[node]
        space = scheme.nodes[node].keys()
        if node in probabilities:
            p = {a: float(probabilities[node].get(a, 0.0)) for a in space}
        elif default is not None:
            p = {a: float(default.get(a, 0.0)) for a in space}
        else:
            p = {a: 1.0 / len(space) for a in space}
        total = sum(p.values())
        if total <= 0:
            raise ValueError(f"{scheme.scheme_id}/{node}: probabilities sum to {total}")
        p = {a: v / total for a, v in p.items()}

        out: dict[str, float] = {}
        for answer, weight in p.items():
            if weight == 0.0:
                continue
            edge = scheme.nodes[node][answer]
            if edge.is_terminal:
                out[edge.target] = out.get(edge.target, 0.0) + weight
            else:
                for terminal, q in dist(edge.target).items():
                    out[terminal] = out.get(terminal, 0.0) + weight * q
        memo[node] = out
        return out

    result = dist(scheme.start)
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))


def enumerate_paths(scheme: Scheme) -> list[tuple[list[tuple[str, str]], str]]:
    """All root-to-terminal paths.  Useful for sanity checks and for reporting how
    many distinct answer vectors map onto the same verdict label."""
    out: list[tuple[list[tuple[str, str]], str]] = []

    def walk(node: str, path: list[tuple[str, str]]) -> None:
        for answer, edge in scheme.nodes[node].items():
            step = path + [(node, answer)]
            if edge.is_terminal:
                out.append((step, edge.target))
            else:
                walk(edge.target, step)

    walk(scheme.start, [])
    return out
