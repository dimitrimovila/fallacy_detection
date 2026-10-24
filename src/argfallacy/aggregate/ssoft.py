"""Path-conditioned expansion and the identical pruning rule for independent CQs."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..schemes import Scheme, SchemeError
from .interface import Aggregation, decide
from .soft import LOGPROB, OUTSIDE, _route, answer_masses

History = tuple[tuple[str, str], ...]
Masses = Mapping[str, float]
Transform = Callable[[str, Masses], Masses]


@dataclass(frozen=True)
class NodeCall:
    """A node reached on one path; depth zero is the first question."""

    cq_id: str
    history: History = ()
    mass: float = 1.0
    probabilities: tuple[float, ...] = ()

    @property
    def path(self) -> str:
        return ">".join(f"{cq}={answer}" for cq, answer in self.history)

    @property
    def depth(self) -> int:
        return len(self.history)


@dataclass(frozen=True)
class Expansion(Aggregation):
    """Terminal, abstention, pruned and pending masses partition the initial mass."""

    pruned: float
    pending: float
    frontier: tuple[NodeCall, ...]
    visited: tuple[NodeCall, ...]
    path_verdict: str | None


def node_masses(
    scheme: Scheme, cq: str, row: Mapping[str, Any] | None, source: str = LOGPROB
) -> dict[str, float]:
    """A parsed CQ row becomes arc masses; verbal confidence is already oriented."""
    return answer_masses(scheme, cq, row, source) or {OUTSIDE: 1.0}


def expand(
    scheme: Scheme,
    answers: Callable[[NodeCall], Masses | None],
    routing: Mapping[str, str],
    threshold: float = 0.02,
    transform: Transform | None = None,
) -> Expansion:
    """Expand known nodes breadth first; return missing nodes as the next frontier.

    A transformation receives a CQ id and its probabilities before burden routing
    and pruning. Each entry can supply its own transformation; None is identity.
    Only arcs to nodes are pruned, by their cumulative path mass. Reconverging
    paths retain separate histories and never pool their mass before pruning.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise SchemeError("the pruning threshold must be finite and between 0 and 1")
    queue = deque([NodeCall(scheme.start)])
    frontier, visited, completed = [], [], []
    terminals: dict[str, float] = {}
    abstention = pruned = pending = 0.0
    chosen = {}
    while queue:
        state = queue.popleft()
        visited.append(state)
        masses = answers(state)
        if masses is None:
            frontier.append(state)
            pending += state.mass
            continue
        if transform is not None:
            masses = transform(state.cq_id, dict(masses))
        allowed = {*scheme.nodes[state.cq_id], OUTSIDE}
        if (
            set(masses) - allowed
            or any(not math.isfinite(p) or p < 0 for p in masses.values())
            or not math.isclose(sum(masses.values()), 1.0, abs_tol=1e-9)
        ):
            raise SchemeError(f"{scheme.scheme_id} {state.cq_id}: invalid expansion masses")
        arcs, outside = _route(scheme, state.cq_id, masses, routing[state.cq_id])
        abstention += state.mass * outside
        best = max(arcs, key=arcs.get)
        if arcs[best] > 0:
            chosen[state.history] = (state.cq_id, best)
        for answer, probability in arcs.items():
            mass = state.mass * probability
            if mass == 0:
                continue
            history = (*state.history, (state.cq_id, answer))
            probabilities = (*state.probabilities, probability)
            edge = scheme.nodes[state.cq_id][answer]
            if edge.is_terminal:
                terminals[edge.target] = terminals.get(edge.target, 0.0) + mass
                completed.append((mass, history, probabilities, edge.target))
            elif mass >= threshold:
                queue.append(NodeCall(edge.target, history, mass, probabilities))
            else:
                pruned += mass
    greedy: History = ()
    path_verdict = None
    while greedy in chosen:
        cq, answer = chosen[greedy]
        greedy = (*greedy, (cq, answer))
        edge = scheme.nodes[cq][answer]
        if edge.is_terminal:
            path_verdict = edge.target
            break
    if completed:
        _, history, probabilities, terminal = max(completed, key=lambda entry: entry[0])
        path = (
            " > ".join(
                f"{cq}={answer} {p:.2f}"
                for (cq, answer), p in zip(history, probabilities, strict=True)
            )
            + f" > {terminal}"
        )
    else:
        path = "no path to a terminal"
    top = sorted(terminals, key=lambda t: (-terminals[t], t))[:3]
    shown = ", ".join(f"{t} {terminals[t]:.2f}" for t in top) or "none"
    explanation = (
        f"{path}; top: {shown}; abstention {abstention:.2f}; "
        f"pruned {pruned:.2f}; pending {pending:.2f}"
    )
    result = decide(terminals, abstention, explanation)
    return Expansion(
        result.verdict,
        result.confidence,
        result.distribution,
        result.abstention,
        result.explanation,
        pruned,
        pending,
        tuple(frontier),
        tuple(visited),
        path_verdict,
    )


def a2_pruned(
    scheme: Scheme,
    rows: Mapping[str, Mapping[str, Any]],
    routing: Mapping[str, str],
    source: str = LOGPROB,
    threshold: float = 0.02,
    transform: Transform | None = None,
) -> Expansion:
    """Independent CQ answers, with exactly the sequential expansion's pruning."""
    return expand(
        scheme,
        lambda state: node_masses(scheme, state.cq_id, rows.get(state.cq_id), source),
        routing,
        threshold,
        transform,
    )


def ssoft(
    scheme: Scheme,
    rows: Iterable[Mapping[str, Any]],
    routing: Mapping[str, str],
    source: str = LOGPROB,
    threshold: float = 0.02,
    history: bool = True,
    transform: Transform | None = None,
) -> Expansion:
    """One item's path rows from ssoft.csv, traversed with their recorded histories.

    Missing path rows remain pending rather than silently becoming abstentions.
    With history disabled every path reads the same independently asked CQ row.
    """
    indexed = {}
    for row in rows:
        path = row.get("path")
        key = (str(row["cq_id"]), path if history and isinstance(path, str) else "")
        if key in indexed:
            raise SchemeError(f"{scheme.scheme_id}: duplicate sequential CQ row {key}")
        indexed[key] = row

    def provide(state: NodeCall) -> Masses | None:
        row = indexed.get((state.cq_id, state.path if history else ""))
        return None if row is None else node_masses(scheme, state.cq_id, row, source)

    return expand(scheme, provide, routing, threshold, transform)
