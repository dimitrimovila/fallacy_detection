"""A2 and A4: the diagram traversed with soft answers.

At every node the answers of an entry become masses on the two arcs and a mass
outside them (``cannot_be_determined`` plus ``na``), which the burden of proof of
the CQ routes.  A mass of 1 enters the first node and collects on the terminals
and in the abstention; the answers to different CQs are treated as independent
given the path, as in ``propagate``, which this routing equals when no mass is
outside the arcs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from ..prompts import CANNOT_BE_DETERMINED, NOT_APPLICABLE
from ..schemes import Scheme
from .burden import ABSTAIN, SPLIT
from .interface import Aggregation, decide

OUTSIDE = "outside"
"""The key of the mass outside the arcs."""
OUTSIDE_ARCS = (CANNOT_BE_DETERMINED, NOT_APPLICABLE)

LOGPROB, SAMPLE, VERBAL = "logprob", "sample", "verbal"
SOURCES = (LOGPROB, SAMPLE, VERBAL)
"""Where an entry's soft answer comes from: ``p_logprob_<answer>``, ``p_sample_<answer>``
or the stated confidence ``p_verbal``."""
_COLUMNS = {LOGPROB: "p_logprob", SAMPLE: "p_sample"}

Rows = Mapping[str, Mapping[str, Any]]
"""The rows of ``summary.csv`` of one entry and one item, by CQ id."""


def _number(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def source_of(model: Mapping[str, Any]) -> str:
    """The soft answer of an entry, by its regime in the manifest of the run.

    Several samples: the shares.  One sample with reasoning off: the logprobs.  One
    sample with reasoning: the stated confidence, since after a reasoning that has
    already decided the logprob of the answer goes to the extremes.
    """
    if int(model.get("samples", 1)) > 1:
        return SAMPLE
    reasoning = model.get("reasoning") or {}
    kwargs = reasoning.get("chat_template_kwargs") or {}
    thinking = reasoning.get("enable_thinking", kwargs.get("enable_thinking"))
    return LOGPROB if thinking is False else VERBAL


def sources_of(manifest: Mapping[str, Any], entries: list[str] | None = None) -> dict[str, str]:
    """The source of every entry of a run, from ``manifest.json``."""
    models = manifest["models"]
    return {entry: source_of(models[entry]) for entry in (entries or list(models))}


def answer_masses(
    scheme: Scheme, cq: str, row: Mapping[str, Any] | None, source: str
) -> dict[str, float] | None:
    """The masses of one entry at one CQ: its two arcs, and ``OUTSIDE``.

    A CQ without masses takes its hard answer as a mass of 1.  An invalid hard
    answer with no masses, or no row at all, gives None.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; known: {list(SOURCES)}")
    if row is None:
        return None
    arcs = list(scheme.answer_space(cq))
    hard = row.get("answer")
    masses = None
    if source == VERBAL:
        stated = _number(row.get("p_verbal"))
        if hard in OUTSIDE_ARCS:
            masses = {OUTSIDE: 1.0}
        elif hard in arcs and stated is not None:
            masses = {arcs[0]: stated, arcs[1]: 1.0 - stated}
    else:
        values = {answer: _number(row.get(f"{_COLUMNS[source]}_{answer}"))
                  for answer in (*arcs, *OUTSIDE_ARCS)}
        if all(value is not None for value in values.values()) and sum(values.values()) > 0:
            masses = {answer: values[answer] for answer in arcs}
            masses[OUTSIDE] = sum(values[answer] for answer in OUTSIDE_ARCS)
    if masses is None:
        if hard in arcs:
            return {hard: 1.0}
        if hard in OUTSIDE_ARCS:
            return {OUTSIDE: 1.0}
        return None
    total = sum(masses.values())
    return {key: value / total for key, value in masses.items()}


def mean_masses(
    scheme: Scheme, rows_by_entry: Mapping[str, Rows], sources: Mapping[str, str]
) -> dict[str, dict[str, float]]:
    """At every CQ, the mean with equal weights of the masses of the entries.

    An entry whose answer is invalid with no masses is left out of the mean; if
    every entry is, the whole mass is outside the arcs.
    """
    out = {}
    for cq in scheme.nodes:
        kept = [masses for entry, rows in rows_by_entry.items()
                if (masses := answer_masses(scheme, cq, rows.get(cq), sources[entry]))
                is not None]
        if not kept:
            out[cq] = {OUTSIDE: 1.0}
            continue
        keys = {key for masses in kept for key in masses}
        out[cq] = {key: sum(m.get(key, 0.0) for m in kept) / len(kept) for key in keys}
    return out


def _route(scheme: Scheme, node: str, masses: Mapping[str, float],
           route: str) -> tuple[dict[str, float], float]:
    """The probabilities of the arcs of ``node`` after routing, and the mass that abstains."""
    arcs = {answer: float(masses.get(answer, 0.0)) for answer in scheme.nodes[node]}
    outside = float(masses.get(OUTSIDE, 0.0))
    if route == ABSTAIN:
        return arcs, outside
    if route == SPLIT:
        return {answer: p + outside / len(arcs) for answer, p in arcs.items()}, 0.0
    arcs[route] += outside
    return arcs, 0.0


def traverse_soft(
    scheme: Scheme, masses: Mapping[str, Mapping[str, float]], routing: Mapping[str, str]
) -> tuple[dict[str, float], float, dict[str, dict[str, float]]]:
    """The terminal masses, the abstention, and the routed probabilities of every arc."""
    routed = {node: _route(scheme, node, masses[node], routing[node]) for node in scheme.nodes}
    memo: dict[str, tuple[dict[str, float], float]] = {}

    def flow(node: str) -> tuple[dict[str, float], float]:
        if node in memo:
            return memo[node]
        arcs, abstention = routed[node]
        terminals: dict[str, float] = {}
        for answer, weight in arcs.items():
            if weight == 0.0:
                continue
            edge = scheme.nodes[node][answer]
            if edge.is_terminal:
                terminals[edge.target] = terminals.get(edge.target, 0.0) + weight
                continue
            below, below_abstention = flow(edge.target)
            for terminal, mass in below.items():
                terminals[terminal] = terminals.get(terminal, 0.0) + weight * mass
            abstention += weight * below_abstention
        memo[node] = (terminals, abstention)
        return memo[node]

    terminals, abstention = flow(scheme.start)
    return terminals, abstention, {node: arcs for node, (arcs, _) in routed.items()}


def _most_probable_path(
    scheme: Scheme, arcs: Mapping[str, Mapping[str, float]]
) -> tuple[float, list[tuple[str, str, float]], str]:
    """The path to a terminal with the highest product of probabilities; ties to the
    arc written first."""
    memo: dict[str, tuple[float, list[tuple[str, str, float]], str]] = {}

    def best(node: str) -> tuple[float, list[tuple[str, str, float]], str]:
        if node not in memo:
            found = None
            for answer, edge in scheme.nodes[node].items():
                weight = arcs[node][answer]
                if edge.is_terminal:
                    candidate = (weight, [(node, answer, weight)], edge.target)
                else:
                    below, steps, end = best(edge.target)
                    candidate = (weight * below, [(node, answer, weight), *steps], end)
                if found is None or candidate[0] > found[0]:
                    found = candidate
            memo[node] = found
        return memo[node]

    return best(scheme.start)


def _explain(scheme: Scheme, terminals: Mapping[str, float], abstention: float,
             arcs: Mapping[str, Mapping[str, float]],
             note: Callable[[str], str] | None = None) -> str:
    probability, steps, end = _most_probable_path(scheme, arcs)
    if probability > 0:
        path = " > ".join(f"{node}={answer} {p:.2f}{note(node) if note else ''}"
                          for node, answer, p in steps) + f" > {end}"
    else:
        path = "no path to a terminal"
    top = sorted(((m, t) for t, m in terminals.items() if m > 0), key=lambda x: (-x[0], x[1]))
    tops = ", ".join(f"{terminal} {mass:.2f}" for mass, terminal in top[:3]) or "none"
    return f"{path}; top: {tops}; abstention {abstention:.2f}"


def a2(scheme: Scheme, rows: Rows, routing: Mapping[str, str],
       source: str = LOGPROB) -> Aggregation:
    """The diagram traversed with the soft answers of one entry.

    An invalid answer with no masses, or a missing CQ, puts all the mass of that
    CQ outside the arcs.  The explanation is the most probable path, with the
    probability of each arc, and the three terminals with the highest mass.
    """
    masses = {cq: answer_masses(scheme, cq, rows.get(cq), source) or {OUTSIDE: 1.0}
              for cq in scheme.nodes}
    terminals, abstention, arcs = traverse_soft(scheme, masses, routing)
    return decide(terminals, abstention, _explain(scheme, terminals, abstention, arcs))


def a4(scheme: Scheme, rows_by_entry: Mapping[str, Rows], sources: Mapping[str, str],
       routing: Mapping[str, str]) -> Aggregation:
    """The traversal of A2 on the mean, with equal weights, of the masses of several entries.

    The explanation is that of A2, with the hard answers of the entries at each node
    of the most probable path.
    """
    masses = mean_masses(scheme, rows_by_entry, sources)
    terminals, abstention, arcs = traverse_soft(scheme, masses, routing)

    def note(node: str) -> str:
        given = [f"{entry} {(rows.get(node) or {}).get('answer', 'missing')}"
                 for entry, rows in rows_by_entry.items()]
        return f" ({', '.join(given)})"

    return decide(terminals, abstention, _explain(scheme, terminals, abstention, arcs, note))
