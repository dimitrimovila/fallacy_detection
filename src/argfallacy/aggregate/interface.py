"""What every aggregator returns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

TIE = 1e-12
"""Two masses closer than this are equal."""


@dataclass(frozen=True)
class Aggregation:
    """The outcome of one aggregator on one item.

    ``distribution`` maps terminals to their mass, and ``abstention`` is what is
    missing to 1.  ``verdict`` is the terminal with the highest mass, a tie going to
    the first terminal id in alphabetical order, or None when no terminal has mass.
    ``confidence`` is the mass of the verdict, None for the aggregators that give a
    single point.
    """

    verdict: str | None
    confidence: float | None
    distribution: dict[str, float]
    abstention: float
    explanation: str


def top_terminal(distribution: Mapping[str, float]) -> str | None:
    """The terminal with the highest mass, ties to the first id in alphabetical order."""
    positive = {terminal: mass for terminal, mass in distribution.items() if mass > 0}
    if not positive:
        return None
    best = max(positive.values())
    return min(terminal for terminal, mass in positive.items() if mass >= best - TIE)


def decide(distribution: Mapping[str, float], abstention: float,
           explanation: str) -> Aggregation:
    """The verdict of a distribution, with its mass as the confidence."""
    verdict = top_terminal(distribution)
    return Aggregation(
        verdict=verdict,
        confidence=None if verdict is None else float(distribution[verdict]),
        distribution=dict(distribution),
        abstention=float(abstention),
        explanation=explanation,
    )
