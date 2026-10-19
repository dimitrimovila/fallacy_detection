"""The aggregators: from the answers to the critical questions of an item to a verdict.

Every aggregator takes the scheme and the rows of ``summary.csv`` of the item and
returns an :class:`Aggregation`: a distribution over the terminals, the abstention,
the verdict, its confidence and an explanation.
"""

from .burden import (
    ABSTAIN,
    BURDEN_DIR,
    POLICIES,
    SPLIT,
    TYPES,
    load_burden,
    routes,
)
from .diagram import a0, good_terminal, pass_arc
from .interface import Aggregation, decide, top_terminal
from .soft import (
    LOGPROB,
    OUTSIDE,
    SAMPLE,
    SOURCES,
    VERBAL,
    a2,
    a4,
    answer_masses,
    mean_masses,
    source_of,
    sources_of,
    traverse_soft,
)

__all__ = [
    "Aggregation", "decide", "top_terminal",
    "a0", "a2", "a4", "pass_arc", "good_terminal",
    "load_burden", "routes", "BURDEN_DIR", "TYPES", "POLICIES", "ABSTAIN", "SPLIT",
    "answer_masses", "mean_masses", "traverse_soft", "source_of", "sources_of",
    "SOURCES", "LOGPROB", "SAMPLE", "VERBAL", "OUTSIDE",
]
