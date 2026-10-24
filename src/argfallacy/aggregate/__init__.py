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
from .interface import Aggregation, decide, macro_f1, top_terminal
from .order import (
    Layout,
    Order,
    a1_out_of_fold,
    choose_order,
    derive_blocks,
    original_order,
    rebuild,
)
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
from .ssoft import Expansion, NodeCall, a2_pruned, expand, ssoft


def __getattr__(name):
    # Serving and parsing need traversal, but never import the learned models.
    if name in {"C_GRID", "a3_out_of_fold", "choose_c", "features"}:
        from . import checklist

        return getattr(checklist, name)
    raise AttributeError(name)

__all__ = [
    "Aggregation", "decide", "top_terminal", "macro_f1",
    "a0", "a2", "a4", "pass_arc", "good_terminal",
    "Layout", "Order", "derive_blocks", "original_order", "rebuild", "choose_order",
    "a1_out_of_fold", "a3_out_of_fold", "choose_c", "features", "C_GRID",
    "load_burden", "routes", "BURDEN_DIR", "TYPES", "POLICIES", "ABSTAIN", "SPLIT",
    "answer_masses", "mean_masses", "traverse_soft", "source_of", "sources_of",
    "SOURCES", "LOGPROB", "SAMPLE", "VERBAL", "OUTSIDE",
    "Expansion", "NodeCall", "expand", "a2_pruned", "ssoft",
]
