"""A decision rule for macro F1: each terminal's mass divided by the frequency of its class.

The terminal with the highest mass maximises the expected accuracy when the masses are
posteriors; dividing each mass by the prior of its class gives the rule that maximises
the expected balanced accuracy, which weighs the classes as macro F1 does.  The masses
of the diagram are not posteriors, so here the rule is an empirical check of how much
the choice of the verdict, rather than the masses, costs on the rare classes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from ..schemes import Scheme
from .interface import Aggregation, decide


def class_priors(
    scheme: Scheme, gold: Mapping[str, str], training: list[str]
) -> dict[str, float]:
    """The frequency of each terminal among the gold verdicts of ``training``, with one
    added to every count so that no terminal has frequency 0; a gold verdict that is not
    a terminal of the scheme is not counted."""
    counts = Counter(gold[i] for i in training if gold[i] in scheme.terminals)
    total = sum(counts.values()) + len(scheme.terminals)
    return {t: (counts[t] + 1) / total for t in scheme.terminals}


def prior_corrected(
    aggregations: Mapping[str, Aggregation],
    schemes: Mapping[str, Scheme],
    scheme_of: Mapping[str, str],
    gold: Mapping[str, str],
    folds: Mapping[str, int],
) -> dict[str, Aggregation]:
    """Every aggregation with its terminal masses divided by the class frequency of the
    items of the same scheme in the other folds, and rescaled to their former sum, so
    that the abstention does not change.  An item with no terminal mass stays as it is."""
    out = {}
    for item, aggregation in aggregations.items():
        scheme = schemes[scheme_of[item]]
        mass = sum(aggregation.distribution.values())
        if mass <= 0:
            out[item] = aggregation
            continue
        training = [i for i in aggregations
                    if folds[i] != folds[item] and scheme_of[i] == scheme.scheme_id]
        priors = class_priors(scheme, gold, training)
        divided = {t: m / priors[t] for t, m in aggregation.distribution.items()}
        scale = mass / sum(divided.values())
        corrected = {t: m * scale for t, m in divided.items()}
        result = decide(corrected, aggregation.abstention, "")
        out[item] = decide(
            corrected,
            aggregation.abstention,
            f"divided by the class frequency: {result.verdict} {result.confidence:.2f}; "
            f"{aggregation.explanation}",
        )
    return out
