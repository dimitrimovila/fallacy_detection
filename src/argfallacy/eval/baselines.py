"""The two comparison lines: majority per scheme and the information ceiling.

Both are the same construction with a different key.  Every item gets the most
frequent gold label among the items with its key, computed on the same items:

* key = scheme: the majority per scheme, an oracle on the prior;
* key = (scheme, answer vector): the information ceiling, an oracle in sample,
  the best accuracy any function of the answer vector can reach on these items.

The result is one prediction per item, scored by the canonical scorer like any
other prediction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence


def majority_by(keys: Sequence[Hashable], gold: Sequence[str]) -> list[str]:
    """For every item, the most frequent gold label among the items with its key.

    Ties go to the first label id in alphabetical order.
    """
    if len(keys) != len(gold):
        raise ValueError(f"{len(keys)} keys but {len(gold)} gold labels")
    counts: dict[Hashable, Counter] = defaultdict(Counter)
    for key, label in zip(keys, gold, strict=True):
        counts[key][label] += 1
    best = {key: min(c, key=lambda label, c=c: (-c[label], label)) for key, c in counts.items()}
    return [best[key] for key in keys]
