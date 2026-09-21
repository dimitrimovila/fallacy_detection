"""Compat mode: reproduce the scorer of the prior runs, and nothing else.

Kept in one function on purpose.  It is not a normalization we believe in — it
has no alias table, no partial credit for coarse labels, no vocabulary lookup at
all — it is a string transform whose only job is to make our numbers line up
with the ``metrics.csv`` files already written, so that any later difference is
attributable to the aggregators and not to the scorer.  Every thesis number
comes from the canonical path instead.

The affixes are not written here: they are read from the family that the
``collapsed_compat`` space merges into, so the only place a label string exists
is still ``labels/fallacies.yaml``.
"""

from __future__ import annotations

from typing import Any

from ..schemes.loader import SchemeError, load_vocabulary
from .vocabulary import light_normalize

COMPAT_SPACE = "collapsed_compat"
"""The label space whose family merge the collapsed row of the prior runs uses."""


def _compat_family_label(vocab: dict[str, Any]) -> str:
    """The one family the compat scorer knew about, taken from the space it merges."""
    targets = set(vocab["spaces"][COMPAT_SPACE].values())
    if len(targets) != 1:
        raise SchemeError(
            f"space {COMPAT_SPACE!r} merges into {sorted(targets)}, expected one family"
        )
    return light_normalize(vocab["families"][targets.pop()]["label"])


def compat_normalize(
    raw_label: str,
    vocabulary: dict[str, Any] | None = None,
    space: str = "fine",
) -> str:
    """The label normalization of the prior runs, as a string transform.

    Lower-case, then exactly one of three affix rules keyed on the family name
    (``ad hominem``), then strip:

    * ``ad hominem (X)``  -> ``X``
    * ``X ad hominem``    -> ``X``
    * ``ad hominem X``    -> ``X``

    The comparison in compat mode is plain equality between two strings put
    through this function.  Nothing else: no aliases, so ``casual reductionism``
    stays a miss, and no partial credit, so a coarse gold label matches nothing.

    ``space="collapsed_compat"`` additionally merges the ad hominem members that
    the collapsed row of the prior runs merges, five of the six: one member is
    left on its own there, which is why the repository also carries a symmetric
    space.

    The third rule (prefix) looks redundant next to the second.  It is there
    for one reason and one reason only: with the first two rules alone, the
    reproduction is exact on seven of the eight
    ``results.csv`` files and off by 0.0577 on ``qwen3.8_27b/zero-shot``; with
    the third, all eight ``final_label`` rows and all eight ``final_label
    (collapsed)`` rows of the ``metrics.csv`` files match.

    That is the whole justification.  The scoring code of the prior runs was
    never read, so this rule is not known to be the rule it applies, only to be
    a rule that produces its numbers.  Nothing else was checked: not whether it
    is what its author intended, not whether some other transform would fit the same eight numbers,
    not whether it generalises to a ninth run.  It is a fit to eight data points,
    and it should be believed exactly that far.
    """
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    family = _compat_family_label(vocab)

    text = str(raw_label).lower()
    if text.startswith(f"{family} (") and text.endswith(")"):
        text = text[len(family) + 2 : -1]
    elif text.endswith(f" {family}"):
        text = text[: -len(family) - 1]
    elif text.startswith(f"{family} "):
        text = text[len(family) + 1 :]
    text = text.strip()

    if space == "fine":
        return text
    if space == COMPAT_SPACE:
        return compat_merge_map(vocab).get(text, text)
    raise SchemeError(f"compat mode knows the spaces 'fine' and {COMPAT_SPACE!r}, not {space!r}")


def compat_merge_map(vocabulary: dict[str, Any] | None = None) -> dict[str, str]:
    """Compat form of each merged terminal -> compat form of its family label."""
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    family = _compat_family_label(vocab)
    merged: dict[str, str] = {}
    for tid, target in vocab["spaces"][COMPAT_SPACE].items():
        key = compat_normalize(vocab["terminals"][tid]["label"], vocab)
        merged[key] = light_normalize(vocab["families"][target]["label"])
    assert all(v == family for v in merged.values())
    return merged
