"""Compat mode: reproduce the scorer of the prior runs, and nothing else.

A port of the two label normalizations of the scoring code that wrote the
``metrics.csv`` files of the prior runs, one for scheme labels and one for
fallacy labels.  It is not a normalization we believe in: no alias table, no
vocabulary lookup, no partial credit.  It exists so that the scorer can be shown
to count like that code on the same input.  Every thesis number comes from the
canonical path instead.

No label string is written here.  The ad hominem family and its members come
from the families, terminals and ``collapsed_compat`` space of
``labels/fallacies.yaml``; the other strings the prior code wrote into its own
source (the prefixes it strips, ``good argument``, the one spelling it repairs)
come from the ``compat`` block of the same file.
"""

from __future__ import annotations

import re
from typing import Any

from ..schemes.loader import SchemeError, load_vocabulary
from .vocabulary import light_normalize

COMPAT_SPACE = "collapsed_compat"
"""The label space whose family merge the collapsed row of the prior runs uses."""


def compat_normalize_scheme(raw_label: str, vocabulary: dict[str, Any] | None = None) -> str:
    """The scheme label normalization of the prior runs.

    Strip and lower-case; remove the leading prefixes of the ``compat`` block in
    order, each at most once; turn every run of ``_``, ``-`` and ``.`` into a
    space; squeeze whitespace; a label that starts with a ``replaced_by_prefix``
    key and a space becomes that entry's value.  Fallacy labels go through it
    first as well.
    """
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    block = vocab["compat"]

    text = str(raw_label).strip().lower()
    for prefix in block["removed_prefixes"]:
        words = r"\s+".join(re.escape(word) for word in prefix.split())
        text = re.sub(rf"^{words}\s+", "", text)
    text = re.sub(r"[_\-.]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for start, whole in block["replaced_by_prefix"].items():
        if text.startswith(f"{start} "):
            text = whole
    return text


def compat_normalize(
    raw_label: str,
    vocabulary: dict[str, Any] | None = None,
    space: str = "fine",
) -> str:
    """The fallacy label normalization of the prior runs.

    The scheme normalization first.  Then the text is matched against a prefix
    pattern (the family name, an optional ``(``, ``:`` or ``-``, the subtype, an
    optional ``)``) and, failing that, against the mirror suffix pattern; the
    captured subtype replaces the text only if it is one of the members of the
    family.  So ``ad hominem (tu quoque)`` and ``tu quoque ad hominem`` are both
    ``tu quoque``, while ``ad hominem red herring`` stays whole, red herring not
    being a member.  With ``space="collapsed_compat"`` the members that space
    merges become the family name.  Last, the ``rewrites`` of the compat block.

    The comparison in compat mode is plain equality between two strings put
    through this function.
    """
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    if space not in ("fine", COMPAT_SPACE):
        raise SchemeError(
            f"compat mode knows the spaces 'fine' and {COMPAT_SPACE!r}, not {space!r}"
        )
    family, patterns, members, merged = _family_rules(vocab)

    text = compat_normalize_scheme(raw_label, vocab)
    for pattern in patterns:
        match = pattern.match(text)
        if match and match.group(1).strip() in members:
            text = match.group(1).strip()
            break
    if space == COMPAT_SPACE and text in merged:
        text = family
    return vocab["compat"]["rewrites"].get(text, text)


def _family_rules(vocab: dict[str, Any]) -> tuple[str, tuple[re.Pattern, ...], set, set]:
    """Family name, the two affix patterns, the members, the members the space merges."""
    merge = vocab["spaces"][COMPAT_SPACE]
    targets = set(merge.values())
    if len(targets) != 1:
        raise SchemeError(
            f"space {COMPAT_SPACE!r} merges into {sorted(targets)}, expected one family"
        )
    family_id = targets.pop()
    family = light_normalize(vocab["families"][family_id]["label"])
    name = re.escape(family)
    patterns = (
        re.compile(rf"^{name}\s*[\(:\-]?\s*(.+?)\s*\)?$"),
        re.compile(rf"^(.+?)\s*[\(:\-]?\s*{name}\)?$"),
    )

    # A member's subtype is its label in compat form with the family name taken off,
    # by the same two patterns: "ad hominem (abusive)" -> "abusive", "ad fidentia" as is.
    subtype: dict[str, str] = {}
    for tid, spec in vocab["terminals"].items():
        if spec.get("family") != family_id:
            continue
        text = compat_normalize_scheme(spec["label"], vocab)
        for pattern in patterns:
            if match := pattern.match(text):
                text = match.group(1).strip()
                break
        subtype[tid] = text
    return family, patterns, set(subtype.values()), {subtype[tid] for tid in merge}
