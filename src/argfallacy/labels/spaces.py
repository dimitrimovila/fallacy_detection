"""Label spaces: the fine 25 terminals and the two collapsed variants.

The maps live in ``labels/fallacies.yaml`` under ``spaces``.  A terminal that no
space mentions maps to itself, so ``collapse`` is total over the 25 ids by
construction and adding a terminal never silently drops it from a space.
"""

from __future__ import annotations

from typing import Any

from ..schemes.loader import INVALID, SchemeError, load_vocabulary


def _space_map(space: str, vocab: dict[str, Any]) -> dict[str, str]:
    spaces = vocab.get("spaces", {})
    if space not in spaces:
        raise SchemeError(f"unknown label space {space!r}; known: {sorted(spaces)}")
    return spaces[space] or {}


def collapse(label_id: str, space: str = "fine", vocabulary: dict[str, Any] | None = None) -> str:
    """Map a predicted id into ``space``.  Ids the space does not mention stay put.

    Three kinds of id may be collapsed, because three kinds can be predicted:

    * a terminal, which the space either merges or leaves alone;
    * a family id, which is already a class of the collapsed spaces and is not a
      class of the fine one, so it maps to itself and matches nothing there;
    * ``invalid``, which maps to itself and is therefore wrong in every space,
      by construction, rather than being quietly dropped.

    A coarse or out-of-scope label is none of those.  It has no place in a label
    space and passing one is a bug, not a default.
    """
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    if label_id in vocab["terminals"]:
        return _space_map(space, vocab).get(label_id, label_id)
    if label_id in vocab.get("families", {}) or label_id == INVALID:
        _space_map(space, vocab)  # still refuse an unknown space name
        return label_id
    raise SchemeError(
        f"{label_id!r} is not a terminal, a family or {INVALID!r}, "
        f"so it has no place in a label space"
    )


def label_space(space: str = "fine", vocabulary: dict[str, Any] | None = None) -> list[str]:
    """The sorted ids of ``space``: 25 for ``fine``, 20 and 21 for the collapsed ones.

    Built from the terminals alone, so the family id is a member of the collapsed
    spaces (something merges into it) and not of the fine one.  ``invalid`` is
    never a member of any space: it is what a prediction is when it is not a class.
    """
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    mapping = _space_map(space, vocab)
    return sorted({mapping.get(tid, tid) for tid in vocab["terminals"]})
