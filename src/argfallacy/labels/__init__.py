"""The label dictionary: one place where a raw string becomes an id.

Fallacy labels come from ``labels/fallacies.yaml``, scheme labels from
``labels/schemes.yaml``.  Nothing here guesses: an unknown string raises
``SchemeError`` naming the string, except inside :func:`audit`, whose job is to
report unknowns rather than trip over them.
"""

from ..schemes.loader import (
    INVALID,
    Label,
    Resolution,
    SchemeError,
    family_members,
    label_matches,
    load_vocabulary,
    normalize_label,
    resolve_for_scheme,
)
from .audit import UNKNOWN, audit, format_audit, invalid_outputs, invalid_share
from .compat import COMPAT_SPACE, compat_merge_map, compat_normalize
from .spaces import collapse, label_space
from .vocabulary import (
    light_normalize,
    load_scheme_vocabulary,
    normalize_scheme_label,
    scheme_ids,
)

__all__ = [
    "SchemeError", "Label", "Resolution", "INVALID",
    "load_vocabulary", "normalize_label", "resolve_for_scheme", "label_matches",
    "family_members",
    "load_scheme_vocabulary", "normalize_scheme_label", "scheme_ids",
    "light_normalize",
    "label_space", "collapse",
    "compat_normalize", "compat_merge_map", "COMPAT_SPACE",
    "audit", "format_audit", "UNKNOWN", "invalid_outputs", "invalid_share",
]
