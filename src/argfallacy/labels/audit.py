"""What a column of raw labels actually contains, and how it translates.

``audit`` is the one function here that never raises on a bad label: its whole
point is to show a column before anything downstream fails on it.  Everything
else in the package still fails loudly.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import pandas as pd

from ..schemes.loader import INVALID, SchemeError, load_vocabulary, normalize_label
from .vocabulary import light_normalize, load_scheme_vocabulary, normalize_scheme_label

Kind = Literal["fallacy", "scheme"]

UNKNOWN = "UNKNOWN"
INVALID_STATUS = INVALID
COLUMNS = ["raw", "count", "id", "status"]


def _markers(values: Iterable[str]) -> set[str]:
    return {light_normalize(v) for v in values}


def _classify_fallacy(raw: str, vocab: dict[str, Any]) -> tuple[str, str]:
    if light_normalize(raw) in _markers(vocab.get("missing_markers", [])):
        return UNKNOWN, "missing"
    try:
        label = normalize_label(raw, vocab)
    except SchemeError:
        return UNKNOWN, UNKNOWN
    return label.id, label.kind


def _classify_scheme(raw: str, vocab: dict[str, Any]) -> tuple[str, str]:
    special = vocab.get("special_values", {})
    key = light_normalize(raw)
    if key in _markers(special.get("missing", [])):
        return UNKNOWN, "missing"
    if key in _markers(special.get("idk", [])):
        return "idk", "idk"
    try:
        sid = normalize_scheme_label(raw, vocab)
    except SchemeError:
        return UNKNOWN, UNKNOWN
    return sid, "none" if sid == "none" else "scheme"


def audit(
    values: Iterable[Any],
    kind: Kind = "fallacy",
    vocabulary: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """One row per distinct raw string: the string, how often, its id, its status.

    Statuses for ``kind="fallacy"``: ``terminal``, ``coarse``, ``family``,
    ``out_of_scope``, ``idk``, ``invalid``, ``missing``, ``UNKNOWN``.  For
    ``kind="scheme"``: ``scheme``, ``none``, ``idk``, ``missing``, ``UNKNOWN``.

    ``UNKNOWN`` means the string translates to nothing and has to be decided on,
    either as a new alias in ``labels/`` or as data to leave alone.  ``invalid``
    means the dictionary knows the string and knows it is not a label: those are
    counted on their own by :func:`invalid_outputs`, since the share of a
    model's column they take up is a property of that model, not of the data.
    """
    if kind == "fallacy":
        vocab = vocabulary or load_vocabulary(check_schemes=False)
        classify = _classify_fallacy
    elif kind == "scheme":
        vocab = vocabulary or load_scheme_vocabulary()
        classify = _classify_scheme
    else:
        raise SchemeError(f"audit kind must be 'fallacy' or 'scheme', not {kind!r}")

    counts = pd.Series(list(values), dtype="object").astype(str).value_counts()
    rows = [
        {"raw": raw, "count": int(n), "id": ident, "status": status}
        for raw, n in counts.items()
        for ident, status in [classify(raw, vocab)]
    ]
    table = pd.DataFrame(rows, columns=COLUMNS)
    if table.empty:
        return table
    return table.sort_values(["count", "raw"], ascending=[False, True]).reset_index(drop=True)


def format_audit(table: pd.DataFrame) -> str:
    """The audit table as a fixed-width report, unknowns last so they are the punchline."""
    if table.empty:
        return "(empty column)"
    width = min(max(len(r) for r in table["raw"].astype(str)), 80)
    ordered = table.assign(_unknown=(table["status"] == UNKNOWN)).sort_values(
        ["_unknown", "count", "raw"], ascending=[True, False, True]
    )
    lines = [f"{'raw':<{width}}  {'count':>6}  {'id':<34}  status", "-" * (width + 52)]
    for row in ordered.itertuples():
        raw = str(row.raw)
        raw = raw if len(raw) <= width else raw[: width - 1] + "…"
        lines.append(f"{raw:<{width}}  {row.count:>6}  {str(row.id):<34}  {row.status}")
    return "\n".join(lines)


def invalid_outputs(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of an audit table that are not labels at all.

    Counted apart from everything else, per column and so per model: an invalid
    output is a prediction the model failed to make, and it stays in the
    denominator as a miss rather than being excluded from the run.
    """
    if table.empty:
        return table
    return table[table["status"] == INVALID_STATUS]


def invalid_share(table: pd.DataFrame) -> float:
    """Share of the audited rows that are invalid outputs.  Zero on an empty table."""
    total = int(table["count"].sum()) if not table.empty else 0
    if not total:
        return 0.0
    return int(invalid_outputs(table)["count"].sum()) / total
