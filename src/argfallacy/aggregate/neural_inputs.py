"""Label-free, scheme-specific CQ features for a pooled neural classifier."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from ..labels import INVALID
from ..prompts import CANNOT_BE_DETERMINED, NOT_APPLICABLE
from ..schemes import Scheme, SchemeError
from .soft import VERBAL, sources_of


@dataclass(frozen=True)
class NeuralInputs:
    item_ids: tuple[str, ...]
    scheme_ids: tuple[str, ...]
    names: tuple[str, ...]
    x: np.ndarray
    audit: dict[str, int]


def _probability(value, context):
    if value is None or value == "" or pd.isna(value):
        return None
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise SchemeError(f"{context}: invalid probability {value}")
    return value


def build_neural_inputs(
    schemes: Mapping[str, Scheme],
    items: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: Mapping,
    entries: Sequence[str],
) -> NeuralInputs:
    """Encode hard answers and available soft signals, without reading verdicts."""
    entries = tuple(sorted(entries))
    if not entries or len(set(entries)) != len(entries):
        raise SchemeError("neural entries must be nonempty and unique")
    if unknown := set(entries) - set(manifest["models"]):
        raise SchemeError(f"unknown neural entries: {sorted(unknown)}")
    own = items.sort_values("item_id").reset_index(drop=True)
    if own.empty or own.item_id.duplicated().any():
        raise SchemeError("neural items must be nonempty with unique IDs")
    if unknown := set(own.gold_scheme) - set(schemes):
        raise SchemeError(f"unknown neural schemes: {sorted(unknown)}")
    sources = sources_of(manifest, list(entries))
    names = [f"scheme/{s}" for s in sorted(schemes)]
    offsets = {}
    for sid, scheme in sorted(schemes.items()):
        for cq in scheme.cq_order:
            options = (*scheme.answer_space(cq), CANNOT_BE_DETERMINED, NOT_APPLICABLE)
            identity = sha256((scheme.schema + scheme.cqs[cq].text).encode()).hexdigest()[:12]
            for entry in entries:
                offsets[sid, cq, entry] = len(names)
                channels = [
                    *(f"hard/{a}" for a in options),
                    "invalid",
                    "missing",
                    *(f"soft/{a}" for a in options),
                    "soft_available",
                ]
                names.extend(
                    f"{sid}/{cq}@{identity}/{entry}/{sources[entry]}/{c}" for c in channels
                )
    selected = summary[
        summary.stage.eq("stage2")
        & summary.scheme_condition.eq("gold")
        & summary.item_id.isin(own.item_id)
        & summary.model.isin(entries)
    ]
    keys = ["item_id", "scheme", "cq_id", "model"]
    if selected.duplicated(keys).any():
        raise SchemeError(
            "duplicate neural CQ rows: "
            + str(selected.loc[selected.duplicated(keys, keep=False), keys].iloc[0].to_dict())
        )
    item_schemes = dict(zip(own.item_id, own.gold_scheme, strict=True))
    rows = {}
    for row in selected.to_dict("records"):
        key = tuple(row[k] for k in keys)
        if row["scheme"] != item_schemes[row["item_id"]]:
            raise SchemeError(f"neural CQ scheme mismatch: {key}")
        if tuple(key[1:]) not in offsets:
            raise SchemeError(f"unknown neural CQ: {key}")
        rows[key] = row
    x = np.zeros((len(own), len(names)), dtype=np.float32)
    audit = {"expected": 0, "missing": 0, "invalid": 0, "soft_available": 0}
    for i, item in enumerate(own.itertuples(index=False)):
        sid = item.gold_scheme
        x[i, names.index(f"scheme/{sid}")] = 1
        for cq in schemes[sid].cq_order:
            options = (*schemes[sid].answer_space(cq), CANNOT_BE_DETERMINED, NOT_APPLICABLE)
            for entry in entries:
                audit["expected"] += 1
                start = offsets[sid, cq, entry]
                row = rows.get((item.item_id, sid, cq, entry))
                if row is None:
                    x[i, start + 5] = 1
                    audit["missing"] += 1
                    continue
                context = f"{item.item_id}/{sid}/{cq}/{entry}"
                hard = row.get("answer")
                if pd.isna(hard) or hard in (INVALID, ""):
                    x[i, start + 4] = 1
                    audit["invalid"] += 1
                elif hard in options:
                    x[i, start + options.index(hard)] = 1
                else:
                    raise SchemeError(f"{context}: unknown hard answer {hard!r}")
                source = sources[entry]
                if source == VERBAL:
                    p = _probability(row.get("p_verbal"), context)
                    values = [p, 1 - p, 0.0, 0.0] if p is not None else None
                else:
                    values = [_probability(row.get(f"p_{source}_{a}"), context) for a in options]
                    if all(v is None for v in values):
                        values = None
                    elif any(v is None for v in values):
                        raise SchemeError(f"{context}: incomplete soft distribution")
                if values is not None:
                    if not np.isclose(sum(values), 1.0, atol=1e-6):
                        raise SchemeError(f"{context}: soft distribution does not sum to one")
                    x[i, start + 6 : start + 10] = values
                    x[i, start + 10] = 1
                    audit["soft_available"] += 1
    return NeuralInputs(tuple(own.item_id), tuple(own.gold_scheme), tuple(names), x, audit)
