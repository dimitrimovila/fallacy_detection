"""Label-free CQ distributions, conditional signals, and stacking features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit

from ..schemes import Scheme, SchemeError
from .soft import LOGPROB, OUTSIDE, SAMPLE, VERBAL, answer_masses, sources_of


@dataclass(frozen=True)
class Signal:
    entry: str
    source: str


@dataclass(frozen=True)
class B1Dataset:
    """One row per item/CQ; entries and signals keep a stable entry/source order.

    Observation codes: 0 first arc, 1 second arc, 2 outside, -1 omitted.
    Binary targets use 1 for the first arc. Splits are missing when their
    signal has no value; base_splits additionally retain A4's hard fallback.
    """

    schemes: Mapping[str, Scheme]
    routing: Mapping[str, Mapping[str, str]]
    items: pd.DataFrame
    pairs: pd.DataFrame
    entries: tuple[str, ...]
    sources: dict[str, str]
    reference: str
    signals: tuple[Signal, ...]
    primary_signals: tuple[int, ...]
    splits: np.ndarray
    base_splits: np.ndarray
    arc_mass: np.ndarray
    outside: np.ndarray
    observations: np.ndarray
    hard: np.ndarray
    x: np.ndarray

    # The pairs never change after construction, so what is derived from them is
    # computed once: nested cross-fitting reads it thousands of times.
    @cached_property
    def pair_keys(self) -> tuple[tuple[str, str, str], ...]:
        """(item, scheme, CQ) of every pair, in order."""
        return tuple(zip(self.pairs.item_id, self.pairs.scheme, self.pairs.cq_id, strict=True))

    @cached_property
    def groups(self) -> tuple[tuple[str, str, tuple[tuple[int, str], ...]], ...]:
        """Each item, in order of first appearance, with its scheme and its (pair, CQ) list."""
        out: dict[str, tuple[str, list]] = {}
        for index, (item, scheme, cq) in enumerate(self.pair_keys):
            out.setdefault(item, (scheme, []))[1].append((index, cq))
        return tuple((item, scheme, tuple(rows)) for item, (scheme, rows) in out.items())

    @cached_property
    def questions(self) -> tuple[tuple[str, str], ...]:
        return tuple(dict.fromkeys((scheme, cq) for _, scheme, cq in self.pair_keys))

    @cached_property
    def question_index(self) -> np.ndarray:
        index = {q: i for i, q in enumerate(self.questions)}
        return np.array([index[(scheme, cq)] for _, scheme, cq in self.pair_keys])

    def with_features(self) -> B1Dataset:
        present = np.isfinite(self.splits)
        logits = np.zeros_like(self.splits)
        logits[present] = logit(np.clip(self.splits[present], 0.005, 0.995))
        return replace(self, x=np.column_stack([np.ones(len(self.pairs)), logits, ~present]))

    def pair_indices(self, ids: Sequence[str]) -> np.ndarray:
        return np.flatnonzero(self.pairs.item_id.isin(set(ids)))

    def targets(self, labels: pd.DataFrame | None) -> np.ndarray:
        target = np.full(len(self.pairs), np.nan)
        if labels is None or labels.empty:
            return target
        keys = list(zip(labels.item_id, labels.scheme, labels.cq_id, strict=True))
        if len(set(keys)) != len(keys):
            raise SchemeError("duplicate CQ labels need explicit provenance")
        if not labels.target.isin([0, 1]).all():
            raise SchemeError("CQ training targets must be binary")
        lookup = dict(zip(keys, labels.target, strict=True))
        return np.array([lookup.get(key, np.nan) for key in self.pair_keys], dtype=float)


def _probability(value, context):
    if value is None or pd.isna(value):
        return np.nan
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise SchemeError(f"{context}: probability outside [0, 1]: {value}")
    return value


def _signal(row, source, arcs, context):
    if source == VERBAL:
        return _probability(row.get("p_verbal"), context)
    a = _probability(row.get(f"p_{source}_{arcs[0]}"), context)
    b = _probability(row.get(f"p_{source}_{arcs[1]}"), context)
    return a / (a + b) if np.isfinite(a + b) and a + b > 0 else np.nan


def build_dataset(
    schemes: Mapping[str, Scheme],
    items: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: Mapping[str, Any],
    reference: str,
    routing: Mapping[str, Mapping[str, str]],
) -> B1Dataset:
    """Read supplied frames only; no labels are consulted and no run is opened."""
    items = items.sort_values("item_id").reset_index(drop=True).copy()
    if items.item_id.duplicated().any():
        raise SchemeError("duplicate B1 item ids")
    entries = tuple(sorted(manifest["models"]))
    if len(entries) != 5 or reference not in entries:
        raise SchemeError("B1 requires five entries including its reference model")
    sources = sources_of(manifest)
    signals = tuple(
        Signal(entry, source)
        for entry in entries
        for source in (LOGPROB, SAMPLE, VERBAL)
        if source != SAMPLE or int(manifest["models"][entry].get("samples", 1)) > 1
    )
    primary = tuple(signals.index(Signal(entry, sources[entry])) for entry in entries)
    own = summary[
        (summary.stage == "stage2")
        & (summary.scheme_condition == "gold")
        & summary.item_id.isin(items.item_id)
        & summary.model.isin(entries)
    ]
    keys = ["item_id", "scheme", "cq_id", "model"]
    if own.duplicated(keys).any():
        row = own.loc[own.duplicated(keys, keep=False), keys].iloc[0]
        raise SchemeError(
            f"duplicate model CQ row: {row.item_id} {row.scheme} {row.cq_id} {row.model}"
        )
    gold = items.set_index("item_id").gold_scheme
    for row in own.itertuples():
        if (
            row.scheme != gold[row.item_id]
            or row.scheme not in schemes
            or row.cq_id not in schemes[row.scheme].cqs
        ):
            raise SchemeError(f"{row.item_id}: incompatible scheme/CQ {row.scheme} {row.cq_id}")
    lookup = {
        (r["item_id"], r["scheme"], r["cq_id"], r["model"]): r for r in own.to_dict("records")
    }
    pairs = []
    for item in items.itertuples(index=False):
        if item.gold_scheme not in schemes:
            raise SchemeError(f"{item.item_id}: unknown gold scheme {item.gold_scheme!r}")
        scheme = schemes[item.gold_scheme]
        if set(routing[scheme.scheme_id]) != set(scheme.cqs):
            raise SchemeError(f"{scheme.scheme_id}: incomplete B1 routing")
        pairs.extend(
            {"item_id": item.item_id, "scheme": scheme.scheme_id, "cq_id": cq}
            for cq in scheme.cq_order
        )
    pairs = pd.DataFrame(pairs, columns=["item_id", "scheme", "cq_id"])
    n, e = len(pairs), len(entries)
    splits, base = np.full((n, len(signals)), np.nan), np.full((n, e), 0.5)
    arc_mass, outside = np.zeros((n, e)), np.ones(n)
    observations, hard = np.full((n, e), -1, dtype=int), np.full((n, e), None, dtype=object)
    for i, pair in enumerate(pairs.itertuples(index=False)):
        scheme, kept_outside = schemes[pair.scheme], []
        arcs = scheme.answer_space(pair.cq_id)
        for j, entry in enumerate(entries):
            row = lookup.get((pair.item_id, pair.scheme, pair.cq_id, entry))
            if row is None:
                continue
            masses = answer_masses(scheme, pair.cq_id, row, sources[entry])
            if masses is None:
                continue
            if any(not np.isfinite(v) or not 0 <= v <= 1 for v in masses.values()):
                raise SchemeError(f"{pair.item_id} {pair.cq_id} {entry}: invalid answer masses")
            hard[i, j] = row.get("answer")
            observations[i, j] = arcs.index(hard[i, j]) if hard[i, j] in arcs else 2
            arc_mass[i, j] = sum(masses.get(a, 0.0) for a in arcs)
            if arc_mass[i, j] > 0:
                base[i, j] = masses.get(arcs[0], 0.0) / arc_mass[i, j]
            kept_outside.append(masses.get(OUTSIDE, 0.0))
            for k, signal in enumerate(signals):
                if signal.entry == entry:
                    splits[i, k] = _signal(
                        row, signal.source, arcs, f"{pair.item_id} {pair.cq_id} {entry}"
                    )
        if kept_outside:
            outside[i] = np.mean(kept_outside)
    return B1Dataset(
        schemes,
        routing,
        items,
        pairs,
        entries,
        sources,
        reference,
        signals,
        primary,
        splits,
        base,
        arc_mass,
        outside,
        observations,
        hard,
        np.empty((n, 0)),
    ).with_features()
