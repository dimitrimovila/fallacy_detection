"""Selective prediction on immutable saved systems and shared item draws."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..schemes import load_all
from .analysis_bootstrap import percentile
from .analysis_inputs import probability, require
from .analysis_tables import metrics_record, table_frame
from .round2 import FALLACY_CORPORA, ORIGINS, SCHEME_CORPORA
from .run_report import good_terminal
from .uncertainty import (
    LEVELS,
    _cut,
    effective_thresholds,
    selective_bootstrap,
)

AUGRC_PAIRS = (
    ("C2", "zeroshot:qwen3_8_27b"),
    ("C4", "A2.pruned"),
    ("C1", "A2"),
    ("A5", "zeroshot:qwen3_8_27b"),
)


def origin_masks(items):
    origin = items.source.str.lower().map(ORIGINS)
    return [
        ("total", np.ones(len(items), bool)),
        *[(name, origin.eq(name).to_numpy()) for name in (FALLACY_CORPORA, SCHEME_CORPORA)],
    ]


def difficulty_ratings(annotations, items):
    """Identical repeats count once; conflicting or non-integer levels fail."""
    result = pd.Series(np.nan, index=items.item_id, dtype=float)
    if annotations is None:
        return result.to_numpy()
    own = annotations[(annotations.annotator == "enrico") & (annotations.field == "difficulty")]
    for item, rows in own.groupby("item_id"):
        values = []
        for raw in rows.value:
            if raw == "" or pd.isna(raw):
                continue
            try:
                value = float(raw)
            except (ValueError, TypeError):
                require(False, f"{item}: invalid difficulty {raw!r}")
            require(
                np.isfinite(value) and value in (1, 2, 3, 4, 5),
                f"{item}: difficulty must be an integer from 1 to 5",
            )
            values.append(int(value))
        require(len(set(values)) <= 1, f"{item}: conflicting difficulty ratings")
        if values and item in result.index:
            result.loc[item] = values[0]
    return result.to_numpy()


def spearman(x, y):
    """Average ranks, undefined for insufficient or constant inputs."""
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return np.nan
    a, b = rankdata(x), rankdata(y)
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def difficulty_correlation(difficulty, confidence, eligible, bootstrap):
    rated = np.asarray(eligible, bool) & np.isfinite(difficulty) & np.isfinite(confidence)
    observed = spearman(difficulty[rated], 1 - confidence[rated])
    spread = np.full(bootstrap.resamples, np.nan)
    start = 0
    if (
        not rated.any()
        or len(np.unique(difficulty[rated])) < 2
        or len(np.unique(confidence[rated])) < 2
    ):
        return {
            "rated_eligible_items": int(rated.sum()),
            "spearman": np.nan,
            "spearman_low": np.nan,
            "spearman_high": np.nan,
            "spearman_undefined": True,
            "spearman_replicates": 0,
            "spearman_omitted": bootstrap.resamples,
        }
    for batch in bootstrap.indices(len(rated)):
        for offset, draw in enumerate(batch):
            draw = draw[rated[draw]]
            spread[start + offset] = spearman(difficulty[draw], 1 - confidence[draw])
        start += len(batch)
    low, high, used = percentile(spread)
    return {
        "rated_eligible_items": int(rated.sum()),
        "spearman": observed,
        "spearman_low": low,
        "spearman_high": high,
        "spearman_undefined": not np.isfinite(observed),
        "spearman_replicates": used,
        "spearman_omitted": bootstrap.resamples - used,
    }


def composition_rows(items, order, attributes=None):
    """Optional attributes are aligned arrays; nulls are unobserved, never negatives."""
    origin = items.source.str.lower().map(ORIGINS).to_numpy()
    groups = {
        "scheme": items.gold_scheme.to_numpy(),
        "corpus": items.source.to_numpy(),
        "gold origin": origin,
        "gold Good Argumentation": items.gold_fallacy.eq(good_terminal()).to_numpy(),
    }
    for name, values in (attributes or {}).items():
        require(name not in groups, f"duplicate composition attribute: {name}")
        require(len(values) == len(items), f"{name}: attribute population mismatch")
        groups[name] = np.asarray(values, dtype=object)
    n = len(order)
    rows = []
    for level in LEVELS:
        cut = _cut(level, n) if n else 0
        kept, rejected = order[:cut], order[cut:]
        for kind, values in groups.items():
            known = pd.notna(values)
            known[known] &= values[known] != ""
            categories = sorted(set(values[order][known[order]]), key=str)
            # An entirely missing optional attribute still reports its availability.
            for value in categories or [None]:
                equal = np.zeros(len(items), bool)
                if value is not None:
                    equal[known] = values[known] == value
                all_count = int(equal[order].sum())
                count = int(equal[rejected].sum())
                rows.append(
                    {
                        "requested_coverage": level,
                        "actual_cut": cut,
                        "actual_coverage": cut / n if n else np.nan,
                        "n_items": n,
                        "n_kept": cut,
                        "n_rejected": n - cut,
                        "kind": kind,
                        "value": value,
                        "count": count,
                        "group_items": all_count,
                        "share_rejected": count / len(rejected) if len(rejected) else np.nan,
                        "rejection_rate": count / all_count if all_count else np.nan,
                        "known_all": int(known[order].sum()),
                        "missing_all": int((~known[order]).sum()),
                        "known_kept": int(known[kept].sum()),
                        "missing_kept": int((~known[kept]).sum()),
                        "known_rejected": int(known[rejected].sum()),
                        "missing_rejected": int((~known[rejected]).sum()),
                    }
                )
    return rows


def terminal_margin(system, items, schemes):
    """Use a genuine saved distribution only; a chosen verdict is insufficient."""
    if "distribution" not in system.rows:
        return None
    margins = np.full(len(items), np.nan)
    complete = True
    for position in np.flatnonzero(system.eligible):
        raw = system.rows.iloc[position].distribution
        if not isinstance(raw, dict) and (pd.isna(raw) or raw == ""):
            complete = False
            continue
        masses = json.loads(raw) if isinstance(raw, str) else raw
        require(isinstance(masses, dict), "invalid terminal distribution")
        terminals = schemes[items.iloc[position].gold_scheme].terminals
        require(set(masses) <= set(terminals), "distribution contains nonterminal labels")
        values = [probability(value, "terminal distribution") for value in masses.values()]
        require(
            np.isfinite(values).all() and 0 <= sum(values) <= 1 + 1e-12,
            "invalid terminal distribution mass",
        )
        top = sorted(values, reverse=True) + [0.0, 0.0]
        confidence = system.rows.iloc[position].confidence
        verdict = system.rows.iloc[position].verdict
        require(
            pd.isna(verdict)
            or pd.isna(confidence)
            or np.isclose(confidence, top[0], rtol=0, atol=1e-12),
            "terminal distribution disagrees with saved confidence",
        )
        require(
            pd.isna(verdict) or np.isclose(masses.get(verdict, -1), top[0], rtol=0, atol=1e-12),
            "terminal distribution disagrees with saved verdict",
        )
        margins[position] = top[0] - top[1]
    return margins if complete else None


def _metadata(evaluation, system, origin, mask):
    return {
        "evaluation": evaluation.name,
        "round": evaluation.round,
        "system": system.name,
        "scheme_version": system.version,
        "origin": origin,
        "n_items": int(mask.sum()),
        "population": "declared eligible items",
        "level": 0.95,
    }


def _missingness(system, mask):
    confidence = system.rows.confidence.to_numpy(float)
    present = system.rows.verdict.notna().to_numpy()
    operational = mask & present & np.isfinite(confidence)
    n = int(mask.sum())
    return {
        "missing_confidence": int((mask & ~np.isfinite(confidence)).sum()),
        "missing_verdicts": int((mask & ~present).sum()),
        "operational_items": int(operational.sum()),
        "maximum_operational_coverage": operational.sum() / n if n else np.nan,
    }


def selective_tables(evaluations, items, bootstrap, difficulty, registry, attributes=None):
    require("difficulty" not in (attributes or {}), "difficulty is a frozen attribute")
    ids, gold = items.item_id.to_numpy(), items.gold_fallacy.to_numpy()
    summary, curves, thresholds, composition, difficulties, differences = [], [], [], [], [], []
    masks = origin_masks(items)
    for evaluation in evaluations:
        for system in evaluation.systems.values():
            if system.role == "oracle":
                continue
            verdicts = system.rows.verdict.to_numpy()
            confidence = system.rows.confidence.to_numpy(float)
            wrong = pd.isna(verdicts) | (verdicts != gold)
            ranked = bool(np.isfinite(confidence[system.eligible]).any())
            margins = (
                terminal_margin(system, items, load_all(registry[system.version]))
                if ranked
                else None
            )
            for origin, origin_mask in masks:
                mask = system.eligible & origin_mask
                meta = _metadata(evaluation, system, origin, mask)
                missingness = _missingness(system, mask)
                if not ranked:
                    count = int(mask.sum())
                    summary.append(
                        {
                            **meta,
                            **missingness,
                            "ranking_available": False,
                            "status": "reference point",
                            "reference_coverage": (mask & pd.notna(verdicts)).sum() / count
                            if count
                            else np.nan,
                            "reference_accuracy": 1 - wrong[mask].mean() if count else np.nan,
                            "margin_status": "unavailable: no terminal distribution",
                        }
                    )
                    continue
                observed, spread, order = selective_bootstrap(
                    ids, wrong, verdicts, confidence, mask, bootstrap
                )
                row = {
                    **meta,
                    **missingness,
                    "ranking_available": True,
                    "status": "computed" if mask.any() else "undefined",
                    **metrics_record(observed, spread),
                }
                del spread
                for level in LEVELS:
                    row[f"actual_cut_{round(100 * level)}"] = (
                        _cut(level, len(order)) if len(order) else 0
                    )
                if margins is not None:
                    check, spread, _ = selective_bootstrap(
                        ids, wrong, verdicts, margins, mask, bootstrap
                    )
                    row.update({"augrc_margin": check["augrc"], "margin_status": "computed"})
                    low, high, used = percentile(spread["augrc"])
                    row.update(
                        augrc_margin_low=low,
                        augrc_margin_high=high,
                        augrc_margin_replicates=used,
                        augrc_margin_omitted=bootstrap.resamples - used,
                    )
                    del spread
                else:
                    row["margin_status"] = "pending: genuine terminal distribution unavailable"
                summary.append(row)
                errors = np.cumsum(wrong[order])
                for k, error in enumerate(errors, 1):
                    curves.append(
                        {
                            **meta,
                            "k": k,
                            "item_id": ids[order[k - 1]],
                            "errors": int(error),
                            "coverage": k / len(order),
                            "selective_risk": error / k,
                            "generalized_risk": error / len(order),
                        }
                    )
                thresholds.extend(
                    {**meta, **missingness, **r}
                    for r in effective_thresholds(ids, wrong, verdicts, confidence, mask)
                )
                attrs = {"difficulty": difficulty, **(attributes or {})}
                composition.extend({**meta, **r} for r in composition_rows(items, order, attrs))
                eligible = mask & pd.notna(verdicts) & np.isfinite(confidence)
                correlation = difficulty_correlation(difficulty, confidence, eligible, bootstrap)
                for level in LEVELS:
                    cut = _cut(level, len(order)) if len(order) else 0
                    availability = {}
                    for name, positions in [
                        ("all", order),
                        ("kept", order[:cut]),
                        ("rejected", order[cut:]),
                    ]:
                        known = int(np.isfinite(difficulty[positions]).sum())
                        availability.update(
                            {
                                f"{name}_items": len(positions),
                                f"{name}_rated": known,
                                f"{name}_missing": len(positions) - known,
                                f"{name}_availability": known / len(positions)
                                if len(positions)
                                else np.nan,
                            }
                        )
                    difficulties.append(
                        {
                            **meta,
                            "requested_coverage": level,
                            "actual_cut": cut,
                            "actual_coverage": cut / len(order) if len(order) else np.nan,
                            **availability,
                            **correlation,
                            "status": "computed"
                            if np.isfinite(difficulty[mask]).any()
                            else "pending",
                        }
                    )
        for name, reference in AUGRC_PAIRS:
            if evaluation.round == 1 and name != "A5":
                continue
            left, right = evaluation.systems.get(name), evaluation.systems.get(reference)
            for origin, origin_mask in masks:
                row = {
                    "evaluation": evaluation.name,
                    "round": evaluation.round,
                    "system": name,
                    "reference": reference,
                    "origin": origin,
                    "kind": "exploratory",
                    "level": 0.95,
                    "status": "pending",
                    "n_items": 0,
                }
                if left is None or right is None:
                    differences.append({**row, "reason": "source system unavailable"})
                    continue
                row.update(scheme_version=left.version, reference_scheme_version=right.version)
                common = left.eligible & right.eligible & origin_mask
                if not all(
                    np.isfinite(s.rows.confidence.to_numpy(float)[s.eligible]).any()
                    for s in (left, right)
                ):
                    differences.append({**row, "reason": "source system without confidence"})
                    continue
                estimates, replicates = [], []
                for system in (left, right):
                    verdicts = system.rows.verdict.to_numpy()
                    observed, spread, _ = selective_bootstrap(
                        ids,
                        pd.isna(verdicts) | (verdicts != gold),
                        verdicts,
                        system.rows.confidence.to_numpy(float),
                        common,
                        bootstrap,
                    )
                    estimates.append(observed["augrc"])
                    replicates.append(spread["augrc"])
                    del spread
                low, high, used = percentile(replicates[0] - replicates[1])
                differences.append(
                    {
                        **row,
                        "status": "computed" if common.any() else "undefined",
                        "n_items": int(common.sum()),
                        "augrc_difference": estimates[0] - estimates[1],
                        "augrc_low": low,
                        "augrc_high": high,
                        "augrc_replicates": used,
                        "augrc_omitted": bootstrap.resamples - used,
                    }
                )
                del replicates
    data = {
        "abstention.csv": (
            summary,
            ["evaluation", "system", "origin", "n_items", "augrc", "aurc", "status"],
        ),
        "abstention_curves.csv": (
            curves,
            [
                "evaluation",
                "system",
                "origin",
                "n_items",
                "k",
                "coverage",
                "selective_risk",
                "generalized_risk",
            ],
        ),
        "abstention_thresholds.csv": (
            thresholds,
            [
                "evaluation",
                "system",
                "origin",
                "n_items",
                "threshold",
                "kept",
                "errors",
                "coverage",
                "selective_risk",
                "generalized_risk",
            ],
        ),
        "abstention_differences.csv": (
            differences,
            [
                "evaluation",
                "system",
                "reference",
                "origin",
                "n_items",
                "augrc_difference",
                "status",
            ],
        ),
        "abstention_composition.csv": (
            composition,
            [
                "evaluation",
                "system",
                "origin",
                "n_items",
                "requested_coverage",
                "actual_cut",
                "kind",
                "value",
                "count",
                "share_rejected",
                "rejection_rate",
            ],
        ),
        "difficulty.csv": (
            difficulties,
            [
                "evaluation",
                "system",
                "origin",
                "n_items",
                "requested_coverage",
                "actual_cut",
                "spearman",
                "status",
            ],
        ),
    }
    return {name: table_frame(rows, columns) for name, (rows, columns) in data.items()}
