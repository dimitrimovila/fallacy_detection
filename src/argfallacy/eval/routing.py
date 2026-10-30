"""Descriptive routing curves on pooled held-out estimates and shared resamples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..aggregate.interface import TIE
from ..aggregate.routing import B2Result
from .evaluate import Bench, Scorer, interval, resample_indices


def routing_curve(
    result: B2Result, items: pd.DataFrame, indices=None, *, seed=20261019, resamples=10000,
    export_dir=None, source_evaluation=None,
) -> pd.DataFrame:
    """Use the smallest pooled threshold attaining at least each requested share.

    Every resample reuses that threshold. Discrete jumps retain their actual
    share, and targets above the largest attainable share have no score interval.
    All test items, including abstentions, enter the denominator of the share.
    """
    if len(result.fixed_sides) != 1:
        raise ValueError("a routing curve requires exactly two sides")
    items = items.sort_values("item_id")
    ids = list(items.item_id)
    if set(result.records.item_id) != set(ids) or result.records.item_id.duplicated().any():
        raise ValueError("routing curve needs exactly the complete test item set")
    indices = (
        resample_indices(seed, len(ids), resamples, 1)
        if indices is None
        else np.asarray(indices, dtype=int)
    )
    if (
        indices.ndim != 2
        or indices.shape[1] != len(ids)
        or (indices.size and (indices.min() < 0 or indices.max() >= len(ids)))
    ):
        raise ValueError("routing bootstrap indices must resample the complete test item set")
    bench = Bench(list(items.gold_fallacy), indices)
    observed = Scorer(list(items.gold_fallacy), np.ones((1, len(ids))))
    thresholds = sorted({-1.0, 1.0, *np.clip(result.records.p_z - result.records.p_d, -1.0, 1.0)})
    rows, by_threshold = [], {}
    for tau in thresholds:
        routed = result.at_tau(tau)
        verdicts = [routed.aggregations[i].verdict for i in ids]
        explained = routed.records.set_index("item_id").explained.reindex(ids).to_numpy(dtype=float)
        metrics = observed.metrics(verdicts)
        row = {
            "tau": tau,
            "explained_share": float(explained.mean()),
            "accuracy": float(metrics["accuracy"][0]),
            "macro_f1": float(metrics["macro_f1"][0]),
            "requested_share": np.nan,
            "attainable": True,
        }
        rows.append(row)
        by_threshold[tau] = verdicts, explained
    for requested in (0.6, 0.7, 0.8, 0.9, 1.0):
        available = [row for row in rows if row["explained_share"] >= requested - TIE]
        if not available:
            rows.append(
                {
                    "tau": np.nan,
                    "explained_share": max(r["explained_share"] for r in rows),
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "requested_share": requested,
                    "attainable": False,
                }
            )
            continue
        selected = min(available, key=lambda row: row["tau"])
        row = {**selected, "requested_share": requested}
        verdicts, explained = by_threshold[selected["tau"]]
        scores = bench.score(verdicts)
        for metric in ("accuracy", "macro_f1"):
            row[f"{metric}_low"], row[f"{metric}_high"] = interval(scores.resampled[metric])
        share = (bench.resampled.weights @ explained) / len(ids)
        row["explained_low"], row["explained_high"] = interval(share)
        rows.append(row)
    if export_dir is not None:
        if source_evaluation is None:
            raise ValueError("routing export needs the source evaluation for verdict verification")
        from .analysis_routing import export_routing_records

        export_routing_records(result, items, export_dir, source_evaluation)
    return pd.DataFrame(rows)
