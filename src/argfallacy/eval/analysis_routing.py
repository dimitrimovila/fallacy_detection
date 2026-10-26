"""Origin views of pooled held-out routing thresholds, from exact side records."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ..aggregate.interface import TIE
from .analysis_bootstrap import percentile, score_population
from .analysis_inputs import require, terminal, unique
from .analysis_tables import metrics_record, views
from .evaluate import Scorer, weights
from .round2 import ORIGINS

ROUTING_COLUMNS = [
    "evaluation",
    "system",
    "scheme_version",
    "subset_kind",
    "subset",
    "n_items",
    "tau",
    "requested_share",
    "attainable",
    "pooled_explained_share",
    "explained_share",
    "explained_low",
    "explained_high",
    "explained_replicates",
    "explained_omitted",
    "accuracy",
    "accuracy_low",
    "accuracy_high",
    "macro_f1",
    "macro_f1_low",
    "macro_f1_high",
    "status",
]


def routed(records, tau):
    d, z = records.diagram_verdict.to_numpy(), records.zero_verdict.to_numpy()
    has_d, has_z = pd.notna(d), pd.notna(z)
    explained = has_d & (
        ~has_z | (d == z) | (records.p_d.to_numpy() + tau >= records.p_z.to_numpy() - TIE)
    )
    return np.where(explained, d, z), explained


def routing_tables(config, evaluations, items, bootstrap, inputs):
    rows, statuses = [], {}
    for evaluation in evaluations:
        if "C2" not in evaluation.systems:
            continue
        path = config.evaluations[evaluation.name]["routing_records"]
        if path is None:
            saved = evaluation.path / "routing_curve.csv"
            if saved.is_file():
                inputs.add(saved)
            statuses[evaluation.name] = {
                "status": "pending",
                "reason": (
                    "exact origin routing needs full-precision side records; "
                    "separate profiled replay or opt-in export required"
                ),
                "saved_curve": str(saved) if saved.is_file() else None,
                "saved_curve_precision": "rounded thresholds; insufficient for subgroup replay",
                "replay_requirements": [
                    "unchanged source folds, seed, grids and nested fitting settings",
                    "verify all saved C2 verdicts before export",
                    "record elapsed time and peak memory separately from resampling",
                ],
            }
            continue
        metadata = inputs.json(path.with_suffix(".manifest.json"))
        require(
            metadata.get("source_manifest_sha256")
            == hashlib.sha256((evaluation.path / "manifest.json").read_bytes()).hexdigest(),
            "routing source manifest mismatch",
        )
        require(
            metadata.get("records_sha256") == hashlib.sha256(path.read_bytes()).hexdigest(),
            "routing records hash mismatch",
        )
        records = inputs.csv(
            path, ["item_id", "fold", "p_d", "p_z", "diagram_verdict", "zero_verdict"]
        )
        unique(records, ["item_id"], "routing records")
        require(set(records.item_id) == set(items.item_id), "routing population mismatch")
        records = records.set_index("item_id").loc[items.item_id].reset_index()
        system = evaluation.systems["C2"]
        require(metadata.get("scheme_version") == system.version, "routing version mismatch")
        require(metadata.get("system") == "C2", "routing system mismatch")
        assignment = evaluation.manifest["folds"]["assignment"]
        require(
            all(assignment[i] == f for i, f in zip(records.item_id, records.fold, strict=True)),
            "routing fold mismatch",
        )
        for column in ("p_d", "p_z"):
            require(
                pd.api.types.is_numeric_dtype(records[column])
                and np.isfinite(records[column]).all()
                and records[column].between(0, 1).all(),
                "invalid routing probability",
            )
        for column in ("diagram_verdict", "zero_verdict"):
            records[column] = pd.Series([terminal(v, None) for v in records[column]], dtype=object)
        primary, _ = routed(records, 0.0)
        require(
            all(a == b for a, b in zip(primary, system.rows.verdict, strict=True)),
            "routing saved verdict mismatch",
        )
        thresholds = sorted({-1.0, 1.0, *np.clip(records.p_z - records.p_d, -1.0, 1.0)})
        pooled = [(tau, routed(records, tau)[1].mean()) for tau in thresholds]
        selected = []
        for share in (0.6, 0.7, 0.8, 0.9, 1.0):
            available = [(t, s) for t, s in pooled if s >= share - TIE]
            selected.append((available[0][0] if available else None, share))
        grid = [(tau, None) for tau in thresholds] + selected
        for tau, requested in grid:
            for view in [v for v in views(items) if v.kind in ("total", "gold origin")]:
                mask = view.mask & system.eligible
                row = {
                    "evaluation": evaluation.name,
                    "system": "C2",
                    "scheme_version": system.version,
                    "subset_kind": view.kind,
                    "subset": view.name,
                    "n_items": int(mask.sum()),
                    "tau": tau,
                    "requested_share": requested,
                    "attainable": tau is not None,
                    "status": "computed" if tau is not None else "unattainable",
                }
                if tau is None:
                    rows.append(row)
                    continue
                predictions, explained = routed(records, tau)
                row.update(
                    pooled_explained_share=float(explained.mean()),
                    explained_share=float(explained[mask].mean()) if mask.any() else np.nan,
                )
                if requested is None:
                    scorer = Scorer(items.gold_fallacy.tolist(), mask.astype(float)[None, :])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        values = scorer.metrics(predictions)
                    row.update({m: v[0] for m, v in values.items()})
                else:
                    observed, spread = score_population(
                        items.gold_fallacy.tolist(), predictions, mask, bootstrap
                    )
                    row.update(metrics_record(observed, spread))
                    shares = np.full(bootstrap.resamples, np.nan)
                    start = 0
                    for indices in bootstrap.indices(len(items)):
                        counts = weights(indices, len(items)) * mask
                        total = counts.sum(axis=1)
                        values = np.divide(
                            counts @ explained.astype(float),
                            total,
                            out=np.full(len(counts), np.nan),
                            where=total > 0,
                        )
                        shares[start : start + len(values)] = values
                        start += len(values)
                    low, high, used = percentile(shares)
                    row.update(
                        explained_low=low,
                        explained_high=high,
                        explained_replicates=used,
                        explained_omitted=bootstrap.resamples - used,
                    )
                rows.append(row)
        statuses[evaluation.name] = {
            "status": "computed",
            "exact_records": str(path),
            "thresholds": thresholds,
            "selected_thresholds": selected,
            "origins": list(set(items.source.str.lower().map(ORIGINS))),
        }
    status = (
        "computed"
        if statuses and all(s["status"] == "computed" for s in statuses.values())
        else "pending"
    )
    return pd.DataFrame(rows, columns=ROUTING_COLUMNS), {
        "status": status,
        "reason": "exact replay inputs required for unavailable curves"
        if status == "pending"
        else "pooled thresholds retained for every origin",
        "sources": statuses,
    }


def export_routing_records(result, items, out, source_evaluation):
    """Opt-in export from an already computed C2Result; performs no fitting.

    The export belongs in a separate analysis/replay directory, then is supplied
    explicitly as input to an analysis using another output root.
    """
    from pathlib import Path

    from .analysis import _json, code_state, peak_memory_bytes
    from .analysis_inputs import Inputs, overlaps

    out, source = Path(out).resolve(), Path(source_evaluation).resolve()
    require(not overlaps(out, source), "routing export output overlaps source evaluation")
    require(
        len(result.fixed_sides) == 1 and result.name == "C2",
        "export requires the primary two-side C2",
    )
    inputs = Inputs()
    manifest = inputs.json(source / "manifest.json")
    saved = inputs.csv(source / "verdicts.csv", ["system", "item_id", "verdict", "fold"])
    main = inputs.csv(source / "main_table.csv", ["system", "schemes"])
    own = saved[saved.system == "C2"].set_index("item_id")
    ordered = items.sort_values("item_id")
    ids = ordered.item_id.tolist()
    require(set(own.index) == set(ids), "routing export saved population mismatch")
    unique(result.records, ["item_id"], "routing calculation")
    primary = result.at_tau(0)
    require(result.folds == manifest["folds"]["assignment"], "routing export folds mismatch")
    require(
        all(primary.aggregations[i].verdict == terminal(own.loc[i, "verdict"], None) for i in ids),
        "routing export saved verdict mismatch",
    )
    records = primary.records.set_index("item_id").loc[ids, ["p_d", "p_z"]].reset_index()
    records["fold"] = [result.folds[i] for i in ids]
    records["diagram_verdict"] = [result.upstream[i].verdict for i in ids]
    records["zero_verdict"] = [
        result.fixed_sides[0].aggregations[i].verdict
        if i in result.fixed_sides[0].aggregations
        else None
        for i in ids
    ]
    records["saved_verdict"] = [primary.aggregations[i].verdict for i in ids]
    metadata = {
        "source_evaluation": str(source),
        "source_manifest_sha256": inputs.files[(source / "manifest.json").resolve()],
        "system": "C2",
        "scheme_version": str(main.loc[main.system == "C2", "schemes"].iloc[0]),
        "analysis": code_state(),
        "folds": result.folds,
        "source_settings": manifest,
        "choices": result.choices,
        "verified_saved_verdicts": len(ids),
        "peak_process_memory_bytes": peak_memory_bytes(),
    }
    inputs.verify()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "routing_records.csv"
    records.to_csv(path, index=False, lineterminator="\n")
    metadata["records_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".manifest.json").write_text(
        json.dumps(metadata, indent=2, default=_json) + "\n", encoding="utf-8"
    )
    return path
