"""Offline thesis breakdowns from saved out-of-fold evaluations, without fitting."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..paths import REPO_ROOT
from .analysis_bootstrap import Bootstrap
from .analysis_diagnostics import integrity, stage_one
from .analysis_inputs import (
    AnalysisConfig,
    Inputs,
    overlaps,
    read_evaluation,
    read_items,
    read_run,
    recover_confidence,
    require,
    scheme_registry,
)
from .analysis_tables import comparisons, main_tables, rare_diagnosis
from .round2 import ORIGINS


def code_state():
    def git(*args):
        result = subprocess.run(
            ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {"commit": git("rev-parse", "HEAD"), "working_tree": git("status", "--porcelain")}


def peak_memory_bytes():
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                *[
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ],
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return int(counters.PeakWorkingSetSize)
        return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def analyze(config: AnalysisConfig, out_root=None, progress=None):
    """Write only to the analysis root. Optional human/D+Q paths are never opened."""
    started = time.perf_counter()
    out = Path(out_root).resolve() if out_root is not None else config.output
    config.guard_output(out)
    say = progress or (lambda _: None)
    inputs = Inputs()
    inputs.add(config.path)
    items = read_items(config, inputs)
    require(set(items.source.str.lower()) <= set(ORIGINS), "corpus without a gold origin")
    registry = scheme_registry(inputs)
    # Record the implementation alongside its Git identity, including uncommitted changes.
    for file in sorted(Path(__file__).parent.glob("analysis*.py")):
        inputs.add(file)
    for name in ("canonical.py", "evaluate.py", "folds.py", "metrics.py", "routing.py"):
        inputs.add(Path(__file__).with_name(name))
    evaluations = [
        read_evaluation(name, source, config, items, registry, inputs)
        for name, source in config.evaluations.items()
    ]
    bootstrap = Bootstrap(config.seed, config.resamples, config.bootstrap_batch_size)
    run_records, integrity_rows, stage_rows = {}, [], []
    for run_id, path in config.parsed_runs.items():
        say(f"reading parsed run {run_id}")
        manifest, scheme_version, summary = read_run(path, run_id, items, registry, inputs)
        for evaluation in evaluations:
            sources = evaluation.manifest.get("runs", {})
            schemes = evaluation.manifest.get("schemes", {})
            if sources.get("answers") == run_id:
                expected = schemes.get("tag")
                folder = schemes.get("answers", schemes.get("folder"))
                require(
                    not expected or str(expected) == scheme_version,
                    f"{evaluation.name}: answers run version mismatch",
                )
                require(
                    not folder or folder == registry[scheme_version].name,
                    f"{evaluation.name}: answers folder/version mismatch",
                )
            if sources.get("answers_schemes_2") == run_id:
                require(
                    schemes.get("answers_schemes_2") == registry[scheme_version].name,
                    f"{evaluation.name}: revised answers version mismatch",
                )
        run_records[run_id] = {
            "path": str(path),
            "manifest": manifest,
            "scheme_version": scheme_version,
            "source_commit_status": "recorded"
            if manifest.get("commit")
            else "unavailable in source",
        }
        recover_confidence(evaluations, run_id, manifest, scheme_version, summary)
        diagnostics = integrity(
            run_id, scheme_version, manifest, summary, path, items, registry, inputs
        )
        recognition = stage_one(run_id, scheme_version, manifest, summary, items)
        for evaluation in evaluations:
            sources = evaluation.manifest.get("runs", {})
            referenced = {
                name
                for value in sources.values()
                for name in ([value] if isinstance(value, str) else (value or []))
            }
            if run_id in referenced:
                integrity_rows.extend({"evaluation": evaluation.name, **row} for row in diagnostics)
                stage_rows.extend({"evaluation": evaluation.name, **row} for row in recognition)
        del summary
    confidence, systems_record = [], {}
    for evaluation in evaluations:
        systems_record[evaluation.name] = {}
        for system in evaluation.systems.values():
            own = system.rows.loc[system.eligible]
            missing = int(own.verdict.isna().sum())
            corpus_counts = items.loc[system.eligible, "source"].value_counts()
            systems_record[evaluation.name][system.name] = {
                "scheme_version": system.version,
                "eligible_item_ids": own.index.tolist(),
                "n_items": len(own),
                "missing_verdicts": missing,
                "missing_confidence": int(own.confidence.isna().sum()),
                "declared_population": "full" if system.eligible.all() else "restricted",
                "small_corpora_total_only": corpus_counts[corpus_counts < 20].to_dict(),
                "confidence_source_runs": sorted(
                    set(own.get("confidence_run", pd.Series(dtype=str)).dropna())
                ),
            }
            for item, row in own.iterrows():
                confidence.append(
                    {
                        "evaluation": evaluation.name,
                        "system": system.name,
                        "scheme_version": system.version,
                        "item_id": item,
                        "verdict": row.verdict,
                        "confidence": row.confidence,
                        "confidence_source": row.confidence_source,
                        "confidence_run": row.get("confidence_run", ""),
                        "p_label": row.p_label,
                        "p_label_lower_bound": row.p_label_lower_bound,
                    }
                )
            integrity_rows.append(
                {
                    "evaluation": evaluation.name,
                    "run": "",
                    "system": system.name,
                    "scheme_version": system.version,
                    "stage": "verdict",
                    "subset": "declared eligible items",
                    "measure": "missing verdicts",
                    "unit": "item",
                    "count": missing,
                    "denominator": len(own),
                    "denominator_basis": "declared eligible items",
                    "rate": missing / len(own) if len(own) else np.nan,
                    "undefined": not len(own),
                    "status": "computed",
                }
            )
    say("scoring saved verdicts sequentially")
    main, per_class = main_tables(evaluations, items, bootstrap)
    say("paired saved comparisons")
    pairs = comparisons(evaluations, items, bootstrap)
    rare = rare_diagnosis(evaluations, items, bootstrap)
    from .analysis_routing import routing_tables

    routing, routing_status = routing_tables(config, evaluations, items, bootstrap, inputs)
    tables = {
        "main_table.csv": main,
        "comparisons.csv": pairs,
        "per_class.csv": per_class,
        "rare_classes.csv": rare,
        "integrity.csv": pd.DataFrame(integrity_rows),
        "stage1.csv": pd.DataFrame(
            stage_rows,
            columns=[
                "evaluation",
                "run",
                "system",
                "scheme_version",
                "subset",
                "condition",
                "n_items",
                "observed_rows",
                "missing_rows",
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "n_classes",
                "excluded_classes",
                "status",
            ],
        ),
        "routing_curve.csv": routing,
        "confidence.csv": pd.DataFrame(confidence),
    }
    parts = {
        name.removesuffix(".csv"): {"status": "computed", "rows": len(table)}
        for name, table in tables.items()
    }
    parts["routing_curve"] = routing_status
    if not stage_rows:
        parts["stage1"] = {"status": "pending", "reason": "no parsed stage-one rows"}
    if rare.empty:
        parts["rare_classes"] = {"status": "pending", "reason": "no saved C1 system"}
    parts.update(
        {
            "direct_verdict": {
                "status": "pending",
                "reason": "reserved inputs; not opened by this component",
                "configured": {k: str(v) if v else None for k, v in config.direct_verdict.items()},
            },
            "dq": {
                "status": "pending",
                "reason": "reserved adapter; not opened by this component",
                "configured": str(config.dq) if config.dq else None,
            },
            "collapsed": {
                "status": "deferred",
                "level": 3,
                "reason": "canonical point scorer exists; existing bootstrap scorer is fine-only",
            },
            "predicted_scheme_verdict": {
                "status": "deferred",
                "level": 3,
                "reason": "unavailable; distinct from stage-one recognition",
            },
            "remaining_stage_two": {
                "status": "pending",
                "reason": "CQ/human, uncertainty and report components are separate",
            },
        }
    )
    without = items.source.str.lower().ne("nlas").to_numpy()
    record = {
        "name": "analyses_v1",
        "configuration_version": 1,
        "configuration": yaml.safe_load(config.path.read_text(encoding="utf-8")),
        "output": str(out),
        "path_rule": "all paths relative to the configuration file directory",
        "analysis": code_state(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": inputs.records(),
        "sources": {
            e.name: {
                "path": str(e.path),
                "round": e.round,
                "manifest": e.manifest,
                "source_commit_status": "recorded"
                if e.manifest.get("commit")
                else "unavailable in source",
            }
            for e in evaluations
        },
        "parsed_runs": run_records,
        "systems": systems_record,
        "bootstrap": {
            "seed": config.seed,
            "resamples": config.resamples,
            "batch_size": config.bootstrap_batch_size,
            "bit_generator": "PCG64",
            "seed_sequence": "SeedSequence(seed).spawn(3)",
            "whole_test_generator": 1,
            "without_nlas_generator": 2,
            "item_ids": items.item_id.tolist(),
            "without_nlas_item_ids": items.loc[without, "item_id"].tolist(),
            "ordering_source": "saved evaluation protocol: ascending item_id",
            "subsets": "filter whole-test draws; never refit",
            "workspace": "one batch; sequential systems; scalar replicates only",
        },
        "denominators": {
            "frozen_test": len(items),
            "without_nlas": int(without.sum()),
            "small_corpora_total_only": items.source.value_counts()[
                items.source.value_counts() < 20
            ].to_dict(),
            "origins": items.source.str.lower().map(ORIGINS).value_counts().to_dict(),
        },
        "parts": parts,
        "parameters": {
            "rare_class_full_test_gold_support_below": 20,
            "main_level": 0.95,
            "new_comparison_level": 0.95,
        },
        "software_versions": {
            p: version(p) for p in ("argfallacy", "numpy", "pandas", "scikit-learn", "scipy")
        },
        "python_version": sys.version,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_memory_bytes": peak_memory_bytes(),
    }
    # No output is touched until every input, supplied contract and metric validates.
    inputs.verify()
    config.guard_output(out)
    for source in inputs.files:
        require(not overlaps(out, source), f"output overlap with input: {source}")
    for filename in [*tables, "manifest.json"]:
        target = out / filename
        require(not target.is_symlink(), f"output overlap through symlink: {target}")
        if target.exists():
            require(
                not any(target.samefile(p) for p in inputs.files),
                f"output overlap through shared file: {target}",
            )
    out.mkdir(parents=True, exist_ok=True)
    for filename, frame in tables.items():
        if filename == "routing_curve.csv":
            frame = frame.copy()
            for column in frame.select_dtypes(include="floating"):
                if column != "tau":
                    frame[column] = frame[column].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        frame.to_csv(
            out / filename,
            index=False,
            lineterminator="\n",
            float_format=None if filename in ("confidence.csv", "routing_curve.csv") else "%.4f",
        )
    record["elapsed_seconds"] = time.perf_counter() - started
    record["peak_process_memory_bytes"] = peak_memory_bytes()
    (out / "manifest.json").write_text(
        json.dumps(record, indent=2, default=_json) + "\n", encoding="utf-8"
    )
    return out


__all__ = ["AnalysisConfig", "analyze"]
