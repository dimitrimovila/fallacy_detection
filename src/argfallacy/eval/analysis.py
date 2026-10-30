"""Offline thesis breakdowns from saved out-of-fold evaluations, without fitting."""

from __future__ import annotations

import ctypes
import hashlib
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
from .analysis_conformal import conformal_tables
from .analysis_diagnostics import integrity, stage_one
from .analysis_direct import direct_tables, read_direct
from .analysis_dq import integrate_dq
from .analysis_human import human_tables, read_frozen_annotations
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
from .analysis_selective import AUGRC_PAIRS, difficulty_ratings, selective_tables
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
    """Write saved-system analyses to a separate root, without inference or fitting."""
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
    for name in (
        "canonical.py",
        "evaluate.py",
        "folds.py",
        "metrics.py",
        "routing.py",
        "uncertainty.py",
        "b1.py",
        "cq.py",
    ):
        inputs.add(Path(__file__).with_name(name))
    for name in ("annotations.py", "aggregate/soft.py", "aggregate/combiners.py"):
        inputs.add(Path(__file__).parent.parent / name)
    evaluations = [
        read_evaluation(name, source, config, items, registry, inputs)
        for name, source in config.evaluations.items()
    ]
    imported_dq, dq_status = integrate_dq(config, evaluations, items, registry, inputs)
    from ..schemes import load_all

    direct, direct_status = read_direct(config, items, load_all(registry["1.1"]), inputs)
    bootstrap = Bootstrap(config.seed, config.resamples, config.bootstrap_batch_size)
    run_records, integrity_rows, stage_rows = {}, [], []
    for run_id, path in config.parsed_runs.items():
        dedicated = {
            r
            for e in evaluations
            for names in e.manifest.get("dq", {}).get("runs", {}).values()
            for r in names
        }
        if config.dq is None and run_id in dedicated:
            continue
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
            elif config.dq is not None and evaluation.path == config.dq and run_id in dedicated:
                integrity_rows.extend({"evaluation": evaluation.name, **row} for row in diagnostics)
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
                "source_table": system.source_table,
                "source_provenance": system.source_provenance,
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
                        "source_table": system.source_table,
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
    say("frozen CQ and historical human tables")
    annotations = read_frozen_annotations(config, inputs)
    from .analysis_cq import cq_tables

    cq, cq_status = cq_tables(config, evaluations, items, registry, annotations, bootstrap, inputs)
    human, human_status = human_tables(
        annotations, evaluations, items, load_all(registry["1.1"]), bootstrap
    )
    new_human = direct_tables(
        direct, annotations, evaluations, items, load_all(registry["1.1"]), bootstrap
    )
    disagreement = new_human.pop("human_disagreement")
    conformal, conformal_status = conformal_tables(
        config,
        evaluations,
        items,
        registry,
        direct if config.direct_verdict["answers"] is not None else None,
        inputs,
    )
    say("selective prediction on saved confidence; sequential systems")
    selective = selective_tables(
        evaluations,
        items,
        bootstrap,
        difficulty_ratings(annotations, items),
        registry,
        {"human disagreement": disagreement}
        if config.direct_verdict["answers"] is not None
        else None,
    )
    post_results = {}
    post_status = {}
    saved_conformal = []
    conformal_columns = [
        "system",
        "kind",
        "subset",
        "n_items",
        "coverage",
        "mean_size",
        "singleton_share",
    ]
    for evaluation in evaluations:
        if evaluation.round != 2:
            continue
        source_conformal = evaluation.path / "conformal.csv"
        if source_conformal.is_file():
            frame = inputs.csv(source_conformal, conformal_columns, dtype=str)
            for column in ("n_items", "coverage", "mean_size", "singleton_share"):
                values = pd.to_numeric(frame.loc[frame[column].ne(""), column], errors="raise")
                require(np.isfinite(values).all(), f"conformal.csv: invalid {column}")
                require((values >= 0).all(), f"conformal.csv: negative {column}")
                if column in ("coverage", "singleton_share"):
                    require((values <= 1).all(), f"conformal.csv: invalid {column}")
            frame["evaluation"] = evaluation.name
            frame["scheme_version"] = [evaluation.systems[n].version for n in frame.system]
            frame["source_table"] = "conformal.csv"
            frame["source_sha256"] = inputs.files[source_conformal.resolve()]
            frame["status"] = "saved aggregate only; exact item exports independent"
            saved_conformal.append(frame)
        for name, columns in {
            "frequent_classes.csv": [
                "system",
                "schemes",
                "n_items",
                "n_classes",
                "macro_f1_frequent",
            ],
            "frequent_classes_differences.csv": [
                "system",
                "reference",
                "n_items",
                "macro_f1_frequent_difference",
            ],
        }.items():
            target_name = "post_results_" + name
            source = evaluation.path / name
            frame = (
                inputs.csv(source, columns) if source.is_file() else pd.DataFrame(columns=columns)
            )
            frame["source_evaluation"] = evaluation.name
            frame["source_table"] = name
            frame["source_sha256"] = inputs.files.get(source.resolve(), "")
            frame["analysis_role"] = "exploratory; added after the results; gold support >=10"
            require(
                target_name not in post_results,
                "multiple post-results sources require distinct table identities",
            )
            post_results[target_name] = frame
            post_status[target_name.removesuffix(".csv")] = {
                "status": "computed" if len(frame) else "pending",
                "rows": len(frame),
                "reason": "selected saved post-results summary"
                if len(frame)
                else "selected source summary unavailable",
            }
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
        **cq,
        **human,
        **selective,
        **new_human,
        **conformal,
        **imported_dq,
        **post_results,
        "source_conformal.csv": pd.concat(saved_conformal, ignore_index=True)
        if saved_conformal
        else pd.DataFrame(
            columns=[
                *conformal_columns,
                "evaluation",
                "scheme_version",
                "source_table",
                "source_sha256",
                "status",
            ]
        ),
    }
    parts = {
        name.removesuffix(".csv"): {"status": "computed", "rows": len(table)}
        for name, table in tables.items()
    }
    parts["routing_curve"] = routing_status
    parts["cq_populations"] = cq_status
    parts["historical_human"] = human_status
    parts["conformal_sources"] = conformal_status
    parts.update(post_status)
    if tables["source_conformal.csv"].empty:
        parts["source_conformal"].update(status="pending", reason="saved B5 aggregates unavailable")
    for name in (*cq, *human, *selective, *new_human, *conformal, *imported_dq):
        frame = tables[name]
        part = parts[name.removesuffix(".csv")]
        if frame.empty:
            part.update(status="pending", reason="no compatible source observations")
        elif "status" in frame:
            part["row_statuses"] = frame.status.value_counts().to_dict()
            pending = frame.status.fillna("").str.startswith("pending")
            if pending.all():
                part.update(
                    status="pending", reason="source observations or references unavailable"
                )
            elif pending.any():
                part["status"] = "partial"
    if annotations is None:
        for name in ("human_agreement", "human_path", "difficulty"):
            parts[name].update(status="pending", reason="frozen annotations not configured")
    if not stage_rows:
        parts["stage1"] = {"status": "pending", "reason": "no parsed stage-one rows"}
    if rare.empty:
        parts["rare_classes"] = {"status": "pending", "reason": "no saved B1 system"}
    parts.update(
        {
            "direct_verdict": {
                **direct_status,
                "configured": {k: str(v) if v else None for k, v in config.direct_verdict.items()},
            },
            "dq": {
                **dq_status,
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
                "reason": (
                    "exact conformal/routing sources and new human labels "
                    "remain optional pending inputs"
                ),
            },
        }
    )
    without = items.source.str.lower().ne("nlas").to_numpy()
    record = {
        "name": "analyses_v1",
        "report_format": "thesis_analyses_v1",
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
            "hard_cq_agreement_minimum_annotated_items": 20,
            "abstention_fixed_coverages": [1.0, 0.95, 0.9, 0.8, 0.7],
            "exploratory_augrc_pairs": AUGRC_PAIRS,
            "abstention_order": (
                "finite confidence descending; verdict without confidence; no verdict; item_id ties"
            ),
            "abstention_integral": "discrete mean over ordered k=1..N; threshold curve separate",
            "difficulty_levels": [1, 2, 3, 4, 5],
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
    record["tables"] = {}
    for filename, frame in tables.items():
        numeric_columns = frame.select_dtypes(include="number").columns.tolist()
        if filename in ("routing_curve.csv", "abstention_thresholds.csv"):
            frame = frame.copy()
            for column in frame.select_dtypes(include="floating"):
                if column not in ("tau", "threshold"):
                    frame[column] = frame[column].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        frame.to_csv(
            out / filename,
            index=False,
            lineterminator="\n",
            float_format=None
            if filename
            in (
                "confidence.csv",
                "routing_curve.csv",
                "abstention_thresholds.csv",
                "conformal_items.csv",
            )
            else "%.4f",
        )
        record["tables"][filename] = {
            "columns": frame.columns.tolist(),
            "numeric_columns": numeric_columns,
            "rows": len(frame),
            "sha256": hashlib.sha256((out / filename).read_bytes()).hexdigest(),
        }
    record["elapsed_seconds"] = time.perf_counter() - started
    record["peak_process_memory_bytes"] = peak_memory_bytes()
    (out / "manifest.json").write_text(
        json.dumps(record, indent=2, default=_json) + "\n", encoding="utf-8"
    )
    return out


__all__ = ["AnalysisConfig", "analyze"]
