"""Genuine B5 item sets from saved distributions or an explicit companion export."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..aggregate.interface import decide
from ..schemes import load_all
from .analysis_inputs import Inputs, overlaps, probability, require, terminal, unique
from .analysis_tables import table_frame
from .round2 import ORIGINS
from .uncertainty import ALPHA, conformal_sets, conformal_summary

ITEM_COLUMNS = [
    "evaluation",
    "system",
    "scheme_version",
    "item_id",
    "fold",
    "terminal_ids",
    "size",
    "gold_included",
    "threshold",
]
SUMMARY_COLUMNS = [
    "evaluation",
    "system",
    "scheme_version",
    "kind",
    "subset",
    "n_items",
    "coverage",
    "mean_size",
    "singleton_share",
]


def selected_systems(evaluation, inputs):
    path = evaluation.path / "conformal.csv"
    if evaluation.round != 2 or not path.is_file():
        return None
    summary = inputs.csv(
        path, ["system", "kind", "subset", "n_items", "coverage", "mean_size", "singleton_share"]
    )
    unique(summary, ["system", "kind", "subset"], "source conformal summaries")
    require(evaluation.manifest.get("conformal_alpha") == ALPHA, "conformal alpha mismatch")
    for name in summary.system.unique():
        require(
            name in evaluation.systems
            and (
                name in ("A2", "A4", "B3.pooled", "B4", "B1", "B1@2.0")
                or (
                    name.startswith("B1.")
                    and (name.endswith(".soft") or name.endswith(".soft@2.0"))
                )
            ),
            f"{name}: not a genuine B5 distribution system",
        )
        require(
            evaluation.systems[name].eligible.all(), f"{name}: B5 requires original full population"
        )
    return summary


def validate_distribution(system, item, aggregation, schemes, scheme):
    masses = aggregation.distribution
    require(isinstance(masses, dict), "invalid conformal distribution format")
    require(
        set(masses) <= set(schemes[scheme].terminals), "conformal distribution outside scheme menu"
    )
    values = [probability(v, "conformal distribution") for v in masses.values()]
    require(
        np.isfinite(values).all() and sum(values) <= 1 + 1e-12,
        "invalid conformal distribution mass",
    )
    chosen = decide(masses, aggregation.abstention, "")
    require(chosen.verdict == aggregation.verdict, "conformal aggregation verdict mismatch")
    saved = system.rows.loc[item]
    require(
        (chosen.verdict is None and pd.isna(saved.verdict)) or chosen.verdict == saved.verdict,
        "conformal distribution/saved verdict disagreement",
    )
    require(
        (chosen.confidence is None and pd.isna(saved.confidence))
        or (
            chosen.confidence is not None
            and pd.notna(saved.confidence)
            and np.isclose(chosen.confidence, saved.confidence, rtol=0, atol=1e-12)
        ),
        "conformal distribution/saved confidence disagreement",
    )


def calculate(evaluation, system, items, registry, aggregations):
    require(set(aggregations) == set(items.item_id), "conformal distribution population mismatch")
    schemes = load_all(registry[system.version])
    gold = items.set_index("item_id").gold_fallacy.to_dict()
    scheme = items.set_index("item_id").gold_scheme.to_dict()
    folds = evaluation.manifest["folds"]["assignment"]
    for item, aggregation in aggregations.items():
        validate_distribution(system, item, aggregation, schemes, scheme[item])
    terminals = {i: sorted(schemes[scheme[i]].terminals) for i in aggregations}
    sets, thresholds = conformal_sets(aggregations, terminals, gold, folds, ALPHA)
    meta = dict(evaluation=evaluation.name, system=system.name, scheme_version=system.version)
    rows = [
        {
            **meta,
            "item_id": i,
            "fold": folds[i],
            "terminal_ids": json.dumps(sets[i]),
            "size": len(sets[i]),
            "gold_included": gold[i] in sets[i],
            "threshold": thresholds[folds[i]],
        }
        for i in sorted(sets)
    ]
    origin = dict(zip(items.item_id, items.source.str.lower().map(ORIGINS), strict=True))
    summary = [
        {**meta, **r}
        for r in conformal_summary(
            system.name, sets, gold, {"gold origin": origin, "scheme": scheme}
        )
    ]
    return rows, summary


def verify_summary(name, rows, source):
    expected = source[source.system == name].set_index(["kind", "subset"])
    observed = pd.DataFrame(rows).set_index(["kind", "subset"])
    require(
        set(expected.index) == set(observed.index), f"{name}: conformal summary population mismatch"
    )
    for key, row in expected.iterrows():
        require(
            int(row.n_items) == int(observed.loc[key, "n_items"]),
            f"{name}: conformal summary count mismatch",
        )
        for metric in ("coverage", "mean_size", "singleton_share"):
            require(
                np.isclose(
                    float(row[metric]),
                    float(observed.loc[key, metric]),
                    rtol=0,
                    atol=0.000050000001,
                ),
                f"{name}: conformal source aggregate differs at serialization precision ({metric})",
            )


def saved_distributions(system):
    if "distribution" not in system.rows:
        return None
    result = {}
    for item, row in system.rows.loc[system.eligible].iterrows():
        raw = row.distribution
        if not isinstance(raw, dict) and (pd.isna(raw) or raw == ""):
            return None
        masses = json.loads(raw) if isinstance(raw, str) else raw
        require(isinstance(masses, dict), "invalid saved conformal distribution")
        masses = {t: probability(v, "conformal distribution") for t, v in masses.items()}
        result[item] = decide(
            masses, max(0.0, 1 - sum(masses.values())), "saved genuine distribution"
        )
    return result


def export_conformal(
    evaluation, items, registry, aggregations, out, dependencies=(), calculation_resources=None
):
    """Opt-in export of already calculated genuine B5 aggregations; no fitting or replay.

    A fitted-system caller must profile its separate deterministic calculation and
    provide its dependency files here. This function never initiates that calculation.
    """
    from .analysis import peak_memory_bytes

    started = time.perf_counter()
    inputs = Inputs()
    inputs.add(evaluation.path / "manifest.json")
    inputs.add(evaluation.path / "verdicts.csv")
    inputs.add(evaluation.path / "main_table.csv")
    saved_predictions = inputs.csv(
        evaluation.path / "verdicts.csv", ["system", "item_id", "fold", "verdict", "confidence"]
    )
    unique(saved_predictions, ["system", "item_id"], "conformal source verdicts")
    for version in {evaluation.systems[n].version for n in aggregations}:
        for file in sorted(registry[version].glob("*.yaml")):
            inputs.add(file)
    source = selected_systems(evaluation, inputs)
    require(source is not None, "conformal source aggregates unavailable")
    require(
        bool(aggregations) and set(aggregations) <= set(source.system),
        "unselected conformal systems",
    )
    for path in dependencies:
        inputs.add(path)
    for path in (Path(__file__), Path(__file__).with_name("uncertainty.py")):
        inputs.add(path)
    out = Path(out).resolve()
    require(not overlaps(out, evaluation.path), "conformal export overlap with source evaluation")
    require(not out.exists(), "conformal export directory already exists")
    rows, distributions = [], []
    for name, values in aggregations.items():
        system = evaluation.systems[name]
        saved = saved_predictions[saved_predictions.system == name].set_index("item_id")
        require(set(saved.index) == set(values), "conformal source prediction population mismatch")
        for item, aggregation in values.items():
            row = saved.loc[item]
            require(
                terminal(row.verdict, None) == aggregation.verdict,
                "conformal export/saved prediction disagreement",
            )
            confidence = probability(row.confidence, "conformal source confidence")
            require(
                (aggregation.confidence is None and pd.isna(confidence))
                or (
                    aggregation.confidence is not None
                    and np.isclose(confidence, aggregation.confidence, rtol=0, atol=1e-12)
                ),
                "conformal export/saved confidence disagreement",
            )
            require(
                row.fold == evaluation.manifest["folds"]["assignment"][item],
                "conformal export/saved fold disagreement",
            )
        own, summary = calculate(evaluation, system, items, registry, values)
        verify_summary(name, summary, source)
        rows.extend(own)
        distributions.extend(
            dict(system=name, item_id=i, distribution=json.dumps(a.distribution, sort_keys=True))
            for i, a in sorted(values.items())
        )
    inputs.verify()
    require(
        not any(overlaps(out, p) for p in inputs.files), "conformal export overlap with dependency"
    )
    out.mkdir(parents=True)
    pd.DataFrame(rows, columns=ITEM_COLUMNS).to_csv(
        out / "conformal_items.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(distributions).to_csv(
        out / "conformal_distributions.csv", index=False, lineterminator="\n"
    )
    exported = Inputs()
    for name in ("conformal_items.csv", "conformal_distributions.csv"):
        exported.add(out / name)
    record = {
        "format": "genuine-conformal-v1",
        "evaluation": evaluation.name,
        "source_manifest_sha256": inputs.files[(evaluation.path / "manifest.json").resolve()],
        "seed": evaluation.manifest["seed"],
        "resamples": evaluation.manifest["resamples"],
        "folds": evaluation.manifest["folds"],
        "alpha": ALPHA,
        "frozen_items_sha256": frozen_fingerprint(items),
        "systems": {n: evaluation.systems[n].version for n in aggregations},
        "inputs": inputs.records(),
        "exports": exported.records(),
        "calculation": (
            "existing uncertainty.conformal_sets on supplied genuine aggregations; no fitting"
        ),
        "replay": "not performed by this export; any fitted calculation is separate",
        "source_calculation_resources": calculation_resources,
        "resource_measurements": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_memory_bytes": peak_memory_bytes(),
        },
    }
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out


def frozen_fingerprint(items):
    columns = ["item_id", "gold_scheme", "gold_fallacy", "source"]
    data = items[columns].sort_values("item_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(data.encode()).hexdigest()


def export_calculated_conformal(
    config, runs_dir, evaluation_path, items, capture, out, dependencies, calculation_resources=None
):
    """Companion hook on the existing round-two calculation, explicitly opted in."""
    from .analysis_inputs import read_evaluation, scheme_registry

    inputs = Inputs()
    registry = scheme_registry(inputs)
    source = dict(path=Path(evaluation_path), round=2, populations={})
    evaluation = read_evaluation(config.name, source, config, items, registry, inputs)
    sources = []
    for key in ("answers", "answers_schemes_2", "ssoft"):
        names = evaluation.manifest.get("runs", {}).get(key) or []
        for name in [names] if isinstance(names, str) else names:
            directory = Path(runs_dir) / name
            sources.extend(
                [
                    directory / "manifest.json",
                    directory / ("ssoft.csv" if key == "ssoft" else "summary.csv"),
                ]
            )
    return export_conformal(
        evaluation,
        items,
        registry,
        capture,
        out,
        [*dependencies, *sources, *inputs.files],
        calculation_resources,
    )


def read_export(path, evaluation, inputs):
    record = inputs.json(path / "manifest.json")
    require(record.get("format") == "genuine-conformal-v1", "unsupported conformal export format")
    require(record.get("evaluation") == evaluation.name, "conformal export evaluation mismatch")
    require(
        record.get("source_manifest_sha256")
        == inputs.files[inputs.add(evaluation.path / "manifest.json")],
        "conformal export source manifest hash mismatch",
    )
    for name in ("seed", "resamples", "folds"):
        require(
            record.get(name) == evaluation.manifest.get(name), f"conformal export {name} mismatch"
        )
    require(record.get("alpha") == ALPHA, "conformal export alpha mismatch")
    for dependency in record.get("inputs", []):
        path_to_input = inputs.add(dependency["path"])
        require(
            inputs.files[path_to_input] == dependency["sha256"],
            "conformal dependency hash mismatch",
        )
    exports = {Path(r["path"]).name: r["sha256"] for r in record.get("exports", [])}
    for name in ("conformal_items.csv", "conformal_distributions.csv"):
        require(
            name in exports and inputs.files[inputs.add(path / name)] == exports[name],
            f"{name}: conformal export hash mismatch",
        )
    rows = inputs.csv(
        path / "conformal_items.csv", ITEM_COLUMNS, dtype={"item_id": str, "scheme_version": str}
    )
    unique(rows, ["system", "item_id"], "conformal item export")
    distributions = inputs.csv(
        path / "conformal_distributions.csv", ["system", "item_id", "distribution"]
    )
    unique(distributions, ["system", "item_id"], "conformal distribution export")
    require(
        set(rows.system) == set(distributions.system) == set(record.get("systems", {})),
        "conformal export system mismatch",
    )
    for name, version in record["systems"].items():
        require(
            name in evaluation.systems and evaluation.systems[name].version == version,
            "conformal export version mismatch",
        )
    return distributions, rows


def conformal_tables(config, evaluations, items, registry, human, inputs):
    rows, summaries, status = [], [], {}
    for evaluation in evaluations:
        export = config.evaluations[evaluation.name].get("conformal_records")
        source = selected_systems(evaluation, inputs)
        if source is None:
            require(export is None, "configured conformal export has no compatible B5 source")
            status[evaluation.name] = {"status": "pending", "reason": "no saved B5 aggregates"}
            continue
        exported, exported_items = (
            read_export(export, evaluation, inputs) if export is not None else (None, None)
        )
        if export is not None:
            record = inputs.json(export / "manifest.json")
            require(
                record.get("frozen_items_sha256") == frozen_fingerprint(items),
                "conformal export frozen population hash mismatch",
            )
        if exported is not None:
            require(
                set(exported.system) <= set(source.system), "unselected conformal export systems"
            )
        for name in source.system.unique():
            system = evaluation.systems[name]
            if exported is not None and name in set(exported.system):
                own = exported[exported.system == name].set_index("item_id").distribution
                values = {
                    i: decide(json.loads(v), 0.0, "validated exact export") for i, v in own.items()
                }
                provenance = "explicit genuine-conformal-v1 export"
            else:
                values = saved_distributions(system)
                provenance = "saved genuine verdict distributions"
            key = f"{evaluation.name}/{name}"
            if values is None:
                status[key] = {
                    "status": "pending",
                    "reason": (
                        "exact item sets unavailable; opt-in export of genuine distributions "
                        "required; "
                        "rounded aggregates/thresholds cannot reconstruct them"
                    ),
                }
                continue
            own_rows, summary = calculate(evaluation, system, items, registry, values)
            verify_summary(name, summary, source)
            if exported_items is not None and name in set(exported_items.system):
                expected = exported_items[exported_items.system == name].set_index("item_id")
                require(
                    set(expected.index) == {r["item_id"] for r in own_rows},
                    "conformal set export population mismatch",
                )
                for row in own_rows:
                    saved = expected.loc[row["item_id"]]
                    for column in ITEM_COLUMNS:
                        if column == "item_id":
                            continue
                        require(
                            saved[column] == row[column],
                            f"conformal exported set/threshold mismatch: {column}",
                        )
            rows.extend({**r, "source": provenance} for r in own_rows)
            summaries.extend(
                {**r, "source": provenance, "status": "computed; source aggregate verified"}
                for r in summary
            )
            if human is not None:
                direct = human.verdict.reindex(items.item_id)
                known = direct.notna().to_numpy()
                disagreement = direct.to_numpy() != items.gold_fallacy.to_numpy()
                frame = pd.DataFrame(own_rows).set_index("item_id").reindex(items.item_id)
                for value in (False, True):
                    chosen = known & (disagreement == value)
                    summaries.append(
                        dict(
                            evaluation=evaluation.name,
                            system=name,
                            scheme_version=system.version,
                            kind="observed human disagreement",
                            subset=str(value),
                            n_items=int(chosen.sum()),
                            mean_size=frame.loc[chosen, "size"].mean(),
                            source=provenance,
                            status="new exploratory analysis" if chosen.any() else "undefined",
                        )
                    )
            status[key] = {"status": "computed", "source": provenance, "aggregate_verified": True}
            del values
    return {
        "conformal_items.csv": table_frame(rows, ITEM_COLUMNS),
        "conformal.csv": table_frame(summaries, SUMMARY_COLUMNS),
    }, status
