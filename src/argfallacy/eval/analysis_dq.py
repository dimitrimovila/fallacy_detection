"""Read dedicated saved D+Q tables; never evaluate or select runs implicitly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..client.run import schemes_hash
from ..schemes import load_all, load_vocabulary
from .analysis_inputs import SavedSystem, boolean, probability, read_run, require, terminal, unique
from .dq import CQ_COLUMNS, MAIN_COLUMNS, RUNS, VERDICT_COLUMNS
from .evaluate import Scorer

DQ_SCHEMAS = {
    "dq_verdicts.csv": VERDICT_COLUMNS,
    "dq_main_table.csv": MAIN_COLUMNS,
    "dq_deciding.csv": [
        "hypothesis",
        "system",
        "reference",
        "level",
        "computed",
        "outcome",
        "reading",
    ],
    "dq_deciding_breakdowns.csv": [
        "hypothesis",
        "kind",
        "subset",
        "system",
        "reference",
        "n_items",
        "macro_f1_difference",
    ],
    "dq_exploratory.csv": [
        "comparison",
        "system",
        "reference",
        "question",
        "subset",
        "n_items",
        "macro_f1_difference",
    ],
    "dq_consistency.csv": [
        "system",
        "kind",
        "subset",
        "n_items",
        "n_paths",
        "n_consistent",
        "consistent",
        "stopped",
    ],
    "dq_departures.csv": [
        "system",
        "scheme",
        "cq_id",
        "answer",
        "reason",
        "gold_origin",
        "count",
        "n_items",
    ],
    "dq_cq_level.csv": CQ_COLUMNS,
    "dq_cq_changes.csv": [
        "system",
        "reference",
        "scheme",
        "cq_id",
        "n_items",
        "changed",
        "first",
        "first_reference",
    ],
    "dq_replication.csv": ["system", "reference", "subset", "n_items", "agreement"],
    "dq_abstention.csv": ["kind", "system", "reference", "n_items", "augrc", "aurc"],
    "dq_abstention_curves.csv": ["system", "k", "coverage", "selective_risk", "generalized_risk"],
    "dq_routing_curve.csv": [
        "tau",
        "explained_share",
        "accuracy",
        "macro_f1",
        "requested_share",
        "attainable",
    ],
    "dq_choices.csv": ["fold", "component", "scheme", "parameter", "value", "training_items"],
}
SOURCE_COLUMNS = ["source_evaluation", "source_table", "source_sha256", "source_key"]


def _population(source, metadata, name, items):
    rule = source["populations"].get(name, metadata.get("populations", {}).get(name))
    if rule is None:
        return np.ones(len(items), bool)
    require(isinstance(rule, dict) and len(rule) == 1, f"{name}: invalid population declaration")
    if "item_ids" in rule:
        require(set(rule["item_ids"]) <= set(items.item_id), f"{name}: unknown population items")
        return items.item_id.isin(rule["item_ids"]).to_numpy()
    require(set(rule) == {"exclude_schemes"}, f"{name}: invalid population rule")
    require(
        set(rule["exclude_schemes"]) <= set(items.gold_scheme),
        f"{name}: unknown population schemes",
    )
    return ~items.gold_scheme.isin(rule["exclude_schemes"]).to_numpy()


def validate_mapping(name, version, value, metadata, evaluation):
    """An @2.0 or rerun identity never falls back to the old zero-shot source."""
    if name in ("C2.dq", "C2@2.0", "C2.dq@2.0"):
        require(version == ("1.1" if name == "C2.dq" else "2.0"), f"{name}: version mismatch")
        require(isinstance(value, dict), f"{name}: routing source mapping required")
        runs = evaluation.manifest.get("runs", {})
        require(
            value.get("fit_answers") == runs.get("answers"), f"{name}: fit source mapping mismatch"
        )
        key = "answers" if version == "1.1" else "answers_schemes_2"
        require(value.get("answers") == runs.get(key), f"{name}: answers mapping mismatch")
        zero = (
            evaluation.manifest.get("zeroshot_sources", {}).get(
                evaluation.manifest["reference_model"]
            )
            if version == "1.1"
            else metadata["systems"].get(f"zeroshot:{evaluation.manifest['reference_model']}@2.0")
        )
        require(
            value.get("zeroshot") == zero and zero is not None,
            f"{name}: zero-shot mapping mismatch",
        )
        if name != "C2@2.0":
            dq = metadata["systems"].get(
                f"DQ:{evaluation.manifest['reference_model']}"
                + ("@2.0" if version == "2.0" else "")
            )
            require(value.get("dq") == dq and dq is not None, f"{name}: D+Q mapping mismatch")
        if name == "C2.dq@2.0":
            require(
                value.get("ssoft") == runs.get("ssoft") and bool(value.get("ssoft")),
                f"{name}: S-soft mapping mismatch",
            )
        return
    require(isinstance(value, str), f"{name}: run mapping must be a string")
    prefix, sep, entry = name.partition(":")
    require(
        bool(sep) and prefix in ("DQ", "DQ.path", "DQ.soft", "zeroshot", "zeroshot.rerun"),
        f"{name}: unsupported D+Q system identity",
    )
    require(version == ("2.0" if name.endswith("@2.0") else "1.1"), f"{name}: version mismatch")
    role = (
        ("dq_schemes_2" if version == "2.0" else "dq")
        if prefix.startswith("DQ")
        else (
            "zeroshot_schemes_2" if prefix == "zeroshot" and version == "2.0" else "zeroshot_rerun"
        )
    )
    require(
        prefix != "zeroshot" or version == "2.0", f"{name}: original baseline is not a D+Q system"
    )
    require(value in metadata["runs"].get(role, []), f"{name}: exact run/version mapping mismatch")


def integrate_dq(config, evaluations, items, registry, inputs):
    empty = {
        name: pd.DataFrame(columns=[*columns, *SOURCE_COLUMNS])
        for name, columns in DQ_SCHEMAS.items()
    }
    if config.dq is None:
        return empty, {
            "status": "pending",
            "reason": "dq explicitly disabled; no dedicated inputs read",
        }
    matches = [e for e in evaluations if e.round == 2 and e.path == config.dq]
    require(
        len(matches) == 1,
        "dq must match exactly one explicitly selected round-two evaluation directory",
    )
    evaluation = matches[0]
    source = config.evaluations[evaluation.name]
    metadata = evaluation.manifest.get("dq")
    require(isinstance(metadata, dict), "missing manifest.dq mapping")
    require(
        metadata.get("family") == {"name": "H_DQ", "m": 1, "level": 0.95}, "invalid H_DQ family"
    )
    require(metadata.get("items") == len(items), "D+Q frozen population mismatch")
    assignment = evaluation.manifest["folds"]["assignment"]
    require(metadata.get("folds") == assignment, "D+Q fold mismatch")
    require(isinstance(metadata.get("systems"), dict), "D+Q systems mapping required")
    require(
        isinstance(metadata.get("runs"), dict) and set(metadata["runs"]) <= set(RUNS),
        "D+Q run mappings invalid",
    )
    run_records = {}
    for role, names in metadata["runs"].items():
        require(
            isinstance(names, list) and len(names) == len(set(names)),
            "D+Q run mapping must be a unique list",
        )
        folder, stage, _ = RUNS[role]
        for run_id in names:
            if run_id not in config.parsed_runs:
                run_records[run_id] = {"status": "pending", "reason": "parsed run not configured"}
                continue
            manifest, version, summary = read_run(
                config.parsed_runs[run_id], run_id, items, registry, inputs
            )
            require(registry[version].name == folder, f"{run_id}: version mapping mismatch")
            require(
                manifest.get("schemes_version", {}).get("content_sha256")
                == schemes_hash(registry[version]),
                f"{run_id}: scheme hash mismatch",
            )
            require(
                manifest.get("config", {}).get("stages") == [stage]
                and manifest.get("prompt_versions") == [stage + "_v1"],
                f"{run_id}: unsupported format",
            )
            require(set(summary.stage) <= {stage}, f"{run_id}: unsupported parsed stages")
            require("p_label" in summary, f"{run_id}: missing p_label column in spec-10 format")
            run_records[run_id] = {
                "status": "validated",
                "models_observed": sorted(set(summary.model)),
                "version": version,
                "source_commit_status": "recorded"
                if manifest.get("commit")
                else "unavailable in source",
            }
    tables = {}
    for name, columns in DQ_SCHEMAS.items():
        # Strings preserve the source's decimal serialization, including rounded summaries.
        tables[name] = inputs.csv(config.dq / name, columns, dtype=str)
    verdicts, main = tables["dq_verdicts.csv"], tables["dq_main_table.csv"]
    unique(verdicts, ["system", "item_id"], "D+Q verdicts")
    unique(main, ["system"], "D+Q main table")
    require(
        set(verdicts.system) == set(main.system) == set(metadata["systems"]),
        "D+Q system mapping/table mismatch",
    )
    require(not (set(main.system) & set(evaluation.systems)), "duplicate base/D+Q system keys")
    require(set(verdicts.item_id) <= set(items.item_id), "unknown D+Q items")
    vocab = load_vocabulary(check_schemes=False)
    for row in main.to_dict("records"):
        name, version = row["system"], str(row["schemes"])
        require(version in registry, f"{name}: unsupported version")
        provenance = metadata["systems"][name]
        validate_mapping(name, version, provenance, metadata, evaluation)
        if isinstance(provenance, str) and run_records[provenance]["status"] == "validated":
            entry = name.split(":", 1)[1].removesuffix("@2.0")
            require(
                entry in run_records[provenance]["models_observed"],
                f"{name}: source model absent from observed run",
            )
        mask = _population(source, metadata, name, items)
        own = verdicts[verdicts.system == name].copy()
        require(
            set(own.item_id) == set(items.loc[mask, "item_id"]),
            f"{name}: declared population mismatch",
        )
        require(int(row["n_items"]) == int(mask.sum()), f"{name}: main population mismatch")
        require(
            all(str(assignment[r.item_id]) == r.fold for r in own.itertuples()),
            f"{name}: verdict fold mismatch",
        )
        own["verdict"] = pd.Series(
            [terminal(v, vocab) for v in own.verdict], index=own.index, dtype=object
        )
        own["confidence"] = [probability(v, name) for v in own.confidence]
        schemes = load_all(registry[version])
        gold_scheme = items.set_index("item_id").gold_scheme
        require(
            all(
                pd.isna(r.verdict) or r.verdict in schemes[gold_scheme[r.item_id]].terminals
                for r in own.itertuples()
            ),
            f"{name}: verdict outside versioned menu",
        )
        own["confidence_source"] = np.where(
            own.confidence.notna(), "saved dq_verdicts.csv", "missing"
        )
        own["p_label"] = np.nan
        own["p_label_lower_bound"] = pd.Series("", index=own.index, dtype=object)
        own = own.set_index("item_id").reindex(items.item_id)
        # Validate any available main point metrics to the saved four-decimal precision.
        point = Scorer(items.gold_fallacy.tolist(), mask.astype(float)[None]).metrics(
            own.verdict.to_numpy()
        )
        for metric, value in point.items():
            if row.get(metric, "") != "":
                require(
                    abs(float(row[metric]) - value[0]) <= 0.000050000001,
                    f"{name}: saved main summary disagreement ({metric})",
                )
        evaluation.systems[name] = SavedSystem(
            name,
            version,
            own,
            mask,
            True,
            row["role"],
            row.get("description", ""),
            "dq_verdicts.csv",
            provenance,
        )
    for table, keys in [
        ("dq_deciding.csv", ["hypothesis"]),
        ("dq_exploratory.csv", ["comparison", "system", "reference", "subset"]),
        ("dq_consistency.csv", ["system", "kind", "subset"]),
        ("dq_abstention_curves.csv", ["system", "k"]),
    ]:
        unique(tables[table], keys, table)
    for table in ("dq_deciding.csv", "dq_exploratory.csv"):
        for row in tables[table].to_dict("records"):
            if table == "dq_deciding.csv":
                require(
                    row["hypothesis"] == "H_DQ" and float(row["level"]) == 0.95,
                    "H_DQ family/level mismatch",
                )
                row.update(kind="deciding", family="H_DQ", family_m=1, read=True)
            else:
                if row["subset"] not in ("all", "total", ""):
                    continue
                row.update(
                    kind="exploratory", family="saved D+Q exploratory", family_m="", level=0.95
                )
            row["source_table"] = table
            evaluation.pairs.append(row)
    imported = {}
    for name, frame in tables.items():
        frame = frame.copy()
        for column in ("system", "reference"):
            if column in frame and name != "dq_cq_level.csv":
                named = set(frame[column]) - {""}
                require(
                    named
                    <= set(evaluation.systems)
                    | set(metadata["missing"] if "missing" in metadata else {})
                    | ({"full run v2", "full run v3"} if name == "dq_cq_changes.csv" else set()),
                    f"{name}: unknown source system keys",
                )
        text_columns = {
            "system",
            "reference",
            "role",
            "schemes",
            "description",
            "hypothesis",
            "component",
            "comparison",
            "question",
            "subset",
            "kind",
            "outcome",
            "reason",
            "reading",
            "scheme",
            "cq_id",
            "answer",
            "gold_origin",
            "parameter",
            "value",
            "training_items",
            "item_id",
            "verdict",
            "explanation",
        }
        bool_columns = {"computed", "improvement", "flagged", "attainable"}
        for column in frame:
            if column in text_columns:
                continue
            for value in frame[column]:
                if value == "":
                    continue
                if column in bool_columns:
                    boolean(value, f"{name}/{column}")
                    continue
                try:
                    number = float(value)
                except ValueError:
                    require(False, f"{name}: invalid numeric value in {column}")
                require(np.isfinite(number) or column == "tau", f"{name}: nonfinite {column}")
                if column in (
                    "abstention",
                    "confidence",
                    "coverage",
                    "accuracy",
                    "macro_f1",
                    "balanced_accuracy",
                    "good_recall",
                    "good_precision",
                ):
                    probability(value, f"{name}/{column}")
                if column in (
                    "n_items",
                    "n",
                    "brier_n",
                    "count",
                    "n_paths",
                    "n_consistent",
                    "n_other",
                    "k",
                    "fold",
                ):
                    require(
                        number >= 0 and number.is_integer(), f"{name}: invalid count in {column}"
                    )
        digest = inputs.files[(config.dq / name).resolve()]
        frame["source_evaluation"], frame["source_table"], frame["source_sha256"] = (
            evaluation.name,
            name,
            digest,
        )
        frame["source_key"] = [f"{evaluation.name}/{name}/{k}" for k in range(len(frame))]
        imported[name] = frame
    return imported, {
        "status": "computed",
        "evaluation": evaluation.name,
        "path": str(config.dq),
        "systems": sorted(metadata["systems"]),
        "runs": run_records,
        "family": metadata["family"],
        "tables": {n: len(f) for n, f in imported.items()},
        "new_breakdowns": "exploratory recomputation from saved item predictions",
        "exact_routing": (
            "pending: rounded dq_routing_curve/dq_choices are summaries; "
            "explicit full-precision export required"
        ),
        "source_commit_status": "recorded"
        if evaluation.manifest.get("commit")
        else "unavailable in source",
    }
