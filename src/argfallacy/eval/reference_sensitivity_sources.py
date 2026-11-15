"""Inventory of frozen verdicts for the integrated-reference evaluation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..env import load_env
from ..paths import REPO_ROOT
from .analysis_inputs import Inputs, require, unique
from .canonical import predicted_scheme_class
from .compat import compat_metrics
from .prior import SPACES, read_prior_runs

EVALS = REPO_ROOT / "evaluations"
ROUND1 = "decision_v2"
REVISED = "exploratory_aggregators_v3"
ROUND2 = "round2"
B3_DIAGNOSTIC = "b3_schemes2_diagnostic_20261029"
GOSDT_DIAGNOSTIC = "gosdt_schemes2_diagnostic_20261030"
ARCHIVE = EVALS / "_archive"
FINE = "fine"
GOLD_SCHEME = "gold_scheme"
PARTIAL_B1 = "B1.annotation_exclusion"
QWEN = "qwen3_8_27b"
METRICS = ("accuracy", "macro_f1", "coverage")


def relative(path: Path) -> str:
    return Path(os.path.relpath(path, REPO_ROOT)).as_posix()


def system_id(name: str, version: str) -> str:
    suffix = f"@{version}"
    return name if name.endswith(suffix) else name + suffix


@dataclass
class System:
    id: str
    model: str
    method: str
    version: str
    condition: str
    role: str
    supervision: str
    cohort: str
    spaces: tuple[str, ...]
    rows: pd.DataFrame
    source: Path
    source_key: str


@dataclass
class Historical:
    system_id: str
    path: Path
    key: str
    n_items: int
    scores: dict[str, float]
    tolerance: float


def method_and_model(name, manifest, description):
    method, _, model = name.split("@", maxsplit=1)[0].partition(":")
    if not model:
        combined = method.startswith(("A4", "B1", "B2", "B3", "GOSDT", "CART.binary"))
        combined |= method == "A3.combined"
        model = "+".join(manifest["entries"]) if combined else manifest["reference_model"]
    if method == "majority":
        model = ""
    if method.startswith("A5"):
        method = "A5.answers_only" if method.endswith("answers_only") else "A5.with_justifications"
    if method in ("majority", "ceiling"):
        supervision = "historical_verdict_in_sample"
    elif method.startswith(("A1", "A3", "A2.prior", "B1.prior", "B2", "B3", "GOSDT", "CART")):
        supervision = "historical_verdict_used_in_fit_or_selection"
    elif "supervision gold" in description:
        supervision = "historical_verdict_used_in_fit_or_selection"
    elif "supervision cq" in description:
        supervision = "human_cq"
    else:
        supervision = "fixed_or_unsupervised"
    return method, model, supervision


class Catalog:
    def __init__(self, items: pd.DataFrame, inputs: Inputs):
        self.items = items
        self.ids = set(items.item_id)
        self.inputs = inputs
        self.systems: dict[str, System] = {}
        self.sources: list[dict] = []
        self.history: list[Historical] = []
        self.compat: list[dict] = []
        self.prior_mapping: dict[str, list[str]] = {}
        self.missing: list[dict] = []

    def omit(self, name, version, path, status, reason):
        self.missing.append(
            {
                "system_id": name,
                "scheme_version": version,
                "status": status,
                "evidence": relative(path),
                "reason": reason,
            }
        )

    def add(self, system: System, expected_ids, evidence: Path, runs=None):
        rows = system.rows
        unique(rows, ["item_id"], f"{system.source} [{system.source_key}]")
        got = set(rows.item_id)
        require(
            got == expected_ids,
            (
                f"{system.source} [{system.source_key}]: unexpected item coverage; "
                f"missing {sorted(expected_ids - got)[:8]}, extra {sorted(got - expected_ids)[:8]}"
            ),
        )
        previous = self.systems.get(system.id)
        if previous:
            old = previous.rows.set_index("item_id").predicted.sort_index()
            new = rows.set_index("item_id").predicted.sort_index()
            require(old.equals(new), f"{system.source}: saved verdicts drift for {system.id}")
        else:
            self.systems[system.id] = system
        self.sources.append(
            {
                "system_id": system.id,
                "source_file": relative(system.source),
                "source_key": system.source_key,
                "source_role": "verified_duplicate" if previous else "scored",
                "n_item_rows": len(rows),
                "manifest_or_protocol": relative(evidence),
                "upstream_runs": json.dumps(runs or {}, sort_keys=True),
            }
        )

    def read_verdicts(self, source, columns):
        return pd.read_csv(
            self.inputs.add(source), usecols=columns, dtype=str, keep_default_na=False
        )

    def evaluation(self, folder, manifest, *, dq=False, diagnostic=False):
        root = EVALS / folder
        prefix = "dq_" if dq else ""
        source = root / f"{prefix}verdicts.csv"
        metric_path = root / ("metrics.csv" if diagnostic else f"{prefix}main_table.csv")
        history = self.inputs.csv(metric_path, ["system", "n_items"])
        if diagnostic:
            history = (
                history[history.subset.eq("all") & history.metric.isin(METRICS)]
                .pivot(index=["system", "n_items"], columns="metric", values="estimate")
                .reset_index()
            )
        unique(history, ["system"], str(metric_path))
        by_name = history.set_index("system").to_dict("index")
        if source.is_file():
            frame = self.read_verdicts(
                source, ["system", "item_id", "verdict", "fold", "abstention"]
            )
        else:
            frame = pd.DataFrame(columns=["system", "item_id", "verdict"])
        missing_names = set(by_name) - set(frame.system)
        for name in sorted(missing_names):
            version = str(by_name[name].get("schemes", manifest["schemes"].get("tag", "2.0")))
            self.omit(
                system_id(name, version),
                version,
                source,
                "not_recalculable",
                "Historical score exists without saved item verdicts in this export.",
            )
        for name, group in frame.groupby("system", sort=False):
            row = by_name.get(name, {})
            version = str(row.get("schemes", manifest["schemes"].get("tag", "2.0")))
            require(version in ("1.1", "2.0"), f"{source}: invalid scheme version for {name}")
            id_ = system_id(name, version)
            expected = self.ids
            cohort = "test"
            if name == PARTIAL_B1:
                excluded = manifest["cq_label_exclusion"]
                expected = set(self.items.loc[~self.items.gold_scheme.isin(excluded), "item_id"])
                cohort = "annotation_exclusion"
            if folder == GOSDT_DIAGNOSTIC and name.startswith(("GOSDT@", "CART.binary@")):
                id_ = system_id(name.split("@")[0] + ".original", version)
                cohort = "superseded_original_tree_run"
            description = row.get(
                "description", manifest.get("primary_configurations", {}).get(name, "")
            )
            method, model, supervision = method_and_model(name, manifest, description)
            role = row.get("role", "diagnostic" if not row or diagnostic else "main")
            if method in ("majority", "ceiling"):
                role = "oracle"
            prediction_columns = ["item_id", "verdict", "fold", "abstention"]
            rows = group[prediction_columns].rename(columns={"verdict": "predicted"})
            system = System(
                id_,
                model,
                method,
                version,
                GOLD_SCHEME,
                role,
                supervision,
                cohort,
                (FINE,),
                rows,
                source,
                name,
            )
            self.add(system, expected, root / "manifest.json", source_runs(name, version, manifest))
            if row:
                self.history.append(
                    Historical(
                        id_,
                        metric_path,
                        name,
                        int(row["n_items"]),
                        {k: float(row[k]) for k in METRICS},
                        1e-10 if diagnostic else 0.000050001,
                    )
                )

    def duplicate(self, path, evidence, key, id_, rows):
        if id_ not in self.systems:
            return
        reference = self.systems[id_]
        duplicate = System(
            id_,
            reference.model,
            reference.method,
            reference.version,
            reference.condition,
            reference.role,
            reference.supervision,
            reference.cohort,
            reference.spaces,
            rows,
            path,
            key,
        )
        self.add(duplicate, set(reference.rows.item_id), evidence)

    def archive(self):
        snapshot = ARCHIVE / "round2_recomputed_spec10_20261026"
        for name in ("verdicts.csv", "dq_verdicts.csv"):
            path = snapshot / name
            if not path.is_file():
                continue
            m = self.inputs.json(snapshot / "manifest.json")
            frame = self.read_verdicts(path, ["system", "item_id", "verdict"])
            main = self.inputs.csv(snapshot / name.replace("verdicts", "main_table"))
            versions = main.set_index("system").schemes.astype(str).to_dict()
            require(
                m["schemes"] == {"answers": "v1", "answers_schemes_2": "v2"},
                f"{snapshot}: unexpected scheme mapping",
            )
            unique(frame, ["system", "item_id"], str(path))
            for key, rows in frame.groupby("system", sort=False):
                prediction = rows[["item_id", "verdict"]].rename(columns={"verdict": "predicted"})
                self.duplicate(
                    path, snapshot / "manifest.json", key, system_id(key, versions[key]), prediction
                )
        aliases = {
            "direction_audit_20261021": {
                "diagram_v1": f"A0:{QWEN}@1.1",
                "diagram_v2": f"A0:{QWEN}@2.0",
                "direct": f"zeroshot:{QWEN}@1.1",
            },
            "judge_audit_20261021": {
                "diagram_full_v2": f"A0:{QWEN}@1.1",
                "diagram_full_v3": f"A0:{QWEN}@2.0",
                "direct": f"zeroshot:{QWEN}@1.1",
                "judge_full_v2": "A5@1.1",
                "judge_full_v2_answers_only": "A5.answers_only@1.1",
                "judge_full_v3": "A5@2.0",
                "judge_full_v3_answers_only": "A5.answers_only@2.0",
            },
        }
        for folder, mapping in aliases.items():
            path = ARCHIVE / folder / "predictions.csv"
            if not path.is_file():
                continue
            frame = self.read_verdicts(path, ["item_id", *mapping])
            evidence = path.with_name("manifest.json")
            if not evidence.is_file():
                evidence = path.with_name("README.md")
            self.inputs.add(evidence)
            for column, id_ in mapping.items():
                self.duplicate(
                    path,
                    evidence,
                    column,
                    id_,
                    frame[["item_id", column]].rename(columns={column: "predicted"}),
                )

    def prior(self):
        load_env()
        directory = os.getenv("PRIOR_RUNS_DIR")
        if not directory or not Path(directory).is_dir():
            self.omit(
                "prior runs",
                "prior",
                REPO_ROOT / "docs/specs/06_scorer.md",
                "not_recalculable",
                "PRIOR_RUNS_DIR is unavailable.",
            )
            return
        mapping = self.items[["item_id", "gold_scheme"]].assign(
            prior_id=self.items.eleni_id.str.split(";").str[0]
        )
        unique(mapping, ["prior_id"], "prior ID mapping")
        for run in read_prior_runs(directory):
            source = self.inputs.add(run.path / "results.csv")
            saved_path = run.path / "metrics.csv"
            saved = self.inputs.csv(saved_path, ["task", "n_samples"])
            self.check_compat(run, saved_path, saved)
            id_ = f"prior/{run.model}/{run.condition}"
            columns = ["id", "predicted_fallacy"] + (["predicted_scheme"] if run.pipeline else [])
            raw = run.rows[columns].rename(
                columns={"id": "prior_id", "predicted_fallacy": "predicted"}
            )
            table = mapping.merge(raw, on="prior_id", how="inner", validate="one_to_one")
            condition = "prior_predicted_scheme" if run.pipeline else "prior_direct_text_only"
            system = System(
                id_,
                run.model,
                run.condition,
                "prior",
                condition,
                "legacy",
                "prior_pipeline_or_direct",
                "test",
                SPACES,
                table,
                source,
                "predicted_fallacy joined by prior ID",
            )
            self.add(system, self.ids, saved_path)
            self.prior_mapping[id_] = sorted(set(raw.prior_id) - set(mapping.prior_id), key=int)
            if run.pipeline:
                matched = table[
                    table.predicted_scheme.map(lambda value: predicted_scheme_class(value)[0]).eq(
                        table.gold_scheme
                    )
                ]
                system = System(
                    id_ + "/scheme-correct",
                    run.model,
                    run.condition,
                    "prior",
                    "prior_scheme_correct",
                    "legacy",
                    "prior_pipeline_or_direct",
                    "scheme_correct",
                    SPACES,
                    matched,
                    source,
                    "predicted_fallacy on matched predicted scheme",
                )
                self.add(system, set(matched.item_id), saved_path)

    def check_compat(self, run, path, saved):
        actual = compat_metrics(run)
        unique(saved, ["task"], str(path))
        joined = saved.merge(
            actual,
            on="task",
            how="outer",
            suffixes=("_saved", "_reproduced"),
            indicator=True,
            validate="one_to_one",
        )
        require(joined._merge.eq("both").all(), f"{path}: compatibility task mismatch")
        for _, row in joined.iterrows():
            record = {
                "source_file": relative(path),
                "task": row["task"],
                "space": "collapsed_compat"
                if "collapsed" in row["task"]
                else ("schemes" if row["task"] == "scheme" else FINE),
            }
            for key in ("n_samples", "n_classes_gold", *METRICS[:2], "precision", "recall"):
                expected, got = row[f"{key}_saved"], row[f"{key}_reproduced"]
                require(abs(expected - got) <= 1e-9, f"{path}: compatibility mismatch on {key}")
                record.update({f"{key}_saved": expected, f"{key}_reproduced": got})
            self.compat.append(record)

    def human_diagnostics(self, notes: Path):
        root = notes / "Diagnosi CQ e diagrammi"
        source = root / "human_model_predictions.csv"
        manifest_path = root / "human_learning_manifest.json"
        if not source.is_file():
            self.omit(
                "human CQ diagnostics",
                "1.1",
                source,
                "not_recalculable",
                "External diagnostic item export unavailable.",
            )
            return
        manifest = self.inputs.json(manifest_path)
        rows = self.inputs.csv(source, ["cohort", "system", "item_id", "verdict"], dtype=str)
        scores = root / "human_model_metrics.csv"
        historical = self.inputs.csv(scores)
        history = historical[historical.scheme.eq("all")].set_index(["cohort", "system"])
        unique(rows, ["cohort", "system", "item_id"], str(source))
        ids = {}
        for cohort, field in (
            ("human_complete476", "items_complete"),
            ("human_resolved438", "items_resolved"),
        ):
            cohort_rows = rows[rows.cohort.eq(cohort)]
            expected = set(cohort_rows.loc[cohort_rows.system.eq("manual_human"), "item_id"])
            require(
                len(expected) == manifest[field] and expected <= self.ids,
                f"{source}: invalid population for {cohort}",
            )
            ids[cohort] = expected
            for key, group in cohort_rows.groupby("system", sort=False):
                id_ = f"human_diagnostic/{cohort}/{key}"
                prediction = group[["item_id", "verdict", "fold"]].rename(
                    columns={"verdict": "predicted"}
                )
                if key.startswith("saved_"):
                    reference = self.systems[system_id(key.removeprefix("saved_"), "1.1")]
                    by_item = reference.rows.set_index("item_id").predicted
                    require(
                        prediction.predicted.eq(prediction.item_id.map(by_item)).all(),
                        f"{source}: restricted verdicts differ for {key}",
                    )
                    model, method = reference.model, reference.method
                    supervision = reference.supervision
                else:
                    model, method = "human CQ answers", key
                    supervision = (
                        "fixed_diagram"
                        if key == "manual_human"
                        else "historical_verdict_used_in_fit_or_selection"
                    )
                system = System(
                    id_,
                    model,
                    method,
                    "1.1",
                    "gold_scheme_human_diagnostic",
                    "diagnostic",
                    supervision,
                    cohort,
                    (FINE,),
                    prediction,
                    source,
                    f"{cohort}/{key}",
                )
                self.add(system, expected, manifest_path)
                prior = history.loc[(cohort, key)]
                self.history.append(
                    Historical(
                        id_,
                        scores,
                        system.source_key,
                        int(prior["n"]),
                        {k: float(prior[k]) for k in METRICS},
                        1e-10,
                    )
                )
        require(
            ids["human_resolved438"] <= ids["human_complete476"],
            f"{source}: resolved cohort is not a subset of complete cases",
        )


def source_runs(name, version, manifest):
    if name in manifest.get("dq", {}).get("systems", {}):
        return manifest["dq"]["systems"][name]
    if name.startswith("zeroshot:"):
        return manifest["zeroshot_sources"][name.split(":")[1]]
    runs = manifest["runs"]
    answer = runs.get("answers_schemes_2" if version == "2.0" else "answers", runs["answers"])
    if name.startswith("A5"):
        answers_only = name.endswith("answers_only")
        judge = [run for run in runs["judge"] if run.endswith("answers_only") == answers_only]
        return {"answers": answer, "judge": judge}
    if name.startswith("B4"):
        return runs["ssoft_nohistory" if name == "B4.nohistory" else "ssoft"]
    if name in ("B2", "B2.A0"):
        return {"answers": answer, "zeroshot": manifest["zeroshot_sources"][QWEN]}
    if name == "majority":
        return {}
    return {"answers": answer}


def unavailable(catalog, round2, revised, notes):
    source = EVALS / ROUND2 / "manifest.json"
    for entry in sorted(set(revised["entries"]) - set(revised["zeroshot_sources"])):
        catalog.omit(
            f"zeroshot:{entry}@2.0",
            "2.0",
            EVALS / REVISED / "manifest.json",
            "no_historical_run",
            "Planned entry without completed 2.0 direct verdicts.",
        )
    for name in (
        "B1.stable_cqs",
        "S-soft with a temperature",
        "S-soft of the other four entries",
        "the predicted scheme condition",
        "conformal sets on the direct-verdict annotation",
    ):
        catalog.omit(name, "", source, "no_historical_run", round2["missing"][name])
    for name in ("GOSDT", "CART.binary"):
        catalog.omit(
            f"{name}@2.0 (F21 tie correction)",
            "2.0",
            EVALS / GOSDT_DIAGNOSTIC / "summary.md",
            "not_recalculable",
            "Corrected aggregate scores are documented; the item export is the "
            "superseded original. Corrected verdicts would require a replay with fits.",
        )
    for name, filename, reason in (
        (
            "B2 routing sweeps",
            "routing_curve.csv",
            "Aggregate sweep; no verified verdict export per policy. Fixed B2 variants are scored.",
        ),
        (
            "D+Q routing sweeps",
            "dq_routing_curve.csv",
            "Aggregate sweep; no verified verdict export per policy. Fixed B2 variants are scored.",
        ),
        (
            "Human CQ substitution variants",
            "substitution.csv",
            "Aggregate interventions and random-draw curves without fixed item verdict exports.",
        ),
        ("B5 conformal", "conformal.csv", "Set-valued endpoint; exact item sets are pending."),
        (
            "B5 selective prediction",
            "abstention_curves.csv",
            "Coverage-indexed curves over already scored verdicts, a different endpoint.",
        ),
    ):
        status = "different_endpoint" if name.startswith("B5") else "not_recalculable"
        catalog.omit(name, "", EVALS / ROUND2 / filename, status, reason)
    catalog.omit(
        "Stage one",
        "",
        EVALS / "analyses_v1/stage1.csv",
        "different_endpoint",
        "Scheme predictions are scored against scheme labels.",
    )
    if notes:
        catalog.omit(
            "Human CQ no-feature control",
            "1.1",
            notes / "Diagnosi CQ e diagrammi/learning.md",
            "not_recalculable",
            "Aggregate scores without saved item verdicts.",
        )
        catalog.omit(
            "E3 selective policies",
            "2.0",
            notes / "E3_astensione/README.md",
            "different_endpoint",
            "Risk and accuracy conditional on a coverage cut. "
            "Underlying fixed verdicts are included from their evaluation sources.",
        )


def inventory(items: pd.DataFrame, inputs: Inputs, notes: Path | None) -> Catalog:
    catalog = Catalog(items, inputs)
    manifests = {
        folder: inputs.json(EVALS / folder / "manifest.json")
        for folder in (ROUND1, REVISED, ROUND2, B3_DIAGNOSTIC, GOSDT_DIAGNOSTIC)
    }
    for folder in (ROUND1, REVISED, ROUND2):
        catalog.evaluation(folder, manifests[folder])
    round2 = manifests[ROUND2]
    catalog.evaluation(ROUND2, round2, dq=True)
    for folder in (B3_DIAGNOSTIC, GOSDT_DIAGNOSTIC):
        manifest = manifests[folder] | {
            "schemes": {"tag": "2.0"},
            "runs": round2["runs"],
            "entries": round2["entries"],
            "reference_model": round2["reference_model"],
        }
        catalog.evaluation(folder, manifest, diagnostic=True)
    for collection in (ROUND1, REVISED, ROUND2):
        manifest = manifests[collection]
        for runs in (manifest["runs"], manifest.get("dq", {}).get("runs", {})):
            for value in runs.values():
                for name in value if isinstance(value, list) else [value]:
                    if name:
                        inputs.json(REPO_ROOT / "runs" / name / "manifest.json")
    catalog.archive()
    catalog.prior()
    if notes:
        catalog.human_diagnostics(notes)
    unavailable(catalog, round2, manifests[REVISED], notes)
    return catalog
