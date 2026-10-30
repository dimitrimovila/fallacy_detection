"""Validated, read-only adapters for saved evaluation and parser contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..client.run import SCHEMES_FOLDERS, schemes_hash, schemes_tag
from ..labels import normalize_scheme_label
from ..parse import INVALID
from ..paths import FALLACIES_FILE, SCHEME_LABELS_FILE, SCHEMES_ROOT
from ..schemes import SchemeError, load_vocabulary
from .canonical import _gold_terminal, predicted_fallacy_class
from .folds import assign_folds


def require(condition, message):
    if not condition:
        raise SchemeError(message)


def mapping(value, name, allowed):
    require(isinstance(value, dict), f"{name}: expected a mapping")
    require(not (unknown := set(value) - set(allowed)), f"{name}: unknown keys {sorted(unknown)}")
    return value


def integer(value, name, minimum=1, maximum=None):
    require(
        type(value) is int and value >= minimum and (maximum is None or value <= maximum),
        f"{name}: invalid integer {value!r}",
    )
    return value


def resolve_path(base, value):
    require(isinstance(value, str) and bool(value.strip()), "path must be a nonempty string")
    return (base / value).resolve()


def overlaps(left: Path, right: Path):
    return left == right or left in right.parents or right in left.parents


@dataclass(frozen=True)
class AnalysisConfig:
    path: Path
    items: Path
    output: Path
    evaluations: dict
    parsed_runs: dict[str, Path]
    seed: int
    resamples: int
    bootstrap_batch_size: int
    direct_verdict: dict
    dq: Path | None
    annotations: Path | None = None

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        mapping(
            raw,
            "analysis",
            {
                "version",
                "items",
                "output",
                "evaluations",
                "parsed_runs",
                "seed",
                "resamples",
                "bootstrap_batch_size",
                "direct_verdict",
                "dq",
                "annotations",
            },
        )
        require(
            type(raw.get("version")) is int and raw["version"] == 1, "analysis version must be 1"
        )
        base = path.parent
        evaluations = raw.get("evaluations")
        require(isinstance(evaluations, dict) and bool(evaluations), "evaluations: empty mapping")
        normal = {}
        for name, source in evaluations.items():
            require(isinstance(name, str) and bool(name), "evaluation identity must be a string")
            mapping(
                source,
                name,
                {"path", "round", "populations", "routing_records", "conformal_records"},
            )
            round_number = integer(source.get("round"), f"{name}.round", maximum=2)
            populations = source.get("populations", {})
            require(isinstance(populations, dict), f"{name}.populations: expected a mapping")
            for system, population in populations.items():
                mapping(population, f"{name}.{system}.population", {"exclude_schemes", "item_ids"})
                require(len(population) == 1, f"{name}.{system}: one population rule required")
                values = next(iter(population.values()))
                require(
                    isinstance(values, list)
                    and all(isinstance(v, str) for v in values)
                    and len(values) == len(set(values)),
                    f"{system}: invalid population list",
                )
            routing = source.get("routing_records")
            normal[name] = {
                "path": resolve_path(base, source.get("path")),
                "round": round_number,
                "populations": populations,
                "routing_records": resolve_path(base, routing) if routing is not None else None,
                "conformal_records": resolve_path(base, source["conformal_records"])
                if source.get("conformal_records") is not None
                else None,
            }
        runs = raw.get("parsed_runs", {})
        require(isinstance(runs, dict), "parsed_runs must be a mapping")
        require(all(isinstance(k, str) and k for k in runs), "invalid parsed run identity")
        direct = raw.get("direct_verdict")
        direct = {} if direct is None else direct
        mapping(direct, "direct_verdict", {"answers", "samples"})
        direct = {
            k: resolve_path(base, direct[k]) if direct.get(k) is not None else None
            for k in ("answers", "samples")
        }
        require(
            (direct["answers"] is None) == (direct["samples"] is None),
            "direct_verdict: both answers and samples must be supplied",
        )
        dq = raw.get("dq")
        return cls(
            path,
            resolve_path(base, raw.get("items")),
            resolve_path(base, raw.get("output", "../evaluations/analyses_v1")),
            normal,
            {k: resolve_path(base, v) for k, v in runs.items()},
            integer(raw.get("seed"), "seed", minimum=0),
            integer(raw.get("resamples", 10000), "resamples"),
            integer(raw.get("bootstrap_batch_size", 128), "bootstrap batch size", maximum=4096),
            direct,
            resolve_path(base, dq) if dq is not None else None,
            resolve_path(base, raw["annotations"]) if raw.get("annotations") is not None else None,
        )

    def guard_output(self, out):
        inputs = [
            self.path,
            self.items,
            *self.parsed_runs.values(),
            *(s["path"] for s in self.evaluations.values()),
            *(p for p in self.direct_verdict.values() if p is not None),
        ]
        if self.dq is not None:
            inputs.append(self.dq)
        if self.annotations is not None:
            inputs.append(self.annotations)
        inputs.extend(
            s["routing_records"]
            for s in self.evaluations.values()
            if s["routing_records"] is not None
        )
        inputs.extend(
            s["conformal_records"]
            for s in self.evaluations.values()
            if s["conformal_records"] is not None
        )
        for source in inputs:
            require(not overlaps(out, source), f"output overlap with input: {source}")


class Inputs:
    """Hash exactly the explicitly configured files consumed."""

    def __init__(self):
        self.files: dict[Path, str] = {}

    def add(self, path):
        path = Path(path).resolve()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        if path in self.files:
            require(self.files[path] == value, f"input changed during analysis: {path}")
        self.files[path] = value
        return path

    def csv(self, path, columns=(), dtype=None):
        path = self.add(path)
        frame = pd.read_csv(
            path,
            keep_default_na=False,
            dtype=dtype or {"item_id": str},
            float_precision="round_trip",
        )
        require(set(columns) <= set(frame), f"{path}: missing columns {set(columns) - set(frame)}")
        return frame

    def json(self, path):
        return json.loads(self.add(path).read_text(encoding="utf-8"))

    def verify(self):
        for path in list(self.files):
            self.add(path)

    def records(self):
        return [{"path": str(p), "sha256": value} for p, value in sorted(self.files.items())]


def unique(frame, keys, name):
    require(not frame.duplicated(keys).any(), f"{name}: duplicate keys {keys}")


def terminal(raw, vocab):
    return predicted_fallacy_class(None if raw in ("", INVALID) else raw, vocabulary=vocab)[0]


def probability(raw, name):
    if raw == "" or raw is None or pd.isna(raw):
        return np.nan
    try:
        result = float(raw)
    except (ValueError, TypeError):
        raise SchemeError(f"{name}: invalid confidence {raw!r}") from None
    require(np.isfinite(result) and 0 <= result <= 1, f"{name}: confidence outside [0, 1]")
    return result


def boolean(raw, name):
    require(
        raw in (True, False, "True", "False", "true", "false", "", 0, 1),
        f"{name}: invalid boolean {raw!r}",
    )
    return raw in (True, "True", "true", 1)


def read_items(config, inputs):
    items = inputs.csv(
        config.items, ["item_id", "in_test_set", "gold_scheme", "gold_fallacy", "source"]
    )
    unique(items, ["item_id"], "frozen items")
    require(items.item_id.ne("").all(), "empty item id")
    all_ids = items.item_id.tolist()
    test = items.in_test_set.map(lambda v: boolean(v, "in_test_set"))
    items = items.loc[test].sort_values("item_id").reset_index(drop=True)
    require(len(items) > 0, "empty frozen test population")
    vocab = load_vocabulary(check_schemes=False)
    items["gold_fallacy"] = [
        _gold_terminal(i, g, vocab) for i, g in zip(items.item_id, items.gold_fallacy, strict=True)
    ]
    for sid in items.gold_scheme:
        require(normalize_scheme_label(sid) == sid, f"noncanonical gold scheme: {sid}")
    items.attrs["all_item_ids"] = all_ids
    return items


def scheme_registry(inputs):
    registry = {}
    for directory in sorted(SCHEMES_ROOT / folder for folder in SCHEMES_FOLDERS):
        if directory.is_dir():
            for path in directory.glob("*.yaml"):
                inputs.add(path)
            registry[str(schemes_tag(directory))] = directory
    inputs.add(FALLACIES_FILE)
    inputs.add(SCHEME_LABELS_FILE)
    return registry


@dataclass
class SavedSystem:
    name: str
    version: str
    rows: pd.DataFrame
    eligible: np.ndarray
    main: bool
    role: str = ""
    description: str = ""
    source_table: str = "verdicts.csv"
    source_provenance: object = None


@dataclass
class SavedEvaluation:
    name: str
    round: int
    path: Path
    manifest: dict
    systems: dict[str, SavedSystem]
    pairs: list[dict] = field(default_factory=list)


def read_evaluation(name, source, config, items, registry, inputs):
    path = source["path"]
    m = inputs.json(path / "manifest.json")
    require(m.get("seed") == config.seed, f"{name}: seed differs from source")
    require(m.get("resamples") == config.resamples, f"{name}: resamples differ from source")
    require(m.get("round", 1) == source["round"], f"{name}: round mismatch")
    require(
        m.get("m") == (5 if source["round"] == 1 else 4),
        f"{name}: deciding family size differs from the recorded protocol",
    )
    require(m.get("items") == len(items), f"{name}: frozen population differs from source")
    require(
        m.get("items_without_nlas") == int(items.source.str.lower().ne("nlas").sum()),
        f"{name}: without-NLAS population differs from source",
    )
    folds = integer(m.get("folds", {}).get("k"), f"{name}.folds", minimum=2)
    assignment = m.get("folds", {}).get("assignment", {})
    require(
        assignment == assign_folds(items, config.seed, folds),
        f"{name}: fold assignment differs from frozen items/seed",
    )
    verdicts = inputs.csv(
        path / "verdicts.csv", ["system", "item_id", "fold", "verdict", "confidence"]
    )
    unique(verdicts, ["system", "item_id"], str(path / "verdicts.csv"))
    require(len(verdicts) > 0, f"{name}: empty saved prediction population")
    require(set(verdicts.item_id) <= set(items.item_id), f"{name}: unknown test item ids")
    require(
        all(assignment[i] == f for i, f in zip(verdicts.item_id, verdicts.fold, strict=True)),
        f"{name}: verdict fold mismatch",
    )
    main = inputs.csv(path / "main_table.csv", ["system", "n_items"])
    unique(main, ["system"], str(path / "main_table.csv"))
    meta = main.set_index("system").to_dict("index")
    default = m.get("schemes", {}).get("tag")
    vocab = load_vocabulary(check_schemes=False)
    systems = {}
    require(
        set(source["populations"])
        <= set(verdicts.system) | set(m.get("dq", {}).get("systems", {})),
        f"{name}: unknown population system",
    )
    for system, frame in verdicts.groupby("system", sort=False):
        row = meta.get(system, {})
        version = str(row.get("schemes", default))
        require(version in registry, f"{name}/{system}: unknown scheme version {version}")
        rule = source["populations"].get(system)
        if rule is None and system == "B1.annotation_exclusion":
            require(bool(m.get("cq_label_exclusion")), f"{name}: absent population declaration")
            rule = {"exclude_schemes": m["cq_label_exclusion"]}
        eligible = np.ones(len(items), bool)
        if rule:
            if "item_ids" in rule:
                require(set(rule["item_ids"]) <= set(items.item_id), "unknown population item ids")
                eligible = items.item_id.isin(rule["item_ids"]).to_numpy()
            else:
                for sid in rule["exclude_schemes"]:
                    require(
                        normalize_scheme_label(sid) == sid, f"noncanonical population scheme: {sid}"
                    )
                eligible = ~items.gold_scheme.isin(rule["exclude_schemes"]).to_numpy()
        declared = set(items.loc[eligible, "item_id"])
        require(set(frame.item_id) == declared, f"{name}/{system}: population mismatch")
        require(
            "n_items" not in row or row["n_items"] == len(declared),
            f"{name}/{system}: declared population count mismatch",
        )
        frame = frame.copy()
        try:
            frame["verdict"] = pd.Series(
                [terminal(v, vocab) for v in frame.verdict], index=frame.index, dtype=object
            )
        except SchemeError as error:
            raise SchemeError(f"{name}/{system}: label: {error}") from None
        frame["confidence"] = [probability(v, f"{name}/{system}") for v in frame.confidence]
        frame["confidence_source"] = np.where(
            frame.confidence.notna(), "saved verdicts.csv", "missing"
        )
        frame["p_label_lower_bound"] = pd.Series("", index=frame.index, dtype=object)
        frame["p_label"] = np.nan
        role = row.get("role", "")
        if not role:
            role = (
                "oracle"
                if system == "majority" or system.startswith("ceiling:")
                else "primary"
                if system in m.get("primary_configurations", {})
                else "reference"
                if system in meta
                else "diagnostic"
            )
        systems[system] = SavedSystem(
            system,
            version,
            frame.set_index("item_id").reindex(items.item_id),
            eligible,
            system in meta,
            role,
            row.get("description", m.get("primary_configurations", {}).get(system, "")),
        )
    require(set(meta) <= set(systems), f"{name}: main table systems missing from predictions")
    result = SavedEvaluation(name, source["round"], path, m, systems)
    for filename, kind in [
        ("deciding.csv", "deciding"),
        ("exploratory.csv", "exploratory"),
        ("diagnostic.csv", "diagnostic"),
    ]:
        if not (path / filename).is_file():
            continue
        pairs = inputs.csv(path / filename, ["system", "reference"])
        unique(pairs, ["system", "reference"], str(path / filename))
        for row in pairs.to_dict("records"):
            row.update(kind=kind, source_table=filename)
            if kind == "deciding":
                require(
                    float(row["level"]) == 1 - 0.05 / m["m"], f"{name}: deciding level mismatch"
                )
            result.pairs.append(row)
            if kind == "deciding" and result.round == 1 and "zeroshot_macro_f1_difference" in row:
                result.pairs.append(
                    {
                        "system": row["system"],
                        "reference": f"zeroshot:{m['reference_model']}",
                        "kind": "exploratory",
                        "source_table": filename,
                        "question": "primary against reference zero-shot",
                    }
                )
    return result


def read_run(path, run_id, items, registry, inputs):
    manifest = inputs.json(path / "manifest.json")
    require(manifest.get("run_id", run_id) == run_id, f"{path}: run identity mismatch")
    version = str(manifest.get("schemes_version", {}).get("tag"))
    require(version in registry, f"{run_id}: unknown scheme version {version}")
    digest = manifest.get("schemes_version", {}).get("content_sha256")
    require(
        not digest or digest == schemes_hash(registry[version]),
        f"{run_id}: scheme content hash/version mismatch",
    )
    summary = inputs.csv(
        path / "summary.csv",
        ["item_id", "model", "stage", "scheme_condition", "scheme", "cq_id", "answer"],
    )
    unique(
        summary,
        ["item_id", "model", "stage", "scheme_condition", "scheme", "cq_id"],
        str(path / "summary.csv"),
    )
    require(
        set(summary.item_id) <= set(items.attrs.get("all_item_ids", items.item_id)),
        f"{run_id}: unknown parsed item ids",
    )
    models = set(manifest.get("models", {})) | set(manifest.get("config", {}).get("models", []))
    require(set(summary.model) <= models, f"{run_id}: unknown model entries")
    allowed = {"stage1", "stage2", "zeroshot", "judge"}
    if "dq" in set(summary.stage):
        require(
            manifest.get("config", {}).get("stages") == ["dq"],
            f"{run_id}: unsupported parsed stages",
        )
        require(manifest.get("prompt_versions") == ["dq_v1"], f"{run_id}: D+Q prompt mismatch")
        require(digest == schemes_hash(registry[version]), f"{run_id}: D+Q scheme hash mismatch")
        require(set(summary.stage) == {"dq"}, f"{run_id}: mixed D+Q stages")
        allowed.add("dq")
    require(
        set(summary.stage) <= allowed,
        f"{run_id}: unsupported parsed stages",
    )
    gold = items.set_index("item_id").gold_scheme
    own = summary[
        (summary.scheme_condition == "gold")
        & (summary.stage != "stage1")
        & summary.item_id.isin(gold.index)
    ]
    require(
        all(gold[i] == s for i, s in zip(own.item_id, own.scheme, strict=True)),
        f"{run_id}: gold scheme mismatch",
    )
    return manifest, version, summary


def recover_confidence(evaluations, run_id, manifest, version, summary):
    vocab = load_vocabulary(check_schemes=False)
    for evaluation in evaluations:
        m = evaluation.manifest
        for system in evaluation.systems.values():
            stage, entry, source = None, None, None
            dq_source = m.get("dq", {}).get("systems", {}).get(system.name)
            if system.source_table == "dq_verdicts.csv" and isinstance(dq_source, str):
                if system.name.startswith(("DQ.path:", "DQ.soft:")):
                    continue
                stage = "dq" if system.name.startswith("DQ:") else "zeroshot"
                entry = system.name.split(":", 1)[1].removesuffix("@2.0")
                source = dq_source
            elif system.name.startswith("zeroshot:"):
                stage, entry = "zeroshot", system.name.split(":", 1)[1]
                source = m.get("zeroshot_sources", {}).get(entry)
            elif system.name in ("A5", "A5.answers_only"):
                settings = manifest.get("config", {}).get("judge") or {}
                judge_runs = m.get("runs", {}).get("judge", [])
                if isinstance(judge_runs, str):
                    judge_runs = [judge_runs]
                if run_id in judge_runs:
                    wanted = system.name == "A5"
                    if settings.get("justifications", True) == wanted:
                        judged = summary[
                            (summary.stage == "judge") & (summary.scheme_condition == "gold")
                        ]
                        require(
                            judged.model.nunique() == 1,
                            f"{run_id}: judge must have exactly one model",
                        )
                        stage, entry, source = "judge", judged.model.iloc[0], run_id
                        require(
                            settings.get("answers_from") == m["runs"]["answers"],
                            f"{run_id}: judge answers run mismatch",
                        )
            if source != run_id:
                continue
            require(system.version == version, f"{run_id}/{system.name}: version mismatch")
            rows = summary[
                (summary.stage == stage)
                & (summary.model == entry)
                & (summary.scheme_condition == "gold")
                & (summary.cq_id == "")
            ]
            unique(rows, ["item_id"], f"{run_id}/{system.name}")
            rows = rows.set_index("item_id")
            own = system.rows.loc[system.eligible]
            require(set(own.index) <= set(rows.index), f"{run_id}: missing confidence source items")
            for item, saved in own.iterrows():
                row = rows.loc[item]
                parsed_verdict = terminal(row.answer, vocab)
                require(
                    (parsed_verdict is None and pd.isna(saved.verdict))
                    or parsed_verdict == saved.verdict,
                    f"{run_id}/{system.name}/{item}: saved verdict disagreement",
                )
                value = probability(row.get("p_label", ""), f"{run_id}/{item}")
                # CSV readers can differ in the last binary digit; retain saved confidence.
                require(
                    pd.isna(saved.confidence)
                    or not np.isfinite(value)
                    or np.isclose(saved.confidence, value, rtol=0, atol=1e-12),
                    f"{run_id}/{system.name}/{item}: saved confidence disagreement",
                )
                lower = row.get("p_label_lower_bound", "")
                if lower != "":
                    boolean(lower, "p_label_lower_bound")
                system.rows.loc[item, "p_label_lower_bound"] = lower
                system.rows.loc[item, "p_label"] = value
                if pd.isna(saved.confidence) and np.isfinite(value):
                    system.rows.loc[item, "confidence"] = value
                    system.rows.loc[item, "confidence_source"] = "summary.csv:p_label"
                system.rows.loc[item, "confidence_run"] = run_id
