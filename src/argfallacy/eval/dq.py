"""D+Q comparisons, written after the independent round-two tables."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..aggregate import LOGPROB, SAMPLE, Aggregation, a0, a2_pruned, load_burden, routes
from ..aggregate.soft import source_of
from ..annotations import load_annotations
from ..client import read_manifest, schemes_hash
from ..parse import INVALID
from ..paths import SCHEMES_ROOT
from ..prompts import DQ, ZEROSHOT, render_zeroshot
from ..schemes import SchemeError, load_all, load_vocabulary
from .b1 import cq_evaluation, cq_labels
from .dq_routing import routing_systems
from .evaluate import METRICS, Scores, difference, weights
from .round2 import ORIGINS, SMALL_SCHEME, Subset, System, _difference, _finite_interval
from .uncertainty import LEVELS, abstention, cumulative_errors

RUNS = {
    "zeroshot_schemes_2": ("v2", ZEROSHOT, "zeroshot"),
    "zeroshot_rerun": ("v1", ZEROSHOT, "zeroshot.rerun"),
    "dq": ("v1", DQ, "DQ"),
    "dq_schemes_2": ("v2", DQ, "DQ"),
}
VERDICT_COLUMNS = [
    "system",
    "item_id",
    "fold",
    "verdict",
    "confidence",
    "abstention",
    "explanation",
]
MAIN_COLUMNS = [
    "system",
    "role",
    "schemes",
    "n_items",
    *[c for m in METRICS for c in (m, f"{m}_low", f"{m}_high")],
    "description",
]
COMPARISON_COLUMNS = [
    "comparison",
    "system",
    "reference",
    "question",
    "subset",
    "n_items",
    "macro_f1_difference",
    "macro_f1_low",
    "macro_f1_high",
]
CQ_COLUMNS = [
    "kind",
    "system",
    "reference",
    "scheme",
    "n",
    "brier_n",
    "agreement",
    "agreement_low",
    "agreement_high",
    "brier",
    "brier_low",
    "brier_high",
    "agreement_difference",
    "brier_difference",
]


def _missing(value):
    return value is None or value == "" or (isinstance(value, float) and np.isnan(value))


def _number(value):
    return None if _missing(value) else float(value)


def _name(prefix, entry, folder):
    return f"{prefix}:{entry}" + ("@2.0" if folder == "v2" else "")


def dq_traversals(scheme, rows, routing, source):
    """The existing hard and pruned soft traversals, without CQ verbal confidence."""
    return a0(scheme, rows), (
        a2_pruned(scheme, rows, routing, source, 0.02) if source in (LOGPROB, SAMPLE) else None
    )


def consistency_case(scheme, rows, declared):
    """The first hard arc losing the declared terminal, or a stop while it is reachable."""
    path = a0(scheme, rows)
    result = {
        "consistent": path.verdict is not None and path.verdict == declared,
        "stopped": path.verdict is None,
        "departure_cq": None,
        "departure_answer": None,
        "departure_reason": None,
    }
    if result["consistent"]:
        return result
    if declared is None:
        result["departure_reason"] = "no declared verdict"
        return result
    reachable = {}

    def terminals(node):
        if node not in reachable:
            reachable[node] = set().union(
                *[
                    {edge.target} if edge.is_terminal else terminals(edge.target)
                    for edge in scheme.nodes[node].values()
                ]
            )
        return reachable[node]

    node = scheme.start
    while True:
        answer = rows.get(node, {}).get("answer", INVALID)
        edge = scheme.nodes[node].get(answer)
        if edge is None:
            reason = "stop"
        elif declared not in ({edge.target} if edge.is_terminal else terminals(edge.target)):
            reason = "arc"
        else:
            if edge.is_terminal:
                return result
            node = edge.target
            continue
        result.update(departure_cq=node, departure_answer=answer, departure_reason=reason)
        return result


def dq_decision(results: Mapping[str, Scores], reference: str, missing=None):
    primary, control = f"DQ:{reference}@2.0", f"zeroshot:{reference}@2.0"
    row = {
        "hypothesis": "H_DQ",
        "component": "DQ",
        "system": primary,
        "reference": control,
        "level": 0.95,
        "computed": False,
        "outcome": "missing",
        "reason": "",
        "reading": "not applicable",
        "secondary_difference": np.nan,
        "improvement": False,
    }
    for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
        row.update(
            dict.fromkeys([f"{metric}_difference", f"{metric}_low", f"{metric}_high"], np.nan)
        )
    if primary not in results or control not in results:
        row["reason"] = "; ".join(
            (missing or {}).get(name, f"no {name}")
            for name in (primary, control)
            if name not in results
        )
        return row
    row["computed"] = True
    for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
        values = difference(results[primary], results[control], metric, 0.05)
        row.update({f"{metric}_{k}": v for k, v in values.items()})
    secondary, repeated = f"DQ:{reference}", f"zeroshot.rerun:{reference}"
    if secondary in results and repeated in results:
        row["secondary_difference"] = (
            results[secondary].observed["macro_f1"] - results[repeated].observed["macro_f1"]
        )
    if row["macro_f1_low"] > 0:
        row.update(
            outcome="improvement",
            improvement=True,
            reading="robust" if row["secondary_difference"] > 0 else "possibly tuning",
        )
    elif row["macro_f1_high"] < 0:
        row["outcome"] = "cost of the CQs in the prompt"
    else:
        row["outcome"] = "no measurable difference"
    return row


@dataclass
class RunRows:
    summary: pd.DataFrame
    models: dict
    runs: dict


def read_dq_runs(config, runs_dir, items):
    """Validate versions before accepting complete reference runs and complete entries."""
    runs_dir = Path(runs_dir)
    ids = set(items.item_id)
    scheme_of = items.set_index("item_id").gold_scheme.to_dict()
    groups, missing = {}, {}
    for key, (folder, stage, _) in RUNS.items():
        names = config.runs.get(key, [])
        if not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names):
            raise SchemeError(f"runs.{key} must be a list of run IDs")
        frames, models, origins = [], {}, {}
        for run in names:
            directory = runs_dir / run
            manifest = read_manifest(directory)
            if manifest is None or not (directory / "summary.csv").is_file():
                raise SchemeError(f"{run}: missing manifest.json or summary.csv")
            version = manifest.get("schemes_version") or {}
            if version.get("folder") != folder:
                raise SchemeError(f"{run}: expected schemes folder {folder}")
            if version.get("content_sha256") != schemes_hash(SCHEMES_ROOT / folder):
                raise SchemeError(f"{run}: missing or mismatched schemes content hash")
            if manifest.get("prompt_versions") != [f"{stage}_v1"]:
                raise SchemeError(f"{run}: expected prompt version {stage}_v1")
            if manifest.get("config", {}).get("stages") != [stage]:
                raise SchemeError(f"{run}: expected the standalone stage {stage}")
            frame = pd.read_csv(directory / "summary.csv", dtype={"item_id": str})
            required = {
                "item_id",
                "stage",
                "scheme_condition",
                "scheme",
                "cq_id",
                "model",
                "answer",
                "p_label",
            }
            if absent := required - set(frame.columns):
                raise SchemeError(f"{run}: summary.csv missing columns {sorted(absent)}")
            frame = frame[
                (frame.stage == stage)
                & (frame.scheme_condition == "gold")
                & frame.item_id.isin(ids)
            ].copy()
            frame["cq_id"] = frame.cq_id.fillna("")
            if frame.duplicated(["model", "item_id", "cq_id"]).any():
                raise SchemeError(f"{run}: duplicate verdict or CQ rows")
            if (frame.scheme != frame.item_id.map(scheme_of)).any():
                raise SchemeError(f"{run}: rows under a scheme other than the gold scheme")
            verdicts = frame[frame.cq_id == ""]
            absent = sorted(
                ids - set(verdicts.loc[verdicts.model == config.reference_model, "item_id"])
            )
            if absent:
                missing[f"{key}/{run}"] = (
                    "incomplete run; missing reference verdict items: " + ", ".join(absent)
                )
                continue
            for entry in sorted(frame.model.unique()):
                own = frame[frame.model == entry]
                absent = sorted(ids - set(own.loc[own.cq_id == "", "item_id"]))
                if absent:
                    missing[f"{key}/{run}/{entry}"] = (
                        "incomplete entry; missing verdict items: " + ", ".join(absent)
                    )
                    continue
                if entry in models:
                    raise SchemeError(f"runs.{key}: two runs supply entry {entry}")
                if entry not in manifest.get("models", {}):
                    raise SchemeError(f"{run}: entry {entry} absent from its manifest")
                frames.append(own)
                models[entry] = manifest["models"][entry]
                origins[entry] = run
        if frames:
            groups[key] = RunRows(pd.concat(frames, ignore_index=True), models, origins)
        elif not names:
            missing[key] = f"no run (runs.{key} is empty)"
    return groups, missing


def _declared(value, scheme, run, item):
    if _missing(value) or value == INVALID:
        return None
    if value not in scheme.terminals:
        raise SchemeError(f"{run}, item {item}: verdict {value!r} is outside its scheme's menu")
    return value


def _systems(groups, schemes_by_folder, items, missing, reference):
    systems, provenance, cq_rows = {}, {}, {}
    scheme_of = items.set_index("item_id").gold_scheme.to_dict()
    ids = list(items.item_id)
    for key, group in groups.items():
        folder, stage, prefix = RUNS[key]
        schemes = schemes_by_folder[folder]
        routing = {
            s: routes(schemes[s], table) for s, table in load_burden(folder, schemes).items()
        }
        for entry in sorted(group.models):
            frame = group.summary[group.summary.model == entry]
            run = group.runs[entry]
            declared, paths, softs, answers = {}, {}, {}, {}
            source = source_of(group.models[entry])
            for item, rows in frame.groupby("item_id", sort=False):
                scheme = schemes[scheme_of[item]]
                verdict = rows[rows.cq_id == ""].iloc[0]
                value = _declared(verdict.answer, scheme, run, item)
                confidence = _number(verdict.p_label)
                if confidence is not None and (
                    not np.isfinite(confidence) or not 0 <= confidence <= 1
                ):
                    raise SchemeError(f"{run}, item {item}: invalid p_label {confidence}")
                declared[item] = Aggregation(
                    value,
                    confidence,
                    {},
                    1.0 if value is None else 0.0,
                    f"declared verdict ({run})",
                )
                if stage == DQ:
                    own = rows[rows.cq_id != ""].set_index("cq_id").to_dict("index")
                    for row in own.values():
                        if _missing(row.get("answer")):
                            row["answer"] = INVALID
                    if unknown := set(own) - set(scheme.cqs):
                        raise SchemeError(f"{run}, item {item}: unknown CQ keys {sorted(unknown)}")
                    answers[item] = own
                    paths[item], soft = dq_traversals(
                        scheme, own, routing[scheme.scheme_id], source
                    )
                    if soft is not None:
                        softs[item] = soft
            name = _name(prefix, entry, folder)
            role = (
                "primary" if stage == DQ and folder == "v2" and entry == reference else "comparison"
            )
            systems[name] = System(
                name,
                declared,
                role,
                "2.0" if folder == "v2" else "1.1",
                f"declared verdict, {entry}, {run}",
            )
            provenance[name] = run
            if stage == DQ:
                cq_rows[name] = answers
                for kind, values in (("DQ.path", paths), ("DQ.soft", softs)):
                    own_name = _name(kind, entry, folder)
                    if values:
                        systems[own_name] = System(
                            own_name,
                            values,
                            "comparison",
                            systems[name].schemes,
                            f"{kind}, {source}, {run}",
                        )
                        provenance[own_name] = run
                    else:
                        missing[own_name] = "no CQ soft source: the entry uses verbal confidence"
            assert set(declared) == set(ids)
    return systems, provenance, cq_rows


class Paired:
    """One item-count matrix, shared by sequential system and subset calculations."""

    def __init__(self, items, indices, without_indices):
        self.items = items
        self.ids = items.item_id.tolist()
        self.gold = items.gold_fallacy.tolist()
        self.indices = indices
        self.counts = weights(indices, len(items))
        self.full = Subset(self.gold, np.arange(len(items)), self.counts)
        self.without_indices = without_indices

    def subset(self, positions, independent=False):
        positions = np.asarray(positions, dtype=int)
        counts = (
            weights(self.without_indices, len(positions))
            if independent
            else self.counts[:, positions]
        )
        return Subset(self.gold, positions, counts)

    def ratio(self, numerator, denominator):
        numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
        total = self.counts @ denominator
        values = np.divide(
            self.counts @ numerator, total, out=np.full(len(total), np.nan), where=total > 0
        )
        low, high = _finite_interval(values)
        observed = numerator.sum() / denominator.sum() if denominator.sum() else np.nan
        return observed, low, high


def _legacy_systems(out, reference, ids):
    wanted = {
        f"A0:{reference}",
        f"A0:{reference}@2.0",
        f"zeroshot:{reference}",
        "A2.pruned",
        "B4",
        "B4.path",
        "B4.nohistory",
    }
    frame = pd.read_csv(
        Path(out) / "verdicts.csv",
        dtype={"item_id": str},
        usecols=["system", "item_id", "verdict", "confidence", "abstention"],
    )
    frame = frame[frame.system.isin(wanted) & frame.item_id.isin(ids)]
    result = {}
    for name, group in frame.groupby("system", sort=False):
        result[name] = {
            r.item_id: Aggregation(
                None if _missing(r.verdict) else r.verdict,
                _number(r.confidence),
                {},
                float(r.abstention),
                "saved round 2",
            )
            for r in group.itertuples(index=False)
        }
    return result


def _groups(items):
    yield "all", "all", np.arange(len(items)), False
    origins = items.source.str.lower().map(ORIGINS)
    for name in sorted(origins.unique()):
        yield "gold origin", name, np.flatnonzero(origins == name), False
    for name in sorted(items.gold_scheme.unique()):
        yield "scheme", name, np.flatnonzero(items.gold_scheme == name), False
    yield "without NLAS", "without NLAS", np.flatnonzero(items.source.str.lower() != "nlas"), True


def _save(out, name, rows, columns=None, precision=True):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    frame.to_csv(
        Path(out) / name,
        index=False,
        float_format="%.4f" if precision else None,
        lineterminator="\n",
    )


def dq_cq_evaluation(items, schemes, labels, predictions, reference, indices):
    """Use the existing item-sum CQ bootstrap, with separate missing-probability support."""
    pairs = predictions[["item_id", "scheme", "cq_id"]].drop_duplicates()
    population = SimpleNamespace(items=items, schemes=schemes, pairs=pairs, reference=reference)
    result = cq_evaluation(population, labels, predictions, indices=indices, allow_missing=True)
    level = result.level.assign(kind="level", reference="")
    differences = result.differences.assign(kind="difference")
    return pd.concat([level, differences], ignore_index=True).reindex(columns=CQ_COLUMNS)


def _consistency_tables(systems, cq_rows, schemes, paired):
    rows, departures = [], []
    ids, gold = paired.ids, paired.full.observed
    scheme_of = paired.items.set_index("item_id").gold_scheme.to_dict()
    for name, answers in cq_rows.items():
        suffix = "@2.0" if name.endswith("@2.0") else ""
        entry = name.split(":", 1)[1].removesuffix(suffix) if suffix else name.split(":", 1)[1]
        path_name, soft_name = f"DQ.path:{entry}{suffix}", f"DQ.soft:{entry}{suffix}"
        declared = systems[name].aggregations
        path = systems[path_name].aggregations
        soft = systems[soft_name].aggregations if soft_name in systems else None
        folder = "v2" if suffix else "v1"
        cases = [
            consistency_case(schemes[folder][scheme_of[i]], answers[i], declared[i].verdict)
            for i in ids
        ]
        consistent = np.array([c["consistent"] for c in cases])
        stopped = np.array([c["stopped"] for c in cases])
        declared_right = gold.correct([declared[i].verdict for i in ids])
        path_right = gold.correct([path[i].verdict for i in ids])
        mass = np.array(
            [soft[i].distribution.get(declared[i].verdict, 0.0) for i in ids]
            if soft is not None
            else np.full(len(ids), np.nan)
        )
        maximum = np.array(
            [
                declared[i].verdict is not None
                and (best := max(soft[i].distribution.values(), default=0.0)) > 0
                and soft[i].distribution.get(declared[i].verdict, 0.0) >= best - 1e-12
                for i in ids
            ]
            if soft is not None
            else np.full(len(ids), np.nan)
        )
        for kind, subset, positions, _ in _groups(paired.items):
            if kind == "without NLAS":
                continue
            mask = np.zeros(len(ids), dtype=bool)
            mask[positions] = True
            row = {
                "system": name,
                "kind": kind,
                "subset": subset,
                "n_items": len(positions),
                "n_paths": int((mask & ~stopped).sum()),
                "n_consistent": int((mask & consistent).sum()),
                "n_other": int((mask & ~consistent).sum()),
            }
            metrics = {
                "consistent": (consistent, mask),
                "stopped": (stopped, mask),
                "agreement_on_paths": (consistent, mask & ~stopped),
                "declared_accuracy_consistent": (declared_right, mask & consistent),
                "path_accuracy_consistent": (path_right, mask & consistent),
                "declared_accuracy_other": (declared_right, mask & ~consistent),
                "path_accuracy_other": (path_right, mask & ~consistent),
            }
            if soft is not None:
                metrics.update(declared_mass=(mass, mask), declared_is_max=(maximum, mask))
            for metric, (values, denominator) in metrics.items():
                value, low, high = paired.ratio(np.asarray(values) * denominator, denominator)
                row.update({metric: value, f"{metric}_low": low, f"{metric}_high": high})
            rows.append(row)
        buckets = defaultdict(list)
        origins = paired.items.source.str.lower().map(ORIGINS).to_numpy()
        for p, case in enumerate(cases):
            if not case["consistent"]:
                buckets[
                    (
                        scheme_of[ids[p]],
                        case["departure_cq"] or "",
                        case["departure_answer"] or "",
                        case["departure_reason"],
                    )
                ].append(p)
        for (sid, cq, answer, reason), positions in sorted(buckets.items()):
            for origin in ["all", *sorted(set(origins))]:
                denominator = paired.items.gold_scheme.to_numpy() == sid
                if origin != "all":
                    denominator &= origins == origin
                numerator = np.zeros(len(ids))
                numerator[positions] = 1
                count = int((numerator * denominator).sum())
                if not count:
                    continue
                value, low, high = paired.ratio(numerator * denominator, denominator)
                departures.append(
                    {
                        "system": name,
                        "scheme": sid,
                        "cq_id": cq,
                        "answer": answer,
                        "reason": reason,
                        "gold_origin": origin,
                        "count": count,
                        "n_items": int(denominator.sum()),
                        "share": value,
                        "share_low": low,
                        "share_high": high,
                    }
                )
    consistency_columns = [
        "system",
        "kind",
        "subset",
        "n_items",
        "n_paths",
        "n_consistent",
        "n_other",
        *[
            c
            for m in (
                "consistent",
                "stopped",
                "agreement_on_paths",
                "declared_accuracy_consistent",
                "path_accuracy_consistent",
                "declared_accuracy_other",
                "path_accuracy_other",
                "declared_mass",
                "declared_is_max",
            )
            for c in (m, f"{m}_low", f"{m}_high")
        ],
    ]
    return (
        pd.DataFrame(rows).reindex(columns=consistency_columns),
        pd.DataFrame(
            departures,
            columns=[
                "system",
                "scheme",
                "cq_id",
                "answer",
                "reason",
                "gold_origin",
                "count",
                "n_items",
                "share",
                "share_low",
                "share_high",
            ],
        ),
    )


def _reference_rows(config, runs_dir, key, items):
    run = config.runs.get(key)
    if not run:
        return {}
    frame = pd.read_csv(Path(runs_dir) / run / "summary.csv", dtype={"item_id": str})
    frame = frame[
        (frame.model == config.reference_model)
        & (frame.stage == "stage2")
        & (frame.scheme_condition == "gold")
        & frame.item_id.isin(items.item_id)
    ]
    return {
        i: group.set_index("cq_id").to_dict("index")
        for i, group in frame.groupby("item_id", sort=False)
    }


def _cq_predictions(items, schemes, answers, system):
    records = []
    for item in items.itertuples(index=False):
        scheme = schemes[item.gold_scheme]
        for cq in scheme.cq_order:
            row = answers.get(item.item_id, {}).get(cq, {})
            value = row.get("p_logprob", row.get(f"p_logprob_{scheme.answer_space(cq)[0]}"))
            records.append(
                {
                    "item_id": item.item_id,
                    "scheme": scheme.scheme_id,
                    "cq_id": cq,
                    "answer": row.get("answer", INVALID),
                    "p_first": np.nan if _missing(value) else float(value),
                    "system": system,
                    "supervision": "none",
                }
            )
    return pd.DataFrame(records)


def _cq_changes(name, comparison, answers, before, items, schemes, paired):
    by_question = defaultdict(list)
    for p, item in enumerate(items.itertuples(index=False)):
        scheme = schemes[item.gold_scheme]
        for cq in scheme.cq_order:
            left, right = (
                answers.get(item.item_id, {}).get(cq),
                before.get(item.item_id, {}).get(cq),
            )
            if left is not None and right is not None:
                a, b = left.get("answer", INVALID), right.get("answer", INVALID)
                first = scheme.answer_space(cq)[0]
                by_question[(scheme.scheme_id, cq)].append((p, a != b, a == first, b == first))
    rows = []
    for (scheme, cq), values in sorted(by_question.items()):
        positions, changed, first, first_before = map(np.asarray, zip(*values, strict=True))
        denominator = np.bincount(positions.astype(int), minlength=len(items))
        row = {
            "system": name,
            "reference": comparison,
            "scheme": scheme,
            "cq_id": cq,
            "n_items": len(values),
        }
        for metric, values in (
            ("changed", changed),
            ("first", first),
            ("first_reference", first_before),
        ):
            numerator = np.bincount(positions.astype(int), weights=values, minlength=len(items))
            value, low, high = paired.ratio(numerator, denominator)
            row.update({metric: value, f"{metric}_low": low, f"{metric}_high": high})
        rows.append(row)
    return rows


def _identical_prompts(items, schemes):
    vocabulary = load_vocabulary(check_schemes=False)
    return np.array(
        [
            render_zeroshot(r._asdict(), schemes["v1"][r.gold_scheme], vocabulary=vocabulary).text
            == render_zeroshot(
                r._asdict(), schemes["v2"][r.gold_scheme], vocabulary=vocabulary
            ).text
            for r in items.itertuples(index=False)
        ]
    )


def _abstention_tables(systems, paired, reference):
    rows, curves, resampled = [], [], {}
    eligible = {name: s for name, s in systems.items() if not name.startswith("DQ.path:")}
    for name, system in eligible.items():
        values = system.aggregations
        wrong = ~paired.full.observed.correct([values[i].verdict for i in paired.ids])
        row, curve, order = abstention(name, paired.ids, wrong, values, paired.indices)
        rows.append({"kind": "system", **row})
        curves.append(curve)
        if name in (
            f"DQ:{reference}",
            f"DQ:{reference}@2.0",
            f"zeroshot:{reference}@2.0",
            f"zeroshot.rerun:{reference}",
        ):
            resampled[name] = cumulative_errors(wrong, order, paired.indices).mean(axis=1) / len(
                paired.ids
            )
    by_name = {r["system"]: r for r in rows}
    for left, right in (
        (f"DQ:{reference}", f"zeroshot.rerun:{reference}"),
        (f"DQ:{reference}@2.0", f"zeroshot:{reference}@2.0"),
    ):
        if left in resampled and right in resampled:
            low, high = _finite_interval(resampled[left] - resampled[right])
            rows.append(
                {
                    "kind": "difference",
                    "system": left,
                    "reference": right,
                    "n_items": len(paired.ids),
                    "augrc_difference": by_name[left]["augrc"] - by_name[right]["augrc"],
                    "augrc_low": low,
                    "augrc_high": high,
                }
            )
    columns = [
        "kind",
        "system",
        "reference",
        "n_items",
        "with_verdict",
        *[
            c
            for m in ("augrc", "aurc", *[f"selective_accuracy_{round(100 * v)}" for v in LEVELS])
            for c in (m, f"{m}_low", f"{m}_high")
        ],
        "augrc_difference",
    ]
    return (
        pd.DataFrame(rows).reindex(columns=columns),
        pd.concat(curves, ignore_index=True)
        if curves
        else pd.DataFrame(
            columns=["system", "k", "coverage", "selective_risk", "generalized_risk"]
        ),
    )


def evaluate_dq(
    config,
    runs_dir,
    out,
    items,
    folds,
    indices,
    without_indices,
    labels=None,
    annotations_path=None,
    progress=None,
):
    """Write only dq_ tables and D+Q metadata; all legacy CSVs are read-only inputs."""
    if progress:
        progress("D+Q")
    out, runs_dir = Path(out), Path(runs_dir)
    items = items.sort_values("item_id").reset_index(drop=True)
    reference = config.reference_model
    groups, missing = read_dq_runs(config, runs_dir, items)
    schemes = {folder: load_all(SCHEMES_ROOT / folder) for folder in ("v1", "v2")}
    systems, provenance, cq_rows = _systems(groups, schemes, items, missing, reference)
    for key, (folder, _, prefix) in RUNS.items():
        name = _name(prefix, reference, folder)
        if name not in systems:
            missing[name] = (
                "; ".join(str(v) for k, v in missing.items() if k == key or k.startswith(key + "/"))
                or f"no {name}"
            )
    paired = Paired(items, indices, without_indices)
    ids = paired.ids
    legacy = _legacy_systems(out, reference, ids)
    if groups and labels is None:
        if annotations_path is None:
            raise SchemeError("D+Q evaluation requires the frozen CQ labels")
        labels = cq_labels(load_annotations(annotations_path), items, schemes["v1"])
    routed, routing_origins, curve, choices, explained, routing_missing = routing_systems(
        config,
        runs_dir,
        items,
        schemes,
        systems,
        legacy,
        provenance,
        cq_rows,
        labels,
        folds,
        indices,
        consistency_case,
        progress,
    )
    systems.update(routed)
    provenance.update(routing_origins)
    missing.update(routing_missing)
    _save(out, "dq_routing_curve.csv", curve)
    _save(out, "dq_choices.csv", choices)
    available = {**legacy, **{n: s.aggregations for n, s in systems.items()}}
    predicted = {
        n: [values.get(i, Aggregation(None, None, {}, 1.0, "missing")).verdict for i in ids]
        for n, values in available.items()
    }
    results = {n: paired.full.score(values) for n, values in predicted.items()}
    main = []
    for name, system in systems.items():
        row = {
            "system": name,
            "role": system.role,
            "schemes": system.schemes,
            "n_items": len(ids),
            "description": system.description,
        }
        for metric in METRICS:
            low, high = _finite_interval(results[name].resampled[metric])
            row.update(
                {
                    metric: results[name].observed[metric],
                    f"{metric}_low": low,
                    f"{metric}_high": high,
                }
            )
        main.append(row)
    _save(out, "dq_main_table.csv", main, MAIN_COLUMNS)
    _save(
        out,
        "dq_verdicts.csv",
        [
            {
                "system": name,
                "item_id": i,
                "fold": folds[i],
                "verdict": a.verdict,
                "confidence": a.confidence,
                "abstention": a.abstention,
                "explanation": a.explanation,
            }
            for name, system in systems.items()
            for i in ids
            for a in [system.aggregations[i]]
        ],
        VERDICT_COLUMNS,
        precision=False,
    )
    decision = dq_decision(results, reference, missing)
    _save(out, "dq_deciding.csv", [decision])
    breakdowns = []
    left, right = f"DQ:{reference}@2.0", f"zeroshot:{reference}@2.0"
    if left in results and right in results:
        for kind, subset, positions, independent in _groups(items):
            if not len(positions):
                continue
            part = paired.subset(positions, independent)
            a, b = part.score(predicted[left]), part.score(predicted[right])
            row = {
                "hypothesis": "H_DQ",
                "kind": kind,
                "subset": subset,
                "system": left,
                "reference": right,
                "n_items": len(positions),
                "n_classes": part.classes,
                "flagged": kind == "scheme" and len(positions) < SMALL_SCHEME,
            }
            for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
                row.update({f"{metric}_{k}": v for k, v in _difference(a, b, metric).items()})
            breakdowns.append(row)
    breakdown_columns = [
        "hypothesis",
        "kind",
        "subset",
        "system",
        "reference",
        "n_items",
        "n_classes",
        "flagged",
        *[
            c
            for m in ("macro_f1", "accuracy", "balanced_accuracy")
            for c in (f"{m}_difference", f"{m}_low", f"{m}_high")
        ],
    ]
    _save(out, "dq_deciding_breakdowns.csv", breakdowns, breakdown_columns)
    consistency, departures = _consistency_tables(systems, cq_rows, schemes, paired)
    _save(out, "dq_consistency.csv", consistency)
    _save(out, "dq_departures.csv", departures)

    # Reference-only comparisons, on the paired draws already used by round two.
    identical = _identical_prompts(items, schemes)
    pairs = [
        ("1", f"DQ:{reference}", f"zeroshot.rerun:{reference}", "D+Q 1.1 against its control"),
        ("3", f"DQ.path:{reference}@2.0", f"A0:{reference}@2.0", "hard CQ traversal, 2.0"),
        ("3", f"DQ.soft:{reference}@2.0", "A2.pruned", "soft CQ traversal, 2.0"),
        ("3", f"DQ.path:{reference}", f"A0:{reference}", "hard CQ traversal, 1.1"),
        ("4", f"DQ.soft:{reference}@2.0", "B4", "D+Q against S-soft"),
        ("4", f"DQ.path:{reference}@2.0", "B4.path", "D+Q against the S-soft path"),
        ("5", f"DQ.soft:{reference}@2.0", "B4.nohistory", "one generation with verdicts in view"),
        ("8", f"zeroshot:{reference}@2.0", f"zeroshot:{reference}", "menus and server"),
        ("B4", "B4", f"zeroshot:{reference}@2.0", "S-soft against the matched 2.0 menus"),
        ("B2", "B2@2.0", f"zeroshot:{reference}@2.0", "two-side routing, 2.0"),
        ("B2", "B2.dq", f"zeroshot:{reference}", "three-side routing, 1.1"),
        ("B2", "B2.dq@2.0", f"zeroshot:{reference}@2.0", "four-side routing, 2.0"),
        ("B2", "B2.dq@2.0", f"DQ:{reference}@2.0", "routing against the D+Q side, 2.0"),
    ]
    exploratory = []
    for comparison, a, b, question in pairs:
        if a not in results or b not in results:
            missing[f"{a} against {b}"] = missing.get(a) or missing.get(b) or "no reference result"
            continue
        subsets = [("all", np.arange(len(items)))]
        if comparison == "8":
            subsets.append(("different prompts", np.flatnonzero(~identical)))
        for subset, positions in subsets:
            if not len(positions):
                continue
            part = paired.full if len(positions) == len(items) else paired.subset(positions)
            d = _difference(part.score(predicted[a]), part.score(predicted[b]), "macro_f1")
            exploratory.append(
                {
                    "comparison": comparison,
                    "system": a,
                    "reference": b,
                    "question": question,
                    "subset": subset,
                    "n_items": len(positions),
                    **{f"macro_f1_{k}": v for k, v in d.items()},
                }
            )
            if a in explained:
                value, low, high = paired.ratio(explained[a], np.ones(len(items)))
                exploratory[-1].update(
                    explained_share=value, explained_low=low, explained_high=high
                )
    _save(
        out,
        "dq_exploratory.csv",
        exploratory,
        [*COMPARISON_COLUMNS, "explained_share", "explained_low", "explained_high"],
    )

    replication = []
    for a, b, subset, mask in [
        (f"zeroshot:{reference}@2.0", f"zeroshot:{reference}", "identical prompts", identical),
        (f"zeroshot.rerun:{reference}", f"zeroshot:{reference}", "all", np.ones(len(ids), bool)),
    ]:
        if a not in available or b not in available:
            continue
        agreement = np.array(
            [
                available[a].get(i) is not None
                and available[a][i].verdict is not None
                and available[a][i].verdict
                == available[b].get(i, Aggregation(None, None, {}, 1, "")).verdict
                for i in ids
            ]
        )
        value, low, high = paired.ratio(agreement * mask, mask)
        replication.append(
            {
                "system": a,
                "reference": b,
                "subset": subset,
                "n_items": int(mask.sum()),
                "agreement": value,
                "agreement_low": low,
                "agreement_high": high,
            }
        )
    _save(
        out,
        "dq_replication.csv",
        replication,
        [
            "system",
            "reference",
            "subset",
            "n_items",
            "agreement",
            "agreement_low",
            "agreement_high",
        ],
    )

    # The CQ comparison retains the parsed p_logprob, including mass outside the arcs.
    old = _reference_rows(config, runs_dir, "answers", items)
    revised = _reference_rows(config, runs_dir, "answers_schemes_2", items)
    cq_table = pd.DataFrame(columns=CQ_COLUMNS)
    primary_cq = f"DQ:{reference}"
    if primary_cq in cq_rows:
        if labels is None:
            if annotations_path is None:
                raise SchemeError("D+Q CQ evaluation requires the frozen CQ labels")
            labels = cq_labels(load_annotations(annotations_path), items, schemes["v1"])
        predictions = pd.concat(
            [
                _cq_predictions(items, schemes["v1"], cq_rows[primary_cq], primary_cq),
                _cq_predictions(items, schemes["v1"], old, f"entry:{reference}"),
            ],
            ignore_index=True,
        )
        cq_table = dq_cq_evaluation(items, schemes["v1"], labels, predictions, reference, indices)
    else:
        missing["CQ labels, comparison 6"] = missing.get(primary_cq, "no D+Q 1.1 run")
    _save(out, "dq_cq_level.csv", cq_table)
    changes = []
    for name, before, version in [(primary_cq, old, "v1"), (f"DQ:{reference}@2.0", revised, "v2")]:
        if name in cq_rows:
            changes += _cq_changes(
                name,
                "full run v2" if version == "v1" else "full run v3",
                cq_rows[name],
                before,
                items,
                schemes[version],
                paired,
            )
    control = []
    for run in config.runs.get("ssoft_nohistory") or []:
        frame = pd.read_csv(runs_dir / run / "ssoft.csv", dtype={"item_id": str})
        control.append(frame[(frame.model == reference) & (frame.scheme_condition == "gold")])
    if control and f"DQ:{reference}@2.0" in cq_rows:
        frame = pd.concat(control, ignore_index=True)
        before = {i: g.set_index("cq_id").to_dict("index") for i, g in frame.groupby("item_id")}
        changes += _cq_changes(
            f"DQ:{reference}@2.0",
            "B4.nohistory",
            cq_rows[f"DQ:{reference}@2.0"],
            before,
            items,
            schemes["v2"],
            paired,
        )
    _save(
        out,
        "dq_cq_changes.csv",
        changes,
        [
            "system",
            "reference",
            "scheme",
            "cq_id",
            "n_items",
            *[
                c
                for m in ("changed", "first", "first_reference")
                for c in (m, f"{m}_low", f"{m}_high")
            ],
        ],
    )
    uncertainty, curves = _abstention_tables(
        {n: s for n, s in systems.items() if n not in routed}, paired, reference
    )
    _save(out, "dq_abstention.csv", uncertainty)
    _save(out, "dq_abstention_curves.csv", curves)
    record = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    record["dq"] = {
        "runs": {key: config.runs.get(key, []) for key in RUNS},
        "systems": provenance,
        "family": {"name": "H_DQ", "m": 1, "level": 0.95},
        "folds": folds,
        "items": len(items),
        "identical_prompts": int(identical.sum()),
        "different_prompts": int((~identical).sum()),
        "missing": missing,
        "cq_probability": "parsed p_logprob, including outside-arc mass; no imputation",
        "bootstrap": "shared round-two item indices; independent without-NLAS indices",
    }
    pointers = {
        "B2 with S-soft as another side": "B2.dq@2.0" in routed,
        "B2 with D+Q as another side": any(n in routed for n in ("B2.dq", "B2.dq@2.0")),
        "B2 on schemes 2.0": "B2@2.0" in routed,
        "B4 against the zero-shot with the 2.0 menus": "B4" in available
        and f"zeroshot:{reference}@2.0" in systems,
    }
    for entry, computed in pointers.items():
        if computed:
            record["missing"][entry] = "computed in the D+Q section (dq_exploratory.csv)"
        elif str(record["missing"].get(entry, "")).startswith("computed in the D+Q section"):
            record["missing"][entry] = "not computed; missing inputs listed in the D+Q section"
    (out / "manifest.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
