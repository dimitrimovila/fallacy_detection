"""``argfallacy evaluate`` on a round 2 configuration: C1 to C5 and their comparisons.

C1, C2 and C3 on full run v2, their exploratory configurations on full run v2 and
full run v3, C4 and its control on the S-soft runs when they exist, the uncertainty
of C5, the four deciding comparisons at level 1 - 0.05/m and what is reported
beside them at 95 per cent.  The command reads ``summary.csv``, ``ssoft.csv`` and
``manifest.json`` of each run, never ``raw.jsonl``.  Folds and resamples come from
the seed as in the first stage, so the folds are those of round 1 and every
comparison is paired.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..aggregate import (
    LOGPROB,
    Aggregation,
    a0,
    a2,
    a2_pruned,
    a4,
    good_terminal,
    load_burden,
    routes,
    source_of,
    sources_of,
    ssoft,
)
from ..aggregate.c1 import (
    C_GRID,
    LAMBDA_GRID,
    PRIMARY_CONFIG,
    C1Result,
    c1_out_of_fold,
    configurations,
    retraverse,
    transfer_c1,
)
from ..aggregate.c1_inputs import build_dataset
from ..aggregate.crossfit import C1CrossFitter, ReferenceInputs
from ..aggregate.interface import decide
from ..aggregate.routing import Side, c2_out_of_fold
from ..aggregate.tree import TREE_GRID, c3_out_of_fold
from ..annotations import ANNOTATIONS_FILE, ITEMS_FILE, load_annotations
from ..client.run import GOLD, read_manifest, schemes_dir_of
from ..prompts import JUDGE
from ..schemes import SchemeError, load_all
from .baselines import majority_by
from .c1 import cq_evaluation, cq_labels, entry_answers, human_path_reference
from .evaluate import ALPHA as FAMILY_ALPHA
from .evaluate import (
    EVALUATIONS_DIR,
    METRICS,
    NLAS,
    Bench,
    EvaluationConfig,
    Scorer,
    Scores,
    _answer_rows,
    _commit,
    _read_run,
    _single,
    _zeroshot_answers,
    deciding,
    difference,
    interval,
    reference_of,
    resample_indices,
    weights,
)
from .folds import assign_folds
from .prior import load_test_set
from .routing import routing_curve
from .uncertainty import (
    ALPHA,
    LEVELS,
    abstention,
    composition,
    conformal_sets,
    conformal_summary,
)

ROUND2_RUNS = ("answers_schemes_2", "ssoft", "ssoft_nohistory")
"""The run keys that only a round 2 configuration has."""
PRUNING = 0.02
"""The pruning threshold of S-soft and of A2.pruned."""
SMALL_SCHEME = 30
"""A scheme with fewer test items is flagged in the breakdowns."""
FALLACY_CORPORA, SCHEME_CORPORA = "fallacy corpora", "scheme corpora"
ORIGINS = {"logic": FALLACY_CORPORA, "reddit": FALLACY_CORPORA,
           "elecdebate": FALLACY_CORPORA, "nlas": SCHEME_CORPORA, "ethix": SCHEME_CORPORA}
"""The gold origin of each corpus: entered with a fallacy label, or with a scheme."""


def is_round2(config: EvaluationConfig) -> bool:
    return any(key in config.runs for key in ROUND2_RUNS)


# --------------------------------------------------------------- deciding


@dataclass(frozen=True)
class Hypothesis:
    """A deciding comparison; ``after`` names the comparison that must pass first."""

    name: str
    component: str
    system: str
    reference: str
    after: str | None = None


def hypotheses(reference: str) -> tuple[Hypothesis, ...]:
    """The six comparisons of the four components, in the order they are read."""
    zero, diagram = f"zeroshot:{reference}", f"A0:{reference}"
    return (
        Hypothesis("H1.1", "C1", "C1", diagram),
        Hypothesis("H1.2", "C1", "C1", zero, "H1.1"),
        Hypothesis("H2", "C2", "C2", zero),
        Hypothesis("H3.1", "C3", "C3", "C1.hard"),
        Hypothesis("H3.2", "C3", "C3", zero, "H3.1"),
        Hypothesis("H4", "C4", "C4", "A2.pruned"),
    )


def _outcome(row: Mapping[str, Any]) -> str:
    if row["macro_f1_low"] > 0:
        return "improvement"
    if row["macro_f1_high"] < 0:
        return ("cost of the diagram" if row["reference"].startswith("zeroshot:")
                else "below the reference")
    return "no measurable difference"


def deciding_family(results: Mapping[str, Scores], family: Sequence[Hypothesis], m: int,
                    missing: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Every comparison of the family at level 1 - 0.05/m, with accuracy and balanced
    accuracy beside it at 95 per cent.

    A comparison whose systems are not computed keeps its row, marked as missing; m
    does not change.  A comparison with ``after`` is read only if that one shows an
    improvement: otherwise its interval is reported and marked as not read.
    """
    missing = missing or {}
    level = 1 - FAMILY_ALPHA / m
    passed: dict[str, bool] = {}
    rows = []
    for h in family:
        row: dict[str, Any] = {"hypothesis": h.name, "component": h.component,
                               "system": h.system, "reference": h.reference, "level": level,
                               "read_after": h.after or ""}
        computed = h.system in results and h.reference in results
        row["computed"] = computed
        for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
            row.update(dict.fromkeys(
                [f"{metric}_difference", f"{metric}_low", f"{metric}_high"], np.nan))
        if computed:
            decision = deciding(results[h.system], results[h.reference], m)
            row.update({"macro_f1_difference": decision["difference"],
                        "macro_f1_low": decision["low"], "macro_f1_high": decision["high"]})
            for metric in ("accuracy", "balanced_accuracy"):
                beside = difference(results[h.system], results[h.reference], metric)
                row.update({f"{metric}_difference": beside["difference"],
                            f"{metric}_low": beside["low"], f"{metric}_high": beside["high"]})
        read = computed and (h.after is None or passed.get(h.after, False))
        row["read"] = read
        row["improvement"] = bool(row["macro_f1_low"] > 0) if read else None
        passed[h.name] = bool(read and row["improvement"])
        if not computed:
            reason = missing.get(h.system) or missing.get(h.reference) or "no result"
            row["outcome"] = f"missing: {reason}"
        elif not read:
            row["outcome"] = f"not read: {h.after} does not pass"
        else:
            row["outcome"] = _outcome(row)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ scoring


def _finite_interval(values: np.ndarray) -> tuple[float, float]:
    # a resample without items of a breakdown gives no value and does not count
    finite = values[np.isfinite(values)]
    return interval(finite) if len(finite) else (np.nan, np.nan)


class Subset:
    """Some of the items, scored on the resamples of the whole test set or on their own."""

    def __init__(self, gold: Sequence[str], positions: np.ndarray, resampled_weights: np.ndarray):
        self.positions = np.asarray(positions, dtype=int)
        self.n = len(self.positions)
        own = [gold[p] for p in self.positions]
        self.observed = Scorer(own, np.ones((1, self.n)))
        self.resampled = Scorer(own, resampled_weights)
        self.classes = len(set(own))

    def score(self, predicted: Sequence[str | None]) -> Scores:
        chosen = [predicted[p] for p in self.positions]
        with np.errstate(invalid="ignore", divide="ignore"):
            observed = {k: float(v[0]) for k, v in self.observed.metrics(chosen).items()}
            return Scores(observed, self.resampled.metrics(chosen))


def _difference(a: Scores, b: Scores, metric: str) -> dict[str, float]:
    low, high = _finite_interval(a.resampled[metric] - b.resampled[metric])
    return {"difference": a.observed[metric] - b.observed[metric], "low": low, "high": high}


# -------------------------------------------------------------------- runs


@dataclass
class System:
    name: str
    aggregations: Mapping[str, Aggregation]
    role: str
    schemes: str
    description: str


def _side(summary: pd.DataFrame, name: str, model: str, stage: str,
          ids: Sequence[str]) -> Side:
    side = Side.from_summary(summary, name, model, stage)
    filled = {i: side.aggregations.get(i, decide({}, 1.0, "no verdict row")) for i in ids}
    return Side(name, filled, side.p_label)


def _ssoft_rows(runs_dir: Path, names: Sequence[str], history: bool, folder: Path,
                reference: str, gold_scheme: Mapping[str, str]) -> tuple[pd.DataFrame, str]:
    """The rows of ``ssoft.csv`` of the reference model in the listed runs, and its source."""
    frames, source = [], None
    for name in names:
        run_dir = runs_dir / name
        if not (run_dir / "ssoft.csv").is_file() or not (run_dir / "manifest.json").is_file():
            raise SchemeError(f"no ssoft.csv or manifest.json in {run_dir}: parse the run "
                              f"where its raw.jsonl is (argfallacy parse {name})")
        manifest = read_manifest(run_dir)
        config = manifest.get("config") or {}
        if config.get("stages") != ["ssoft"] or bool(config.get("history", True)) != history:
            raise SchemeError(f"{name} is not an S-soft run "
                              f"{'with' if history else 'without'} history")
        if schemes_dir_of(manifest) != folder:
            raise SchemeError(f"{name}: its schemes differ from those of answers_schemes_2")
        if not np.isclose(float(config.get("threshold", PRUNING)), PRUNING):
            raise SchemeError(f"{name}: pruning threshold {config.get('threshold')}, "
                              f"not {PRUNING}")
        if reference not in manifest.get("models", {}):
            continue
        source = source_of(manifest["models"][reference])
        frame = pd.read_csv(run_dir / "ssoft.csv")
        frame = frame[(frame.model == reference) & (frame.scheme_condition == GOLD)
                      & frame.item_id.isin(gold_scheme)]
        wrong = frame[frame.scheme != frame.item_id.map(gold_scheme)]
        if len(wrong):
            raise SchemeError(f"{name}: {wrong.item_id.iloc[0]} answered under another scheme")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), LOGPROB
    rows = pd.concat(frames, ignore_index=True)
    rows["path"] = rows["path"].fillna("") if "path" in rows else ""
    if rows.duplicated(["item_id", "cq_id", "path"]).any():
        raise SchemeError("the S-soft runs answer the same node of the same path twice")
    return rows, source or LOGPROB


def _path(history) -> str:
    return " > ".join(f"{cq}={answer}" for cq, answer in history)


def ssoft_systems(schemes, routing, ids, gold_scheme, v3_rows, history_rows, control_rows,
                  source=LOGPROB, threshold=PRUNING) -> tuple[dict[str, dict], pd.DataFrame, int]:
    """A2.pruned on the answers of full run v3 and, when their rows exist, C4, C4.path
    and C4.nohistory; with the answers that change on the most probable path of C4,
    and the number of items with pending mass."""
    out: dict[str, dict] = {"A2.pruned": {
        i: a2_pruned(schemes[gold_scheme[i]], v3_rows[i], routing[gold_scheme[i]], source,
                     threshold) for i in ids}}
    changes, pending = [], 0
    if len(history_rows):
        by_item = {i: g.to_dict("records") for i, g in history_rows.groupby("item_id")}
        out["C4"], out["C4.path"] = {}, {}
        for i in ids:
            scheme = schemes[gold_scheme[i]]
            rows = by_item.get(i, [])
            result = ssoft(scheme, rows, routing[scheme.scheme_id], source, threshold)
            pending += result.pending > 0
            out["C4"][i] = result
            out["C4.path"][i] = _single(result.path_verdict,
                                        f"single path: {_path(result.single_path) or 'none'}")
            indexed = {(r["cq_id"], r["path"]): r for r in rows}
            for depth, (cq, _) in enumerate(result.best_path):
                prefix = ">".join(f"{c}={a}" for c, a in result.best_path[:depth])
                given = indexed[(cq, prefix)].get("answer")
                before = (v3_rows[i].get(cq) or {}).get("answer")
                changes.append({"scheme": scheme.scheme_id, "cq_id": cq,
                                "changed": given != before})
    if len(control_rows):
        by_item = {i: g.to_dict("records") for i, g in control_rows.groupby("item_id")}
        out["C4.nohistory"] = {
            i: ssoft(schemes[gold_scheme[i]], by_item.get(i, []), routing[gold_scheme[i]],
                     source, threshold, history=False) for i in ids}
    table = pd.DataFrame(changes, columns=["scheme", "cq_id", "changed"])
    if len(table):
        per_cq = table.groupby(["scheme", "cq_id"]).changed.agg(["size", "sum"]).reset_index()
        total = pd.DataFrame([{"scheme": "all", "cq_id": "all", "size": len(table),
                               "sum": int(table.changed.sum())}])
        table = pd.concat([per_cq, total], ignore_index=True).rename(
            columns={"size": "nodes", "sum": "changed"})
        table["share"] = table.changed / table.nodes
    return out, table, pending


# ----------------------------------------------------------------- systems


def _c1_family(data, labels, folds, seed, configs) -> dict[str, C1Result]:
    """Each combination fitted once; its other traversal reuses the fit."""
    results, fitted = {}, {}
    for config in configs:
        key = (config.combiner, config.supervision, config.exclude_schemes, config.unstable)
        if key in fitted:
            results[config.name] = retraverse(data, fitted[key], config)
        else:
            results[config.name] = fitted[key] = c1_out_of_fold(data, labels, folds, seed,
                                                                config)
    return results


def _choices(component: str, choices: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Long rows: fold, component, scheme, parameter, value."""
    rows = []
    for choice in choices:
        if "parameter" in choice:
            # C1 names its hyperparameter by its combiner: lambda of the weights, or C
            choice = {**choice}
            choice["lambda" if choice.get("signals") else "C"] = choice.pop("parameter")
        base = {"fold": choice.get("fold"), "component": choice.get("component", component),
                "scheme": choice.get("scheme", "")}
        for key, value in choice.items():
            if key in ("fold", "component", "scheme") or value is None:
                continue
            if isinstance(value, Mapping):
                rows += [{**base, "parameter": f"{key} {k}", "value": v}
                         for k, v in value.items()]
            elif isinstance(value, list | tuple):
                if value:
                    rows.append({**base, "parameter": key,
                                 "value": "; ".join(map(str, value))})
            else:
                rows.append({**base, "parameter": key, "value": value})
    return rows


def _pairs(reference: str, systems: Mapping[str, System]) -> list[tuple[str, str, str]]:
    """The exploratory comparisons: those named in spec 09, then every other exploratory
    system against the diagram of the reference model on its own answers."""
    zero, diagram, revised = (f"zeroshot:{reference}", f"A0:{reference}",
                              f"A0:{reference}@2.0")
    named = [
        ("C1", "A4", "learned combination against equal weights"),
        ("C1", "A2", "the five entries combined against the reference model alone"),
        ("C1.weighted.cq.soft", "A4", "weights per CQ against equal weights"),
        ("C1", "C1.majority.none.soft", "supervision (b) against (a), majority"),
        ("C1", "C1.dawid_skene.none.soft", "supervision (b) against (a), Dawid and Skene"),
        ("C1.stacking.gold.soft", "C1", "supervision (c) against (b), soft traversal"),
        ("C1.stacking.gold.hard", "C1.hard", "supervision (c) against (b), hard traversal"),
        ("C1.annotation_exclusion", "C1",
         "without the CQ labels of the excluded schemes, on the other schemes"),
        ("C1.stable_cqs", "C1", "without the unstable CQs"),
        ("C2.A0", zero, "routing with A0 of the reference model as the diagram side"),
        ("C3.pooled", "C1", "pooled recalibration of the terminal masses of C1"),
        ("A3.tree", diagram, "a tree on the hard answers of the reference model"),
        ("C3", "A3.tree", "a tree on the combined answers against one on the reference model"),
        ("C4", "C4.nohistory", "the history alone"),
        ("C4.nohistory", "A2.pruned", "the list of the CQs alone"),
        ("C4.path", revised, "the single path of S-soft against the diagram 2.0"),
        ("C4", revised, "S-soft against the diagram 2.0 on full run v3"),
    ]
    pairs = [p for p in named if p[0] in systems and p[1] in systems]
    paired = {p[0] for p in named}
    for name, system in systems.items():
        if system.role == "exploratory" and name not in paired:
            against = revised if system.schemes == "2.0" else diagram
            if against in systems:
                pairs.append((name, against, "against the diagram on the same run"))
    return pairs


# ---------------------------------------------------------------- evaluate


def evaluate_round2(config: EvaluationConfig, runs_dir: str | Path,
                    out_root: str | Path = EVALUATIONS_DIR, items_path: str | Path = ITEMS_FILE,
                    annotations_path: str | Path = ANNOTATIONS_FILE,
                    progress: Callable[[str], None] | None = None) -> Path:
    """Compute round 2 and write it into ``out_root/<name>/``."""
    say = progress or (lambda message: None)
    runs_dir = Path(runs_dir)
    items = load_test_set(items_path).sort_values("item_id").reset_index(drop=True)
    if unknown := sorted(set(items.source.str.lower()) - set(ORIGINS)):
        raise SchemeError(f"corpora without a gold origin: {unknown}")
    ids = items["item_id"].tolist()
    gold = items["gold_fallacy"].tolist()
    gold_of = dict(zip(ids, gold, strict=True))
    scheme_of = dict(zip(ids, items["gold_scheme"], strict=True))
    frame = items[["item_id", "gold_scheme", "gold_fallacy"]]
    folds = assign_folds(items, config.seed, config.folds)
    indices = resample_indices(config.seed, len(ids), config.resamples, 1)
    reference = config.reference_model
    missing: dict[str, str] = {}

    # full run v2: the schemes 1.1, the entries, the reference model
    answers_run = config.runs["answers"]
    manifest, summary = _read_run(runs_dir, answers_run)
    rows, entries = _answer_rows(summary, manifest, answers_run, ids)
    folder = schemes_dir_of(manifest)
    schemes = load_all(folder)
    burden = load_burden(folder.name, schemes)
    routing = {sid: routes(schemes[sid], burden[sid]) for sid in schemes}
    systems: dict[str, System] = {}

    def add(name, aggregations, role, schemes_version, description):
        systems[name] = System(name, aggregations, role, schemes_version, description)

    a0_of = {e: {i: a0(schemes[scheme_of[i]], rows[e][i]) for i in ids} for e in entries}
    observed = Scorer(gold, np.ones((1, len(ids))))
    a0_f1 = {e: float(observed.metrics([a0_of[e][i].verdict for i in ids])["macro_f1"][0])
             for e in entries}
    if (found := reference_of(a0_f1)) != reference:
        raise SchemeError(f"the reference model is {found} (A0 macro F1 {a0_f1[found]:.4f}), "
                          f"not {reference} as the configuration says")
    sources = sources_of(manifest, entries)
    add(f"A0:{reference}", a0_of[reference], "reference", "1.1",
        f"the diagram 1.1 with the hard answers of {reference}, full run v2")
    add("A2", {i: a2(schemes[scheme_of[i]], rows[reference][i], routing[scheme_of[i]],
                     LOGPROB) for i in ids}, "reference", "1.1",
        f"A2 of round 1: soft traversal of the p_logprob of {reference}")
    add("A4", {i: a4(schemes[scheme_of[i]], {e: rows[e][i] for e in entries}, sources,
                     routing[scheme_of[i]]) for i in ids}, "reference", "1.1",
        "A4 of round 1: the mean with equal weights of the five entries, soft traversal")

    zeroshot_run = config.runs.get("zeroshot")
    _, zeroshot_sources = _zeroshot_answers(runs_dir, zeroshot_run, ids)
    zero = None
    if reference in zeroshot_sources:
        _, zeroshot_summary = _read_run(runs_dir, zeroshot_sources[reference])
        if "p_label" not in zeroshot_summary.columns:
            raise SchemeError(
                f"the summary.csv of {zeroshot_sources[reference]} has no p_label, a feature "
                f"of C2 and the confidence of the zero-shot: parse the run again where its "
                f"raw.jsonl is (argfallacy parse {zeroshot_sources[reference]})")
        zero = _side(zeroshot_summary, f"zeroshot:{reference}", reference, "zeroshot", ids)
        add(zero.name, zero.aggregations, "reference", "1.1",
            f"the zero-shot of {reference} ({zeroshot_sources[reference]})")
    else:
        missing[f"zeroshot:{reference}"] = "no zero-shot run of the reference model"
        missing["C2"] = "no zero-shot run of the reference model"

    for run_id in config.runs.get("judge") or []:
        judge_manifest, judge_summary = _read_run(runs_dir, run_id)
        settings = judge_manifest["config"].get("judge") or {}
        if settings.get("answers_from") != answers_run:
            raise SchemeError(f"judge run {run_id}: answers_from does not match {answers_run}")
        if schemes_dir_of(judge_manifest) != folder:
            raise SchemeError(f"judge run {run_id}: schemes differ from the answers run")
        name = "A5" if settings.get("justifications", True) else "A5.answers_only"
        if name in systems:
            raise SchemeError(f"judge run {run_id}: another run already supplies {name}")
        models = judge_summary.loc[judge_summary.stage == JUDGE, "model"].unique()
        if len(models) != 1:
            raise SchemeError(f"the judge run {run_id} has {len(models)} entries, not one")
        side = _side(judge_summary, name, str(models[0]), JUDGE, ids)
        add(name, side.aggregations, "reference", "1.1", f"the LLM judge of {run_id}")
    for name in ("A5", "A5.answers_only"):
        if name not in systems:
            missing[name] = "no judge run " + ("with" if name == "A5" else "without") \
                + " justifications"
    majority = majority_by([scheme_of[i] for i in ids], gold)
    add("majority", {i: _single(v, f"the most frequent gold of {scheme_of[i]}")
                     for i, v in zip(ids, majority, strict=True)}, "oracle", "1.1",
        "the most frequent gold of each scheme, in sample")

    # C1 on full run v2
    say("C1 on full run v2")
    data = build_dataset(schemes, frame, summary, manifest, reference, routing)
    annotations = load_annotations(annotations_path)
    labels = cq_labels(annotations, items, schemes)
    v2 = _c1_family(data, labels, folds, config.seed, configurations(
        excluded_label_schemes=tuple(config.cq_label_exclusion)))
    missing["C1.stable_cqs"] = "the re-annotation of schemes 1.1 does not exist"
    for name, result in v2.items():
        role = ("primary" if name == "C1" else "comparison" if name == "C1.hard"
                else "exploratory")
        c = result.config
        add(name, result.aggregations, role, "1.1",
            f"{c.combiner}, supervision {c.supervision}, {c.traversal} traversal, full run v2"
            + (f", without the CQ labels of {', '.join(c.exclude_schemes)}"
               if c.exclude_schemes else ""))

    # full run v3: the schemes 2.0
    say("C1 on full run v3")
    revised_run = config.runs.get("answers_schemes_2")
    v3: dict[str, C1Result] = {}
    if revised_run:
        manifest3, summary3 = _read_run(runs_dir, revised_run)
        rows3, _ = _answer_rows(summary3, manifest3, revised_run, ids)
        if reference not in rows3:
            raise SchemeError(f"{revised_run} has no answers of {reference}")
        folder3 = schemes_dir_of(manifest3)
        schemes3 = load_all(folder3)
        burden3 = load_burden(folder3.name, schemes3)
        routing3 = {sid: routes(schemes3[sid], burden3[sid]) for sid in schemes3}
        add(f"A0:{reference}@2.0", {i: a0(schemes3[scheme_of[i]], rows3[reference][i])
                                    for i in ids}, "comparison", "2.0",
            f"the diagram 2.0 with the hard answers of {reference}, full run v3")
        data3 = build_dataset(schemes3, frame, summary3, manifest3, reference, routing3)
        v3 = _c1_family(data3, None, folds, config.seed,
                        configurations(schemes_version="2.0"))
        v3["C1@2.0"] = transfer_c1(v2["C1"], data3)
        v3["C1.hard@2.0"] = transfer_c1(v2["C1.hard"], data3)
        for name, result in v3.items():
            c = result.config
            add(name, result.aggregations, "exploratory", "2.0",
                f"{c.combiner}, supervision {c.supervision}, {c.traversal} traversal, "
                f"full run v3" + (", coefficients fitted on full run v2" if name.startswith(
                    ("C1@", "C1.hard@")) else ""))
    else:
        missing[f"A0:{reference}@2.0"] = missing["A2.pruned"] = "no answers_schemes_2 run"

    # C2 and C3, with C1 cross-fitted inside each training partition
    source = C1CrossFitter(data, labels, config.seed, PRIMARY_CONFIG)
    # the transferred C1 keeps the choices of full run v2, written once
    choices = [row for name, result in [*v2.items(), *v3.items()]
               if result.config.traversal == "soft" and name != "C1@2.0"
               for row in _choices(result.config.name, result.choices)]
    curve = None
    if zero is not None:
        say("C2")
        c2 = c2_out_of_fold(source, zero, folds)
        add("C2", c2.aggregations, "primary", "1.1",
            f"routing between C1 and zeroshot:{reference}, threshold 0")
        curve = routing_curve(c2, items, indices)
        c2_a0 = c2_out_of_fold(ReferenceInputs(data, config.seed, hard=True), zero, folds,
                               name="C2.A0")
        add("C2.A0", c2_a0.aggregations, "exploratory", "1.1",
            f"routing between A0 of {reference} and its zero-shot")
        choices += _choices("C2", c2.choices) + _choices("C2.A0", c2_a0.choices)
    say("C3")
    c3 = c3_out_of_fold(source, folds)
    pooled = c3_out_of_fold(source, folds, kind="pooled")
    tree = c3_out_of_fold(ReferenceInputs(data, config.seed), folds, name="A3.tree")
    add("C3", c3.aggregations, "primary", "1.1",
        "a tree of depth at most 3 per scheme on the combined answers of C1, cross-fitted")
    add("C3.pooled", pooled.aggregations, "exploratory", "1.1",
        "a pooled recalibration of the terminal masses of C1")
    add("A3.tree", tree.aggregations, "exploratory", "1.1",
        f"a tree of depth at most 3 per scheme on the hard answers of {reference}")
    choices += sum((_choices(r.name, r.choices) for r in (c3, pooled, tree)), [])

    # C4: S-soft and its control, against A2.pruned on full run v3
    say("C4")
    changes, pending = pd.DataFrame(), 0
    if revised_run:
        history_rows, source_name = _ssoft_rows(runs_dir, config.runs.get("ssoft") or [], True,
                                                folder3, reference, scheme_of)
        control_rows, _ = _ssoft_rows(runs_dir, config.runs.get("ssoft_nohistory") or [],
                                      False, folder3, reference, scheme_of)
        found, changes, pending = ssoft_systems(
            schemes3, routing3, ids, scheme_of, rows3[reference], history_rows, control_rows,
            source_name)
        described = {
            "A2.pruned": ("comparison", f"A2 on the answers of {reference} in full run v3, "
                                        f"burden 2.0, pruning {PRUNING}"),
            "C4": ("primary", f"S-soft, {reference}, schemes 2.0, burden 2.0, pruning "
                              f"{PRUNING}"),
            "C4.path": ("comparison", "the single path of S-soft"),
            "C4.nohistory": ("comparison", "S-soft without the history of the path"),
        }
        for name, aggregations in found.items():
            add(name, aggregations, described[name][0], "2.0", described[name][1])
    for name, key in (("C4", "ssoft"), ("C4.path", "ssoft"), ("C4.nohistory", "ssoft_nohistory")):
        if name not in systems:
            missing[name] = f"no S-soft run (runs.{key} is empty)"
    if "C4" not in systems:
        missing["the answers that change on the path of C4"] = missing["C4"]
    missing.update({
        "C2 with S-soft as another side": "the diagram 2.0 of example has a terminal that the "
                                          "diagram 1.1 of C1 does not have",
        "C2 with D+Q as another side": "no D+Q run (spec 10)",
        "C2 on schemes 2.0": "no zero-shot run with the 2.0 menus (spec 10)",
        "C4 against the zero-shot with the 2.0 menus": "no such run (spec 10)",
        "S-soft with a temperature": "the out-of-fold temperature is not implemented",
        "S-soft of the other four entries": "no runs",
        "the predicted scheme condition": "not available (spec 08, P1)",
        "conformal sets on the direct-verdict annotation": "the annotation has not been read",
    })

    # scores: every system on its own items, every comparison on the items of both
    say("scores")
    full_weights = weights(indices, len(ids))
    bench = Bench(gold, indices)
    position = {i: p for p, i in enumerate(ids)}
    verdicts = {name: [s.aggregations[i].verdict if i in s.aggregations else None for i in ids]
                for name, s in systems.items()}
    subsets: dict[tuple, Subset] = {}

    def subset(positions: tuple[int, ...]) -> Subset:
        if positions not in subsets:
            subsets[positions] = Subset(gold, np.array(positions),
                                        full_weights[:, list(positions)])
        return subsets[positions]

    def items_of(name: str) -> tuple[int, ...]:
        return tuple(sorted(position[i] for i in systems[name].aggregations))

    results: dict[str, Scores] = {}
    for name in systems:
        own = items_of(name)
        results[name] = (bench.score(verdicts[name]) if len(own) == len(ids)
                         else subset(own).score(verdicts[name]))

    def compared(a: str, b: str) -> tuple[Scores, Scores, int]:
        common = tuple(sorted(set(items_of(a)) & set(items_of(b))))
        if len(common) == len(ids):
            return results[a], results[b], len(ids)
        part = subset(common)
        return part.score(verdicts[a]), part.score(verdicts[b]), len(common)

    main = []
    order = {"reference": 0, "oracle": 1, "primary": 2, "comparison": 3, "exploratory": 4}
    for name in sorted(systems, key=lambda n: order[systems[n].role]):
        row: dict[str, Any] = {"system": name, "role": systems[name].role,
                               "schemes": systems[name].schemes,
                               "n_items": len(items_of(name))}
        for metric in METRICS:
            low, high = _finite_interval(results[name].resampled[metric])
            row.update({metric: results[name].observed[metric], f"{metric}_low": low,
                        f"{metric}_high": high})
        row["description"] = systems[name].description
        main.append(row)
    main = pd.DataFrame(main)

    family = hypotheses(reference)
    decide_table = deciding_family(results, family, config.m, missing)

    origin = {i: ORIGINS[s.lower()] for i, s in zip(ids, items.source, strict=True)}
    nlas_free = [p for p, s in enumerate(items.source) if s.lower() != NLAS]
    groups = [("gold origin", value, [p for p, i in enumerate(ids) if origin[i] == value])
              for value in (FALLACY_CORPORA, SCHEME_CORPORA)]
    groups += [("scheme", sid, [p for p, i in enumerate(ids) if scheme_of[i] == sid])
               for sid in sorted(set(scheme_of.values()))]
    parts = [(kind, value, subset(tuple(p))) for kind, value, p in groups]
    without = Subset(gold, np.array(nlas_free), weights(
        resample_indices(config.seed, len(nlas_free), config.resamples, 2), len(nlas_free)))
    parts.append(("without NLAS", "without NLAS", without))
    breakdown = []
    for h in family:
        if h.system not in systems or h.reference not in systems:
            continue
        for kind, value, part in parts:
            a, b = part.score(verdicts[h.system]), part.score(verdicts[h.reference])
            row = {"hypothesis": h.name, "system": h.system, "reference": h.reference,
                   "kind": kind, "subset": value, "n_items": part.n, "n_classes": part.classes,
                   "flagged": kind == "scheme" and part.n < SMALL_SCHEME}
            for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
                d = _difference(a, b, metric)
                row.update({f"{metric}_difference": d["difference"], f"{metric}_low": d["low"],
                            f"{metric}_high": d["high"]})
            breakdown.append(row)
    breakdown = pd.DataFrame(breakdown)

    exploratory = []
    named_pairs = _pairs(reference, systems)
    for a, b, question in named_pairs:
        left, right, n = compared(a, b)
        d = _difference(left, right, "macro_f1")
        exploratory.append({"system": a, "reference": b, "question": question, "n_items": n,
                            "macro_f1_difference": d["difference"], "macro_f1_low": d["low"],
                            "macro_f1_high": d["high"]})
    done = {(a, b) for a, b, _ in named_pairs}
    for a, b in [("C4", "C4.nohistory"), ("C4.nohistory", "A2.pruned"),
                 ("C4.path", f"A0:{reference}@2.0"), ("C4", f"A0:{reference}@2.0")]:
        if (a, b) not in done:
            missing[f"{a} against {b}"] = missing.get(a) or missing.get(b) or "no result"
    exploratory = pd.DataFrame(exploratory, columns=[
        "system", "reference", "question", "n_items", "macro_f1_difference", "macro_f1_low",
        "macro_f1_high"])

    # the level of each CQ, and the path of the human answers as a reference line
    say("CQ level")
    predictions = [entry_answers(data)]
    for result in v2.values():
        c = result.config
        if c.traversal == "soft" and not c.exclude_schemes:
            predictions.append(result.combined.assign(system=f"C1.{c.combiner}.{c.supervision}"))
    cq = cq_evaluation(data, labels, pd.concat(predictions, ignore_index=True), indices=indices)
    human = human_path_reference(annotations, items, schemes)
    score = human.pop("score")
    human.update({"accuracy": score.accuracy if score else np.nan,
                  "macro_f1": score.macro_f1 if score else np.nan,
                  "balanced_accuracy": score.macro_recall if score else np.nan})
    combined = pd.concat([
        r.combined.assign(system=name, schemes="1.1" if name in v2 else "2.0")
        for name, r in [*v2.items(), *v3.items()] if r.config.traversal == "soft"],
        ignore_index=True)[["system", "schemes", "combiner", "supervision", "item_id", "scheme",
                            "cq_id", "fold", "p_first", "outside"]]

    # C5: abstention curves and conformal sets
    say("C5")
    by_item = {"good argumentation": {i: "yes" if gold_of[i] == good_terminal() else "no"
                                      for i in ids},
               "scheme": scheme_of, "corpus": dict(zip(ids, items.source, strict=True)),
               "gold origin": origin}
    soft_c1 = [n for n, r in {**v2, **v3}.items()
               if r.config.traversal == "soft" and len(items_of(n)) == len(ids)]
    distributions = [n for n in [*soft_c1, "C3.pooled", "C4", "A2", "A4"] if n in systems]
    abstaining = [n for n, s in systems.items() if len(items_of(n)) == len(ids)
                  and s.role != "oracle" and not n.startswith("A0:") and n != "A5.answers_only"
                  and any(a.confidence is not None for a in s.aggregations.values())]
    abstention_rows, curves, kinds = [], [], []
    for name in abstaining:
        wrong = ~bench.observed.correct(verdicts[name])
        row, own_curve, ranked = abstention(name, ids, wrong, systems[name].aggregations,
                                            indices, LEVELS, margin=name in distributions)
        abstention_rows.append(row)
        curves.append(own_curve)
        kinds += composition(name, ids, ranked, by_item, LEVELS)
    conformal = []
    for name in distributions:
        version_schemes = schemes if systems[name].schemes == "1.1" else schemes3
        terminals = {i: sorted(version_schemes[scheme_of[i]].terminals) for i in ids}
        sets, thresholds = conformal_sets(systems[name].aggregations, terminals, gold_of, folds,
                                          ALPHA)
        conformal += conformal_summary(name, sets, gold_of,
                                       {"gold origin": origin, "scheme": scheme_of})
        choices += [{"fold": f, "component": f"conformal {name}", "scheme": "",
                     "parameter": "threshold", "value": q} for f, q in thresholds.items()]

    # output
    say("writing")
    out = Path(out_root) / config.name
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "main_table.csv": main, "deciding.csv": decide_table,
        "deciding_breakdowns.csv": breakdown, "exploratory.csv": exploratory,
        "cq_combined.csv": combined, "round2_choices.csv": pd.DataFrame(
            choices, columns=["fold", "component", "scheme", "parameter", "value"]),
        "cq_level.csv": cq.level, "cq_level_differences.csv": cq.differences,
        "human_path.csv": pd.DataFrame([human]),
        "routing_curve.csv": curve if curve is not None else pd.DataFrame(),
        "conformal.csv": pd.DataFrame(conformal), "abstention.csv": pd.DataFrame(abstention_rows),
        "abstention_curves.csv": pd.concat(curves, ignore_index=True) if curves
        else pd.DataFrame(),
        "abstention_composition.csv": pd.DataFrame(kinds),
        "ssoft_changes.csv": changes,
    }
    for file_name, table in tables.items():
        table.to_csv(out / file_name, index=False, float_format="%.4f", lineterminator="\n")
    pd.DataFrame([
        {"system": name, "item_id": i, "fold": folds[i], "verdict": result.verdict,
         "confidence": result.confidence, "abstention": result.abstention,
         "explanation": result.explanation}
        for name, system in systems.items() for i in ids
        if (result := system.aggregations.get(i)) is not None
    ]).to_csv(out / "verdicts.csv", index=False, lineterminator="\n")

    record = {
        "name": config.name, "round": 2,
        "runs": {key: config.runs.get(key) for key in
                 ("answers", "answers_schemes_2", "zeroshot", "judge", "ssoft",
                  "ssoft_nohistory")},
        "zeroshot_sources": zeroshot_sources,
        "schemes": {"answers": folder.name,
                    "answers_schemes_2": folder3.name if revised_run else None},
        "entries": sources, "reference_model": reference, "a0_macro_f1": a0_f1,
        "items": len(ids), "items_without_nlas": len(nlas_free),
        "seed": config.seed, "resamples": config.resamples,
        "folds": {"k": config.folds, "assignment": folds},
        "m": config.m, "deciding_level": 1 - FAMILY_ALPHA / config.m,
        "primary_configurations": {n: systems[n].description for n in ("C1", "C2", "C3", "C4")
                                   if n in systems},
        "hypotheses": [{"name": h.name, "system": h.system, "reference": h.reference,
                        "after": h.after} for h in family],
        "grids": {"C1 stacking C": list(C_GRID), "C1 weighted lambda": list(LAMBDA_GRID),
                  "C2 C": list(C_GRID), "C3 max_depth, min_samples_leaf": list(TREE_GRID),
                  "C3.pooled C": list(C_GRID)},
        "cq_label_exclusion": list(config.cq_label_exclusion),
        "cq_labels": len(labels),
        "pruning": PRUNING, "conformal_alpha": ALPHA, "coverage_levels": list(LEVELS),
        "ssoft_items_with_pending_mass": pending,
        "missing": missing,
        "commit": _commit(), "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "software_versions": {package: version(package)
                              for package in ("numpy", "pandas", "scikit-learn", "scipy")},
    }
    (out / "manifest.json").write_text(json.dumps(record, indent=2, default=str) + "\n",
                                       encoding="utf-8")
    from .html_report import write_summary

    write_summary(out)
    return out
