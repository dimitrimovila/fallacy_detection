"""``argfallacy evaluate CONFIG``, first stage: what the decision on the aggregators needs.

The gold scheme condition, the fine space, the main table with a 95 per cent
bootstrap interval on every metric, the deciding comparisons of each primary
configuration against A0 of the reference model at level 1 - 0.05/m, the
diagnostic and exploratory comparisons, and the same main table and deciding
comparisons without the NLAS items.

The command reads ``summary.csv`` and ``manifest.json`` of each run, never
``raw.jsonl``; of the zero-shot and judge runs it reads only the hard answer.  One
seed gives the folds and the resamples: the indices of the resamples are drawn
once and every system is scored on them, which makes every comparison paired.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..aggregate import (
    LOGPROB,
    POLICIES,
    Aggregation,
    a0,
    a1_out_of_fold,
    a2,
    a3_out_of_fold,
    a4,
    good_terminal,
    load_burden,
    routes,
    sources_of,
)
from ..aggregate.interface import TIE
from ..annotations import ITEMS_FILE
from ..client.run import GOLD, read_manifest, schemes_dir_of
from ..labels import label_space
from ..parse import ALL_OPTIONS, INVALID
from ..paths import REPO_ROOT
from ..prompts import STAGE2, ZEROSHOT
from ..schemes import SchemeError, load_all, load_vocabulary
from .baselines import majority_by
from .canonical import _gold_terminal, predicted_fallacy_class
from .folds import FOLDS, assign_folds, generators
from .prior import load_test_set

EVALUATIONS_DIR = REPO_ROOT / "evaluations"
NLAS = "nlas"
"""The ``source`` of the synthetic corpus, left out in the ablation without NLAS."""

METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "good_recall", "good_precision",
           "coverage")
PRIMARY = ("A1", "A2", "A3", "A4", "A5")
DECIDE = ("A2", "A3", "A4")
"""The answer to the question of the decision is yes if one of these passes."""
EXPLORATORY = ("A2.stop", "A2.branch", "A2.charitable", "A5.answers_only")
ALPHA = 0.05


# ------------------------------------------------------------------- config


@dataclass
class EvaluationConfig:
    name: str
    runs: dict[str, Any]
    reference_model: str
    seed: int
    resamples: int = 10000
    folds: int = FOLDS
    m: int = 5

    @classmethod
    def load(cls, path: str | Path) -> EvaluationConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise SchemeError(f"{Path(path).name}: unknown configuration keys {unknown}")
        return cls(**data)


# ---------------------------------------------------------------- resamples


def resample_indices(seed: int, n: int, b: int, generator: int) -> np.ndarray:
    """B resamples of N indices, with replacement, from one of the generators of the seed.

    Generator 1 draws the resamples of the test set, generator 2 those of the items
    without NLAS (generator 0 draws the folds).
    """
    return generators(seed)[generator].integers(0, n, size=(b, n))


def weights(indices: np.ndarray, n: int) -> np.ndarray:
    """How many times each resample draws each item: B by N."""
    b = indices.shape[0]
    flat = (indices + np.arange(b)[:, None] * n).ravel()
    return np.bincount(flat, minlength=b * n).reshape(b, n).astype(float)


class Scorer:
    """The canonical metrics of every prediction set, on every resample at once.

    A metric on a resample counts each item as many times as it is drawn; the macro
    averages run over the classes with support in the resample.  With every weight
    1 it is the canonical scorer.
    """

    def __init__(self, gold: Sequence[str], weights: np.ndarray):
        self.vocab = load_vocabulary(check_schemes=False)
        self.classes = label_space("fine", self.vocab)
        self.index = {label: k for k, label in enumerate(self.classes)}
        terminals = [_gold_terminal(str(n), g, self.vocab) for n, g in enumerate(gold)]
        self.gold = np.array([self.index[t] for t in terminals])
        self.weights = np.asarray(weights, dtype=float)
        self.gold_onehot = np.eye(len(self.classes))[self.gold]
        self.support = self.weights @ self.gold_onehot
        self.total = self.weights.sum(axis=1)
        self.good = self.index[good_terminal()]
        self._classes: dict[Any, int] = {}

    def _class(self, raw: Any) -> int:
        # the parser writes invalid for an answer it could not read: no verdict
        missing = raw is None or raw == INVALID or (isinstance(raw, float) and np.isnan(raw))
        key = None if missing else raw
        if key not in self._classes:
            label, _ = predicted_fallacy_class(key, "fine", self.vocab)
            self._classes[key] = -1 if label is None else self.index[label]
        return self._classes[key]

    def metrics(self, predicted: Sequence[str | None]) -> dict[str, np.ndarray]:
        p = np.array([self._class(raw) for raw in predicted])
        if len(p) != len(self.gold):
            raise ValueError(f"{len(self.gold)} gold labels but {len(p)} predictions")
        emitted_onehot = np.zeros_like(self.gold_onehot)
        emitted_onehot[np.flatnonzero(p >= 0), p[p >= 0]] = 1.0
        correct = (p == self.gold).astype(float)
        emitted = self.weights @ emitted_onehot
        tp = self.weights @ (self.gold_onehot * correct[:, None])
        supported = self.support > 0
        f1 = np.divide(2 * tp, self.support + emitted, out=np.zeros_like(tp), where=supported)
        recall = np.divide(tp, self.support, out=np.zeros_like(tp), where=supported)
        n_supported = supported.sum(axis=1)
        g = self.good
        return {
            "accuracy": (self.weights @ correct) / self.total,
            "macro_f1": f1.sum(axis=1) / n_supported,
            "balanced_accuracy": recall.sum(axis=1) / n_supported,
            "good_recall": np.divide(tp[:, g], self.support[:, g], out=np.zeros(len(tp)),
                                     where=self.support[:, g] > 0),
            "good_precision": np.divide(tp[:, g], emitted[:, g], out=np.zeros(len(tp)),
                                        where=emitted[:, g] > 0),
            "coverage": (self.weights @ (p >= 0).astype(float)) / self.total,
        }


@dataclass
class Scores:
    """The metrics of one system on the observed sample and on every resample."""

    observed: dict[str, float]
    resampled: dict[str, np.ndarray] = field(repr=False)


class Bench:
    """The items of one table, with their resamples, scored once per system."""

    def __init__(self, gold: Sequence[str], indices: np.ndarray):
        self.n = len(gold)
        self.observed = Scorer(gold, np.ones((1, len(gold))))
        self.resampled = Scorer(gold, weights(indices, len(gold)))

    def score(self, predicted: Sequence[str | None]) -> Scores:
        observed = {k: float(v[0]) for k, v in self.observed.metrics(predicted).items()}
        return Scores(observed, self.resampled.metrics(predicted))


def scores(gold: Sequence[str], indices: np.ndarray, predicted: Sequence[str | None]) -> Scores:
    return Bench(gold, indices).score(predicted)


def interval(values: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """The percentile interval at level 1 - alpha, with the default method of numpy."""
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def difference(a: Scores, b: Scores, metric: str, alpha: float = ALPHA) -> dict[str, float]:
    """The paired difference a - b on ``metric``, with its interval."""
    low, high = interval(a.resampled[metric] - b.resampled[metric], alpha)
    return {"difference": a.observed[metric] - b.observed[metric], "low": low, "high": high}


def deciding(system: Scores, reference: Scores, m: int) -> dict[str, Any]:
    """The difference in macro F1 at level 1 - 0.05/m; an improvement only if the interval
    excludes zero from above."""
    out: dict[str, Any] = difference(system, reference, "macro_f1", ALPHA / m)
    out["improvement"] = out["low"] > 0
    return out


def reference_of(macro_f1: Mapping[str, float]) -> str:
    """The entry whose A0 has the highest macro F1; equal ones go to the first name."""
    best = max(macro_f1.values())
    return min(entry for entry, value in macro_f1.items() if value >= best - TIE)


def without_nlas(items: pd.DataFrame) -> pd.DataFrame:
    return items[items["source"].str.lower() != NLAS].reset_index(drop=True)


# -------------------------------------------------------------------- runs


def _read_run(runs_dir: Path, run_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = runs_dir / run_id
    for name in ("summary.csv", "manifest.json"):
        if not (run_dir / name).is_file():
            raise SchemeError(f"no {name} in {run_dir}: parse the run where its raw.jsonl is "
                              f"(argfallacy parse {run_id}) and copy its tables here")
    return read_manifest(run_dir), pd.read_csv(run_dir / "summary.csv")


def _answer_rows(summary: pd.DataFrame, manifest: Mapping[str, Any], run_id: str,
                 ids: Sequence[str]) -> tuple[dict[str, dict[str, dict[str, dict]]], list[str]]:
    """entry -> item -> CQ -> row of ``summary.csv``, for the CQs of stage two, gold scheme."""
    needed = [f"p_logprob_{o}" for o in ALL_OPTIONS] + [f"p_sample_{o}" for o in ALL_OPTIONS]
    if missing := [c for c in needed if c not in summary.columns]:
        raise SchemeError(f"the summary.csv of {run_id} has no {missing[0]}: it was written by "
                          f"an earlier parser; parse the run again (argfallacy parse {run_id})")
    stage2 = summary[(summary["stage"] == STAGE2) & (summary["scheme_condition"] == GOLD)]
    entries = [e for e in manifest["config"]["models"] if e in set(stage2["model"])]
    rows: dict[str, dict[str, dict[str, dict]]] = {e: {i: {} for i in ids} for e in entries}
    for record in stage2.to_dict("records"):
        if record["model"] in rows and record["item_id"] in rows[record["model"]]:
            rows[record["model"]][record["item_id"]][record["cq_id"]] = record
    return rows, entries


def _hard_answers(summary: pd.DataFrame, ids: Sequence[str],
                  stage: str | None = None) -> dict[str, dict[str, str | None]]:
    """entry -> item -> hard answer, from the verdict rows of a zero-shot or judge run."""
    rows = summary if stage is None else summary[summary["stage"] == stage]
    out: dict[str, dict[str, str | None]] = {}
    for model, own in rows.groupby("model"):
        given = dict(zip(own["item_id"], own["answer"], strict=True))
        out[str(model)] = {i: (None if pd.isna(given.get(i)) or given.get(i) == INVALID
                               else given[i]) for i in ids}
    return out


def _single(verdict: str | None, explanation: str) -> Aggregation:
    return Aggregation(verdict, None, {} if verdict is None else {verdict: 1.0},
                       1.0 if verdict is None else 0.0, explanation)


def _commit() -> str | None:
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


# ------------------------------------------------------------------ output


def _cell(observed: float, low: float, high: float) -> str:
    # rounding first, so that a difference of -1e-17 reads 0.000, not -0.000
    observed, low, high = (round(value, 3) + 0.0 for value in (observed, low, high))
    return f"{observed:.3f} [{low:.3f}, {high:.3f}]"


def _main_table(names: Sequence[str], bench: Bench, results: Mapping[str, Scores]) -> pd.DataFrame:
    rows = []
    for name in names:
        row: dict[str, Any] = {"system": name, "n_items": bench.n}
        for metric in METRICS:
            low, high = interval(results[name].resampled[metric])
            row.update({metric: results[name].observed[metric], f"{metric}_low": low,
                        f"{metric}_high": high})
        rows.append(row)
    return pd.DataFrame(rows)


def _deciding_table(present: Sequence[str], results: Mapping[str, Scores], reference: str,
                    m: int) -> pd.DataFrame:
    baseline, zeroshot = results[f"A0:{reference}"], results.get(f"zeroshot:{reference}")
    rows = []
    for name in present:
        decision = deciding(results[name], baseline, m)
        row: dict[str, Any] = {"system": name, "reference": f"A0:{reference}",
                               "level": 1 - ALPHA / m,
                               "macro_f1_difference": decision["difference"],
                               "macro_f1_low": decision["low"], "macro_f1_high": decision["high"],
                               "improvement": decision["improvement"]}
        for metric in ("accuracy", "balanced_accuracy"):
            beside = difference(results[name], baseline, metric)
            row.update({f"{metric}_difference": beside["difference"],
                        f"{metric}_low": beside["low"], f"{metric}_high": beside["high"]})
        if zeroshot is not None:
            against = difference(results[name], zeroshot, "macro_f1")
            row.update({"zeroshot_macro_f1_difference": against["difference"],
                        "zeroshot_macro_f1_low": against["low"],
                        "zeroshot_macro_f1_high": against["high"]})
        rows.append(row)
    return pd.DataFrame(rows)


def _pairs_table(pairs: Sequence[tuple[str, str]], results: Mapping[str, Scores]) -> pd.DataFrame:
    rows = []
    for name, reference in pairs:
        compared = difference(results[name], results[reference], "macro_f1")
        rows.append({"system": name, "reference": reference,
                     "macro_f1_difference": compared["difference"],
                     "macro_f1_low": compared["low"], "macro_f1_high": compared["high"]})
    return pd.DataFrame(rows, columns=["system", "reference", "macro_f1_difference",
                                       "macro_f1_low", "macro_f1_high"])


def _markdown(frame: pd.DataFrame, columns: Mapping[str, str]) -> str:
    """A table whose cells are a value with its interval, from the columns ``<x>``,
    ``<x>_low`` and ``<x>_high``; ``columns`` maps a column of the frame to its title."""
    titles = ["system", *columns.values()]
    lines = ["| " + " | ".join(titles) + " |", "|" + "---|" * len(titles)]
    for row in frame.to_dict("records"):
        cells = [row["system"]]
        for column in columns:
            if column.endswith("_difference"):
                base = column.removesuffix("_difference")
                value = row.get(column)
                cells.append("" if value is None or pd.isna(value) else
                             _cell(value, row[f"{base}_low"], row[f"{base}_high"]))
            elif column in ("improvement",):
                cells.append("yes" if row[column] else "no")
            else:
                cells.append(_cell(row[column], row[f"{column}_low"], row[f"{column}_high"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


MAIN_COLUMNS = {"accuracy": "accuracy", "macro_f1": "macro F1",
                "balanced_accuracy": "balanced accuracy",
                "good_recall": "recall of Good Argumentation",
                "good_precision": "precision of Good Argumentation", "coverage": "coverage"}
MAIN_NOTE = ("Fine space, gold scheme condition. Every value with its 95 per cent percentile "
             "bootstrap interval. No verdict counts as wrong; macro F1 and balanced accuracy "
             "average over the classes with support. `majority` (the most frequent gold of each "
             "scheme) and `ceiling:<entry>` (the most frequent gold of each scheme and answer "
             "vector) are oracles computed in sample.")


def _summary(card: list[tuple[str, str]], main: pd.DataFrame, decide: pd.DataFrame,
             answer: str, diagnostic: pd.DataFrame, exploratory: pd.DataFrame,
             main_n: pd.DataFrame, decide_n: pd.DataFrame, missing: list[str], m: int,
             n_all: int, n_without: int) -> str:
    level = f"{100 * (1 - ALPHA / m):g}"
    decide_columns = {"macro_f1_difference": f"difference in macro F1 [{level}%]",
                      "improvement": "improvement",
                      "accuracy_difference": "difference in accuracy [95%]",
                      "balanced_accuracy_difference": "difference in balanced accuracy [95%]",
                      "zeroshot_macro_f1_difference": "against the zero-shot, macro F1 [95%]"}
    pair_columns = {"macro_f1_difference": "difference in macro F1 [95%]"}
    out = ["# Evaluation, first stage", "", "## 1. Run card", "", "| | |", "|---|---|"]
    out += [f"| {key} | {value} |" for key, value in card]
    out += ["", f"## 2. Main table ({n_all} items)", "", _markdown(main, MAIN_COLUMNS), "",
            MAIN_NOTE, "", "## 3. Deciding comparisons", "", answer, "",
            _markdown(decide, decide_columns) if len(decide) else "No primary configuration.",
            "", f"Each primary configuration against A0 of the reference model, paired on the "
            f"same items. The difference in macro F1 decides, with its interval at level "
            f"{level} per cent (Bonferroni, m = {m}): an improvement only if the interval "
            f"excludes zero. The other differences are reported beside it and decide nothing.",
            "", "## 4. Diagnostic comparisons", "",
            _markdown(diagnostic, {"macro_f1_difference": "difference in macro F1 [95%]"})
            if len(diagnostic) else "None.", "",
            "A2 of each other entry against A0 of the same entry: a negative result here "
            "comes from the aggregation, not from the choice of the model. For A1, A2, A3 and "
            "A5 the A0 of their model is the reference, so the deciding comparison is also "
            "the diagnostic one.", "", "## 5. Exploratory comparisons", "",
            _markdown(exploratory, pair_columns) if len(exploratory) else "None.", "",
            "Exploratory: they support no claim of improvement.", "",
            f"## 6. Without NLAS ({n_without} items)", "",
            "The verdicts of every system as computed on the whole test set, resampled on the "
            "items without NLAS.", "", _markdown(main_n, MAIN_COLUMNS), "",
            _markdown(decide_n, decide_columns) if len(decide_n) else "", "",
            "## 7. Systems missing", ""]
    out += [f"* {item}" for item in missing] or ["None."]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- evaluate


def evaluate(config: EvaluationConfig, runs_dir: str | Path, out_root: str | Path = EVALUATIONS_DIR,
             items_path: str | Path = ITEMS_FILE) -> Path:
    """Compute the first stage of the evaluation and write it into ``out_root/<name>/``."""
    runs_dir = Path(runs_dir)
    items = load_test_set(items_path).sort_values("item_id").reset_index(drop=True)
    ids = items["item_id"].tolist()
    gold = items["gold_fallacy"].tolist()
    scheme_of = dict(zip(ids, items["gold_scheme"], strict=True))
    frame = items[["item_id", "gold_scheme", "gold_fallacy"]]

    answers_run = config.runs["answers"]
    manifest, summary = _read_run(runs_dir, answers_run)
    rows, entries = _answer_rows(summary, manifest, answers_run, ids)
    folder = schemes_dir_of(manifest)
    schemes = load_all(folder)
    table = load_burden(folder.name, schemes)
    routing = {policy: {sid: routes(schemes[sid], table[sid], policy) for sid in schemes}
               for policy in POLICIES}
    sources = sources_of(manifest, entries)
    folds = assign_folds(items, config.seed, config.folds)

    systems: dict[str, dict[str, Aggregation]] = {}
    for entry in entries:
        systems[f"A0:{entry}"] = {i: a0(schemes[scheme_of[i]], rows[entry][i]) for i in ids}

    bench = Bench(gold, resample_indices(config.seed, len(ids), config.resamples, 1))

    def verdicts(name: str, subset: Sequence[str] = ids) -> list[str | None]:
        return [systems[name][i].verdict for i in subset]

    a0_f1 = {e: bench.observed.metrics(verdicts(f"A0:{e}"))["macro_f1"][0] for e in entries}
    reference = reference_of(a0_f1)
    if reference != config.reference_model:
        raise SchemeError(f"the reference model is {reference} (A0 macro F1 "
                          f"{a0_f1[reference]:.4f}), not {config.reference_model} as the "
                          f"configuration says")

    own = rows[reference]
    systems["A1"], a1_orders = a1_out_of_fold(schemes, frame, own, folds)
    burden = routing["burden"]
    systems["A2"] = {i: a2(schemes[scheme_of[i]], own[i], burden[scheme_of[i]], LOGPROB)
                     for i in ids}
    systems["A3"], a3_choices = a3_out_of_fold(schemes, frame, own, folds, config.seed, LOGPROB)
    systems["A4"] = {i: a4(schemes[scheme_of[i]], {e: rows[e][i] for e in entries}, sources,
                           burden[scheme_of[i]]) for i in ids}
    descriptions = {
        "A1": f"the diagram with the order of its blocks chosen out of fold; hard answers of "
              f"{reference}",
        "A2": f"probabilistic traversal of p_logprob of {reference}, burden of proof of each CQ",
        "A3": f"multinomial logistic regression per scheme on p_logprob of {reference}",
        "A4": f"mean with equal weights of {', '.join(entries)}, each with the soft answer of "
              f"its regime, then the traversal of A2",
    }

    for policy in ("stop", "branch", "charitable"):
        systems[f"A2.{policy}"] = {i: a2(schemes[scheme_of[i]], own[i],
                                         routing[policy][scheme_of[i]], LOGPROB) for i in ids}
    for entry in entries:
        if entry != reference:
            systems[f"A2:{entry}"] = {i: a2(schemes[scheme_of[i]], rows[entry][i],
                                            burden[scheme_of[i]], sources[entry]) for i in ids}

    missing = []
    zeroshot_run = config.runs.get("zeroshot")
    zeroshot = _hard_answers(_read_run(runs_dir, zeroshot_run)[1], ids, ZEROSHOT) \
        if zeroshot_run else {}
    for entry in entries:
        if entry in zeroshot:
            systems[f"zeroshot:{entry}"] = {i: _single(v, "the verdict asked directly")
                                            for i, v in zeroshot[entry].items()}
        else:
            missing.append(f"zeroshot:{entry}: no zero-shot run of this entry")
    for run_id in config.runs.get("judge") or []:
        judge_manifest, judge_summary = _read_run(runs_dir, run_id)
        settings = judge_manifest["config"].get("judge") or {}
        name = "A5" if settings.get("justifications", True) else "A5.answers_only"
        judged = _hard_answers(judge_summary, ids)
        if len(judged) != 1:
            raise SchemeError(f"the judge run {run_id} has {len(judged)} entries, not one")
        ((judge_model, given),) = judged.items()
        systems[name] = {i: _single(v, "the verdict of the judge") for i, v in given.items()}
        if name == "A5":
            descriptions["A5"] = (f"LLM judge: {judge_model} reads the answers of "
                                  f"{settings.get('answers_model')} with their justifications "
                                  f"({run_id})")
    for name in ("A5", "A5.answers_only"):
        if name not in systems:
            missing.append(f"{name}: no judge run "
                           f"{'with' if name == 'A5' else 'without'} justifications")

    majority = majority_by([scheme_of[i] for i in ids], gold)
    systems["majority"] = {i: _single(v, f"the most frequent gold of {scheme_of[i]}")
                           for i, v in zip(ids, majority, strict=True)}
    for entry in entries:
        keys = [(scheme_of[i], tuple((rows[entry][i].get(cq) or {}).get("answer", "missing")
                                     for cq in schemes[scheme_of[i]].cq_order)) for i in ids]
        ceiling = majority_by(keys, gold)
        systems[f"ceiling:{entry}"] = {
            i: _single(v, "the most frequent gold of the scheme and answer vector")
            for i, v in zip(ids, ceiling, strict=True)}

    present = [p for p in PRIMARY if p in systems]
    main_names = ([f"A0:{e}" for e in entries] + present
                  + [f"zeroshot:{e}" for e in entries if f"zeroshot:{e}" in systems]
                  + ["majority"] + [f"ceiling:{e}" for e in entries])
    exploratory_names = [n for n in EXPLORATORY if n in systems]
    diagnostic_pairs = [(f"A2:{e}", f"A0:{e}") for e in entries if f"A2:{e}" in systems]
    names = main_names + [n for n in systems if n not in main_names]

    results = {name: bench.score(verdicts(name)) for name in names}
    nlas_free = without_nlas(items)
    ids_n = nlas_free["item_id"].tolist()
    bench_n = Bench(nlas_free["gold_fallacy"].tolist(),
                    resample_indices(config.seed, len(ids_n), config.resamples, 2))
    results_n = {name: bench_n.score(verdicts(name, ids_n)) for name in main_names}

    main = _main_table(main_names, bench, results)
    decide = _deciding_table(present, results, reference, config.m)
    diagnostic = _pairs_table(diagnostic_pairs, results)
    exploratory = _pairs_table([(n, f"A0:{reference}") for n in exploratory_names], results)
    main_n = _main_table(main_names, bench_n, results_n)
    decide_n = _deciding_table(present, results_n, reference, config.m)

    passing = dict(zip(decide["system"], decide["improvement"], strict=True)) if len(decide) \
        else {}
    decided = [p for p in DECIDE if p in passing]
    winners = [p for p in decided if passing[p]]
    answer = (f"Do soft answers and several models raise the ceiling? "
              f"**{'Yes' if winners else 'No'}**. "
              + (f"{', '.join(winners)} pass{'es' if len(winners) == 1 else ''} the deciding "
                 f"comparison." if winners else
                 f"None of {', '.join(decided) or 'A2, A3, A4'} passes the deciding comparison."))
    if "A1" in passing:
        answer += (" A1 " + ("passes: the order of the diagram was a problem." if passing["A1"]
                             else "does not pass."))
    if "A5" in passing:
        answer += (" A5 " + ("passes: an LLM judge beats the diagram, to be read against the "
                             "zero-shot." if passing["A5"] else "does not pass."))
    missing = [f"{p}: no result" for p in PRIMARY if p not in systems and p != "A5"] + missing

    out = Path(out_root) / config.name
    out.mkdir(parents=True, exist_ok=True)
    tables = {"main_table.csv": main, "deciding.csv": decide, "diagnostic.csv": diagnostic,
              "exploratory.csv": exploratory, "main_table_without_nlas.csv": main_n,
              "deciding_without_nlas.csv": decide_n,
              "a1_orders.csv": pd.DataFrame(a1_orders), "a3_choices.csv": pd.DataFrame(a3_choices)}
    for file_name, table_frame in tables.items():
        table_frame.to_csv(out / file_name, index=False, float_format="%.4f",
                           lineterminator="\n")
    pd.DataFrame([
        {"system": name, "item_id": i, "fold": folds[i], "verdict": result.verdict,
         "confidence": result.confidence, "abstention": result.abstention,
         "explanation": result.explanation}
        for name in names for i in ids for result in [systems[name][i]]
    ]).to_csv(out / "verdicts.csv", index=False, lineterminator="\n")

    date = datetime.now(UTC).isoformat(timespec="seconds")
    commit = _commit()
    record = {
        "name": config.name,
        "runs": {"answers": answers_run, "zeroshot": zeroshot_run,
                 "judge": list(config.runs.get("judge") or [])},
        "schemes": {"folder": folder.name,
                    "tag": (manifest.get("schemes_version") or {}).get("tag")},
        "prompt_versions": manifest.get("prompt_versions"),
        "entries": {e: sources[e] for e in entries},
        "items": len(ids), "items_without_nlas": len(ids_n),
        "seed": config.seed, "resamples": config.resamples,
        "folds": {"k": config.folds, "assignment": folds},
        "reference_model": reference, "a0_macro_f1": a0_f1,
        "primary_configurations": {p: descriptions[p] for p in present},
        "m": config.m, "commit": commit, "date": date,
    }
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    card = [("name", config.name), ("date", date), ("commit", commit or "unknown"),
            ("runs", f"answers {answers_run}; zero-shot {zeroshot_run or 'none'}; judge "
                     f"{', '.join(config.runs.get('judge') or []) or 'none'}"),
            ("schemes", f"{folder.name} ({record['schemes']['tag']})"),
            ("entries", ", ".join(f"{e} ({sources[e]})" for e in entries)),
            ("reference model", f"{reference} (A0 macro F1 {a0_f1[reference]:.3f})"),
            ("items", f"{len(ids)}, {len(ids_n)} without NLAS"),
            ("seed, resamples, folds, m",
             f"{config.seed}, {config.resamples}, {config.folds}, {config.m}")]
    (out / "summary.md").write_text(
        _summary(card, main, decide, answer, diagnostic, exploratory, main_n, decide_n,
                 missing, config.m, len(ids), len(ids_n)), encoding="utf-8")
    return out
