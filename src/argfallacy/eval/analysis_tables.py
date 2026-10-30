"""Thesis breakdowns on immutable saved verdicts and common eligible items."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ..labels import label_space
from .analysis_bootstrap import percentile, score_population
from .analysis_inputs import boolean
from .canonical import score_fallacies
from .round2 import FALLACY_CORPORA, ORIGINS, SCHEME_CORPORA
from .run_report import good_terminal


def table_frame(rows, columns):
    """Keep promised CSV headers when observations or estimates are pending."""
    result = pd.DataFrame(rows)
    return result.reindex(columns=[*columns, *(c for c in result if c not in columns)])


@dataclass(frozen=True)
class View:
    kind: str
    name: str
    mask: np.ndarray
    generator: int = 1
    origin: str = "total"


def views(items, corpus=False):
    all_items = np.ones(len(items), bool)
    origin = items.source.str.lower().map(ORIGINS)
    output = [View("total", "total", all_items)]
    output += [
        View("gold origin", name, origin.eq(name).to_numpy(), origin=name)
        for name in (FALLACY_CORPORA, SCHEME_CORPORA)
    ]
    output += [
        View("scheme", str(name), items.gold_scheme.eq(name).to_numpy())
        for name in sorted(items.gold_scheme.unique())
    ]
    if corpus:
        output += [
            View("corpus", str(name), items.source.eq(name).to_numpy())
            for name, count in sorted(items.source.value_counts().items())
            if count >= 20
        ]
    output.append(
        View("without NLAS", "without NLAS", items.source.str.lower().ne("nlas").to_numpy(), 2)
    )
    detail = [v for v in output if v.kind in ("scheme", "corpus", "without NLAS")]
    output += [
        View(v.kind, v.name, v.mask & origin.eq(name).to_numpy(), v.generator, name)
        for name in (FALLACY_CORPORA, SCHEME_CORPORA)
        for v in detail
    ]
    return output


def population(items, predictions, eligible, view):
    mask = eligible & view.mask
    if view.generator == 2:
        # Generator 2 draws from the original ordered without-NLAS population,
        # before filtering a system's declared eligible items.
        base = items.source.str.lower().ne("nlas").to_numpy()
        return (items.loc[base].reset_index(drop=True), predictions[base], mask[base], mask)
    return items, predictions, mask, mask


def identity(evaluation, system, view, items, mask, vocabulary=None):
    gold = items.loc[mask, "gold_fallacy"]
    supported = sorted(gold.unique())
    return {
        "evaluation": evaluation.name,
        "round": evaluation.round,
        "system": system.name,
        "scheme_version": system.version,
        "source_table": system.source_table,
        "analysis_role": "new exploratory analysis"
        if system.source_table == "dq_verdicts.csv"
        else "saved-system breakdown",
        "condition": "gold",
        "space": "fine",
        "subset_kind": view.kind,
        "subset": view.name,
        "origin": view.origin,
        "n_items": int(mask.sum()),
        "n_classes": len(supported),
        "supported_classes": json.dumps(supported),
        "excluded_classes": json.dumps(
            sorted(set(label_space("fine", vocabulary)) - set(supported))
        ),
        "small_scheme": view.kind == "scheme" and mask.sum() < 30,
        "status": "computed" if mask.any() else "undefined",
        "generator": view.generator,
    }


def metrics_record(observed, spread, level=0.95):
    row = {}
    for metric, value in observed.items():
        low, high, used = percentile(spread[metric], level)
        row.update(
            {
                metric: value,
                f"{metric}_low": low,
                f"{metric}_high": high,
                f"{metric}_replicates": used,
                f"{metric}_omitted": len(spread[metric]) - used,
            }
        )
    return row


def class_rows(evaluation, system, view, items, predictions, mask, vocabulary=None):
    selected = items.loc[mask]
    score = score_fallacies(
        pd.DataFrame({"item_id": selected.item_id, "predicted": predictions[mask]}),
        selected,
        vocabulary=vocabulary,
    )
    metadata = identity(evaluation, system, view, items, mask, vocabulary)
    counts = pd.Series(predictions[mask], dtype=object).value_counts()
    rows = []
    for row in score.classes.to_dict("records"):
        support = row.pop("support")
        count = int(counts.get(row["class"], 0))
        rows.append(
            {
                **metadata,
                **row,
                "gold_support": support,
                "prediction_count": count,
                "absent": support == 0,
                "precision_undefined": count == 0,
                "recall_undefined": support == 0,
                "f1_undefined": support + count == 0,
            }
        )
    return rows


def main_tables(evaluations, items, bootstrap):
    rows, per_class = [], []
    all_views = views(items, corpus=True)
    vocabulary = bootstrap.scorer(items.gold_fallacy.tolist()).vocab
    for evaluation in evaluations:
        for system in evaluation.systems.values():
            predictions = system.rows.verdict.to_numpy()
            for view in all_views:
                own, chosen, eligible, mask = population(items, predictions, system.eligible, view)
                if view.kind == "corpus" and mask.sum() < 20:
                    continue
                metadata = identity(evaluation, system, view, items, mask, vocabulary)
                if system.main:
                    observed, spread = score_population(
                        own.gold_fallacy.tolist(), chosen, eligible, bootstrap, view.generator
                    )
                    good_support = int((items.loc[mask, "gold_fallacy"] == good_terminal()).sum())
                    good_predictions = int((predictions[mask] == good_terminal()).sum())
                    rows.append(
                        {
                            **metadata,
                            "role": system.role,
                            "description": system.description,
                            "level": 0.95,
                            **metrics_record(observed, spread),
                            "good_gold_support": good_support,
                            "good_prediction_count": good_predictions,
                            "good_recall_undefined": good_support == 0,
                            "good_precision_undefined": good_predictions == 0,
                            "population": "full" if system.eligible.all() else "restricted",
                        }
                    )
                    del observed, spread
                per_class.extend(
                    class_rows(evaluation, system, view, items, predictions, mask, vocabulary)
                )
    return pd.DataFrame(rows), pd.DataFrame(per_class)


def comparisons(evaluations, items, bootstrap):
    rows = []
    vocabulary = bootstrap.scorer(items.gold_fallacy.tolist()).vocab
    for evaluation in evaluations:
        for pair in evaluation.pairs:
            left = evaluation.systems.get(pair["system"])
            right = evaluation.systems.get(pair["reference"])
            for view in views(items):
                deciding = pair["kind"] == "deciding" and view.kind == "total"
                row = {
                    "evaluation": evaluation.name,
                    "round": evaluation.round,
                    "system": pair["system"],
                    "reference": pair["reference"],
                    "scheme_version": left.version if left else "",
                    "reference_scheme_version": right.version if right else "",
                    "condition": "gold",
                    "space": "fine",
                    "subset_kind": view.kind,
                    "subset": view.name,
                    "origin": view.origin,
                    "kind": "deciding" if deciding else "exploratory",
                    "parent_kind": pair["kind"],
                    "source_table": pair["source_table"],
                    "family": pair.get("family", f"round {evaluation.round}"),
                    "family_m": pair.get("family_m", evaluation.manifest["m"]),
                    "source_reading": pair.get("reading", ""),
                    "source_secondary_difference": pair.get("secondary_difference", ""),
                    "hypothesis": pair.get("hypothesis", ""),
                    "question": pair.get("question", ""),
                    "read_after": pair.get("read_after", ""),
                    "source_read": pair.get("read", ""),
                    "source_outcome": pair.get("outcome", ""),
                    "level": float(pair["level"]) if deciding else 0.95,
                    "generator": view.generator,
                    "read": boolean(pair.get("read", True), "read") if deciding else True,
                    "improvement": "",
                    "status": "pending",
                    "n_items": 0,
                }
                if left is None or right is None:
                    row["reason"] = "source system unavailable"
                    rows.append(row)
                    continue
                common = left.eligible & right.eligible
                predictions = left.rows.verdict.to_numpy()
                references = right.rows.verdict.to_numpy()
                own, a, eligible, mask = population(items, predictions, common, view)
                _, b, _, _ = population(items, references, common, view)
                metadata = identity(evaluation, left, view, items, mask, vocabulary)
                row.update(
                    {
                        key: metadata[key]
                        for key in (
                            "n_items",
                            "n_classes",
                            "supported_classes",
                            "excluded_classes",
                            "small_scheme",
                            "status",
                        )
                    }
                )
                observed_a, spread_a = score_population(
                    own.gold_fallacy.tolist(), a, eligible, bootstrap, view.generator
                )
                observed_b, spread_b = score_population(
                    own.gold_fallacy.tolist(), b, eligible, bootstrap, view.generator
                )
                for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
                    level = row["level"] if metric == "macro_f1" else 0.95
                    delta = spread_a[metric] - spread_b[metric]
                    low, high, used = percentile(delta, level)
                    row.update(
                        {
                            f"{metric}_difference": observed_a[metric] - observed_b[metric],
                            f"{metric}_low": low,
                            f"{metric}_high": high,
                            f"{metric}_level": level,
                            f"{metric}_replicates": used,
                            f"{metric}_omitted": bootstrap.resamples - used,
                        }
                    )
                gold = items.gold_fallacy.to_numpy()
                correct_a, correct_b = (
                    predictions[mask] == gold[mask],
                    references[mask] == gold[mask],
                )
                a_only = int((correct_a & ~correct_b).sum())
                b_only = int((~correct_a & correct_b).sum())
                row.update(
                    mcnemar_a_only=a_only,
                    mcnemar_b_only=b_only,
                    mcnemar_denominator=a_only + b_only,
                    mcnemar_p=(
                        binomtest(a_only, a_only + b_only).pvalue if a_only + b_only else 1.0
                    )
                    if mask.any()
                    else np.nan,
                )
                if deciding and row["read"] and mask.all():
                    row["improvement"] = row["macro_f1_low"] > 0
                elif deciding and not mask.all():
                    # A changed population cannot silently acquire the original claim.
                    row.update(kind="exploratory", level=0.95, read=True)
                    low, high, _ = percentile(spread_a["macro_f1"] - spread_b["macro_f1"])
                    row.update(macro_f1_low=low, macro_f1_high=high, macro_f1_level=0.95)
                rows.append(row)
                del spread_a, spread_b
    return pd.DataFrame(rows)


def rare_diagnosis(evaluations, items, bootstrap):
    """Fix the <20 list on the full test set; compare B1's four common populations."""
    vocabulary = bootstrap.scorer(items.gold_fallacy.tolist()).vocab
    classes = label_space("fine", vocabulary)
    support = items.gold_fallacy.value_counts()
    rare = [c for c in classes if support.get(c, 0) < 20]
    rows = []
    for evaluation in evaluations:
        names = [
            "B1",
            f"A0:{evaluation.manifest['reference_model']}",
            "A2",
            f"zeroshot:{evaluation.manifest['reference_model']}",
        ]
        if "B1" not in evaluation.systems:
            continue
        available = [evaluation.systems[n] for n in names if n in evaluation.systems]
        common = np.logical_and.reduce([s.eligible for s in available])
        for system in available:
            predictions = system.rows.verdict.to_numpy()
            for view in views(items):
                mask = common & view.mask
                supported = [c for c in rare if ((items.gold_fallacy == c) & mask).any()]
                per = class_rows(evaluation, system, view, items, predictions, mask, vocabulary)
                recalls = [r["recall"] for r in per if r["class"] in supported]
                rows.append(
                    {
                        **identity(evaluation, system, view, items, mask, vocabulary),
                        "rare_definition": "gold support <20 on full frozen test",
                        "rare_classes": json.dumps(rare),
                        "supported_rare_classes": json.dumps(supported),
                        "n_supported_rare_classes": len(supported),
                        "mean_recall": float(np.mean(recalls)) if recalls else np.nan,
                        "undefined": not bool(supported),
                        "requested_systems": json.dumps(names),
                        "missing_systems": json.dumps(
                            [n for n in names if n not in evaluation.systems]
                        ),
                    }
                )
    columns = [
        "evaluation",
        "round",
        "system",
        "scheme_version",
        "condition",
        "space",
        "subset_kind",
        "subset",
        "origin",
        "n_items",
        "n_classes",
        "supported_classes",
        "excluded_classes",
        "small_scheme",
        "status",
        "generator",
        "rare_definition",
        "rare_classes",
        "supported_rare_classes",
        "n_supported_rare_classes",
        "mean_recall",
        "undefined",
        "requested_systems",
        "missing_systems",
    ]
    return pd.DataFrame(rows, columns=columns)
