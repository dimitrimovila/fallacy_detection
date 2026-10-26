"""Frozen historical verdict agreement and gold-scheme human paths."""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

from ..annotations import ANNOTATOR
from ..labels import label_space
from ..schemes import traverse
from .analysis_bootstrap import contribution_ratios, score_population
from .analysis_inputs import require
from .analysis_selective import origin_masks
from .analysis_tables import metrics_record, table_frame
from .c1 import gold_scheme_answers
from .cq import agreement
from .evaluate import weights
from .run_report import good_terminal

PEOPLE = ("enrico", "eleni", ANNOTATOR)


def read_frozen_annotations(config, inputs):
    if config.annotations is None:
        return None
    notes = inputs.csv(
        config.annotations, ["item_id", "annotator", "sheet", "row", "field", "value"]
    )
    require(notes.item_id.ne("").all(), "empty annotation item id")
    return notes


def historical_ratings(annotations, items, schemes):
    """Deduplicate across sheets, excluding each conflicting item/rater rating."""
    ratings = pd.DataFrame(None, index=items.item_id, columns=PEOPLE, dtype=object)
    conflict = pd.DataFrame(False, index=items.item_id, columns=PEOPLE)
    coarse = conflict.copy()
    terminals = set().union(*(s.terminals for s in schemes.values()))
    own = annotations[
        (annotations.field == "verdict")
        & annotations.annotator.isin(PEOPLE)
        & annotations.item_id.isin(items.item_id)
    ]
    for (item, person), rows in own.groupby(["item_id", "annotator"]):
        values = set(rows.loc[rows.value.ne("") & rows.value.notna(), "value"])
        if len(values) > 1:
            conflict.loc[item, person] = True
        elif values:
            value = next(iter(values))
            if value in terminals:
                ratings.loc[item, person] = value
            else:
                coarse.loc[item, person] = True
    return ratings, conflict, coarse


def agreement_bootstrap(units, eligible, bootstrap):
    """Nominal coincidences and observed rater pairs, clustered by item."""
    clean = [[x for x in u if x is not None and not pd.isna(x) and x != ""] for u in units]
    categories = sorted({x for u in clean if len(u) >= 2 for x in u})
    counts = np.array(
        [[u.count(c) if len(u) >= 2 else 0 for c in categories] for u in clean], float
    )
    counts = counts.reshape(len(clean), len(categories))
    different = np.array([sum(a != b for a, b in combinations(u, 2)) for u in clean], float)
    pairs = np.array([len(u) * (len(u) - 1) / 2 for u in clean])
    disagreement = np.array(
        [2 * d / (len(u) - 1) if len(u) >= 2 else 0 for u, d in zip(clean, different, strict=True)]
    )
    mask = np.asarray(eligible, bool)
    observed = agreement([clean[i] for i in np.flatnonzero(mask)])
    spread = {key: np.full(bootstrap.resamples, np.nan) for key in ("alpha", "raw_agreement")}
    start = 0
    for batch in bootstrap.indices(len(units)):
        multiplicities = weights(batch, len(units)) * mask
        category_counts = multiplicities @ counts
        total = category_counts.sum(axis=1)
        expected = np.divide(
            total**2 - (category_counts**2).sum(axis=1),
            total - 1,
            out=np.zeros(len(batch)),
            where=total > 1,
        )
        alpha = 1 - np.divide(
            multiplicities @ disagreement,
            expected,
            out=np.full(len(batch), np.nan),
            where=expected > 0,
        )
        denominator = multiplicities @ pairs
        raw = np.divide(
            multiplicities @ (pairs - different),
            denominator,
            out=np.full(len(batch), np.nan),
            where=denominator > 0,
        )
        stop = start + len(batch)
        spread["alpha"][start:stop], spread["raw_agreement"][start:stop] = alpha, raw
        start = stop
    return observed, spread


def reference_support(reference, predictions, mask):
    """Class denominators and undefined flags for a classification reference."""
    truth, predicted = np.asarray(reference, object)[mask], np.asarray(predictions, object)[mask]
    supported = sorted(set(truth))
    good_support = int((truth == good_terminal()).sum())
    good_predictions = int((predicted == good_terminal()).sum())
    return {
        "n_classes": len(supported),
        "supported_classes": json.dumps(supported),
        "excluded_classes": json.dumps(sorted(set(label_space("fine")) - set(supported))),
        "good_reference_support": good_support,
        "good_prediction_count": good_predictions,
        "good_recall_undefined": good_support == 0,
        "good_precision_undefined": good_predictions == 0,
        "classification_undefined": not len(truth),
    }


def human_tables(annotations, evaluations, items, schemes, bootstrap):
    agreement_columns = [
        "evaluation",
        "system",
        "scheme_version",
        "origin",
        "comparison",
        "population",
        "n_items",
        "units",
        "pairs",
        "alpha",
        "raw_agreement",
        "distribution",
        "status",
    ]
    path_columns = [
        "evaluation",
        "system",
        "reference",
        "scheme_version",
        "origin",
        "population",
        "n_items",
        "annotated_items",
        "usable_items",
        "test_items",
        "accuracy",
        "status",
    ]
    pending = {
        "second_human_cq": "CQ ratings by the second human annotator unavailable",
        "blind_repeats": "identified blind repeat annotations unavailable",
    }
    if annotations is None:
        return {
            "human_agreement.csv": pd.DataFrame(columns=agreement_columns),
            "human_path.csv": pd.DataFrame(columns=path_columns),
        }, {"status": "pending", "reason": "frozen annotations not configured", **pending}
    ratings, conflicts, coarse = historical_ratings(annotations, items, schemes)
    agreement_rows, path_rows = [], []
    for origin, mask in origin_masks(items):
        for people in [PEOPLE, *combinations(PEOPLE, 2)]:
            units = ratings[list(people)].to_numpy().tolist()
            observed, spread = agreement_bootstrap(units, mask, bootstrap)
            agreement_rows.append(
                {
                    "evaluation": "frozen human reference",
                    "system": "+".join(people),
                    "scheme_version": "1.1",
                    "origin": origin,
                    "comparison": "+".join(people),
                    "population": "historical verdicts across sheets; any judged scheme",
                    "n_items": int(mask.sum()),
                    "level": 0.95,
                    **observed,
                    **metrics_record({k: observed[k] for k in spread}, spread),
                    "rated_items": int(ratings.loc[mask, list(people)].notna().any(axis=1).sum()),
                    "missing_ratings": int(ratings.loc[mask, list(people)].isna().sum().sum()),
                    "conflicting_ratings_excluded": int(
                        conflicts.loc[mask, list(people)].sum().sum()
                    ),
                    "nonterminal_ratings_excluded": int(coarse.loc[mask, list(people)].sum().sum()),
                    "all_ratings_distribution": json.dumps(
                        {p: ratings.loc[mask, p].dropna().value_counts().to_dict() for p in people},
                        sort_keys=True,
                    ),
                    "status": "computed" if observed["pairs"] else "undefined",
                }
            )
            del spread
        for part, reason in pending.items():
            agreement_rows.append(
                {
                    "evaluation": "frozen human reference",
                    "system": part,
                    "scheme_version": "1.1",
                    "origin": origin,
                    "comparison": part,
                    "population": reason,
                    "status": "pending",
                    "reason": reason,
                }
            )
    human = gold_scheme_answers(annotations, items, schemes)
    paths = pd.Series(None, index=items.item_id, dtype=object)
    annotated = items.item_id.isin(human.item_id).to_numpy()
    gold_scheme = items.set_index("item_id").gold_scheme
    for item, rows in human.groupby("item_id"):
        paths.loc[item] = traverse(
            schemes[gold_scheme[item]], rows.set_index("cq_id").human.to_dict()
        ).verdict
    usable = paths.notna().to_numpy()
    positions = np.arange(len(items))
    for origin, mask in origin_masks(items):
        base = {
            "scheme_version": "1.1",
            "origin": origin,
            "level": 0.95,
            "annotated_items": int((mask & annotated).sum()),
            "usable_items": int((mask & usable).sum()),
            "test_items": int(mask.sum()),
            "population": "gold-scheme sheet and positive scheme judgement",
        }
        observed, spread = score_population(
            items.gold_fallacy.tolist(), paths.to_numpy(), mask & usable, bootstrap
        )
        row = {
            **base,
            "evaluation": "frozen human reference",
            "system": "human_path",
            "reference": "gold",
            "n_items": int((mask & usable).sum()),
            "status": "computed" if (mask & usable).any() else "undefined",
            **metrics_record(observed, spread),
            **reference_support(items.gold_fallacy, paths.to_numpy(), mask & usable),
        }
        del spread
        for name, denominator in [
            ("coverage_annotated", mask & annotated),
            ("coverage_test", mask),
        ]:
            observed, spread = contribution_ratios(
                positions,
                (mask & usable).astype(float),
                denominator.astype(float),
                len(items),
                bootstrap,
            )
            row.update(metrics_record({name: observed}, {name: spread}))
        path_rows.append(row)
        for person in PEOPLE:
            common = mask & usable & ratings[person].notna().to_numpy()
            units = np.column_stack([paths.to_numpy(), ratings[person].to_numpy()]).tolist()
            observed, spread = agreement_bootstrap(units, common, bootstrap)
            path_rows.append(
                {
                    **base,
                    "evaluation": "frozen human reference",
                    "system": "human_path",
                    "reference": f"historical:{person}",
                    "n_items": int(common.sum()),
                    "population": (
                        "common usable gold-scheme paths and historical verdicts; any judged scheme"
                    ),
                    **observed,
                    **metrics_record({k: observed[k] for k in spread}, spread),
                    "status": "computed" if common.any() else "undefined",
                }
            )
            del spread
        for evaluation in evaluations:
            for system in evaluation.systems.values():
                common = mask & usable & system.eligible
                # Missing paths are outside the mask and cannot affect class supports.
                reference = paths.fillna(good_terminal()).tolist()
                observed, spread = score_population(
                    reference, system.rows.verdict.to_numpy(), common, bootstrap
                )
                path_rows.append(
                    {
                        **base,
                        "evaluation": evaluation.name,
                        "round": evaluation.round,
                        "system": system.name,
                        "system_scheme_version": system.version,
                        "reference": "human_path@1.1",
                        "n_items": int(common.sum()),
                        "population": (
                            "common system eligibility and usable gold-scheme human paths"
                        ),
                        **metrics_record(observed, spread),
                        **reference_support(reference, system.rows.verdict.to_numpy(), common),
                        "status": "computed" if common.any() else "undefined",
                    }
                )
                del spread
    return {
        "human_agreement.csv": table_frame(agreement_rows, agreement_columns),
        "human_path.csv": table_frame(path_rows, path_columns),
    }, {
        "status": "computed",
        "annotation_version": "1.1",
        **pending,
        "historical_verdicts": "cross-sheet judgements are not gold-scheme paths",
        "conflicting_verdict_ratings_excluded": int(conflicts.sum().sum()),
        "nonterminal_verdict_ratings_excluded": int(coarse.sum().sum()),
    }
