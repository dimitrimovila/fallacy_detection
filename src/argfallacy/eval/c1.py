"""CQ-label provenance and paired item-bootstrap evaluation of C1 answers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from ..aggregate.c1_inputs import C1Dataset
from ..annotations import gold_scheme_rows
from ..schemes import SchemeError, traverse
from .canonical import score_fallacies
from .cq import human_answers
from .folds import generators


def gold_scheme_answers(annotations, items, schemes):
    """Gold-scheme sheet rows whose scheme judgement also agrees with the gold."""
    return human_answers(gold_scheme_rows(items, annotations, schemes), items, schemes)


def cq_labels(annotations, items, schemes):
    """Binary CQ answers under the gold scheme, including the row's scheme judgement."""
    human = gold_scheme_answers(annotations, items, schemes)
    records = []
    for row in human.itertuples(index=False):
        arcs = schemes[row.scheme].answer_space(row.cq_id)
        if row.human in arcs:
            records.append(
                {
                    "item_id": row.item_id,
                    "scheme": row.scheme,
                    "cq_id": row.cq_id,
                    "answer": row.human,
                    "target": int(row.human == arcs[0]),
                }
            )
    return pd.DataFrame(records, columns=["item_id", "scheme", "cq_id", "answer", "target"])


def entry_answers(data: C1Dataset):
    """Each entry with its regime's distribution, including outside hard answers."""
    frames = []
    for e, entry in enumerate(data.entries):
        frame = data.pairs.copy()
        frame["p_first"], frame["answer"] = data.base_splits[:, e], data.hard[:, e]
        frame["system"], frame["supervision"] = f"entry:{entry}", "none"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class CQEvaluation:
    level: pd.DataFrame
    differences: pd.DataFrame
    resamples: np.ndarray


def cq_evaluation(
    data: C1Dataset,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    indices=None,
    seed=20261019,
    resamples=10000,
) -> CQEvaluation:
    """Pair-weighted scores; every sampled item brings all of its CQ pairs together.

    Supply the evaluation's existing resample indices to share them with other
    tables. Scheme breakdowns reuse whole-test resamples without stratification.
    """
    item_ids = list(data.items.item_id)
    if indices is None:
        indices = generators(seed)[1].integers(0, len(item_ids), size=(resamples, len(item_ids)))
    indices = np.asarray(indices, dtype=int)
    if (
        indices.ndim != 2
        or indices.shape[1] != len(item_ids)
        or (indices.size and (indices.min() < 0 or indices.max() >= len(item_ids)))
    ):
        raise SchemeError("CQ bootstrap indices must resample the complete item set")
    counts = np.zeros((len(indices) + 1, len(item_ids)))
    counts[0] = 1
    for i, row in enumerate(indices, 1):
        counts[i] = np.bincount(row, minlength=len(item_ids))
    positions = {item: i for i, item in enumerate(item_ids)}

    def resampled(item_ids_of_rows, *values):
        """The sum over the pairs of each value, in the observed sample and in every resample.

        A pair weighs as many times as its item is drawn, so the values are first summed per
        item and then multiplied by the counts of the items: the same sums, without building
        a matrix of resamples by pairs."""
        index = np.fromiter((positions[i] for i in item_ids_of_rows), dtype=int)
        per_item = [np.bincount(index, weights=v, minlength=len(item_ids)) for v in values]
        return [counts @ v for v in per_item]

    keys = ["item_id", "scheme", "cq_id"]
    if labels.duplicated(keys).any() or predictions.duplicated(["system", *keys]).any():
        raise SchemeError("duplicate paired CQ evaluation rows")
    joined = predictions.merge(labels[keys + ["target"]], on=keys, validate="many_to_one")
    metadata = predictions.groupby("system").supervision.first().to_dict()
    systems = sorted(joined.system.unique())
    reference = f"entry:{data.reference}"
    comparisons = {(s, reference) for s in systems if s != reference and reference in systems}
    comparisons.update(
        (a, b)
        for a, b in combinations(systems, 2)
        if not a.startswith("entry:") and not b.startswith("entry:") and metadata[a] != metadata[b]
    )
    levels, differences = [], []
    schemes = ["all", *sorted(data.pairs.scheme.unique())]
    for scheme in schemes:
        part = joined if scheme == "all" else joined[joined.scheme == scheme]
        arrays = {}
        for system in systems:
            rows = part[part.system == system]
            if rows.empty:
                continue
            if not np.isfinite(rows.p_first).all() or not rows.p_first.between(0, 1).all():
                raise SchemeError(f"{system}: invalid CQ evaluation probabilities")
            truth = [
                data.schemes[r.scheme].answer_space(r.cq_id)[0 if r.target else 1]
                for r in rows.itertuples(index=False)
            ]
            correct = (rows.answer.to_numpy() == np.array(truth)).astype(float)
            squared = (rows.p_first.to_numpy() - rows.target.to_numpy()) ** 2
            total, correct_sum, squared_sum = resampled(
                rows.item_id, np.ones(len(rows)), correct, squared
            )
            metrics = {
                "agreement": np.divide(
                    correct_sum, total, out=np.full(len(total), np.nan), where=total > 0
                ),
                "brier": np.divide(
                    squared_sum, total, out=np.full(len(total), np.nan), where=total > 0
                ),
            }
            arrays[system] = (rows, metrics)
            record = {"system": system, "scheme": scheme, "n": len(rows)}
            for name, values in metrics.items():
                finite = values[1:][np.isfinite(values[1:])]
                low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
                record.update({name: values[0], f"{name}_low": low, f"{name}_high": high})
            levels.append(record)
        for system, reference in sorted(comparisons):
            if system not in arrays or reference not in arrays:
                continue
            left, right = arrays[system][0], arrays[reference][0]
            # Paired comparisons use the same available item/CQ pairs on both sides.
            common = left.merge(right, on=keys, suffixes=("", "_reference"), validate="one_to_one")
            if common.empty:
                continue
            truth = np.array(
                [
                    data.schemes[r.scheme].answer_space(r.cq_id)[0 if r.target else 1]
                    for r in common.itertuples(index=False)
                ]
            )
            agreement = (common.answer.to_numpy() == truth).astype(float) - (
                common.answer_reference.to_numpy() == truth
            ).astype(float)
            brier = (
                (common.p_first - common.target) ** 2
                - (common.p_first_reference - common.target) ** 2
            ).to_numpy()
            record = {"system": system, "reference": reference, "scheme": scheme, "n": len(common)}
            total, agreement_sum, brier_sum = resampled(
                common.item_id, np.ones(len(common)), agreement, brier
            )
            for name, delta_sum in [("agreement", agreement_sum), ("brier", brier_sum)]:
                values = np.divide(
                    delta_sum, total, out=np.full(len(total), np.nan), where=total > 0
                )
                finite = values[1:][np.isfinite(values[1:])]
                low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
                record.update(
                    {f"{name}_difference": values[0], f"{name}_low": low, f"{name}_high": high}
                )
            differences.append(record)
    return CQEvaluation(pd.DataFrame(levels), pd.DataFrame(differences), indices)


def human_path_reference(annotations, items, schemes):
    """A descriptive reference, with coverage over annotated items and the whole test set."""
    human = gold_scheme_answers(annotations, items, schemes)
    predictions = {}
    gold = items.set_index("item_id")
    for item, rows in human.groupby("item_id"):
        scheme = schemes[gold.loc[item, "gold_scheme"]]
        predictions[item] = traverse(scheme, rows.set_index("cq_id").human.to_dict()).verdict
    usable = {item: verdict for item, verdict in predictions.items() if verdict is not None}
    observed = items[items.item_id.isin(usable)]
    verdicts = pd.DataFrame(
        {"item_id": observed.item_id, "predicted": [usable[i] for i in observed.item_id]}
    )
    scored = score_fallacies(verdicts, observed) if len(observed) else None
    return {
        "role": "descriptive reference",
        "annotated_items": len(predictions),
        "usable_items": len(usable),
        "test_items": len(items),
        "coverage_annotated": len(usable) / len(predictions) if predictions else np.nan,
        "coverage_test": len(usable) / len(items) if len(items) else np.nan,
        "score": scored,
    }
