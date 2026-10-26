"""How much a better CQ answer is worth in verdicts: the human answers put in place of the
model's, through the diagram, one CQ at a time and at random shares of the disagreements.

The diagram is traversed with the rule of A0.  Only binary human answers enter (the CQ
labels of round 2); the other answers of each item stay those of the model.  Agreement
counts an answer outside the arcs as a disagreement, as the evaluation at the level of
each CQ does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..schemes import Scheme, traverse
from .evaluate import interval

FRACTIONS = tuple(k / 10 for k in range(11))
"""The shares of the disagreeing pairs that take the human answer, in the curve."""
DRAWS = 50
"""The random draws of the disagreeing pairs at each share."""


def substitution_rng(seed: int) -> np.random.Generator:
    """The fourth generator of the seed: the first three draw the folds and the two sets
    of resamples, so the curve changes none of them."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[3])


def _verdicts(schemes: Mapping[str, Scheme], scheme_of: Mapping[str, str], ids: Sequence[str],
              answers: Mapping[str, Mapping[str, Any]]) -> list[str | None]:
    return [traverse(schemes[scheme_of[i]], answers[i]).verdict for i in ids]


def _replaced(base: Mapping[str, Mapping[str, Any]], pairs: pd.DataFrame
              ) -> dict[str, dict[str, Any]]:
    answers = {i: dict(given) for i, given in base.items()}
    for item, cq, human in zip(pairs.item_id, pairs.cq_id, pairs.answer, strict=True):
        answers[item][cq] = human
    return answers


def labelled_pairs(rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
                   labels: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
    """The CQ labels of the items in ``ids``, each with the hard answer of the model."""
    own = labels[labels.item_id.isin(set(ids))].copy()
    own["model"] = [(rows[i].get(cq) or {}).get("answer")
                    for i, cq in zip(own.item_id, own.cq_id, strict=True)]
    own["agrees"] = own.model == own.answer
    return own.reset_index(drop=True)


def substitution(schemes: Mapping[str, Scheme], scheme_of: Mapping[str, str],
                 ids: Sequence[str], rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
                 labels: pd.DataFrame, bench, frequent: Sequence[int], seed: int,
                 fractions: Sequence[float] = FRACTIONS, draws: int = DRAWS
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The table by CQ and the curve.

    ``rows`` is item -> CQ -> row of the model; ``bench`` scores a set of verdicts on the
    observed items and on the resamples of the evaluation; ``frequent`` are the indices of
    the classes of the macro F1 over the frequent classes.  By CQ: every labelled pair of
    that CQ takes the human answer, and the last row does it for every CQ at once; each
    with the verdicts that change and the paired differences from the diagram on the
    model's answers, at 95 per cent.  The curve: at each share, ``draws`` random subsets
    of that share of the disagreeing pairs take the human answer; the mean of each
    metric over the draws, with the 2.5 and 97.5 percentiles of the draws.
    """
    base = {i: {cq: row["answer"] for cq, row in rows[i].items() if row is not None}
            for i in ids}
    pairs = labelled_pairs(rows, labels, ids)
    before = _verdicts(schemes, scheme_of, ids, base)
    reference = bench.score(before)
    right_before = bench.observed.correct(before)

    def frequent_f1(scorer, verdicts):
        f1, supported = scorer.class_f1(verdicts)
        n = supported[:, frequent].sum(axis=1)
        return np.divide(f1[:, frequent].sum(axis=1), n, out=np.full(len(f1), np.nan),
                         where=n > 0)

    reference_frequent = frequent_f1(bench.resampled, before)
    observed_frequent = float(frequent_f1(bench.observed, before)[0])

    table = []
    groups = [((s, q), own) for (s, q), own in pairs.groupby(["scheme", "cq_id"], sort=True)]
    groups.append((("all", "all"), pairs))
    for (scheme_id, cq), own in groups:
        after = _verdicts(schemes, scheme_of, ids, _replaced(base, own))
        scored = bench.score(after)
        right_after = bench.observed.correct(after)
        changed = np.array([a != b for a, b in zip(after, before, strict=True)])
        row = {"scheme": scheme_id, "cq_id": cq, "labelled": len(own),
               "disagreeing": int((~own.agrees).sum()),
               "verdicts_changed": int(changed.sum()),
               "wrong_to_right": int((~right_before & right_after).sum()),
               "right_to_wrong": int((right_before & ~right_after).sum())}
        for metric in ("macro_f1", "accuracy"):
            low, high = interval(scored.resampled[metric] - reference.resampled[metric])
            row.update({f"{metric}_difference": scored.observed[metric]
                        - reference.observed[metric], f"{metric}_low": low,
                        f"{metric}_high": high})
        spread = frequent_f1(bench.resampled, after) - reference_frequent
        finite = spread[np.isfinite(spread)]
        # with no frequent class among the items there is no value to resample
        low, high = interval(finite) if len(finite) else (np.nan, np.nan)
        row.update({"macro_f1_frequent_difference":
                    float(frequent_f1(bench.observed, after)[0]) - observed_frequent,
                    "macro_f1_frequent_low": low, "macro_f1_frequent_high": high})
        table.append(row)

    rng = substitution_rng(seed)
    disagreeing = pairs[~pairs.agrees].reset_index(drop=True)
    base_agreement = float(pairs.agrees.mean()) if len(pairs) else np.nan
    curve = []
    for share in fractions:
        k = int(round(share * len(disagreeing)))
        values = {"macro_f1": [], "macro_f1_frequent": [], "accuracy": []}
        for _ in range(draws):
            chosen = disagreeing.iloc[np.sort(rng.choice(len(disagreeing), k, replace=False))]
            after = _verdicts(schemes, scheme_of, ids, _replaced(base, chosen))
            metrics = bench.observed.metrics(after)
            values["macro_f1"].append(float(metrics["macro_f1"][0]))
            values["accuracy"].append(float(metrics["accuracy"][0]))
            values["macro_f1_frequent"].append(float(frequent_f1(bench.observed, after)[0]))
        row = {"share": share, "pairs_replaced": k,
               "agreement": base_agreement + k / len(pairs) if len(pairs) else np.nan}
        for metric, own in values.items():
            row[metric] = float(np.mean(own))
            row[f"{metric}_low"], row[f"{metric}_high"] = (
                float(v) for v in np.percentile(own, [2.5, 97.5]))
        curve.append(row)
    return pd.DataFrame(table), pd.DataFrame(curve)


__all__ = ["DRAWS", "FRACTIONS", "labelled_pairs", "substitution", "substitution_rng"]
