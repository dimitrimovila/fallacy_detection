"""Explicit app answers and associations with observed human disagreement."""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..annotation_samples import CQ_V2, DIRECT
from ..schemes import traverse
from .analysis_bootstrap import contribution_ratios, score_population
from .analysis_human import historical_ratings, reference_support
from .analysis_inputs import require, unique
from .analysis_selective import origin_masks
from .analysis_tables import metrics_record, table_frame
from .b1 import gold_scheme_answers
from .round2 import ORIGINS
from .run_report import good_terminal
from .uncertainty import LEVELS, _cut, saved_ranking


def csv_integer(value, name, minimum=0):
    require(re.fullmatch(r"[0-9]+", str(value)) is not None, f"{name}: invalid integer")
    result = int(value)
    require(result >= minimum, f"{name}: integer below {minimum}")
    return result


def read_direct(config, items, schemes, inputs):
    """Read exactly the two configured CSVs, selecting revisions without time ordering."""
    paths = config.direct_verdict
    require(
        (paths["answers"] is None) == (paths["samples"] is None),
        "direct_verdict: both answers and samples are required",
    )
    columns = ["position", "verdict", "answered_at", "revision"]
    empty = pd.DataFrame(columns=columns, index=pd.Index([], name="item_id"))
    if paths["answers"] is None:
        return empty, {"status": "pending", "reason": "explicit human inputs not configured"}
    samples = inputs.csv(paths["samples"], ["item_id", "sample", "scheme", "round", "position"])
    require(set(samples["sample"]) <= {DIRECT, CQ_V2}, "unknown sample identifier")
    unique(samples, ["item_id"], "sample item ids")
    gold = items.set_index("item_id").gold_scheme
    require(set(samples.item_id) <= set(gold.index), "unknown sample items")
    require(
        all(gold[r.item_id] == r.scheme for r in samples.itertuples()),
        "sample gold scheme mismatch",
    )
    planned = samples[samples["sample"] == DIRECT].copy()
    for name in ("position", "round"):
        planned[name] = [csv_integer(v, f"sample {name}", 1) for v in planned[name]]
    unique(planned, ["position"], "sample position")
    planned = planned.set_index("item_id")
    answers = inputs.csv(paths["answers"], ["item_id", *columns])
    require(set(answers.item_id) <= set(gold.index), "unknown answer items")
    require(set(answers.item_id) <= set(planned.index), "answer outside direct sample")
    seen = {}
    for row in answers.itertuples():
        position = csv_integer(row.position, "answer position", 1)
        revision = csv_integer(row.revision, "answer revision")
        require(position == planned.loc[row.item_id, "position"], "answer/sample position mismatch")
        try:
            stamp = datetime.fromisoformat(str(row.answered_at))
        except (TypeError, ValueError):
            require(False, "invalid answered_at timestamp")
        require(stamp.utcoffset() is not None, "answered_at needs a time zone offset")
        # Canonical ids only: display names and unknown ids must not be normalized here.
        all_terminals = set().union(*(s.terminals for s in schemes.values()))
        require(row.verdict in all_terminals, "human verdict must be a canonical terminal id")
        require(
            row.verdict in schemes[gold[row.item_id]].terminals, "human verdict outside 1.1 menu"
        )
        key = (row.item_id, revision)
        require(key not in seen or seen[key] == row.verdict, "conflicting (item_id, revision)")
        seen[key] = row.verdict
    for name in ("position", "revision"):
        answers[name] = [csv_integer(v, name, int(name == "position")) for v in answers[name]]
    chosen = answers.sort_values("revision", kind="stable").drop_duplicates("item_id", keep="last")
    chosen = chosen.set_index("item_id")[columns].sort_index()
    # Empty files and explicit finals have the same typed interface as an answer log.
    chosen.index.name = "item_id"
    complete, partial = [], []
    for number, group in planned.groupby("round", sort=True):
        (complete if set(group.index) <= set(chosen.index) else partial).append(int(number))
    counts = planned.assign(origin=items.set_index("item_id").source.str.lower().map(ORIGINS))
    counts["answered"] = counts.index.isin(chosen.index)
    by_group = [
        dict(
            scheme=s,
            origin=o,
            planned=len(g),
            answered=int(g.answered.sum()),
            missing=int((~g.answered).sum()),
        )
        for (s, o), g in counts.groupby(["scheme", "origin"], sort=True)
    ]
    return chosen, {
        "status": "computed",
        "planned": len(planned),
        "answered": len(chosen),
        "missing": len(planned) - len(chosen),
        "complete_rounds": complete,
        "partial_rounds": partial,
        "by_scheme_origin": by_group,
        "planned_minimum": 200,
        "planned_minimum_reached": len(chosen) >= 200,
        "population": "explicit direct sample; unanswered items have no inferred label",
    }


def auroc(disagreement, uncertainty):
    """Mann-Whitney AUROC, with average ranks and half credit for score ties."""
    target = np.asarray(disagreement, bool)
    positive, negative = int(target.sum()), int((~target).sum())
    if not positive or not negative:
        return np.nan
    ranks = rankdata(uncertainty)
    return float((ranks[target].sum() - positive * (positive + 1) / 2) / (positive * negative))


def disagreement_statistics(
    ids, verdicts, confidence, eligible, human, bootstrap, levels=LEVELS, subgroup=None
):
    """Rank global eligible draws first; intersect observed humans and origins afterwards."""
    confidence, human = np.asarray(confidence, float), np.asarray(human, object)
    eligible = np.asarray(eligible, bool)
    subgroup = np.ones(len(ids), bool) if subgroup is None else np.asarray(subgroup, bool)
    known = pd.notna(human)
    target = np.zeros(len(ids), bool)
    target[known] = human[known].astype(bool)
    order = saved_ranking(ids, verdicts, confidence)
    rank = np.empty(len(ids), int)
    rank[order] = np.arange(len(ids))
    operational = pd.notna(verdicts) & np.isfinite(confidence)
    ranked = bool((eligible & operational).any())
    descriptors = [("all", None), ("agreement", None), ("disagreement", None)]
    if ranked:
        descriptors += [(selection, c) for c in levels for selection in ("kept", "rejected")]

    def calculate(draw):
        global_draw = draw[eligible[draw]]
        global_draw = global_draw[np.argsort(rank[global_draw], kind="stable")]
        output = []
        for selection, level in descriptors:
            selected = global_draw
            if level is not None:
                cut = _cut(level, len(global_draw)) if len(global_draw) else 0
                selected = global_draw[:cut] if selection == "kept" else global_draw[cut:]
            selected = selected[subgroup[selected]]
            observed = selected[known[selected]]
            if selection in ("agreement", "disagreement"):
                observed = observed[target[observed] == (selection == "disagreement")]
            finite = observed[np.isfinite(confidence[observed])]
            valid = observed[operational[observed]]
            output.append(
                {
                    "selection": selection,
                    "ranking_available": ranked,
                    "requested_coverage": level,
                    "eligible_items": len(selected),
                    "observed_items": len(observed),
                    "missing_human": int((~known[selected]).sum()),
                    "confidence_items": len(finite),
                    "missing_confidence": len(observed) - len(finite),
                    "missing_verdicts": int(pd.isna(np.asarray(verdicts)[observed]).sum()),
                    "auroc_items": len(valid),
                    "disagreement_prevalence": float(target[observed].mean())
                    if len(observed)
                    else np.nan,
                    "mean_confidence": float(confidence[finite].mean()) if len(finite) else np.nan,
                    "auroc": auroc(target[valid], 1 - confidence[valid])
                    if selection == "all"
                    else np.nan,
                }
            )
        return output

    observed = calculate(np.arange(len(ids)))
    names = ("disagreement_prevalence", "mean_confidence", "auroc")
    spread = [{name: np.full(bootstrap.resamples, np.nan) for name in names} for _ in observed]
    start = 0
    for batch in bootstrap.indices(len(ids)):
        for offset, draw in enumerate(batch):
            for j, row in enumerate(calculate(draw)):
                for name in names:
                    spread[j][name][start + offset] = row[name]
        start += len(batch)
    for row, values in zip(observed, spread, strict=True):
        row.update(metrics_record({name: row[name] for name in names}, values))
        row.update({name + "_undefined": not np.isfinite(row[name]) for name in names})
    return observed


def human_paths(annotations, items, schemes):
    paths = pd.Series(None, index=items.item_id, dtype=object)
    if annotations is not None:
        human = gold_scheme_answers(annotations, items, schemes)
        gold = items.set_index("item_id").gold_scheme
        for item, rows in human.groupby("item_id"):
            paths.loc[item] = traverse(
                schemes[gold[item]], rows.set_index("cq_id").human.to_dict()
            ).verdict
    return paths


def direct_tables(observed, annotations, evaluations, items, schemes, bootstrap):
    direct = observed.verdict.reindex(items.item_id)
    known = direct.notna().to_numpy()
    attributes = np.full(len(items), None, object)
    attributes[known] = direct.to_numpy()[known] != items.gold_fallacy.to_numpy()[known]
    columns = [
        "evaluation",
        "system",
        "reference",
        "scheme_version",
        "origin",
        "comparison",
        "n_items",
        "accuracy",
        "status",
    ]
    uncertainty_columns = [
        "evaluation",
        "system",
        "scheme_version",
        "origin",
        "selection",
        "observed_items",
        "auroc",
        "mean_confidence",
        "disagreement_prevalence",
        "interpretation",
        "status",
    ]
    output, uncertainty = [], []
    paths = human_paths(annotations, items, schemes)
    ratings = (
        historical_ratings(annotations, items, schemes)[0] if annotations is not None else None
    )
    for origin, mask in origin_masks(items):
        primary = mask & known
        meta = dict(scheme_version="1.1", origin=origin, level=0.95, kind="exploratory")

        def score(
            name,
            reference_name,
            truth,
            predictions,
            common,
            comparison,
            evaluation="human reference",
            meta=meta,
        ):
            reference = pd.Series(np.asarray(truth, object)).fillna(good_terminal()).to_numpy()
            point, spread = score_population(
                reference.tolist(), np.asarray(predictions, object), common, bootstrap
            )
            output.append(
                {
                    **meta,
                    "evaluation": evaluation,
                    "system": name,
                    "reference": reference_name,
                    "comparison": comparison,
                    "n_items": int(common.sum()),
                    "status": "computed" if common.any() else "undefined",
                    **metrics_record(point, spread),
                    **reference_support(reference, predictions, common),
                }
            )
            return point, spread

        if known.any():
            score(
                "human_direct", "gold", items.gold_fallacy, direct, primary, "direct against gold"
            )
            common = primary & paths.notna().to_numpy()
            a, da = score(
                "human_direct",
                "gold",
                items.gold_fallacy,
                direct,
                common,
                "paired direct/path against gold",
            )
            b, db = score(
                "human_path",
                "gold",
                items.gold_fallacy,
                paths,
                common,
                "paired direct/path against gold",
            )
            output.append(
                {
                    **meta,
                    "evaluation": "human reference",
                    "system": "human_direct-human_path",
                    "reference": "gold",
                    "comparison": "paired direct/path against gold",
                    "n_items": int(common.sum()),
                    "interpretation": "same-person procedural comparison",
                    **metrics_record(
                        {k + "_difference": a[k] - b[k] for k in a},
                        {k + "_difference": da[k] - db[k] for k in a},
                    ),
                    "status": "computed" if common.any() else "pending human paths",
                }
            )
            for label, reference in [
                ("human_path", paths),
                *(
                    [(f"historical:{p}", ratings[p]) for p in ratings]
                    if ratings is not None
                    else []
                ),
            ]:
                common = primary & reference.notna().to_numpy()
                estimate, spread = contribution_ratios(
                    np.arange(len(items)),
                    (direct.to_numpy() == reference.to_numpy()) & common,
                    common.astype(float),
                    len(items),
                    bootstrap,
                )
                output.append(
                    {
                        **meta,
                        "evaluation": "human reference",
                        "system": "human_direct",
                        "reference": label,
                        "comparison": "observed verdict agreement",
                        "n_items": int(common.sum()),
                        "interpretation": "same-person procedural comparison"
                        if label in ("human_path", "historical:dumitru")
                        else "other-rater comparison",
                        **metrics_record({"agreement": estimate}, {"agreement": spread}),
                        "status": "computed" if common.any() else "pending",
                    }
                )
        for evaluation in evaluations:
            for system in evaluation.systems.values():
                if known.any():
                    score(
                        system.name,
                        "human_direct@1.1",
                        direct,
                        system.rows.verdict.to_numpy(),
                        primary & system.eligible,
                        "second reference; primary gold unchanged",
                        evaluation.name,
                    )
                    output[-1]["system_scheme_version"] = system.version
                if system.role == "oracle" or not known.any():
                    continue
                for row in disagreement_statistics(
                    items.item_id.to_numpy(),
                    system.rows.verdict.to_numpy(),
                    system.rows.confidence.to_numpy(float),
                    system.eligible,
                    attributes,
                    bootstrap,
                    subgroup=mask,
                ):
                    uncertainty.append(
                        {
                            **meta,
                            "evaluation": evaluation.name,
                            "system": system.name,
                            "scheme_version": system.version,
                            **row,
                            "reference": "direct human != gold",
                            "interpretation": "association with observed disagreement",
                            "ranking_population": (
                                "global system eligibility before human/origin intersection"
                            ),
                            "status": "computed" if row["observed_items"] else "undefined",
                        }
                    )
    return {
        "human_direct.csv": table_frame(output, columns),
        "uncertainty_disagreement.csv": table_frame(uncertainty, uncertainty_columns),
        "human_disagreement": attributes,
    }
