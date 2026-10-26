"""Parser integrity denominators and frozen-test scheme recognition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..parse import INVALID, valid_samples
from ..prompts import CANNOT_BE_DETERMINED, NOT_APPLICABLE
from ..schemes import load_all
from .analysis_inputs import boolean, require, unique
from .canonical import score_schemes


def stage_one(run_id, version, manifest, summary, items):
    rows = summary[(summary.stage == "stage1") & summary.item_id.isin(items.item_id)]
    output = []
    for entry, frame in rows.groupby("model", sort=False):
        unique(frame, ["item_id"], f"{run_id}/stage1/{entry}")
        hard = frame.set_index("item_id").answer.reindex(items.item_id).replace("", None)
        hard = hard.where(hard.notna(), None).replace(INVALID, None)
        score = score_schemes(
            pd.DataFrame({"item_id": items.item_id, "predicted": hard.to_numpy()}), items
        )
        output.append(
            {
                "run": run_id,
                "system": f"stage1:{entry}",
                "scheme_version": version,
                "subset": "frozen test",
                "condition": "scheme recognition",
                "n_items": score.n_items,
                "observed_rows": len(frame),
                "missing_rows": len(items) - len(frame),
                "accuracy": score.accuracy,
                "macro_f1": score.macro_f1,
                "balanced_accuracy": score.macro_recall,
                "n_classes": score.n_classes_gold,
                "excluded_classes": ";".join(score.left_out),
                "status": "computed",
            }
        )
    return output


def integrity(run_id, version, manifest, summary, path, items, registry, inputs):
    """Report sample, expected independent CQ-row and item units separately.

    Sequential histories have observed-node denominators: unvisited CQs are not
    missing. No raw responses or human annotations are read.
    """
    output = []
    if not (path / "answers.csv").is_file():
        return [
            {
                "run": run_id,
                "system": "",
                "scheme_version": version,
                "stage": "",
                "measure": "sample integrity",
                "unit": "sample",
                "count": np.nan,
                "denominator": np.nan,
                "rate": np.nan,
                "status": "pending",
                "reason": "parsed answers.csv unavailable",
            }
        ]
    answers = inputs.csv(
        path / "answers.csv",
        [
            "item_id",
            "model",
            "stage",
            "scheme_condition",
            "scheme",
            "cq_id",
            "sample_index",
            "answer",
            "parse_ok",
            "finish_reason",
        ],
    )
    answers = answers[answers.item_id.isin(items.item_id) & (answers.scheme_condition == "gold")]
    answers = answers.copy()
    answers["parse_ok"] = answers.parse_ok.map(lambda v: boolean(v, "parse_ok"))
    keys = ["item_id", "model", "stage", "scheme_condition", "scheme", "cq_id", "sample_index"]
    if "path" in answers:
        keys.append("path")
    unique(answers, keys, f"{run_id}/answers.csv")
    schemes = load_all(registry[version])
    entries = set(answers.model) | set(summary.model)

    def add(entry, stage, measure, unit, count, denominator, basis):
        output.append(
            {
                "run": run_id,
                "system": f"{stage}:{entry}",
                "scheme_version": version,
                "stage": stage,
                "subset": "frozen test",
                "measure": measure,
                "unit": unit,
                "count": count,
                "denominator": denominator,
                "denominator_basis": basis,
                "rate": count / denominator if denominator else np.nan,
                "undefined": denominator == 0,
                "status": "computed",
            }
        )

    for entry in sorted(entries):
        samples = (
            manifest.get("models", {})
            .get(entry, {})
            .get("samples", manifest.get("config", {}).get("samples", 1))
        )
        require(type(samples) is int and samples > 0, f"{run_id}/{entry}: invalid samples")
        stages = sorted(
            set(answers.loc[answers.model == entry, "stage"])
            | set(summary.loc[summary.model == entry, "stage"])
        )
        for stage in stages:
            own = answers[(answers.model == entry) & (answers.stage == stage)]
            summed = summary[
                (summary.model == entry)
                & (summary.stage == stage)
                & summary.item_id.isin(items.item_id)
                & (summary.scheme_condition == "gold")
            ]
            require(
                all(type(v) in (int, np.int64) and 0 <= v < samples for v in own.sample_index),
                f"{run_id}/{entry}: sample index out of range",
            )
            if stage == "stage2":
                expected = {
                    (row.item_id, cq)
                    for row in items.itertuples()
                    for cq in schemes[row.gold_scheme].cq_order
                }
                observed = set(zip(summed.item_id, summed.cq_id, strict=True))
                sample_questions = set(zip(own.item_id, own.cq_id, strict=True))
                require(
                    observed <= expected and sample_questions <= expected,
                    f"{run_id}/{entry}: unknown CQ row",
                )
                denominator = len(expected)
                add(
                    entry,
                    stage,
                    "missing CQ rows",
                    "CQ row",
                    len(expected - observed),
                    denominator,
                    "all frozen-test independent CQs",
                )
                for answer, measure in [
                    (INVALID, "invalid CQ rows"),
                    (CANNOT_BE_DETERMINED, "cannot_be_determined"),
                    (NOT_APPLICABLE, "na"),
                ]:
                    add(
                        entry,
                        stage,
                        measure,
                        "CQ row",
                        int(summed.answer.eq(answer).sum()),
                        denominator,
                        "all frozen-test independent CQs",
                    )
                expected_samples = denominator * samples
            elif stage == "ssoft":
                node_keys = ["item_id", "cq_id"] + (["path"] if "path" in own else [])
                denominator = len(own.drop_duplicates(node_keys))
                expected_samples = denominator * samples
                for answer, measure in [
                    (CANNOT_BE_DETERMINED, "cannot_be_determined"),
                    (NOT_APPLICABLE, "na"),
                ]:
                    add(
                        entry,
                        stage,
                        measure,
                        "sample",
                        int(own.answer.eq(answer).sum()),
                        expected_samples,
                        "observed sequential histories only",
                    )
            else:
                denominator = len(items)
                expected_samples = denominator * samples
            require(len(own) <= expected_samples, f"{run_id}/{entry}: too many samples")
            valid = valid_samples(own)
            basis = "observed histories" if stage == "ssoft" else "frozen-test planned samples"
            add(
                entry,
                stage,
                "invalid samples",
                "sample",
                int(own.answer.eq(INVALID).sum()),
                expected_samples,
                basis,
            )
            add(
                entry,
                stage,
                "unusable samples",
                "sample",
                len(own) - len(valid),
                expected_samples,
                basis,
            )
            add(
                entry,
                stage,
                "missing samples",
                "sample",
                expected_samples - len(own),
                expected_samples,
                basis,
            )
    return output
