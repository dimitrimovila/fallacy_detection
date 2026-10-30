"""Exploratory B2 families with version-matched fixed sides."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from ..aggregate import load_burden, routes
from ..aggregate.b1_inputs import build_dataset
from ..aggregate.crossfit import B1CrossFitter
from ..aggregate.routing import Side, b2_out_of_fold
from .evaluate import _read_run, _zeroshot_answers
from .round2 import System, _choices
from .routing import routing_curve

CHOICE_COLUMNS = ["fold", "component", "scheme", "parameter", "value", "training_items"]
CURVE_COLUMNS = [
    "tau",
    "explained_share",
    "accuracy",
    "macro_f1",
    "requested_share",
    "attainable",
    "accuracy_low",
    "accuracy_high",
    "macro_f1_low",
    "macro_f1_high",
    "explained_low",
    "explained_high",
]


def routing_systems(
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
    consistent,
    progress=None,
):
    """Reuse exact-training-set B1 cross-fitting, freeing each scheme family in turn."""
    reference = config.reference_model
    zero1, zero2 = f"zeroshot:{reference}", f"zeroshot:{reference}@2.0"
    dq1, dq2 = f"DQ:{reference}", f"DQ:{reference}@2.0"
    required = {
        "B2.dq": (zero1, dq1),
        "B2@2.0": (zero2,),
        "B2.dq@2.0": (zero2, dq2, "B4"),
    }
    available = {**legacy, **{n: s.aggregations for n, s in systems.items()}}
    missing, planned = {}, set()
    for name, needs in required.items():
        absent = [n for n in needs if n not in available]
        if not config.runs.get("answers"):
            absent.append("answers")
        if name != "B2.dq" and not config.runs.get("answers_schemes_2"):
            absent.append("answers_schemes_2")
        if absent:
            missing[name] = "missing inputs: " + ", ".join(absent)
        else:
            planned.add(name)
    added, origins, choices, shares = {}, {}, [], {}
    curve = pd.DataFrame(columns=CURVE_COLUMNS)
    if not planned:
        return added, origins, curve, pd.DataFrame(columns=CHOICE_COLUMNS), shares, missing
    ids = list(items.item_id)
    scheme_of = items.set_index("item_id").gold_scheme.to_dict()

    def dataset(key, folder):
        manifest, summary = _read_run(runs_dir, config.runs[key])
        routing = {
            sid: routes(schemes[folder][sid], table)
            for sid, table in load_burden(folder, schemes[folder]).items()
        }
        return build_dataset(schemes[folder], items, summary, manifest, reference, routing)

    def side(name, version, *, dq=False, explained=False):
        values = available[name]
        flags = (
            {
                i: consistent(
                    schemes["v1" if version == "1.1" else "v2"][scheme_of[i]],
                    cq_rows[name][i],
                    values[i].verdict,
                )["consistent"]
                for i in ids
            }
            if dq
            else None
        )
        return Side(
            name,
            values,
            {i: np.nan if values[i].confidence is None else values[i].confidence for i in ids},
            consistent=flags,
            explained=explained,
            schemes_version=version,
        )

    fit_data = dataset("answers", "v1")
    for folder, names in [("v1", ("B2.dq",)), ("v2", ("B2@2.0", "B2.dq@2.0"))]:
        names = [n for n in names if n in planned]
        if not names:
            continue
        version = "1.1" if folder == "v1" else "2.0"
        target = fit_data if folder == "v1" else dataset("answers_schemes_2", "v2")
        source = B1CrossFitter(target, labels, config.seed, fit_data=fit_data)
        zero_name, dq_name = (zero1, dq1) if folder == "v1" else (zero2, dq2)
        if folder == "v1":
            # Keep the original round-two zero-shot, including its own confidence.
            _, zero_runs = _zeroshot_answers(runs_dir, config.runs.get("zeroshot"), ids)
            zero_run = zero_runs[reference]
        else:
            zero_run = provenance[zero_name]
        zero = side(zero_name, version)
        for name in names:
            if progress:
                progress(name + " (exploratory)")
            extras = () if name == "B2@2.0" else (side(dq_name, version, dq=True),)
            if name == "B2.dq@2.0":
                extras += (side("B4", version, explained=True),)
            result = b2_out_of_fold(
                source, zero, folds, extra_sides=extras, name=name, schemes_version=version
            )
            added[name] = System(
                name,
                result.aggregations,
                "exploratory",
                version,
                "cross-fitted correctness routing: B1, "
                + ", ".join([zero_name, *[s.name for s in extras]])
                + "; threshold 0",
            )
            origins[name] = {
                "answers": config.runs["answers" if folder == "v1" else "answers_schemes_2"],
                "fit_answers": config.runs["answers"],
                "zeroshot": zero_run,
            }
            if extras:
                origins[name]["dq"] = provenance[dq_name]
            if name == "B2.dq@2.0":
                origins[name]["ssoft"] = config.runs.get("ssoft", [])
            choices += _choices(name, result.choices)
            shares[name] = result.records.set_index("item_id").explained.reindex(ids).to_numpy()
            if name == "B2@2.0":
                curve = routing_curve(result, items, indices)
            del result
        # Include every upstream fit and its selected penalty, including inner partitions.
        for choice in source.choices:
            training = "; ".join(choice["training_items"])
            choices.append(
                {
                    "component": "B1" if folder == "v1" else "B1@2.0",
                    "scheme": "",
                    "training_items": training,
                    "parameter": "C",
                    "value": choice["parameter"],
                }
            )
        del source, target
        gc.collect()
    return added, origins, curve, pd.DataFrame(choices, columns=CHOICE_COLUMNS), shares, missing
