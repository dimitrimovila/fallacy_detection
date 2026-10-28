"""Explicit CPU-only analysis, report regeneration or isolated primary-C2 replay."""

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from argfallacy.eval.analysis import AnalysisConfig, analyze, peak_memory_bytes  # noqa: E402
from argfallacy.eval.analysis_inputs import (  # noqa: E402
    Inputs,
    read_evaluation,
    read_items,
    read_run,
    require,
    scheme_registry,
)
from argfallacy.eval.html_report import write_summary  # noqa: E402


def replay(config, evaluation_name, out, check_only):
    """Refit only primary C1/C2, export after saved agreement, never evaluate a round."""
    from argfallacy.aggregate import load_burden, routes
    from argfallacy.aggregate.c1 import C_GRID, PRIMARY_CONFIG, c1_out_of_fold
    from argfallacy.aggregate.c1_inputs import build_dataset
    from argfallacy.aggregate.crossfit import C1CrossFitter
    from argfallacy.aggregate.routing import C_GRID as C2_GRID
    from argfallacy.aggregate.routing import c2_out_of_fold
    from argfallacy.eval.analysis_conformal import export_conformal, selected_systems
    from argfallacy.eval.analysis_human import read_frozen_annotations
    from argfallacy.eval.analysis_routing import export_routing_records
    from argfallacy.eval.c1 import cq_labels
    from argfallacy.eval.round2 import _side
    from argfallacy.schemes import load_all

    started = time.perf_counter()
    config.guard_output(out)
    require(not out.exists(), "replay output must be a new directory")
    require(evaluation_name in config.evaluations, "explicit source evaluation is required")
    source = config.evaluations[evaluation_name]
    require(source["round"] == 2, "primary C2 replay needs a round-two source")
    inputs = Inputs()
    inputs.add(config.path)
    items = read_items(config, inputs)
    registry = scheme_registry(inputs)
    evaluation = read_evaluation(evaluation_name, source, config, items, registry, inputs)
    metadata = evaluation.manifest
    require(
        metadata.get("grids", {}).get("C1 stacking C") == list(C_GRID),
        "replay C1 grid/settings differ from saved source",
    )
    require(
        metadata.get("grids", {}).get("C2 C") == list(C2_GRID),
        "replay C2 grid/settings differ from saved source",
    )
    reference = metadata["reference_model"]
    answers_id = metadata["runs"]["answers"]
    zero_id = metadata.get("zeroshot_sources", {}).get(reference)
    require(
        answers_id in config.parsed_runs and zero_id in config.parsed_runs,
        "explicit parsed answers and zero-shot sources required",
    )
    answer_manifest, answer_version, summary = read_run(
        config.parsed_runs[answers_id], answers_id, items, registry, inputs
    )
    _, zero_version, zero = read_run(config.parsed_runs[zero_id], zero_id, items, registry, inputs)
    require(answer_version == zero_version == "1.1", "primary replay requires schemes 1.1")
    selected = zero[
        (zero.stage == "zeroshot")
        & (zero.model == reference)
        & zero.scheme_condition.eq("gold")
        & zero.cq_id.eq("")
    ]
    require(
        set(selected.item_id) == set(items.item_id) and "p_label" in selected,
        "replay unavailable: complete genuine zero-shot p_label export required; "
        "never derive probabilities from rounded tables",
    )
    import pandas as pd

    probabilities = pd.to_numeric(selected.p_label, errors="coerce")
    require(
        probabilities.notna().all() and probabilities.between(0, 1).all(),
        "replay unavailable: missing/invalid zero-shot p_label",
    )
    for r in selected.itertuples():
        saved = evaluation.systems["zeroshot:" + reference].rows.loc[r.item_id]
        require(r.answer == saved.verdict, "replay zero-shot saved prediction disagreement")
    notes = read_frozen_annotations(config, inputs)
    require(notes is not None, "replay requires frozen historical CQ targets")
    dependencies = list(inputs.files)
    audit = dict(
        source=evaluation_name,
        source_path=str(source["path"]),
        source_commit_status="recorded" if metadata.get("commit") else "unavailable in source",
        configuration=str(config.path),
        inputs=inputs.records(),
        folds=metadata["folds"],
        settings=metadata["grids"],
        primary_C1=vars(PRIMARY_CONFIG),
        scope="primary C2 and C1 conformal only; other C5 systems remain pending",
        resource_status="fitted replay not profiled until this separate command runs",
    )
    inputs.verify()
    if check_only:
        return audit
    schemes = load_all(registry["1.1"])
    burden = load_burden(registry["1.1"].name, schemes)
    routing = {sid: routes(schemes[sid], burden[sid]) for sid in schemes}
    frame = items[["item_id", "gold_scheme", "gold_fallacy"]]
    data = build_dataset(schemes, frame, summary, answer_manifest, reference, routing)
    labels = cq_labels(notes, items, schemes)
    folds = metadata["folds"]["assignment"]
    upstream = C1CrossFitter(data, labels, config.seed, PRIMARY_CONFIG)
    side = _side(zero, "zeroshot:" + reference, reference, "zeroshot", items.item_id.tolist())
    print("Fitting isolated primary C2 with unchanged nested cross-fitting", flush=True)
    result = c2_out_of_fold(upstream, side, folds)
    import numpy as np

    for item, aggregation in result.aggregations.items():
        saved = evaluation.systems["C2"].rows.loc[item]
        require(
            aggregation.verdict == saved.verdict
            and (
                (aggregation.confidence is None and pd.isna(saved.confidence))
                or (
                    aggregation.confidence is not None
                    and np.isclose(aggregation.confidence, saved.confidence, rtol=0, atol=1e-12)
                )
            ),
            f"primary C2 replay/saved prediction or confidence disagreement: {item}",
        )
    inputs.verify()
    routing_file = export_routing_records(result, items, out / "routing", source["path"])
    audit["routing_records"] = str(routing_file)
    del result, upstream
    gc.collect()
    selected_c5 = selected_systems(evaluation, inputs)
    if selected_c5 is not None and "C1" in set(selected_c5.system):
        print("Fitting primary C1 for a verified genuine C5 companion export", flush=True)
        c1 = c1_out_of_fold(data, labels, folds, config.seed, PRIMARY_CONFIG)
        inputs.verify()
        export_conformal(
            evaluation,
            items,
            registry,
            {"C1": c1.aggregations},
            out / "conformal",
            dependencies,
            dict(
                elapsed_seconds=time.perf_counter() - started,
                peak_process_memory_bytes=peak_memory_bytes(),
            ),
        )
        audit["conformal_records"] = str(out / "conformal")
    audit.update(
        elapsed_seconds=time.perf_counter() - started, peak_process_memory_bytes=peak_memory_bytes()
    )
    inputs.verify()
    (out / "replay.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, help="deployment label only; no site defaults")
    parser.add_argument("--environment", required=True, help="explicit CPU environment identity")
    parser.add_argument("--mode", required=True, choices=("analysis", "report", "replay"))
    parser.add_argument(
        "--input", required=True, type=Path, help="config, or saved analysis folder for report"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluation", help="explicit round-two identity for replay")
    parser.add_argument(
        "--check-only", action="store_true", help="replay readiness without fitting"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    if args.mode == "report":
        require(
            args.output.resolve() == args.input.resolve() / "summary.html",
            "report output must be INPUT/summary.html",
        )
        result = str(write_summary(args.input, report_format="analysis"))
    else:
        config = AnalysisConfig.load(args.input)
        require(
            all(p is None for p in config.direct_verdict.values()),
            "CPU batch entry point excludes external human sources",
        )
        if args.mode == "analysis":
            require(not args.check_only, "--check-only is replay-specific")
            result = str(analyze(config, args.output, progress=lambda s: print(s, flush=True)))
        else:
            result = replay(config, args.evaluation, args.output.resolve(), args.check_only)
    print(
        json.dumps(
            dict(
                site=args.site,
                environment=args.environment,
                mode=args.mode,
                elapsed_seconds=time.perf_counter() - started,
                peak_process_memory_bytes=peak_memory_bytes(),
                result=result,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
