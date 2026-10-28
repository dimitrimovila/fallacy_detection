"""Audit explicitly selected saved sources and protected hashes without new analyses."""

import argparse
import json
from pathlib import Path

from argfallacy.eval.analysis import AnalysisConfig, code_state
from argfallacy.eval.analysis_dq import integrate_dq
from argfallacy.eval.analysis_human import read_frozen_annotations
from argfallacy.eval.analysis_inputs import (
    Inputs,
    read_evaluation,
    read_items,
    read_run,
    recover_confidence,
    require,
    scheme_registry,
)
from argfallacy.eval.analysis_selective import difficulty_ratings


def audit(config):
    require(
        all(v is None for v in config.direct_verdict.values()),
        "preflight excludes new human answers and samples",
    )
    inputs = Inputs()
    inputs.add(config.path)
    items = read_items(config, inputs)
    registry = scheme_registry(inputs)
    for file in sorted(Path(__file__).parent.parent.joinpath('schemes').rglob('*.yaml')):
        inputs.add(file)
    if config.annotations:
        notes = read_frozen_annotations(config, inputs)
        difficulty_ratings(notes, items)
    for source in config.evaluations.values():
        for file in sorted(source["path"].glob("*.csv")):
            inputs.add(file)
        inputs.add(source["path"] / "manifest.json")
    for path in config.parsed_runs.values():
        for filename in ("manifest.json", "summary.csv", "answers.csv", "ssoft.csv"):
            if (path / filename).is_file():
                inputs.add(path / filename)
    for file in sorted(Path(__file__).parent.parent.joinpath("src").rglob("*.py")):
        inputs.add(file)
    evaluations = [
        read_evaluation(name, source, config, items, registry, inputs)
        for name, source in config.evaluations.items()
    ]
    _, dq = integrate_dq(config, evaluations, items, registry, inputs)
    issues, runs = [], {}
    for run_id, directory in config.parsed_runs.items():
        try:
            manifest, version, summary = read_run(directory, run_id, items, registry, inputs)
            recover_confidence(evaluations, run_id, manifest, version, summary)
            runs[run_id] = dict(
                scheme_version=version,
                rows=len(summary),
                columns=summary.columns.tolist(),
                source_commit=manifest.get("commit") or "unavailable in source",
            )
        except (ValueError, RuntimeError) as error:
            issues.append(dict(run=run_id, reason=str(error)))
    inputs.verify()
    return dict(
        configuration=str(config.path),
        configuration_values=config.path.read_text(),
        analysis_code=code_state(),
        items=len(items),
        seed=config.seed,
        resamples=config.resamples,
        batch_size=config.bootstrap_batch_size,
        sources={
            e.name: dict(
                path=str(e.path),
                round=e.round,
                settings=e.manifest,
                systems=len(e.systems),
                source_commit_status="recorded"
                if e.manifest.get("commit")
                else "unavailable in source",
            )
            for e in evaluations
        },
        dq=dq,
        parsed_runs=runs,
        issues=issues,
        protected_files=inputs.records(),
        scope="saved source contract and agreement validation only; no new analysis results",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument(
        "--verify", type=Path, help="earlier audit whose protected hashes must match"
    )
    args = parser.parse_args()
    config = AnalysisConfig.load(args.config)
    config.guard_output(args.audit.resolve())
    result = audit(config)
    if args.verify:
        previous = json.loads(args.verify.read_text(encoding="utf-8"))
        now = {r["path"]: r["sha256"] for r in result["protected_files"]}
        changes = [
            r["path"] for r in previous["protected_files"] if now.get(r["path"]) != r["sha256"]
        ]
        require(not changes, f"protected source hashes changed: {changes}")
        result["verified_against"] = str(args.verify)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    require(not args.audit.exists(), "audit output already exists")
    args.audit.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                audit=str(args.audit),
                issues=result["issues"],
                protected_files=len(result["protected_files"]),
                dq=result["dq"]["status"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
