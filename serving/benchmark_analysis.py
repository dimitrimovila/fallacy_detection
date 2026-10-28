"""Profile the complete saved-result pipeline and report in a fresh CPU process."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import yaml

# Limit BLAS oversubscription before importing NumPy through the package.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from synthetic_analysis import build  # noqa: E402

from argfallacy.eval.analysis import AnalysisConfig, analyze, peak_memory_bytes  # noqa: E402
from argfallacy.eval.html_report import write_summary  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True, help="new synthetic directory")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--compare", type=Path, help="previous benchmark output for invariance")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.compare:
        old = json.loads((args.compare / "benchmark.json").read_text(encoding="utf-8"))
        workload = old["workload"]
        assert workload["items"] == 601 and workload["resamples"] == 10000
        # Compare the same frozen invented exports, independent of generator changes.
        shutil.copytree(
            args.compare, args.workdir, ignore=shutil.ignore_patterns("out", "benchmark.json")
        )
        path = args.workdir / "analysis.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["bootstrap_batch_size"] = args.batch_size
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    else:
        path, workload = build(args.workdir, args.batch_size)
    out = analyze(AnalysisConfig.load(path), progress=lambda step: print(step, flush=True))
    analysis_done = time.perf_counter()
    write_summary(out, report_format="analysis")
    done = time.perf_counter()
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.csv")}
    if args.compare:
        # Input source hashes/keys are provenance, not numerical estimates.
        import pandas as pd

        for name in hashes:
            a, b = (
                pd.read_csv(folder / name, keep_default_na=False)
                for folder in (out, args.compare / "out")
            )
            columns = [c for c in a if c not in ("source_sha256", "source_key")]
            pd.testing.assert_frame_equal(a[columns], b[columns], check_exact=True)
        workload["batch_invariance"] = dict(
            compared_to=str(args.compare), passed=True, previous_batch=old["batch_size"]
        )
    record = dict(
        workload=workload,
        timing_scope="synthetic preparation, saved-result analysis and report; imports excluded",
        memory_scope="peak process working set on Windows; process lifetime including imports",
        batch_size=args.batch_size,
        setup_and_analysis_seconds=analysis_done - started,
        report_seconds=done - analysis_done,
        total_seconds=done - started,
        peak_process_memory_bytes=peak_memory_bytes(),
        table_hashes=hashes,
    )
    (args.workdir / "benchmark.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in record.items() if k != "table_hashes"}, indent=2), flush=True
    )


if __name__ == "__main__":
    main()
