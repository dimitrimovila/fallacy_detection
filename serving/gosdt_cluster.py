"""Check a CPU allocation on invented inputs, then gate the frozen experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from argfallacy.aggregate.sparse_tree import GRID
from argfallacy.eval.b3_diagnostic import json_write
from argfallacy.eval.gosdt_diagnostic import _snapshot, configuration, preflight, run
from argfallacy.eval.gosdt_process import SolverProcess
from argfallacy.paths import REPO_ROOT


def verify_inputs(root, reference=None):
    reference = reference or REPO_ROOT / "serving/gosdt_input_checksums.sha256"
    manifest = (root / "input_checksums.sha256").read_bytes()
    if manifest != reference.read_bytes():
        raise ValueError("transferred input manifest differs from the frozen reference")
    entries = manifest.decode().splitlines()
    if len(entries) != 8:
        raise ValueError("expected exactly eight archived input/reference files")
    for line in entries:
        expected, name = line.split("  ", 1)
        file = (root / name).resolve()
        if not file.is_relative_to(root.resolve()):
            raise ValueError("input checksum escaped the experiment checkout")
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"transferred input hash differs: {name}")


def synthetic_grid(python, vendor, folder):
    rng = np.random.default_rng(20261019)
    x = rng.integers(0, 2, size=(96, 45), dtype=np.int8).astype(bool)
    y = np.array(["a", "b", "c", "d"])[2 * x[:, 0].astype(int) + x[:, 1]]
    noise = rng.random(len(y)) < 0.10
    y[noise] = rng.choice(["a", "b", "c", "d"], noise.sum())
    process = SolverProcess(python, vendor, folder)
    records = []
    for inner, (train, validation) in enumerate(
        KFold(3, shuffle=True, random_state=20261019).split(x)
    ):
        for depth, coefficient in GRID:
            result = process(
                f"inner{inner}_d{depth}_a{coefficient}",
                x[train],
                y[train],
                depth,
                coefficient,
                x[validation],
            )
            records.append(
                {"inner": inner, "depth": depth, "coefficient": coefficient, "result": result}
            )
            json_write(folder / "synthetic_grid.json", records)
            if not result.get("valid") or not result.get("evaluation_replay_verified"):
                raise RuntimeError(
                    "a synthetic candidate failed; no benchmark fit is authorized by this check"
                )
    return records


def timeout_check(python, vendor, folder):
    rng = np.random.default_rng(20261019)
    x = rng.integers(0, 2, size=(256, 60), dtype=np.int8).astype(bool)
    y = rng.choice(["a", "b", "c", "d"], len(x))
    process = SolverProcess(python, vendor, folder, seconds=1)
    result = process("short_deadline", x, y, 4, 2, x[:9])
    status = (result.get("native") or {}).get("status")
    if status not in {"Status.TIMEOUT", "Status.CONVERGED"}:
        raise RuntimeError("short native deadline returned no interpretable solver status")
    if status == "Status.CONVERGED" and not result.get("valid"):
        raise RuntimeError("short converged solution failed validation")
    # An empty TIMEOUT incumbent is an explicitly observed failure, never a valid fit.
    return {
        "native_status": status,
        "valid_incumbent": result.get("valid", False),
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "run", "resume"))
    parser.add_argument("--vendor", required=True)
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or os.name == "nt":
        raise RuntimeError("submit the CPU batch script; do not fit on a login node")
    os.environ["ARGFALLACY_RUNS_DIR"] = str(REPO_ROOT / "runs")
    verify_inputs(REPO_ROOT)
    config_path = REPO_ROOT / "configs/gosdt_schemes2_diagnostic.yaml"
    config = configuration(config_path)
    vendor = Path(args.vendor).resolve()
    snapshot = _snapshot(config_path, config, sys.executable, vendor)
    receipt = REPO_ROOT / "scratch/gosdt_cluster_preflight/approved.json"
    if args.mode == "check":
        folder = receipt.parent / os.environ["SLURM_JOB_ID"]
        report = preflight(config, sys.executable, vendor, folder / "runtime")
        if not report["ready"]:
            raise RuntimeError(f"resource/runtime check failed: {report}")
        cap = report["runtime"]["linux_cgroup"]
        timeout = timeout_check(sys.executable, vendor, folder / "deadline")
        grid = synthetic_grid(sys.executable, vendor, folder / "grid")
        json_write(
            receipt,
            {
                "snapshot": snapshot,
                "job": os.environ["SLURM_JOB_ID"],
                "scope": "invented data and file hashes only; no benchmark fitting",
                "grid_fits": len(grid),
                "memory_check": cap,
                "short_deadline": timeout,
            },
        )
        print(
            "PREFLIGHT PASSED: 18 invented-data fits, deadline and effective cgroup checks",
            flush=True,
        )
    else:
        if not receipt.exists() or json.loads(receipt.read_text())["snapshot"] != snapshot:
            raise RuntimeError(
                "successful preflight with identical inputs/code/runtime is required"
            )
        output = run(config_path, sys.executable, vendor, resume=args.mode == "resume")
        print(output / "summary.md", flush=True)


if __name__ == "__main__":
    main()
