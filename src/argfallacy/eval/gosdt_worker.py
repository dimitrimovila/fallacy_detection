"""Standalone worker: a supplied runtime/vendor directory, arrays in, tree out."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("--vendor", required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, args.vendor)
    request_path = Path(args.request)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = request_path.parent / "result.json"
    start = time.perf_counter()
    cgroup, linux = None, None
    try:
        if os.name != "nt":
            linux = load_module("linux_resources", Path(__file__).with_name("gosdt_linux.py"))
            if request.get("backend") != "slurm_cgroup_v2":
                raise RuntimeError("Linux workers require verified Slurm cgroup supervision")
            cgroup = linux.enforcement(request["memory_bytes"])
            write(request_path.parent / "started.json", cgroup)
        # Load the pure adapter without importing the project's main runtime dependencies.
        path = Path(__file__).resolve().parents[1] / "aggregate" / "sparse_tree.py"
        adapter = load_module("sparse_adapter", path)
        import numpy as np

        arrays = np.load(request_path.parent / "input.npz", allow_pickle=False)
        if os.name != "nt":
            import signal

            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.setitimer(signal.ITIMER_REAL, request["seconds"] + request["grace_seconds"])
        write(request_path.parent / "ready.json", {"ready": True})
        result = adapter.fit_gosdt(
            arrays["x"],
            arrays["y"],
            request["depth"],
            request["coefficient"],
            request["seconds"],
            evaluation_x=arrays["evaluation_x"],
        )
        result["runtime"] = {
            "python": sys.version,
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("gosdt", "scikit-learn", "numpy", "scipy")
            },
        }
        if result["runtime"]["packages"]["gosdt"] != "1.0.4":
            raise ValueError("the checked release is gosdt 1.0.4")
    except Exception as error:
        result = {
            "valid": False,
            "status": "worker_error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    if os.name != "nt":
        import resource
        import signal

        signal.setitimer(signal.ITIMER_REAL, 0)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        private = None
        if cgroup:
            result["linux_cgroup"] = linux.memory_report(cgroup)
    else:
        import psutil

        memory = psutil.Process().memory_info()
        peak = getattr(memory, "peak_wset", memory.rss)
        private = getattr(memory, "peak_pagefile", None)
    result.update(
        elapsed_seconds=time.perf_counter() - start,
        peak_working_set_bytes=peak,
        peak_private_bytes=private,
    )
    write(output, result)


if __name__ == "__main__":
    main()
