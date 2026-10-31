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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("--vendor", required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, args.vendor)
    # Load the pure adapter without importing the project's main runtime dependencies.
    path = Path(__file__).resolve().parents[1] / "aggregate" / "sparse_tree.py"
    spec = importlib.util.spec_from_file_location("sparse_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    import numpy as np
    import psutil

    request_path = Path(args.request)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = request_path.parent / "result.json"
    start = time.perf_counter()
    (request_path.parent / "ready.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    try:
        arrays = np.load(request_path.parent / "input.npz", allow_pickle=False)
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
    memory = psutil.Process().memory_info()
    result.update(
        elapsed_seconds=time.perf_counter() - start,
        peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss),
        peak_private_bytes=getattr(memory, "peak_pagefile", None),
    )
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    temp.replace(output)


if __name__ == "__main__":
    main()
