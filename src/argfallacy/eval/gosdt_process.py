"""Sequential solver supervision; never install dependencies or launch cluster jobs."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from .b3_diagnostic import digest, json_write


def available_memory():
    if os.name == "nt":

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError("cannot read available physical memory")
        return memory.available
    import psutil

    physical = psutil.virtual_memory().available
    from .gosdt_linux import headroom

    try:
        remaining = headroom()
    except (OSError, RuntimeError):
        if os.environ.get("SLURM_JOB_ID"):
            raise
        remaining = None
    return min(physical, remaining) if remaining is not None else physical


def require_memory(minimum=3 * 1024**3):
    free = available_memory()
    if free < minimum:
        raise MemoryError(
            f"available RAM {free / 1024**3:.2f} GiB; "
            f"local preflight requires {minimum / 1024**3:.2f} GiB"
        )
    return free


def process_memory(pid):
    if os.name != "nt":
        import psutil

        try:
            process = psutil.Process(pid)
            rss = process.memory_info().rss
            private = process.memory_full_info().uss
            status = Path(f"/proc/{pid}/status").read_text()
            peak = next(
                (
                    int(line.split()[1]) * 1024
                    for line in status.splitlines()
                    if line.startswith("VmHWM:")
                ),
                rss,
            )
            return {"rss": rss, "private": private, "peak_rss": peak}
        except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError):
            return None
    from ctypes import wintypes as w

    class Counters(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "rss",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
                "private",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    read = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    read.argtypes = [w.HANDLE, ctypes.POINTER(Counters), w.DWORD]
    read.restype = w.BOOL
    handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    try:
        if not handle or not read(handle, ctypes.byref(counters), counters.cb):
            return None
        return {"rss": counters.rss, "private": counters.private, "peak_rss": counters.peak}
    finally:
        if handle:
            kernel.CloseHandle(handle)


def windows_job(pid, budget):
    """Set a hard committed-memory cap; caller keeps the handle alive."""
    from ctypes import wintypes as w

    class Basic(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_longlong),
            ("job_time", ctypes.c_longlong),
            ("flags", w.DWORD),
            ("min_ws", ctypes.c_size_t),
            ("max_ws", ctypes.c_size_t),
            ("active", w.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", w.DWORD),
            ("scheduling", w.DWORD),
        ]

    class IO(ctypes.Structure):
        _fields_ = [(str(i), ctypes.c_ulonglong) for i in range(6)]

    class Limits(ctypes.Structure):
        _fields_ = [
            ("basic", Basic),
            ("io", IO),
            ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t),
            ("peak_process", ctypes.c_size_t),
            ("peak_job", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    process = kernel.OpenProcess(0x0100 | 0x0001, False, pid)
    limits = Limits()
    limits.basic.flags = 0x100 | 0x2000
    limits.process_memory = budget
    ok = (
        handle
        and process
        and kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
        and kernel.AssignProcessToJobObject(handle, process)
    )
    if process:
        kernel.CloseHandle(process)
    if not ok:
        if handle:
            kernel.CloseHandle(handle)
        raise OSError(f"memory job setup failed: Windows error {ctypes.get_last_error()}")
    return lambda: kernel.CloseHandle(handle)


class SolverProcess:
    def __init__(
        self,
        python,
        vendor,
        folder,
        *,
        seconds=60,
        memory=4_000_000_000,
        grace=5,
    ):
        self.python, self.vendor, self.folder = Path(python), Path(vendor), Path(folder)
        self.seconds, self.memory, self.grace = seconds, memory, grace

    def __call__(self, key, x, y, depth, coefficient, evaluation_x):
        folder = self.folder / key
        folder.mkdir(parents=True, exist_ok=True)
        arrays = folder / "input.npz"
        request = {
            "depth": depth,
            "coefficient": coefficient,
            "seconds": self.seconds,
            "memory_bytes": self.memory,
            "grace_seconds": self.grace,
            "worker_sha256": digest(Path(__file__).with_name("gosdt_worker.py")),
            "adapter_sha256": digest(Path(__file__).parents[1] / "aggregate/sparse_tree.py"),
        }
        if os.name != "nt":
            request["backend"] = "slurm_cgroup_v2"
            request["supervisor_sha256"] = digest(Path(__file__).with_name("gosdt_linux.py"))
        if arrays.exists():
            stored_hash = json.loads((folder / "input_hash.json").read_text(encoding="utf-8"))
            if stored_hash["sha256"] != digest(arrays):
                raise ValueError("checkpoint input hash differs")
            saved = np.load(arrays, allow_pickle=False)
            if not np.array_equal(saved["x"], x) or not np.array_equal(saved["y"], y):
                raise ValueError("checkpoint arrays differ from the current partition")
            if not np.array_equal(saved["evaluation_x"], evaluation_x):
                raise ValueError("checkpoint evaluation arrays differ")
            if json.loads((folder / "request.json").read_text(encoding="utf-8")) != request:
                raise ValueError("checkpoint solver configuration differs")
            if (folder / "result.json").exists() and (folder / "supervision.json").exists():
                record = json.loads((folder / "supervision.json").read_text(encoding="utf-8"))
                if record["result_sha256"] != digest(folder / "result.json"):
                    raise ValueError("checkpoint result hash differs")
                return json.loads((folder / "result.json").read_text(encoding="utf-8"))
            raise ValueError("unfinished solver checkpoint; preserve it and review before retry")
        np.savez_compressed(
            arrays,
            x=np.asarray(x, bool),
            y=np.asarray(y, str),
            evaluation_x=np.asarray(evaluation_x, bool),
        )
        json_write(folder / "request.json", request)
        json_write(folder / "input_hash.json", {"sha256": digest(arrays)})
        worker = Path(__file__).with_name("gosdt_worker.py")
        start = time.monotonic()
        ready_at, reason, release = None, None, None
        peak_rss, peak_private = 0, 0
        environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = "1"
        command = [
            str(self.python),
            str(worker),
            str(folder / "request.json"),
            "--vendor",
            str(self.vendor),
        ]
        if os.name != "nt":
            from .gosdt_linux import step_command

            command = step_command(
                self.python, worker, folder / "request.json", self.vendor, self.memory
            )
        identity = None

        def terminate():
            # Cancel only the worker's numbered step, never the enclosing allocation.
            if os.name != "nt" and identity:
                subprocess.run(
                    ["scancel", "--signal=KILL", f"{identity['job']}.{identity['step']}"],
                    check=False,
                    timeout=5,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            if child.poll() is None:
                if os.name != "nt":
                    os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.kill()

        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=os.name != "nt",
            )
            try:
                if os.name == "nt":
                    release = windows_job(child.pid, self.memory)
                while child.poll() is None:
                    if identity is None and (folder / "started.json").exists():
                        candidate = json.loads(
                            (folder / "started.json").read_text(encoding="utf-8")
                        )
                        if (
                            candidate["job"] != environment.get("SLURM_JOB_ID")
                            or not candidate["step"].isdigit()
                        ):
                            raise RuntimeError("worker step identity differs from the allocation")
                        identity = candidate
                    usage = process_memory(identity["pid"] if identity else child.pid)
                    if usage:
                        peak_rss = max(peak_rss, usage["peak_rss"])
                        peak_private = max(peak_private, usage["private"])
                        if usage["rss"] > self.memory:
                            reason = "worker_rss_limit"
                            terminate()
                    if ready_at is None and (folder / "ready.json").exists():
                        ready_at = time.monotonic()
                    elapsed = time.monotonic() - (ready_at or start)
                    if (ready_at is None and elapsed > 60) or (
                        ready_at is not None and elapsed > self.seconds + self.grace
                    ):
                        reason = "worker_wall_timeout"
                        terminate()
                    if available_memory() < 1024**3:
                        reason = "system_low_memory"
                        terminate()
                    if reason:
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                    time.sleep(0.05)
                code = child.wait()
            except BaseException:
                try:
                    terminate()
                finally:
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                raise
            finally:
                if release:
                    release()
        if identity is None and (folder / "started.json").exists():
            candidate = json.loads((folder / "started.json").read_text(encoding="utf-8"))
            if (
                candidate["job"] != environment.get("SLURM_JOB_ID")
                or not candidate["step"].isdigit()
            ):
                raise RuntimeError("worker step identity differs from the allocation")
            identity = candidate
        if not (folder / "result.json").exists():
            json_write(
                folder / "result.json",
                {
                    "valid": False,
                    "status": "process_failed",
                    "error": reason or f"worker exit {code}",
                },
            )
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if reason:
            result.update(valid=False, error=reason)
        if code and result.get("valid"):
            result.update(valid=False, error=f"worker exit {code}")
        json_write(folder / "result.json", result)
        json_write(
            folder / "supervision.json",
            {
                "exit_code": code,
                "reason": reason,
                "elapsed_seconds": time.monotonic() - start,
                "private_memory_hard_cap": self.memory if os.name == "nt" else None,
                "backend": "windows_job" if os.name == "nt" else "slurm_cgroup_v2",
                "worker_step": identity,
                "system_free_reserve": 1024**3,
                "result_sha256": digest(folder / "result.json"),
                "observed_peak_rss_bytes": peak_rss,
                "observed_peak_private_bytes": peak_private,
            },
        )
        return result
