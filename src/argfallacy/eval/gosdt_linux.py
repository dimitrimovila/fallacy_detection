"""Linux cgroup-v2 inspection, shared by the standalone worker and its supervisor."""

from __future__ import annotations

import os
from pathlib import Path


def cgroup_path(proc=Path("/proc/self/cgroup"), mount=Path("/sys/fs/cgroup")):
    entries = [line[3:] for line in proc.read_text().splitlines() if line.startswith("0::/")]
    if len(entries) != 1 or ".." in Path(entries[0]).parts:
        raise RuntimeError("a readable cgroup-v2 hierarchy is required")
    path = mount / entries[0].lstrip("/")
    if not path.is_dir():
        raise RuntimeError("process cgroup is not visible at /sys/fs/cgroup")
    return path


def hierarchy(path, mount=Path("/sys/fs/cgroup")):
    if not path.is_relative_to(mount):
        raise RuntimeError("cgroup path escaped its mount")
    for directory in (path, *path.parents):
        if not directory.is_relative_to(mount):
            break
        maximum = directory / "memory.max"
        if maximum.exists():
            raw = maximum.read_text().strip()
            if raw != "max":
                yield directory, int(raw), int((directory / "memory.current").read_text())


def headroom(path=None, mount=Path("/sys/fs/cgroup")):
    path = path or cgroup_path(mount=mount)
    return min(
        (max(0, limit - current) for _, limit, current in hierarchy(path, mount)), default=None
    )


def step_command(python, worker, request, vendor, budget, environment=None):
    environment = os.environ if environment is None else environment
    job = environment.get("SLURM_JOB_ID", "")
    if not job.isdigit() or int(environment.get("SLURM_JOB_NUM_NODES", "0")) != 1:
        raise RuntimeError("Linux GOSDT requires a single-node Slurm allocation")
    if int(environment.get("SLURM_CPUS_ON_NODE", "0")) < 2:
        raise RuntimeError("reserve two CPUs for the driver and sequential worker")
    mib = budget // 1024**2
    if mib <= 0:
        raise ValueError("invalid worker memory budget")
    return [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--exact",
        "--exclusive",
        "--mpi=none",
        "--immediate=30",
        "--kill-on-bad-exit=1",
        f"--mem={mib}M",
        str(python),
        str(worker),
        str(request),
        "--vendor",
        str(vendor),
    ]


def enforcement(budget, *, path=None, mount=Path("/sys/fs/cgroup"), environment=None):
    environment = os.environ if environment is None else environment
    job, step = environment.get("SLURM_JOB_ID", ""), environment.get("SLURM_STEP_ID", "")
    if not job.isdigit() or not step.isdigit():
        raise RuntimeError("native worker must run inside a numbered Slurm job step")
    path = path or cgroup_path(mount=mount)
    limits = list(hierarchy(path, mount))
    candidates = [(p, cap, current) for p, cap, current in limits if 0 < cap <= budget]
    if not candidates:
        raise RuntimeError("no enforced cgroup memory.max at or below the worker budget")
    owner, cap, _ = min(candidates, key=lambda entry: entry[1])
    if cap != (budget // 1024**2) * 1024**2:
        raise RuntimeError("cgroup hard limit differs from the MiB-rounded request")
    return {
        "job": job,
        "step": step,
        "pid": os.getpid(),
        "path": str(path),
        "limit_path": str(owner),
        "hard_limit_bytes": cap,
        "requested_budget_bytes": budget,
        "accounting": "cgroup charged RAM including page cache; not Windows private commit",
    }


def memory_report(record):
    path = Path(record["limit_path"])
    report = dict(record)
    for name in ("memory.current", "memory.peak", "memory.swap.max", "memory.events"):
        file = path / name
        report[name] = file.read_text().strip() if file.exists() else None
    return report
