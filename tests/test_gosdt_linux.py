"""Resource isolation and safety gates, without a scheduler or benchmark fits."""

import importlib.util
from pathlib import Path

import pytest

from argfallacy.eval.gosdt_linux import cgroup_path, enforcement, headroom, step_command


def limit(path, maximum, current):
    path.mkdir(parents=True, exist_ok=True)
    (path / "memory.max").write_text(str(maximum))
    (path / "memory.current").write_text(str(current))


def test_effective_limit_and_headroom_include_ancestors(tmp_path):
    mount = tmp_path / "cgroup"
    limit(mount, 8_000_000_000, 1_000_000_000)
    job = mount / "job"
    limit(job, 7_000_000_000, 6_000_000_000)
    step = job / "step"
    limit(step, 3_999_268_864, 100_000_000)
    task = step / "task"
    limit(task, "max", 90_000_000)
    assert headroom(task, mount) == 1_000_000_000
    record = enforcement(
        4_000_000_000,
        path=task,
        mount=mount,
        environment={"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "7"},
    )
    assert record["hard_limit_bytes"] == 3_999_268_864
    assert record["limit_path"] == str(step)
    with pytest.raises(RuntimeError, match="numbered"):
        enforcement(
            4_000_000_000,
            path=task,
            mount=mount,
            environment={"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "batch"},
        )
    with pytest.raises(RuntimeError, match="no enforced"):
        enforcement(
            4_000_000_000,
            path=job,
            mount=mount,
            environment={"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "7"},
        )


def test_cgroup_resolution_rejects_legacy_and_traversal(tmp_path):
    mount = tmp_path / "mount"
    path = mount / "job/task"
    path.mkdir(parents=True)
    proc = tmp_path / "self_cgroup"
    proc.write_text("0::/job/task\n")
    assert cgroup_path(proc, mount) == path
    proc.write_text("2:memory:/job/task\n")
    with pytest.raises(RuntimeError, match="cgroup-v2"):
        cgroup_path(proc, mount)
    proc.write_text("0::/../other\n")
    with pytest.raises(RuntimeError):
        cgroup_path(proc, mount)


def test_step_never_creates_an_allocation_or_exceeds_fit_memory():
    environment = {"SLURM_JOB_ID": "123", "SLURM_JOB_NUM_NODES": "1", "SLURM_CPUS_ON_NODE": "2"}
    command = step_command("python", "worker", "request", "vendor", 4_000_000_000, environment)
    assert command[0] == "srun" and "--exact" in command
    assert "--cpus-per-task=1" in command and "--nodes=1" in command
    cap = int(next(value for value in command if value.startswith("--mem="))[6:-1]) * 1024**2
    assert cap == 3_999_268_864 and cap <= 4_000_000_000
    with pytest.raises(RuntimeError, match="allocation"):
        step_command("p", "w", "r", "v", 4_000_000_000, {})
    with pytest.raises(RuntimeError, match="two CPUs"):
        step_command("p", "w", "r", "v", 4_000_000_000, {**environment, "SLURM_CPUS_ON_NODE": "1"})


def test_transferred_input_gate_checks_content_and_escape(tmp_path):
    import hashlib

    file = Path("serving/gosdt_cluster.py")
    spec = importlib.util.spec_from_file_location("cluster_gate", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = [f"input{i}" for i in range(8)]
    for name in names:
        (tmp_path / name).write_bytes(b"invented")
    sha = hashlib.sha256(b"invented").hexdigest()
    checksum = tmp_path / "input_checksums.sha256"
    checksum.write_text("".join(f"{sha}  {name}\n" for name in names))
    reference = tmp_path / "expected.sha256"
    reference.write_bytes(checksum.read_bytes())
    module.verify_inputs(tmp_path, reference)
    (tmp_path / names[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        module.verify_inputs(tmp_path, reference)
    checksum.write_text("".join(f"{sha}  ../{name}\n" for name in names))
    with pytest.raises(ValueError, match="frozen reference"):
        module.verify_inputs(tmp_path, reference)
    reference.write_bytes(checksum.read_bytes())
    with pytest.raises(ValueError, match="escaped"):
        module.verify_inputs(tmp_path, reference)


def test_fast_worker_oom_preserves_step_proof_even_before_first_poll(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import numpy as np

    from argfallacy.eval import gosdt_linux, gosdt_process

    monkeypatch.setattr(
        gosdt_process, "os", SimpleNamespace(name="posix", environ={"SLURM_JOB_ID": "123"})
    )
    monkeypatch.setattr(gosdt_linux, "step_command", lambda *args: ["invented_srun"])
    folder = tmp_path / "probe"

    class Child:
        pid = 42

        def __init__(self, *args, **kwargs):
            assert kwargs["start_new_session"]
            kwargs["stdout"].write("Detected oom-kill event\n")
            (folder / "started.json").write_text(
                json.dumps({"pid": 43, "job": "123", "step": "7", "hard_limit_bytes": 127_926_272})
            )

        def poll(self):
            return 137

        def wait(self):
            return 137

    monkeypatch.setattr(gosdt_process.subprocess, "Popen", Child)
    process = gosdt_process.SolverProcess("unused", "unused", tmp_path, memory=128_000_000)
    result = process(
        "probe", np.zeros((2, 1), bool), np.array(["a", "b"]), 2, 2, np.zeros((1, 1), bool)
    )
    assert not result["valid"]
    report = json.loads((folder / "supervision.json").read_text())
    assert report["exit_code"] == 137 and report["worker_step"]["step"] == "7"
