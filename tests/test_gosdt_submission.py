"""Submission ordering with fake scheduler commands; never submit a real job."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", ["none", "admission", "experiment"])
def test_submission_checks_admission_and_keeps_experiment_conditional(tmp_path, failure):
    bash = shutil.which("bash")
    if not bash:
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if not candidate.exists():
            pytest.skip("Bash is not installed")
        bash = str(candidate)
    serving = tmp_path / "serving"
    serving.mkdir()
    script = serving / "submit.sh"
    script.write_bytes(Path("serving/gosdt_cluster_submit.sh").read_bytes())
    python = tmp_path / ".venv-gosdt/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    (tmp_path / "scratch/gosdt-worker-vendor/gosdt").mkdir(parents=True)
    binaries = tmp_path / "fakes"
    binaries.mkdir()
    sbatch = binaries / "sbatch"
    sbatch.write_bytes(b"""#!/bin/bash
printf '%s\n' "$*" >> "$GOSDT_TEST_CALLS"
if [[ "$*" == *--test-only* ]]; then
    [[ "$GOSDT_TEST_FAILURE" != admission ]]
    exit $?
fi
if [[ "${@: -1}" == check ]]; then
    echo 101
elif [[ "$GOSDT_TEST_FAILURE" == experiment ]]; then
    exit 1
else
    echo 102
fi
""")
    sbatch.chmod(0o755)
    cancel = binaries / "scancel"
    cancel.write_bytes(b'#!/bin/bash\nprintf "cancel %s\\n" "$*" >> "$GOSDT_TEST_CALLS"\n')
    cancel.chmod(0o755)
    log = tmp_path / "calls.log"
    result = subprocess.run(
        [
            bash,
            "-c",
            'export PATH="/usr/bin:/bin:$PATH"; test_prefix=$1; '
            'if command -v cygpath >/dev/null; then test_prefix=$(cygpath -u "$1"); fi; '
            'export PATH="$test_prefix:$PATH"; "$BASH" "$2"',
            "test",
            binaries.as_posix(),
            script.as_posix(),
        ],
        env={**os.environ, "GOSDT_TEST_CALLS": log.as_posix(), "GOSDT_TEST_FAILURE": failure},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert log.exists(), result.stdout + result.stderr
    calls = log.read_text().splitlines()
    if failure == "admission":
        assert result.returncode != 0
        assert len(calls) == 1 and "--test-only" in calls[0]
    else:
        assert "--test-only" in calls[0] and "--test-only" in calls[1]
        assert calls[2].endswith("check") and "--parsable" in calls[2]
        assert "--dependency=afterok:101" in calls[3]
        assert "--kill-on-invalid-dep=yes" in calls[3]
        if failure == "experiment":
            assert result.returncode != 0 and calls[4] == "cancel 101"
        else:
            assert result.returncode == 0 and len(calls) == 4
