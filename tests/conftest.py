"""Shared fixtures.

``tests/`` goes on ``sys.path`` so the tests can import ``fakes.llm`` and the
scheme snapshot in ``validate_schemes.py``, which both live here.

The ``.env`` at the repository root is read here, once, before any fixture looks
at the environment: it is the one place the paths to data outside the repository
are configured.  It does not override a variable already set, so a one-off
``PRIOR_RUNS_DIR=... pytest`` still wins over the file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# tests/fakes/ holds the stand-in backend, imported as `fakes.llm`
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from argfallacy.env import load_env  # noqa: E402
from argfallacy.labels import load_scheme_vocabulary  # noqa: E402
from argfallacy.schemes import load_all, load_vocabulary  # noqa: E402

load_env()


@pytest.fixture(scope="session")
def schemes():
    return load_all()


@pytest.fixture(scope="session")
def vocab():
    return load_vocabulary()


@pytest.fixture(scope="session")
def scheme_vocab():
    return load_scheme_vocabulary()


@pytest.fixture(scope="session")
def prior_dir() -> Path:
    """The folder of the prior runs, or skip.  Never a path written in the code."""
    raw = os.environ.get("PRIOR_RUNS_DIR")
    if not raw:
        pytest.skip("PRIOR_RUNS_DIR is set neither in the environment nor in .env")
    path = Path(raw)
    if not path.is_dir():
        pytest.skip(f"PRIOR_RUNS_DIR points at {path}, which is not a directory")
    return path


@pytest.fixture(scope="session")
def prior_runs(prior_dir: Path) -> list[Path]:
    """Every folder of the prior runs that holds both a results.csv and a metrics.csv."""
    runs = sorted(p.parent for p in prior_dir.rglob("results.csv")
                  if (p.parent / "metrics.csv").exists())
    if not runs:
        pytest.skip(f"no results.csv + metrics.csv pair under {prior_dir}")
    return runs
