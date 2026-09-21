"""Where the versioned data files live.

The YAML files are data of the thesis, not package resources: they sit at the
repository root (``schemes/``, ``labels/``) so that a diff on them is readable
as a change to the encoding rather than to the code.  One module resolves them
once, so no other module has to count ``parent`` hops.
"""

from __future__ import annotations

from pathlib import Path

# src/argfallacy/paths.py -> parents[0]=argfallacy, [1]=src, [2]=repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMES_DIR = REPO_ROOT / "schemes"
LABELS_DIR = REPO_ROOT / "labels"

FALLACIES_FILE = LABELS_DIR / "fallacies.yaml"
SCHEME_LABELS_FILE = LABELS_DIR / "schemes.yaml"

ENV_FILE = REPO_ROOT / ".env"
"""Where the paths to data outside the repository live.  Git ignores it."""

__all__ = [
    "REPO_ROOT",
    "SCHEMES_DIR",
    "LABELS_DIR",
    "FALLACIES_FILE",
    "SCHEME_LABELS_FILE",
    "ENV_FILE",
]
