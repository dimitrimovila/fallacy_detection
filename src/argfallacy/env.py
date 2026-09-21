"""Reading the ``.env`` at the repository root.

Paths to data outside the repository are configuration, not code: no module
writes one, and this is the one place that turns the file into settings.
``.env`` is git-ignored, so the values never leave the machine that holds the
data.

Written by hand rather than pulled in as a dependency: the project admits a new
one only when it saves more than fifty lines, and the file this reads is one we write
ourselves.  The format is therefore deliberately small — ``KEY=VALUE`` one per
line, ``#`` starts a comment line, blank lines ignored, a matched pair of
surrounding quotes stripped.  No escape sequences, no interpolation, no
multi-line values, no ``export`` prefix.  A line that is none of those raises
rather than being skipped, because a path silently dropped is a test that
silently turns into a skip.
"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import ENV_FILE


class EnvError(ValueError):
    """Raised when the .env file has a line that is not a setting."""


def parse_env(text: str, filename: str = ".env") -> dict[str, str]:
    """Settings of an env file, in the order they appear."""
    settings: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise EnvError(f"{filename}:{number}: not a KEY=VALUE setting: {raw!r}")
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            raise EnvError(f"{filename}:{number}: setting with no name: {raw!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        settings[key] = value
    return settings


def load_env(path: str | Path = ENV_FILE, override: bool = False) -> dict[str, str]:
    """Put the settings of ``path`` into ``os.environ``; return what was applied.

    A missing file is not an error: on a machine that sets the variables some
    other way there is nothing to read, and that is a normal way to run.

    ``override`` is off by default, so a variable already in the environment
    wins over the file.  That is what makes a one-off
    ``PRIOR_RUNS_DIR=... pytest`` still work, and what lets a cluster job
    carry its own paths without editing anything.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    settings = parse_env(path.read_text(encoding="utf-8"), path.name)
    applied = {k: v for k, v in settings.items() if override or k not in os.environ}
    os.environ.update(applied)
    return applied
