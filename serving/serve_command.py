#!/usr/bin/env python3
"""The part of the ``vllm serve`` command that comes from ``serving/models.yaml``.

One entry in, one line out: the served repository, its pinned revision and the
entry's ``serve_args``, plus ``--max-model-len 8192`` when ``serve_args`` does not
set one.  The caller adds what depends on the node: port, host, tensor
parallelism, dtype.

    python serving/serve_command.py qwen3_8_27b_think
    Qwen/Qwen3.8-27B --revision 1d4bf0f2... --reasoning-parser qwen3 --max-model-len 16384

An entry whose revision is still a placeholder, or that is not served with vLLM,
prints nothing on the standard output and exits with a code other than zero, so
that a job reading the line has nothing to launch.

It reads the YAML file with PyYAML alone, without the package of the repository,
so it runs in the environment of the server as well as in that of the client.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import yaml

MODELS_FILE = Path(__file__).resolve().parent / "models.yaml"
PLACEHOLDER = "PLACEHOLDER"
MAX_MODEL_LEN = "--max-model-len"
DEFAULT_MAX_MODEL_LEN = "8192"


class Refusal(ValueError):
    """The entry cannot be served as it stands."""


def command(entry: str, models_file: Path = MODELS_FILE) -> list[str]:
    """The arguments of ``vllm serve`` that the entry decides, in order."""
    models = yaml.safe_load(models_file.read_text(encoding="utf-8"))["models"]
    if entry not in models:
        raise Refusal(f"unknown entry {entry!r}; serving/models.yaml has {sorted(models)}")
    spec = models[entry]
    revision = str(spec.get("revision") or "")
    if not revision or PLACEHOLDER in revision:
        raise Refusal(
            f"{entry}: no pinned revision in serving/models.yaml ({revision or 'empty'}); "
            f"pin the Hugging Face commit before serving it"
        )
    if "serve_args" not in spec:
        raise Refusal(f"{entry}: no serve_args in serving/models.yaml: not served with vLLM")
    args = [str(spec["model_id"]), "--revision", revision]
    args += [str(arg) for arg in spec["serve_args"] or []]
    if MAX_MODEL_LEN not in args:
        args += [MAX_MODEL_LEN, DEFAULT_MAX_MODEL_LEN]
    return args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("entry", help="a key of serving/models.yaml")
    args = parser.parse_args(argv)
    try:
        line = shlex.join(command(args.entry))
    except Refusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
