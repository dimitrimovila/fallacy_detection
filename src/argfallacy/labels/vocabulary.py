"""The two label dictionaries, and normalization of raw scheme labels.

Fallacy labels are handled by :mod:`argfallacy.schemes.loader`, which already
owns ``normalize_label``; this module adds the stage-one counterpart for scheme
labels and a single place to read ``labels/schemes.yaml``.

No label string is ever written in code.  Both dictionaries are data files, and
an unknown string fails loudly rather than being guessed at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..paths import SCHEME_LABELS_FILE
from ..schemes.loader import SchemeError


def light_normalize(raw: object) -> str:
    """Lower-case and squeeze runs of whitespace.  The only preprocessing there is."""
    return " ".join(str(raw).strip().lower().split())


def load_scheme_vocabulary(path: str | Path = SCHEME_LABELS_FILE) -> dict[str, Any]:
    """Read ``labels/schemes.yaml`` and check that its aliases are unambiguous."""
    vocab = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _check_scheme_vocabulary(vocab, Path(path).name)
    return vocab


def _check_scheme_vocabulary(vocab: dict[str, Any], filename: str) -> None:
    problems: list[str] = []
    seen: dict[str, str] = {}
    for sid, spec in vocab["schemes"].items():
        for raw in [sid, *spec.get("aliases", [])]:
            key = light_normalize(raw)
            if key in seen and seen[key] != sid:
                problems.append(f"alias {key!r} points at both {seen[key]!r} and {sid!r}")
            seen[key] = sid
    if problems:
        raise SchemeError(f"{filename}:\n  - " + "\n  - ".join(problems))


def scheme_alias_table(vocabulary: dict[str, Any] | None = None) -> dict[str, str]:
    """Light-normalized raw string -> scheme id, for every id and every alias."""
    vocab = vocabulary or load_scheme_vocabulary()
    table: dict[str, str] = {}
    for sid, spec in vocab["schemes"].items():
        table[light_normalize(sid)] = sid
        for alias in spec.get("aliases", []):
            table[light_normalize(alias)] = sid
    return table


def normalize_scheme_label(raw_label: str, vocabulary: dict[str, Any] | None = None) -> str:
    """Map a raw stage-one label onto a scheme id, or ``none``.

    Same shape as ``normalize_label`` for fallacies: light normalization, the
    alias table, then one deterministic rule (spaces to underscores).  Nothing
    approximate, and no silent default — an unknown string raises.
    """
    vocab = vocabulary or load_scheme_vocabulary()
    table = scheme_alias_table(vocab)
    key = light_normalize(raw_label)

    if key in table:
        return table[key]
    snake = key.replace(" ", "_")
    if snake in table:
        return table[snake]
    raise SchemeError(f"scheme label {raw_label!r} is not in the vocabulary")


def scheme_ids(vocabulary: dict[str, Any] | None = None) -> list[str]:
    """The eight scheme ids plus ``none``, sorted."""
    vocab = vocabulary or load_scheme_vocabulary()
    return sorted(vocab["schemes"])
