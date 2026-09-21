"""Argumentation schemes as executable data.

The diagrams live in ``schemes/*.yaml`` at the repository root; ``loader``
parses and validates them, ``interpreter`` executes them.
"""
from .interpreter import Traversal, enumerate_paths, propagate, traverse
from .loader import (
                          SCHEMES_DIR,
                          CriticalQuestion,
                          Edge,
                          Label,
                          Resolution,
                          Scheme,
                          SchemeError,
                          check_vocabulary_against_schemes,
                          family_members,
                          label_matches,
                          load_all,
                          load_scheme,
                          load_vocabulary,
                          normalize_label,
                          resolve_for_scheme,
)

__all__ = [
    "SchemeError", "CriticalQuestion", "Edge", "Scheme", "SCHEMES_DIR",
    "load_scheme", "load_all", "load_vocabulary", "check_vocabulary_against_schemes",
    "family_members",
    "Label", "normalize_label", "Resolution", "resolve_for_scheme", "label_matches",
    "Traversal", "traverse", "propagate", "enumerate_paths",
]
