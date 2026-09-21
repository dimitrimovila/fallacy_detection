"""Explainable fallacy detection over Walton's argumentation schemes.

Two stages: recognise the scheme, then answer its critical questions with
several models and combine the answers into a verdict.  This package holds the
pieces; the YAML under ``schemes/`` and ``labels/`` holds the encoding.
"""

from .paths import FALLACIES_FILE, LABELS_DIR, REPO_ROOT, SCHEME_LABELS_FILE, SCHEMES_DIR
from .schemes import SchemeError, load_all, load_scheme, load_vocabulary

__all__ = [
    "REPO_ROOT", "SCHEMES_DIR", "LABELS_DIR", "FALLACIES_FILE", "SCHEME_LABELS_FILE",
    "SchemeError", "load_scheme", "load_all", "load_vocabulary",
]
