"""The acceptance criterion of a rewritten question, and the content-validity file.

A candidate is **accepted** only when all three conditions hold:

(i)   the paired bootstrap interval of BA(candidate) minus BA(original CQ) lies entirely
      above zero;
(ii)  its balanced accuracy is not lower than that of the best lexical baseline
      (TF-IDF + CART, TF-IDF + RuleFit), globally and inside every evaluable source;
(iii) every question of the candidate (one per outer fold for an ACT candidate) has a
      content-validity score of at least ``min_content_validity`` (scale 1 to 3) from every
      rater listed in ``content_validity.csv``.

It is **rejected** as soon as one condition fails, and **pending** when (i) and (ii) hold
but the content-validity scores are missing.

``content_validity.csv`` is filled in by hand. Columns: ``boundary``, ``question``,
``content_validity`` (1, 2 or 3; empty means not rated yet), and optionally ``rater`` and
``notes``. Questions are matched after lower-casing and squeezing whitespace.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import PairedDifference

PASS, FAIL, PENDING = "pass", "fail", "pending"


@dataclass(frozen=True)
class CriterionResult:
    status: str
    detail: str


@dataclass
class Acceptance:
    status: str  # "accepted", "rejected" or "pending"
    criteria: dict[str, CriterionResult] = field(default_factory=dict)

    def summary(self) -> str:
        return "; ".join(f"({k}) {v.status}: {v.detail}" for k, v in self.criteria.items())


@dataclass(frozen=True)
class SourceComparison:
    source: str
    candidate_ba: float
    best_baseline: str
    best_baseline_ba: float


def normalise_question(question: str) -> str:
    return " ".join(question.lower().split())


def load_content_validity(path: Path) -> dict[tuple[str, str], list[int]] | None:
    """Scores per (boundary, normalised question); ``None`` when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    scores: dict[tuple[str, str], list[int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"boundary", "question", "content_validity"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: the columns {sorted(required)} are required")
        for n, row in enumerate(reader, start=2):
            value = (row.get("content_validity") or "").strip()
            key = ((row["boundary"] or "").strip(), normalise_question(row["question"] or ""))
            scores.setdefault(key, [])
            if not value:
                continue
            try:
                score = int(float(value))
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{n}: content_validity {value!r} is not 1, 2 or 3"
                ) from exc
            if score not in (1, 2, 3):
                raise ValueError(f"{path}:{n}: content_validity {score} is not 1, 2 or 3")
            scores[key].append(score)
    return scores


def content_validity_status(
    boundary: str,
    questions: Iterable[str],
    scores: dict[tuple[str, str], list[int]] | None,
    min_score: int,
) -> CriterionResult:
    if scores is None:
        return CriterionResult(PENDING, "content_validity.csv not found")
    missing, low = [], []
    for q in questions:
        rated = scores.get((boundary, normalise_question(q)), [])
        if not rated:
            missing.append(q)
        elif min(rated) < min_score:
            low.append(f"{min(rated)} for {q!r}")
    if low:
        return CriterionResult(FAIL, f"score below {min_score}: {'; '.join(low)}")
    if missing:
        return CriterionResult(PENDING, f"{len(missing)} question(s) not rated yet")
    return CriterionResult(PASS, f"every rater gave at least {min_score}")


def assess(
    *,
    boundary: str,
    questions: Sequence[str],
    difference: PairedDifference,
    candidate_ba: float,
    best_baseline: str,
    best_baseline_ba: float,
    sources: Sequence[SourceComparison],
    validity_scores: dict[tuple[str, str], list[int]] | None,
    min_content_validity: int = 2,
) -> Acceptance:
    """The three conditions are checked and combined."""
    criteria: dict[str, CriterionResult] = {}
    if math.isnan(difference.low):
        criteria["i"] = CriterionResult(FAIL, "the paired difference is undefined")
    elif difference.excludes_zero_in_favour:
        criteria["i"] = CriterionResult(
            PASS, f"delta {difference.value:+.3f}, CI [{difference.low:+.3f}, "
                  f"{difference.high:+.3f}] above zero"
        )
    else:
        criteria["i"] = CriterionResult(
            FAIL, f"delta {difference.value:+.3f}, CI [{difference.low:+.3f}, "
                  f"{difference.high:+.3f}] does not exclude zero in favour of the candidate"
        )

    failures = []
    if math.isnan(candidate_ba) or candidate_ba < best_baseline_ba:
        failures.append(f"global {candidate_ba:.3f} < {best_baseline} {best_baseline_ba:.3f}")
    for s in sources:
        if math.isnan(s.candidate_ba) or s.candidate_ba < s.best_baseline_ba:
            failures.append(
                f"{s.source} {s.candidate_ba:.3f} < {s.best_baseline} {s.best_baseline_ba:.3f}"
            )
    scope = (
        f"global and {len(sources)} evaluable source(s)" if sources
        else "global only (no evaluable source)"
    )
    criteria["ii"] = (
        CriterionResult(FAIL, "; ".join(failures))
        if failures
        else CriterionResult(PASS, f"not below the best lexical baseline ({scope})")
    )
    criteria["iii"] = content_validity_status(
        boundary, questions, validity_scores, min_content_validity
    )

    statuses = [c.status for c in criteria.values()]
    if FAIL in statuses:
        status = "rejected"
    elif PENDING in statuses:
        status = "pending"
    else:
        status = "accepted"
    return Acceptance(status=status, criteria=criteria)


def write_validity_template(path: Path, rows: Iterable[tuple[str, str, str]]) -> None:
    """A template with one row per (boundary, candidate, question) and empty scores."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["boundary", "candidate_id", "question", "rater", "content_validity",
                         "notes"])
        seen = set()
        for boundary, candidate_id, question in rows:
            key = (boundary, normalise_question(question))
            if key in seen:
                continue
            seen.add(key)
            writer.writerow([boundary, candidate_id, question, "", "", ""])
