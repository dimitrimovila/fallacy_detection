"""Evaluation of the candidate questions on the outer folds.

Every item is predicted once, by a question chosen without that item: fixed questions (the
original CQ, the hand-written rewrites, the controls) are the same in every fold, while an
ACT candidate uses, in outer fold ``f``, the question selected on the training folds of
``f``. The out-of-fold predictions are pooled before computing the balanced accuracy.

Per source, the pooled predictions are restricted to the items of that source. A source is
*evaluable* when it has at least ``min_items_per_class_in_source`` items of each class;
otherwise only counts and class recalls are reported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .acceptance import Acceptance, SourceComparison, assess
from .act import answer_items
from .answers import Answer
from .baselines import BaselineResult, LexicalPredictability, lexical_predictability
from .config import Boundary, ControlQuestion, EvaluationConfig
from .constraints import check_question
from .data import LabeledItem
from .llm import LLMClient
from .metrics import (
    McNemar,
    PairedDifference,
    balanced_accuracy,
    mcnemar_exact,
    paired_bootstrap_ba,
    recalls,
)

LEXICAL_BASELINES = ("tfidf_cart", "tfidf_rulefit")


@dataclass
class QuestionCandidate:
    candidate_id: str
    kind: str  # "original", "manual", "act", "control"
    questions: dict[int, str]  # outer fold -> question

    @property
    def distinct_questions(self) -> list[str]:
        return list(dict.fromkeys(self.questions[f] for f in sorted(self.questions)))


def build_candidates(
    boundary: Boundary,
    manual: Sequence[str],
    controls: Sequence[ControlQuestion],
    act_selected: dict[int, dict[int, str]],
    folds: Sequence[int],
) -> tuple[list[QuestionCandidate], list[str]]:
    """The candidates of a boundary. ``act_selected`` maps seed -> fold -> question.

    An ACT seed whose selected questions do not cover every outer fold is left out; the
    returned notes say which.
    """
    notes = []
    fixed = lambda q: {f: q for f in folds}  # noqa: E731
    out = [QuestionCandidate("original", "original", fixed(boundary.seed_question))]
    out += [QuestionCandidate(f"manual_{k}", "manual", fixed(q)) for k, q in
            enumerate(manual, start=1)]
    for seed in sorted(act_selected):
        chosen = act_selected[seed]
        missing = [f for f in folds if f not in chosen]
        if missing:
            notes.append(f"ACT seed {seed} left out: no selected question for folds {missing}")
            continue
        out.append(QuestionCandidate(f"act_seed{seed}", "act", {f: chosen[f] for f in folds}))
    if not act_selected:
        notes.append("no ACT result found: run `act-pilot run-act` first")
    out += [QuestionCandidate(c.id, "control", fixed(c.question)) for c in controls]
    return out, notes


@dataclass
class CandidatePredictions:
    candidate: QuestionCandidate
    answers: list[Answer]
    raw: list[str]
    predictions: list[int]


def predict(
    answerer: LLMClient,
    candidate: QuestionCandidate,
    items: Sequence[LabeledItem],
    policy: str,
    boundary_id: str,
) -> CandidatePredictions:
    """Every item is answered with the question of its outer fold."""
    answers: list[Answer | None] = [None] * len(items)
    raw: list[str] = [""] * len(items)
    preds = [0] * len(items)
    by_question: dict[str, list[int]] = {}
    for k, item in enumerate(items):
        by_question.setdefault(candidate.questions[item.outer_fold], []).append(k)
    for question, idx in by_question.items():
        ev = answer_items(
            answerer,
            question,
            [items[k] for k in idx],
            policy,
            {"boundary": boundary_id, "purpose": "evaluation",
             "candidate": candidate.candidate_id},
        )
        for j, k in enumerate(idx):
            answers[k], raw[k], preds[k] = ev.answers[j], ev.raw[j], ev.predictions[j]
    return CandidatePredictions(candidate, [a for a in answers if a is not None], raw, preds)


@dataclass
class SourceRow:
    name: str
    source: str
    n_positive: int
    n_negative: int
    evaluable: bool
    ba: float
    recall_positive: float
    recall_negative: float
    difference: PairedDifference | None = None
    mcnemar: McNemar | None = None


@dataclass
class QuestionSummary:
    candidate: QuestionCandidate
    n: int
    ba: float
    recall_positive: float
    recall_negative: float
    per_fold_ba: dict[int, float]
    n_undetermined: int
    n_invalid: int
    difference: PairedDifference
    mcnemar: McNemar
    lexical: LexicalPredictability
    violations: list[str]
    sources: list[SourceRow] = field(default_factory=list)
    acceptance: Acceptance | None = None


@dataclass
class BaselineSummary:
    result: BaselineResult
    ba: float
    recall_positive: float
    recall_negative: float
    sources: list[SourceRow] = field(default_factory=list)


@dataclass
class BoundaryEvaluation:
    boundary: Boundary
    items: list[LabeledItem]
    questions: list[QuestionSummary]
    baselines: list[BaselineSummary]
    predictions: list[CandidatePredictions]
    label_predictability: LexicalPredictability
    sources: dict[str, tuple[int, int]]
    evaluable_sources: list[str]
    notes: list[str]


def _source_rows(
    name: str,
    items: Sequence[LabeledItem],
    preds: Sequence[int],
    evaluable_sources: Sequence[str],
    reference: Sequence[int] | None,
    cfg: EvaluationConfig,
) -> list[SourceRow]:
    rows = []
    for source in sorted({i.source for i in items}):
        idx = [k for k, i in enumerate(items) if i.source == source]
        y = [items[k].label for k in idx]
        p = [preds[k] for k in idx]
        r_pos, r_neg = recalls(y, p)
        row = SourceRow(
            name=name,
            source=source,
            n_positive=sum(y),
            n_negative=len(y) - sum(y),
            evaluable=source in evaluable_sources,
            ba=balanced_accuracy(y, p) if source in evaluable_sources else float("nan"),
            recall_positive=r_pos,
            recall_negative=r_neg,
        )
        if row.evaluable and reference is not None:
            ref = [reference[k] for k in idx]
            row.difference = paired_bootstrap_ba(
                y, p, ref, n_boot=cfg.bootstrap, seed=cfg.bootstrap_seed, level=cfg.ci_level
            )
            row.mcnemar = mcnemar_exact(y, p, ref)
        rows.append(row)
    return rows


def summarise_boundary(
    boundary: Boundary,
    items: list[LabeledItem],
    predictions: list[CandidatePredictions],
    baselines: list[BaselineResult],
    cfg: EvaluationConfig,
    *,
    validity_scores: dict | None,
    max_logical_ops: int,
    max_question_chars: int,
    notes: Sequence[str] = (),
) -> BoundaryEvaluation:
    """Metrics, paired comparisons, per-source rows and the acceptance of each candidate."""
    labels = [i.label for i in items]
    sources: dict[str, tuple[int, int]] = {}
    for i in items:
        pos, neg = sources.get(i.source, (0, 0))
        sources[i.source] = (pos + i.label, neg + 1 - i.label)
    evaluable = sorted(
        s for s, (pos, neg) in sources.items()
        if pos >= cfg.min_items_per_class_in_source and neg >= cfg.min_items_per_class_in_source
    )
    reference = next(p for p in predictions if p.candidate.kind == "original").predictions

    baseline_rows = []
    for b in baselines:
        r_pos, r_neg = recalls(labels, b.predictions)
        baseline_rows.append(
            BaselineSummary(
                result=b,
                ba=balanced_accuracy(labels, b.predictions),
                recall_positive=r_pos,
                recall_negative=r_neg,
                sources=_source_rows(b.name, items, b.predictions, evaluable, None, cfg),
            )
        )
    lexical = [b for b in baseline_rows if b.result.name in LEXICAL_BASELINES]

    summaries = []
    for cp in predictions:
        cand = cp.candidate
        r_pos, r_neg = recalls(labels, cp.predictions)
        per_fold = {}
        for f in sorted(cand.questions):
            idx = [k for k, i in enumerate(items) if i.outer_fold == f]
            per_fold[f] = balanced_accuracy([labels[k] for k in idx],
                                            [cp.predictions[k] for k in idx])
        violations = sorted(
            {
                v
                for q in cand.distinct_questions
                for v in check_question(
                    q,
                    seed_question=boundary.seed_question,
                    banned_terms=boundary.banned_terms,
                    max_logical_ops=max_logical_ops,
                    max_chars=max_question_chars,
                )
            }
        ) if cand.kind != "control" else []
        summary = QuestionSummary(
            candidate=cand,
            n=len(items),
            ba=balanced_accuracy(labels, cp.predictions),
            recall_positive=r_pos,
            recall_negative=r_neg,
            per_fold_ba=per_fold,
            n_undetermined=sum(a is Answer.UNDETERMINED for a in cp.answers),
            n_invalid=sum(a is Answer.INVALID for a in cp.answers),
            difference=paired_bootstrap_ba(
                labels, cp.predictions, reference,
                n_boot=cfg.bootstrap, seed=cfg.bootstrap_seed, level=cfg.ci_level,
            ),
            mcnemar=mcnemar_exact(labels, cp.predictions, reference),
            lexical=lexical_predictability(items, cp.predictions, cfg.lexical),
            violations=violations,
            sources=_source_rows(cand.candidate_id, items, cp.predictions, evaluable,
                                 None if cand.kind == "original" else reference, cfg),
        )
        if cand.kind in ("manual", "act") and lexical:
            best = max(lexical, key=lambda b: (-1 if math.isnan(b.ba) else b.ba))
            comparisons = []
            for source in evaluable:
                cand_row = next(r for r in summary.sources if r.source == source)
                base = [
                    (b.result.name, next(r for r in b.sources if r.source == source).ba)
                    for b in lexical
                ]
                name, ba = max(base, key=lambda t: -1 if math.isnan(t[1]) else t[1])
                comparisons.append(SourceComparison(source, cand_row.ba, name, ba))
            summary.acceptance = assess(
                boundary=boundary.id,
                questions=cand.distinct_questions,
                difference=summary.difference,
                candidate_ba=summary.ba,
                best_baseline=best.result.name,
                best_baseline_ba=best.ba,
                sources=comparisons,
                validity_scores=validity_scores,
                min_content_validity=cfg.min_content_validity,
            )
        summaries.append(summary)

    return BoundaryEvaluation(
        boundary=boundary,
        items=items,
        questions=summaries,
        baselines=baseline_rows,
        predictions=predictions,
        label_predictability=lexical_predictability(items, labels, cfg.lexical),
        sources=dict(sorted(sources.items())),
        evaluable_sources=evaluable,
        notes=list(notes),
    )
