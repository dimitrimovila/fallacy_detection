"""A depth-one reimplementation of the ACT loop (Grari et al., arXiv:2509.26433).

The algorithm follows the description of the paper; no code of the original repository is
used. At depth one the tree is a single question, so the loop reduces to refining that
question:

1. the seed is the original critical question;
2. at each step the answerer answers the current question on every item of the inner
   training set and of the inner validation set; the weighted Gini impurity and the
   balanced accuracy are computed on both;
3. for each of the two groups (answered yes, answered no) up to ``m`` correct and ``m``
   wrong training examples are sampled with the run's seed; the optimizer writes a
   critique and then a rewritten question;
4. the rewritten question is checked against the constraints; a violating proposal is sent
   back with the reasons, at most ``max_rewrite_attempts`` times; if every attempt fails the
   step is recorded as rejected and the current question is kept;
5. after ``k`` steps the question with the best balanced accuracy on the inner validation
   set is selected, ties going to the earliest step. Training scores are recorded but never
   used for the selection (the paper selects on training impurity; this pilot does not).

Polarity: "yes" stands for the positive class.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from .answers import Answer, answer_prompt, parse_answer, reasoning_of, to_prediction
from .config import ActConfig, Boundary
from .constraints import check_question, document_frequency
from .data import LabeledItem
from .llm import LLMClient, Request
from .metrics import balanced_accuracy, gini, weighted_gini
from .prompts import (
    GroupSummary,
    ShownExample,
    critique_prompt,
    critique_text,
    parse_rewrite,
    rewrite_prompt,
    rewrite_text,
)


@dataclass
class Evaluation:
    """The answers of one question on one set of items."""

    answers: list[Answer]
    raw: list[str]
    predictions: list[int]
    balanced_accuracy: float
    weighted_gini: float
    n_undetermined: int
    n_invalid: int


@dataclass
class Candidate:
    step: int
    question: str
    origin: str  # "seed" or "rewrite"
    status: str  # "evaluated", "rejected" or "duplicate"
    violations: list[str] = field(default_factory=list)
    rewrite_attempts: int = 0
    critique: str = ""
    train_ba: float = float("nan")
    train_gini: float = float("nan")
    val_ba: float = float("nan")
    val_gini: float = float("nan")
    train_undetermined: int = 0
    val_undetermined: int = 0
    train_invalid: int = 0
    val_invalid: int = 0
    selected: bool = False


@dataclass
class ActRun:
    boundary: str
    outer_fold: int
    seed: int
    n_inner_train: int
    n_inner_val: int
    candidates: list[Candidate]
    selected_step: int
    selected_question: str

    def as_rows(self) -> list[dict]:
        rows = []
        for c in self.candidates:
            row = {"boundary": self.boundary, "outer_fold": self.outer_fold, "seed": self.seed}
            row.update(asdict(c))
            row["violations"] = "; ".join(c.violations)
            rows.append(row)
        return rows


def answer_items(
    answerer: LLMClient,
    question: str,
    items: Sequence[LabeledItem],
    policy: str,
    meta: dict,
) -> Evaluation:
    """Every item is answered with ``question``; the split statistics are computed."""
    prompt = answer_prompt(question)
    requests = [
        Request(prompt=prompt, text=item.text, meta={**meta, "item_id": item.item_id})
        for item in items
    ]
    raw = answerer.ask_many(requests)
    answers = [parse_answer(r) for r in raw]
    predictions = [to_prediction(a, policy) for a in answers]
    labels = [item.label for item in items]
    return Evaluation(
        answers=answers,
        raw=raw,
        predictions=predictions,
        balanced_accuracy=balanced_accuracy(labels, predictions),
        weighted_gini=weighted_gini(labels, predictions),
        n_undetermined=sum(a is Answer.UNDETERMINED for a in answers),
        n_invalid=sum(a is Answer.INVALID for a in answers),
    )


def _group(
    name: str,
    pred_value: int,
    items: Sequence[LabeledItem],
    evaluation: Evaluation,
    m: int,
    rng: random.Random,
) -> GroupSummary:
    idx = [i for i, p in enumerate(evaluation.predictions) if p == pred_value]
    labels = [items[i].label for i in idx]
    correct = [i for i in idx if items[i].label == pred_value]
    wrong = [i for i in idx if items[i].label != pred_value]

    def pick(pool: list[int]) -> list[int]:
        # A random subset in a random order: with fewer than m examples all of them are
        # shown, and the seed still changes their order (and so the prompt).
        return rng.sample(pool, min(m, len(pool)))

    def shown(i: int) -> ShownExample:
        return ShownExample(
            text=items[i].text,
            expected="yes" if items[i].label == 1 else "no",
            answer=evaluation.answers[i].value,
            reasoning=reasoning_of(evaluation.raw[i]),
        )

    return GroupSummary(
        name=name,
        size=len(idx),
        n_positive=sum(labels),
        n_negative=len(labels) - sum(labels),
        gini=gini(labels),
        correct=[shown(i) for i in pick(correct)],
        wrong=[shown(i) for i in pick(wrong)],
    )


def _history_entry(c: Candidate) -> tuple[str, str]:
    """A past question with its training outcome; validation scores are not disclosed."""
    if c.status == "rejected":
        return c.question or "(no question was produced)", "rejected: " + "; ".join(
            c.violations)
    return c.question, f"training balanced accuracy {c.train_ba:.2f}"


def _normalise(question: str) -> str:
    return " ".join(question.lower().split()).rstrip("?").strip()


def run_act(
    boundary: Boundary,
    inner_train: Sequence[LabeledItem],
    inner_val: Sequence[LabeledItem],
    answerer: LLMClient,
    optimizer: LLMClient,
    settings: ActConfig,
    *,
    seed: int,
    outer_fold: int,
    policy: str = "negative",
) -> ActRun:
    """One ACT run for one boundary, one outer fold and one seed."""
    rng = random.Random(seed)
    corpus_df = document_frequency([i.text for i in inner_train])
    base_meta = {"boundary": boundary.id, "outer_fold": outer_fold, "seed": seed}
    candidates: list[Candidate] = []
    seen: dict[str, Candidate] = {}
    question = boundary.seed_question
    current: Candidate | None = None
    current_eval: Evaluation | None = None
    pending = Candidate(step=0, question=question, origin="seed", status="evaluated")

    for step in range(settings.steps + 1):
        if pending is not None:
            key = _normalise(pending.question)
            if key in seen:
                # The answers come from the cache; the step is kept to show the repetition.
                pending.status = "duplicate"
            meta = {**base_meta, "step": step, "purpose": "act_answer"}
            train_eval = answer_items(answerer, pending.question, inner_train, policy,
                                      {**meta, "split": "inner_train"})
            val_eval = answer_items(answerer, pending.question, inner_val, policy,
                                    {**meta, "split": "inner_val"})
            pending.train_ba = train_eval.balanced_accuracy
            pending.train_gini = train_eval.weighted_gini
            pending.val_ba = val_eval.balanced_accuracy
            pending.val_gini = val_eval.weighted_gini
            pending.train_undetermined = train_eval.n_undetermined
            pending.val_undetermined = val_eval.n_undetermined
            pending.train_invalid = train_eval.n_invalid
            pending.val_invalid = val_eval.n_invalid
            seen.setdefault(key, pending)
            candidates.append(pending)
            current, current_eval = pending, train_eval
            question = pending.question
            pending = None
        if step == settings.steps:
            break

        assert current is not None and current_eval is not None
        groups = [
            _group("answered yes", 1, inner_train, current_eval, settings.examples_per_group,
                   rng),
            _group("answered no", 0, inner_train, current_eval, settings.examples_per_group,
                   rng),
        ]
        critique = optimizer.ask(
            critique_prompt(),
            critique_text(
                boundary,
                question,
                groups,
                weighted_gini=current_eval.weighted_gini,
                balanced_accuracy=current_eval.balanced_accuracy,
                max_example_chars=settings.max_example_chars,
                show_reasoning=settings.show_error_reasoning,
            ),
            {**base_meta, "step": step, "purpose": "critique"},
        )
        shown_texts = [e.text for g in groups for e in (*g.correct, *g.wrong)]
        history = [_history_entry(c) for c in candidates]
        violations: list[str] = []
        rejected: str | None = None
        proposal: str | None = None
        attempts = 0
        for attempt in range(settings.max_rewrite_attempts):
            attempts = attempt + 1
            reply = optimizer.ask(
                rewrite_prompt(),
                rewrite_text(
                    boundary,
                    question,
                    critique,
                    max_logical_ops=settings.max_logical_ops,
                    previous_violations=violations,
                    rejected_question=rejected,
                    history=history,
                ),
                {**base_meta, "step": step, "purpose": "rewrite", "attempt": attempts},
            )
            proposal = parse_rewrite(reply)
            violations = (
                ["no question found in the reply"]
                if proposal is None
                else check_question(
                    proposal,
                    seed_question=boundary.seed_question,
                    banned_terms=boundary.banned_terms,
                    max_logical_ops=settings.max_logical_ops,
                    max_chars=settings.max_question_chars,
                    example_texts=shown_texts,
                    corpus_df=corpus_df,
                    max_df=settings.specific_term_max_df,
                )
            )
            if not violations:
                break
            rejected = proposal or (reply or "")[:200]
        if violations:
            candidates.append(
                Candidate(
                    step=step + 1,
                    question=proposal or "",
                    origin="rewrite",
                    status="rejected",
                    violations=violations,
                    rewrite_attempts=attempts,
                    critique=critique,
                )
            )
            continue
        assert proposal is not None
        pending = Candidate(
            step=step + 1,
            question=proposal,
            origin="rewrite",
            status="evaluated",
            rewrite_attempts=attempts,
            critique=critique,
        )

    selected = select_candidate(candidates)
    selected.selected = True
    return ActRun(
        boundary=boundary.id,
        outer_fold=outer_fold,
        seed=seed,
        n_inner_train=len(inner_train),
        n_inner_val=len(inner_val),
        candidates=candidates,
        selected_step=selected.step,
        selected_question=selected.question,
    )


def select_candidate(candidates: Sequence[Candidate]) -> Candidate:
    """The evaluated candidate with the best validation balanced accuracy; earliest on ties."""
    eligible = [c for c in candidates if c.status in ("evaluated", "duplicate")]
    scored = [c for c in eligible if not math.isnan(c.val_ba)]
    if not scored:
        raise ValueError("no candidate has a defined validation balanced accuracy")
    best = max(c.val_ba for c in scored)
    return min((c for c in scored if c.val_ba == best), key=lambda c: c.step)
