"""The two prompts of the optimizer: the critique and the rewrite.

Each step of the loop makes two optimizer calls. The critique call receives the current
question, the statistics of the two groups it produces on the inner training set and up to
``m`` correct and ``m`` wrong examples per group; it returns a short diagnosis. The rewrite
call receives the critique and the constraints and returns one new question. The class
names are never shown to the optimizer, only their descriptions, and the constraints forbid
naming them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .answers import Prompt
from .config import Boundary

CRITIQUE_PROMPT_VERSION = "critique-v1"
REWRITE_PROMPT_VERSION = "rewrite-v1"

OPTIMIZER_SYSTEM = (
    "You help to improve a critical question of an argumentation scheme. The question is "
    "answered yes or no by another model, and it should separate two kinds of arguments "
    "that belong to the same scheme. The question must keep its role in the scheme: it is a "
    "test of whether the argument meets one condition of the scheme, not a topic detector."
)


@dataclass(frozen=True)
class ShownExample:
    text: str
    expected: str  # "yes" or "no": the answer the positive/negative class should receive
    answer: str  # what the answerer said
    reasoning: str = ""


@dataclass(frozen=True)
class GroupSummary:
    name: str  # "answered yes" or "answered no"
    size: int
    n_positive: int
    n_negative: int
    gini: float
    correct: Sequence[ShownExample]
    wrong: Sequence[ShownExample]


def _block(example: ShownExample, max_chars: int, with_reasoning: bool) -> str:
    text = " ".join(example.text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    line = f"- [should be {example.expected.upper()}, answered {example.answer}] {text}"
    if with_reasoning and example.reasoning:
        line += f"\n  (answerer's reasoning: {example.reasoning})"
    return line


def _boundary_context(boundary: Boundary) -> str:
    return (
        f"Argumentation scheme:\n{boundary.scheme_text}\n\n"
        f"Original critical question: {boundary.seed_question}\n"
        f"Role of the question in the scheme: {boundary.question_function}\n\n"
        f"Arguments that should receive YES: {boundary.positive_description}\n"
        f"Arguments that should receive NO: {boundary.negative_description}"
    )


def critique_prompt() -> Prompt:
    return Prompt(system=OPTIMIZER_SYSTEM, user_template="{text}", version=CRITIQUE_PROMPT_VERSION)


def rewrite_prompt() -> Prompt:
    return Prompt(system=OPTIMIZER_SYSTEM, user_template="{text}", version=REWRITE_PROMPT_VERSION)


def critique_text(
    boundary: Boundary,
    question: str,
    groups: Sequence[GroupSummary],
    *,
    weighted_gini: float,
    balanced_accuracy: float,
    max_example_chars: int,
    show_reasoning: bool,
) -> str:
    """The user message of the critique call."""
    parts = [
        _boundary_context(boundary),
        f"\nCurrent question: {question}",
        f"On the training examples the current question has a weighted Gini impurity of "
        f"{weighted_gini:.3f} (0 is a perfect split, 0.5 the worst) and a balanced accuracy "
        f"of {balanced_accuracy:.3f}.",
    ]
    for g in groups:
        parts.append(
            f"\nGroup '{g.name}': {g.size} examples, {g.n_positive} should be YES and "
            f"{g.n_negative} should be NO (Gini {g.gini:.3f})."
        )
        if g.correct:
            parts.append("Correctly answered examples:")
            parts.extend(_block(e, max_example_chars, False) for e in g.correct)
        if g.wrong:
            parts.append("Wrongly answered examples:")
            parts.extend(_block(e, max_example_chars, show_reasoning) for e in g.wrong)
    parts.append(
        "\nExplain briefly why the current question answers the wrong examples wrongly. "
        "Focus on at most two properties of the arguments that the condition of the scheme "
        "is about, and that should generalise to new arguments on other topics. Do not "
        "propose properties tied to the topics, people, places or wording of these examples. "
        "Keep the critique under 150 words."
    )
    return "\n".join(parts)


def constraints_list(boundary: Boundary, max_logical_ops: int) -> list[str]:
    return [
        "It is a single yes/no question that ends with a question mark.",
        "A 'yes' answer must stand for the arguments described as YES above.",
        "It remains a reformulation of the original critical question and keeps the same "
        "role in the scheme.",
        "It does not name the fallacy, the class of the argument or the argumentation scheme.",
        "It does not mention topics, people, places or words that are specific to the "
        "examples.",
        f"It uses at most {max_logical_ops} logical conjunctions ('and', 'or').",
        "It avoids vague words such as 'possibly', 'might' or 'somehow'.",
    ]


def rewrite_text(
    boundary: Boundary,
    question: str,
    critique: str,
    *,
    max_logical_ops: int,
    previous_violations: Sequence[str] = (),
    rejected_question: str | None = None,
    history: Sequence[tuple[str, str]] = (),
) -> str:
    """The user message of the rewrite call.

    ``history`` lists the questions of the earlier steps with a short outcome (the training
    balanced accuracy, or the reason for a rejection). Validation scores are never shown.
    Besides informing the optimizer, the history makes the prompt of every step different,
    so that a step after a rejected one does not repeat the same cached reply.
    """
    parts = [
        _boundary_context(boundary),
        f"\nCurrent question: {question}",
        f"\nCritique of the current question:\n{critique.strip()}",
    ]
    if history:
        parts.append("\nQuestions tried in earlier steps (do not repeat them):")
        parts.extend(f"- {q} ({outcome})" for q, outcome in history)
    parts.append("\nWrite an improved question. Constraints:")
    parts.extend(f"- {c}" for c in constraints_list(boundary, max_logical_ops))
    if previous_violations and rejected_question:
        parts.append(
            f"\nYour previous proposal was rejected: {rejected_question}\n"
            f"Reasons: {'; '.join(previous_violations)}."
        )
    parts.append("\nReply with one line only, in the form: Question: <the new question>")
    return "\n".join(parts)


_QUESTION_LINE = re.compile(r"question\s*[:：]\s*(.+)", re.IGNORECASE)


def parse_rewrite(raw: str | None) -> str | None:
    """The new question from a rewrite reply, or ``None`` when no question can be found."""
    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [m.group(1) for m in _QUESTION_LINE.finditer(text)]
    if not candidates:
        candidates = [line for line in text.splitlines() if line.strip().endswith("?")]
    if not candidates:
        return None
    question = candidates[-1].strip()
    question = re.sub(r"^[\s*_`'\"“”]+|[\s*_`'\"“”]+$", "", question)
    question = re.sub(r"\s*\((yes/no|yes or no)\)\s*\??\s*$", "?", question, flags=re.I)
    question = " ".join(question.split())
    return question or None
