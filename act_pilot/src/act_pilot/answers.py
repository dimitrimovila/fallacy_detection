"""The answer prompt and the parser of the answers.

The answerer sees the text and the question, reasons briefly and ends with
``Answer: yes`` or ``Answer: no``. The parser reads the last ``Answer:`` line. A reply that
says the answer cannot be determined is kept as a third category, ``cannot_determine``;
a reply that cannot be read at all is ``invalid``. Both are counted in the report. For the
metrics they are mapped to a class by ``evaluation.undetermined_policy`` (by default to the
negative class, because only "yes" stands for the positive class).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class Answer(StrEnum):
    YES = "yes"
    NO = "no"
    UNDETERMINED = "cannot_determine"
    INVALID = "invalid"


ANSWER_PROMPT_VERSION = "answer-v1"

ANSWER_SYSTEM = (
    "You read a short argumentative text and answer one yes/no question about it. "
    "Judge only what the text itself says or clearly implies. Reason briefly, in at most "
    "three sentences, and then write a final line of the form 'Answer: yes' or "
    "'Answer: no'."
)

ANSWER_USER_TEMPLATE = (
    "Text:\n<<<\n{text}\n>>>\n\n"
    "Question: {question}\n\n"
    "Reason briefly, then end with exactly one line: 'Answer: yes' or 'Answer: no'."
)


@dataclass(frozen=True)
class Prompt:
    """A prompt split into the part that does not depend on the text and the text slot.

    ``user_template`` contains the placeholder ``{text}``; everything else in it is already
    filled. The cache key is built from the model, ``system``, ``user_template`` and the
    text, so the same question on the same text is asked only once.
    """

    system: str
    user_template: str
    version: str

    def render(self, text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user_template.replace("{text}", text)},
        ]


def answer_prompt(question: str) -> Prompt:
    """The answer prompt for one question; the text is filled in later."""
    question = " ".join(question.split())
    return Prompt(
        system=ANSWER_SYSTEM,
        user_template=ANSWER_USER_TEMPLATE.replace("{question}", question),
        version=ANSWER_PROMPT_VERSION,
    )


_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER = re.compile(r"answer\s*[:：]\s*(.*)", re.IGNORECASE)
_DECORATION = re.compile(r"^[\s*_`'\"“”‘’\[\](){}<>]+")
_UNDETERMINED = re.compile(
    r"^(cannot|can't|can not|could not|unable|undetermin|indetermin|unclear|unknown|"
    r"not (enough|sure|clear|determin)|insufficient|neither|n/?a\b|uncertain)",
    re.IGNORECASE,
)
_YES = re.compile(r"^yes\b", re.IGNORECASE)
_NO = re.compile(r"^no\b", re.IGNORECASE)
_BARE = re.compile(r"^[\s*_`'\"]*(yes|no)[\s*_`'\".!]*$", re.IGNORECASE)


def _classify(fragment: str) -> Answer:
    fragment = _DECORATION.sub("", fragment).strip()
    head = fragment[:40].lower()
    if re.match(r"^(yes\s*(/|or)\s*no|no\s*(/|or)\s*yes)\b", head):
        return Answer.INVALID
    if _YES.match(head):
        return Answer.YES
    if _NO.match(head):
        return Answer.NO
    if _UNDETERMINED.match(head):
        return Answer.UNDETERMINED
    return Answer.INVALID


def parse_answer(raw: str | None) -> Answer:
    """The final answer is read from a raw reply.

    Reasoning blocks between ``<think>`` tags are removed first; a reply whose reasoning
    block is never closed is ``invalid``. The last ``Answer:`` occurrence decides. When there
    is none, a last line that is only "yes" or "no" is accepted.
    """
    if not raw:
        return Answer.INVALID
    text = _THINK.sub("", raw)
    if re.search(r"<think>", text, re.IGNORECASE):
        return Answer.INVALID
    matches = list(_ANSWER.finditer(text))
    if matches:
        return _classify(matches[-1].group(1))
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if lines:
        bare = _BARE.match(lines[-1])
        if bare:
            return Answer.YES if bare.group(1).lower() == "yes" else Answer.NO
    return Answer.INVALID


def to_prediction(answer: Answer, policy: str = "negative") -> int:
    """The binary prediction of an answer: 1 for "yes", 0 for "no", ``policy`` otherwise."""
    if answer is Answer.YES:
        return 1
    if answer is Answer.NO:
        return 0
    return 1 if policy == "positive" else 0


def reasoning_of(raw: str | None, limit: int = 300) -> str:
    """The reasoning part of a reply, without the answer line, shortened to ``limit``."""
    if not raw:
        return ""
    text = _THINK.sub("", raw)
    text = _ANSWER.split(text)[0] if _ANSWER.search(text) else text
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
