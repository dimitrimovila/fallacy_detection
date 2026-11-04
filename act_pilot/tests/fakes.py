"""Fake chat backends and synthetic data for the tests. No network call is ever made."""

from __future__ import annotations

import csv
import hashlib
import re
import threading
from collections.abc import Callable
from pathlib import Path

_TEXT = re.compile(r"Text:\n<<<\n(.*)\n>>>", re.DOTALL)
_QUESTION = re.compile(r"\nQuestion: (.*)\n")


def split_answer_request(messages: list[dict[str, str]]) -> tuple[str, str]:
    user = messages[-1]["content"]
    text = _TEXT.search(user).group(1)
    question = _QUESTION.search(user).group(1)
    return question, text


def stable_hash(*parts: str) -> int:
    return int(hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:8], 16)


def default_rule(question: str, text: str) -> str:
    """A deterministic answer that depends on cue words of the question and of the text."""
    q, t = question.lower(), text.lower()
    if "longer than three sentences" in q:
        return "yes" if t.count(".") > 3 else "no"
    if "contain a question" in q:
        return "yes" if "?" in t else "no"
    if "reason why the reference group" in q or "independent" in q:
        return "yes" if ("expert" in t or "studies" in t) else "no"
    if "in advance" in q or "before the person" in q or "pre-emptive" in q:
        return "yes" if ("before" in t or "do not listen" in t) else "no"
    h = stable_hash(question, text)
    if h % 23 == 0:
        return "cannot determine"
    return "yes" if h % 2 == 0 else "no"


class FakeAnswerer:
    """Replies like an answerer: a sentence of reasoning and an 'Answer:' line."""

    def __init__(self, rule: Callable[[str, str], str] = default_rule):
        self.rule = rule
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def complete(self, *, model, messages, temperature, max_tokens, extra_body):
        question, text = split_answer_request(messages)
        with self._lock:
            self.calls.append({"model": model, "question": question, "text": text,
                               "temperature": temperature})
        verdict = self.rule(question, text)
        return f"The text was read with care.\nAnswer: {verdict}"


GENERIC_POOL = [
    f"Does the argument rest on {a} {b} rather than on the number of supporters alone?"
    for a in ("independent", "verifiable", "explicit", "relevant", "stated", "external",
              "documented", "shared", "considered", "reasoned", "careful", "specific")
    for b in ("grounds", "evidence", "reasons", "support", "justification", "warrant")
]


class FakeOptimizer:
    """Critiques with a fixed text; rewrites with questions taken from a pool, each once.

    ``scripted`` gives replies for the first rewrite calls, before the pool is used.
    """

    def __init__(self, pool: list[str] | None = None, scripted: list[str] | None = None):
        self.pool = list(pool or GENERIC_POOL)
        self.scripted = list(scripted or [])
        self.calls: list[dict] = []
        self._next = 0
        self._lock = threading.Lock()

    def complete(self, *, model, messages, temperature, max_tokens, extra_body):
        user = messages[-1]["content"]
        with self._lock:
            kind = "rewrite" if "Write an improved question." in user else "critique"
            self.calls.append({"model": model, "kind": kind, "user": user})
            if kind == "critique":
                return "The question confuses popularity with the grounds of the claim."
            if self.scripted:
                return self.scripted.pop(0)
            question = self.pool[self._next % len(self.pool)]
            self._next += 1
            return f"Question: {question}"


class FlakyBackend:
    """Fails with the given exceptions first, then answers 'Answer: yes'."""

    def __init__(self, failures: list[Exception]):
        self.failures = list(failures)
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "Reasoning.\nAnswer: yes"


GOOD_TEMPLATES = [
    "Most doctors recommend the vaccine, and the studies they rely on show it works for {x}.",
    "Many experts in the field agree that {x} helps, after years of careful trials.",
    "Most residents who use the park daily say {x} is safe, from their direct experience.",
]
POPULUM_TEMPLATES = [
    "Everyone is buying {x} these days, so it must be the best choice.",
    "Millions of people believe in {x}, so it has to be true.",
    "All my friends think {x} is great, therefore it is great.",
]
POISON_TEMPLATES = [
    "Before you hear what my opponent says about {x}, remember that he lies about everything.",
    "Do not listen to her views on {x}: she has never told the truth in her life.",
]
ABUSIVE_TEMPLATES = [
    "His plan for {x} is wrong because he is an ignorant fool.",
    "You cannot trust her claim about {x}; she is a lazy hippie.",
]
TOPICS = ["recycling", "public transport", "the new library", "solar panels", "school meals",
          "bike lanes", "the museum", "online classes", "the local market", "tap water",
          "electric cars", "the city budget", "the night bus", "green roofs", "the harbour",
          "free clinics", "the old bridge", "street lights", "the theatre", "rain gardens"]


def synthetic_items() -> list[dict[str, str]]:
    """Items for B1 and B2 with sources chosen so that one source of B1 is evaluable."""
    rows: list[dict[str, str]] = []

    def add(n: int, scheme: str, gold: str, source: str, templates: list[str], tag: str):
        for k in range(n):
            text = templates[k % len(templates)].format(x=TOPICS[k % len(TOPICS)])
            if k % 4 == 0:
                text += " Is that not obvious? It is. Think about it. Really."
            rows.append({"item_id": f"{tag}{k:03d}", "text": text, "scheme": scheme,
                         "gold": gold, "source": source})

    add(12, "popular_opinion", "good_argumentation", "NLAS", GOOD_TEMPLATES, "gn")
    add(10, "popular_opinion", "ad_populum", "NLAS", POPULUM_TEMPLATES, "pn")
    add(10, "popular_opinion", "good_argumentation", "EthiX", GOOD_TEMPLATES, "ge")
    add(14, "popular_opinion", "ad_populum", "LOGIC", POPULUM_TEMPLATES, "pl")
    add(16, "ad_hominem", "poisoning_the_well", "Reddit", POISON_TEMPLATES, "hp")
    add(16, "ad_hominem", "ad_hominem_abusive", "LOGIC", ABUSIVE_TEMPLATES, "ha")
    add(5, "expert_opinion", "good_argumentation", "NLAS", GOOD_TEMPLATES, "xo")
    return rows


def write_synthetic(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    rows = synthetic_items()
    items = directory / "items.csv"
    folds = directory / "folds.csv"
    with items.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "text", "scheme", "gold",
                                                    "source"])
        writer.writeheader()
        writer.writerows(rows)
    with folds.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "outer_fold"])
        for k, row in enumerate(sorted(rows, key=lambda r: r["item_id"])):
            writer.writerow([row["item_id"], k % 5])
    return items, folds
