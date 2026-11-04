"""Checks applied to every rewritten question before it is evaluated.

The rewrite prompt states the constraints; the same constraints are also checked here,
because a prompt alone does not guarantee them. The checks are deliberately simple and
conservative. The one on example-specific words flags a content word of the new question
when (a) it is not in the seed question, (b) it is not a generic word of argumentation,
(c) it occurs in one of the example texts shown to the optimizer and (d) it is rare in the
training texts (document frequency at most ``max_df``). Rare words that come straight from
the examples are the typical sign of a topical shortcut. A capitalised word in the middle of
the question, other than the variables of the scheme, is flagged as a possible proper name.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

AUXILIARIES = {
    "is", "are", "was", "were", "am", "do", "does", "did", "has", "have", "had", "can",
    "could", "would", "will", "shall", "should", "may", "might", "must",
}
LOGICAL_WORDS = ("and", "or", "nor")
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "even", "every", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "may", "might", "more", "most", "must", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", "own", "rather",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "whether", "which",
    "while", "who", "whom", "why", "will", "with", "within", "without", "would", "you",
    "your", "yes", "instead", "beyond", "whatever", "whoever", "anything", "something",
}
# Words of argumentation that may appear in any rewrite without being example-specific.
GENERIC_WORDS = {
    "argument", "arguments", "arguer", "author", "speaker", "text", "claim", "claims",
    "conclusion", "premise", "premises", "reason", "reasons", "reasoning", "evidence",
    "support", "supports", "supported", "person", "people", "opponent", "audience", "group",
    "majority", "many", "most", "accept", "accepted", "accepts", "acceptance", "believe",
    "belief", "beliefs", "true", "false", "truth", "presumption", "attack", "attacks",
    "discredit", "discussion", "debate", "position", "view", "views", "opinion", "opinions",
    "character", "trust", "distrust", "credibility", "relevant", "irrelevant", "issue",
    "topic", "matter", "fact", "facts", "statement", "statements", "point", "points",
    "offered", "offers", "given", "gives", "presented", "used", "made", "makes", "says",
    "said", "put", "forward", "reply", "replies", "respond", "responds", "response",
}
SCHEME_VARIABLES = {"A", "B", "I"}

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def content_words(text: str) -> set[str]:
    return {w for w in tokens(text) if len(w) >= 4 and w not in STOPWORDS}


def document_frequency(texts: Sequence[str]) -> dict[str, float]:
    """The share of texts in which each lower-cased word occurs."""
    n = len(texts) or 1
    counts: dict[str, int] = {}
    for text in texts:
        for word in set(tokens(text)):
            counts[word] = counts.get(word, 0) + 1
    return {w: c / n for w, c in counts.items()}


def count_logical_words(question: str) -> int:
    return sum(1 for w in tokens(question) if w in LOGICAL_WORDS)


def _contains_term(question: str, term: str) -> bool:
    pattern = r"(?<![a-z])" + re.escape(term.lower()) + r"(?![a-z])"
    return re.search(pattern, question.lower()) is not None


def check_question(
    question: str,
    *,
    seed_question: str,
    banned_terms: Iterable[str],
    max_logical_ops: int,
    max_chars: int,
    example_texts: Sequence[str] = (),
    corpus_df: dict[str, float] | None = None,
    max_df: float = 0.05,
) -> list[str]:
    """The list of violated constraints; an empty list means the question is admissible."""
    q = " ".join(question.split())
    violations: list[str] = []
    if not q:
        return ["empty question"]
    if len(q) > max_chars:
        violations.append(f"longer than {max_chars} characters")
    if not q.endswith("?"):
        violations.append("does not end with a question mark")
    if q.count("?") > 1:
        violations.append("contains more than one question")
    first = tokens(q[:30])[:1]
    if not first or first[0] not in AUXILIARIES:
        violations.append("does not start like a yes/no question (Is/Does/Can ...)")
    n_logical = count_logical_words(q)
    if n_logical > max_logical_ops:
        violations.append(f"uses {n_logical} logical conjunctions (at most {max_logical_ops})")
    for term in banned_terms:
        if term and _contains_term(q, term):
            violations.append(f"names a forbidden term: {term!r}")
    seed_words = set(tokens(seed_question))
    example_words: set[str] = set()
    for text in example_texts:
        example_words.update(tokens(text))
    df = corpus_df or {}
    specific = sorted(
        w
        for w in content_words(q)
        if w not in seed_words
        and w not in GENERIC_WORDS
        and w in example_words
        and df.get(w, 0.0) <= max_df
    )
    if specific:
        violations.append(f"uses words specific to the examples: {', '.join(specific)}")
    words = _WORD.findall(q)
    names = sorted(
        {
            w
            for i, w in enumerate(words)
            if i > 0
            and w[0].isupper()
            and w not in SCHEME_VARIABLES
            and w.lower() not in seed_words
        }
    )
    if names:
        violations.append(f"contains capitalised words that may be names: {', '.join(names)}")
    return violations
