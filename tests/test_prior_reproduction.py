"""Our diagrams against the prior pipeline: 2416 verdicts out of 2428 agree.

Each of the four pipeline runs holds, for every item, the scheme the model chose,
its answer to each critical question and the verdict that pipeline drew from them.
Walking our diagrams on the same answers gives the same verdict everywhere except
in twelve known places:

* seven on expert opinion, where that pipeline has the CQ4 and CQ4.1 arcs inverted
  with respect to the diagram in Enrico's thesis;
* two on analogy, where it does not implement the return from CQ2 ``no`` to CQ1.1;
* three where the pipeline recorded an error instead of a verdict and one CQ
  answer is missing, so our walk stops without a verdict.

Skips when ``PRIOR_RUNS_DIR`` is missing or lacks the four pipeline runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from argfallacy.labels import normalize_label, normalize_scheme_label
from argfallacy.schemes import traverse

MODELS = ("deepseek", "gpt-5", "llama3.3_70B", "qwen3.8_27b")
EXPECTED_PAIRS = 2428
EXPECTED_AGREE = 2416
EXPECTED_DIVERGENCES = {
    # (run, id in results.csv, scheme, verdict of the prior pipeline, ours)
    ("deepseek", 95, "popular_opinion", "invalid", None),
    ("deepseek", 101, "popular_opinion", "invalid", None),
    ("deepseek", 354, "example", "invalid", None),
    ("gpt-5", 256, "analogy", "weak_analogy", "appeal_to_extremes"),
    ("gpt-5", 277, "analogy", "weak_analogy", "appeal_to_extremes"),
    ("gpt-5", 388, "expert_opinion", "non_sequitur", "good_argumentation"),
    ("gpt-5", 390, "expert_opinion", "good_argumentation", "non_sequitur"),
    ("gpt-5", 399, "expert_opinion", "good_argumentation", "non_sequitur"),
    ("gpt-5", 414, "expert_opinion", "non_sequitur", "good_argumentation"),
    ("gpt-5", 417, "expert_opinion", "non_sequitur", "good_argumentation"),
    ("llama3.3_70B", 100, "expert_opinion", "non_sequitur", "good_argumentation"),
    ("llama3.3_70B", 561, "expert_opinion", "non_sequitur", "good_argumentation"),
}


@pytest.fixture(scope="module")
def pipeline_runs(prior_dir):
    paths = {model: prior_dir / model / "pipeline" / "results.csv" for model in MODELS}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        pytest.skip(f"PRIOR_RUNS_DIR lacks the pipeline runs: {missing}")
    return {model: pd.read_csv(path) for model, path in paths.items()}


def _answers(group: pd.DataFrame, scheme_id: str) -> dict[str, str]:
    """The recorded answers keyed as in our YAML: ``cq2_1`` becomes ``CQ2.1``."""
    answers: dict[str, str] = {}
    for _, row in group.iterrows():
        if pd.isna(row["cq_number"]):
            continue
        cq_id = str(row["cq_number"]).upper().replace("_", ".")
        answer = str(row["cq_answer"]).strip().lower()
        if (scheme_id, cq_id) == ("ad_hominem", "CQ1"):
            # the prior pipeline asked CQ1 as yes or no; yes takes its first alternative, negative
            answer = {"yes": "negative", "no": "positive"}.get(answer, answer)
        answers[cq_id] = answer
    return answers


@pytest.fixture(scope="module")
def comparison(pipeline_runs, schemes, vocab, scheme_vocab):
    pairs = 0
    divergences: set[tuple] = set()
    for model, frame in pipeline_runs.items():
        for row_id, group in frame.groupby("id", sort=False):
            pairs += 1
            first = group.iloc[0]
            scheme_id = normalize_scheme_label(str(first["predicted_scheme"]), scheme_vocab)
            theirs = normalize_label(str(first["predicted_fallacy"]), vocab).id
            ours = traverse(schemes[scheme_id], _answers(group, scheme_id)).verdict
            if ours != theirs:
                divergences.add((model, int(row_id), scheme_id, theirs, ours))
    return pairs, divergences


def test_every_item_of_every_run_is_compared(comparison):
    assert comparison[0] == EXPECTED_PAIRS


def test_2416_of_2428_verdicts_agree(comparison):
    pairs, divergences = comparison
    assert pairs - len(divergences) == EXPECTED_AGREE


def test_the_twelve_divergences_are_the_known_ones(comparison):
    assert comparison[1] == EXPECTED_DIVERGENCES
