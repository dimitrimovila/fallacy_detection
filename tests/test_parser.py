"""The parser for the aggregators, on lines of ``raw.jsonl`` written by hand.

Every value checked here is worked out by hand from the lines, so a change in how
the parser averages or orients shows up as a failed assertion.  The logprob blocks
have the shape of the fake backend: the tokens of the object, with the alternatives
at the token where the value begins.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from argfallacy.eval import run_report
from argfallacy.labels import scheme_ids
from argfallacy.parse import ALL_OPTIONS, parse_answers, summarise
from argfallacy.prompts import STAGE1, STAGE2, ZEROSHOT, answer_schema, verdict_schema
from argfallacy.schemes import traverse
from fakes.llm import first_token


def _block(key: str, chosen: str, masses: dict[str, float]) -> dict:
    """The tokens of ``{"key": "chosen", ...}``, with ``masses`` at the value token."""
    alternatives = [{"token": first_token(t), "logprob": math.log(p)} for t, p in masses.items()]
    return {"content": [
        {"token": '{"', "logprob": -0.01, "top_logprobs": []},
        {"token": key, "logprob": -0.01, "top_logprobs": []},
        {"token": '": "', "logprob": -0.01, "top_logprobs": []},
        {"token": first_token(chosen), "logprob": math.log(masses.get(chosen, 0.5)),
         "top_logprobs": alternatives},
        {"token": '", "confidence": 1, "justification": "x"}', "logprob": -0.01,
         "top_logprobs": []},
    ]}


def _line(schemes, *, stage=STAGE2, model="m", item="i1", scheme="analogy", cq="CQ1",
          sample=0, answer="yes", confidence=80, masses=None, finish="stop", content=None,
          logprobs=None) -> dict:
    key = "scheme" if stage == STAGE1 else "answer"
    if content is None:
        content = json.dumps({key: answer, "confidence": confidence,
                              "justification": f"j{sample}"})
    if stage == STAGE2:
        schema = answer_schema(schemes[scheme], cq)
    elif stage == ZEROSHOT:
        schema = verdict_schema(schemes[scheme])
    else:
        schema = None
    if logprobs is None and masses is not None:
        logprobs = _block(key, answer, masses)
    return {
        "run_id": "r", "item_id": item, "stage": stage, "scheme_condition": "gold",
        "scheme": None if stage == STAGE1 else scheme,
        "cq_id": cq if stage == STAGE2 else "", "sample_index": sample, "model": model,
        "model_id": "x", "prompt_version": "v", "params": {}, "json_schema": schema,
        "cache_key": f"{model}{item}{stage}{cq}{sample}", "from_cache": False,
        "latency_s": 0.1, "error": None,
        "raw_response": {"content": content, "finish_reason": finish, "logprobs": logprobs},
    }


def _tables(schemes, lines):
    answers = parse_answers(lines, schemes)
    return answers, summarise(answers, schemes)


def _row(summary, **keys) -> pd.Series:
    mask = pd.Series(True, index=summary.index)
    for key, value in keys.items():
        mask &= summary[key] == value
    assert mask.sum() == 1, keys
    return summary[mask].iloc[0]


FOUR = {"yes": 0.7, "no": 0.2, "cannot_be_determined": 0.05, "na": 0.05}


# --------------------------------------------------------------- one sample


def test_one_sample_keeps_the_values_of_the_minimal_parser(schemes):
    lines = [
        _line(schemes, cq="CQ1", answer="yes", confidence=80, masses=FOUR),
        _line(schemes, cq="CQ2", answer="no", confidence=70, masses=FOUR),
        _line(schemes, cq="CQ3", answer="cannot_be_determined", confidence=60, masses=FOUR),
        _line(schemes, cq="CQ4", answer="na", confidence=90, masses=FOUR),
        _line(schemes, scheme="ad_hominem", cq="CQ1", answer="negative", confidence=90,
              masses={"positive": 0.1, "negative": 0.9}),
        _line(schemes, item="i2", cq="CQ1", content="no object here"),
    ]
    answers, summary = _tables(schemes, lines)

    # hard answer, p_logprob, p_verbal and p_sample as the minimal parser gave them
    expected = {  # (scheme, item, cq): (answer, p_logprob, p_verbal)
        ("analogy", "i1", "CQ1"): ("yes", 0.7, 0.8),
        ("analogy", "i1", "CQ2"): ("no", 0.7, 0.3),
        ("analogy", "i1", "CQ3"): ("cannot_be_determined", 0.7, None),
        ("analogy", "i1", "CQ4"): ("na", 0.7, None),
        ("ad_hominem", "i1", "CQ1"): ("negative", 0.1, 0.1),
        ("analogy", "i2", "CQ1"): ("invalid", None, None),
    }
    for (scheme, item, cq), (answer, p_logprob, p_verbal) in expected.items():
        row = _row(summary, scheme=scheme, item_id=item, cq_id=cq)
        assert row["answer"] == answer
        for column, value in (("p_logprob", p_logprob), ("p_verbal", p_verbal)):
            if value is None:
                assert pd.isna(row[column]), (cq, column)
            else:
                assert row[column] == pytest.approx(value, abs=1e-9), (cq, column)
        assert pd.isna(row["p_sample"])
        assert row["idk"] == (answer == "cannot_be_determined")

        # the masses of sample 0 as they are, empty where the question does not admit them
        zero = _row(answers, scheme=scheme, item_id=item, cq_id=cq)
        admitted = [o for o in ALL_OPTIONS if pd.notna(zero[f"p_logprob_{o}"])]
        for option in ALL_OPTIONS:
            if option in admitted:
                assert row[f"p_logprob_{option}"] == zero[f"p_logprob_{option}"]
            else:
                assert pd.isna(row[f"p_logprob_{option}"])
            assert pd.isna(row[f"p_sample_{option}"])
        if answer == "invalid":
            assert pd.isna(row["justification"])
        else:
            assert row["justification"] == "j0"
    assert pd.isna(_row(summary, scheme="ad_hominem", cq_id="CQ1")["p_logprob_yes"])


# ---------------------------------------------------------- several samples


def test_several_samples_average_over_the_valid_ones(schemes):
    masses = {
        0: {"yes": 0.2, "no": 0.7, "cannot_be_determined": 0.05, "na": 0.05},
        1: {"yes": 0.9, "no": 0.05, "cannot_be_determined": 0.025, "na": 0.025},
        2: {"yes": 0.1, "no": 0.1, "cannot_be_determined": 0.1, "na": 0.7},
        3: {"yes": 0.99, "no": 0.005, "cannot_be_determined": 0.0025, "na": 0.0025},
        4: {"yes": 0.6, "no": 0.3, "cannot_be_determined": 0.05, "na": 0.05},
    }
    lines = [
        _line(schemes, sample=0, answer="no", confidence=70, masses=masses[0]),
        # cut off by max_tokens: parsed, but not a valid sample
        _line(schemes, sample=1, answer="yes", confidence=95, masses=masses[1], finish="length"),
        _line(schemes, sample=2, answer="na", confidence=60, masses=masses[2]),
        # outside its JSON schema: invalid, though its masses can be read
        _line(schemes, sample=3, answer="yes", confidence=150, masses=masses[3]),
        _line(schemes, sample=4, answer="yes", confidence=80, masses=masses[4]),
        # valid, with no logprobs
        _line(schemes, sample=5, answer="yes", confidence=50),
    ]
    answers, summary = _tables(schemes, lines)
    assert answers.loc[answers["sample_index"] == 3, "p_logprob_yes"].notna().all()
    row = summary.iloc[0]

    assert row["answer"] == "yes"
    assert row["n_samples_ok"] == 4 and row["invalid_count"] == 1
    used = [masses[0], masses[2], masses[4]]  # valid and with masses
    for option in ("yes", "no", "cannot_be_determined", "na"):
        mean = sum(m[option] for m in used) / 3
        assert row[f"p_logprob_{option}"] == pytest.approx(mean, abs=1e-9)
    assert row["p_logprob"] == pytest.approx(row["p_logprob_yes"])
    assert sum(row[f"p_logprob_{o}"] for o in ("yes", "no", "cannot_be_determined", "na")
               ) == pytest.approx(1.0, abs=1e-9)
    assert pd.isna(row["p_logprob_positive"]) and pd.isna(row["p_sample_negative"])

    shares = {"yes": 2 / 4, "no": 1 / 4, "cannot_be_determined": 0.0, "na": 1 / 4}
    for option, share in shares.items():
        assert row[f"p_sample_{option}"] == pytest.approx(share)
    assert row["p_sample"] == pytest.approx(0.5)

    # no at 70 is 0.3 of yes; yes at 80 and at 50; the na adds nothing
    assert row["p_verbal"] == pytest.approx((0.3 + 0.8 + 0.5) / 3)
    # sample 1 gave yes first, but it was cut off
    assert row["justification"] == "j4"


def test_several_samples_with_no_masses_and_no_answer_on_an_arc(schemes):
    lines = [_line(schemes, sample=n, answer="cannot_be_determined", confidence=40)
             for n in range(3)]
    row = _tables(schemes, lines)[1].iloc[0]
    assert row["answer"] == "cannot_be_determined" and row["idk"]
    assert all(pd.isna(row[f"p_logprob_{o}"]) for o in ALL_OPTIONS)
    assert pd.isna(row["p_logprob"]) and pd.isna(row["p_verbal"])
    assert row["p_sample_cannot_be_determined"] == 1.0 and row["p_sample_yes"] == 0.0
    assert row["justification"] == "j0"


def test_no_valid_sample_leaves_the_shares_empty(schemes):
    lines = [_line(schemes, sample=n, answer="yes", masses=FOUR, finish="length")
             for n in range(2)]
    row = _tables(schemes, lines)[1].iloc[0]
    assert row["answer"] == "invalid"
    assert all(pd.isna(row[f"p_sample_{o}"]) for o in ALL_OPTIONS)
    assert all(pd.isna(row[f"p_logprob_{o}"]) for o in ALL_OPTIONS)
    assert pd.isna(row["justification"])


# ---------------------------------------------------------------- stage one


def test_stage_one_reads_the_probabilities_of_the_scheme_chosen(schemes):
    lines = [
        _line(schemes, stage=STAGE1, model="one", answer="analogy", confidence=80,
              masses={"analogy": 0.7, "example": 0.2, "none": 0.1}),
        _line(schemes, stage=STAGE1, model="five", sample=0, answer="example", confidence=70,
              masses={"example": 0.5, "analogy": 0.4, "none": 0.1}),
        _line(schemes, stage=STAGE1, model="five", sample=1, answer="analogy", confidence=90,
              masses={"analogy": 0.8, "example": 0.2}),
        _line(schemes, stage=STAGE1, model="five", sample=2, answer="analogy", confidence=60,
              masses={"analogy": 0.6, "example": 0.3, "none": 0.1}),
        _line(schemes, stage=STAGE1, model="five", sample=3, answer="analogy", confidence=99,
              masses={"analogy": 0.99, "none": 0.01}, finish="length"),
    ]
    lines += [_line(schemes, model=m, answer="yes", masses=FOUR) for m in ("one", "five")]
    answers, summary = _tables(schemes, lines)

    columns = [f"p_logprob_{s}" for s in scheme_ids()]
    assert len(columns) == 9
    stage1 = answers[answers["stage"] == STAGE1]
    assert (stage1[columns].sum(axis=1) - 1.0).abs().max() < 1e-9
    assert stage1.loc[stage1["model"] == "one", "p_logprob_analogy"].iloc[0] == pytest.approx(0.7)
    stage2 = answers[answers["stage"] == STAGE2]
    assert stage2[columns].isna().all().all()
    assert stage1[[f"p_logprob_{o}" for o in ALL_OPTIONS]].isna().all().all()

    one = _row(summary, stage=STAGE1, model="one")
    assert one["answer"] == "analogy"
    assert one["p_logprob"] == pytest.approx(0.7)
    assert one["p_verbal"] == pytest.approx(0.8)
    assert pd.isna(one["p_sample"])

    five = _row(summary, stage=STAGE1, model="five")
    assert five["answer"] == "analogy"
    assert five["p_sample"] == pytest.approx(2 / 3)
    assert five["p_logprob"] == pytest.approx((0.4 + 0.8 + 0.6) / 3)
    assert five["p_verbal"] == pytest.approx((0.0 + 0.9 + 0.6) / 3)
    assert pd.isna(five["p_label"])

    data = run_report.RunData(
        run_id="r", answers=answers, summary=summary,
        manifest={"config": {"models": ["one", "five"]}}, items=pd.DataFrame(),
        annotations=pd.DataFrame(), schemes=schemes, raw_path=None)
    checks = run_report.parser_checks(data)
    assert checks["stage1_p_sample_not_modal"].tolist() == [0, 0]


# ------------------------------------------------------------ verdict stage


def _label_tokens(pieces):
    """A logprob block whose value is written as (token, logprob, alternatives)."""
    head = [('{"', -0.01, []), ("answer", -0.01, []), ('": "', -0.5, [])]
    tail = [(', "confidence": 70, "justification": "x"}', -0.4, [])]
    return {"content": [
        {"token": token, "logprob": logprob,
         "top_logprobs": [{"token": t, "logprob": lp} for t, lp in alternatives]}
        for token, logprob, alternatives in head + pieces + tail
    ]}


NATURE_SHARED = [  # "Appeal to" is shared with Tradition: a lower bound
    ("Appe", -0.2, [("Appe", -0.2), ("Ap", -2.0), ("Ad", -3.0)]),
    ("al to", -0.1, []),
    (" Nature", -0.3, [(" Nature", -0.3), (" Nat", -2.5), (" Tradition", -1.6)]),
    ('"', -0.01, []),
]
NATURE_WHOLE = [  # the whole name in one token: exact
    ("Appeal to Nature", -0.1, [("Appeal to Nature", -0.1), ("Ad", -2.5)]),
    ('"', -0.01, []),
]


def test_a_verdict_stage_reads_the_probability_of_the_label(schemes):
    shared = math.exp(-0.3) * (math.exp(-0.3) + math.exp(-2.5))
    whole = math.exp(-0.1)

    def zeroshot(model, sample, shown, pieces=None, finish="stop"):
        return _line(schemes, stage=ZEROSHOT, model=model, scheme="popular_opinion",
                     sample=sample, answer=shown, confidence=70, finish=finish,
                     logprobs=None if pieces is None else _label_tokens(pieces))

    lines = [
        zeroshot("one", 0, "Appeal to Nature", NATURE_SHARED),
        zeroshot("five", 0, "Appeal to Nature", NATURE_SHARED),
        zeroshot("five", 1, "Appeal to Nature"),  # no logprobs: left out of p_label
        zeroshot("five", 2, "Ad Populum"),
        zeroshot("five", 3, "Appeal to Nature", NATURE_WHOLE),
        zeroshot("five", 4, "Appeal to Nature", NATURE_WHOLE, finish="length"),
    ]
    answers, summary = _tables(schemes, lines)
    assert answers.loc[answers["sample_index"] == 3, "p_label"].iloc[0] == pytest.approx(whole)

    one = _row(summary, model="one")
    assert one["answer"] == "appeal_to_nature"
    assert one["p_label"] == pytest.approx(shared) and one["p_label_lower_bound"]
    assert one["p_verbal"] == pytest.approx(0.7)
    assert pd.isna(one["p_sample"]) and pd.isna(one["p_logprob"])

    five = _row(summary, model="five")
    assert five["answer"] == "appeal_to_nature"
    assert five["p_label"] == pytest.approx((shared + 0.0 + whole) / 3)
    assert five["p_label_lower_bound"]
    assert five["p_verbal"] == pytest.approx((0.7 + 0.7 + 0.0 + 0.7) / 4)
    assert five["p_sample"] == pytest.approx(3 / 4)
    assert pd.isna(five["p_logprob"])
    assert all(pd.isna(five[f"p_logprob_{o}"]) for o in ALL_OPTIONS)

    exact = _tables(schemes, [zeroshot("five", n, "Appeal to Nature", NATURE_WHOLE)
                              for n in range(2)])[1].iloc[0]
    assert exact["p_label"] == pytest.approx(whole) and not exact["p_label_lower_bound"]


# ----------------------------------------------------- the diagram rule


def test_the_diagram_rule_traverses_the_same_hard_answers(schemes):
    """Sample 0 for one sample, the majority of the valid samples for several, as before."""
    scheme = schemes["expert_opinion"]
    hard = {  # the hard answers the minimal parser gave on these lines
        ("one", "a"): {"CQ1": "yes", "CQ2": "no", "CQ3": "yes", "CQ4": "no"},
        ("one", "b"): {"CQ1": "yes", "CQ2": "no", "CQ3": "no", "CQ3.1": "yes"},
        ("five", "a"): {"CQ1": "yes", "CQ2": "yes"},
        ("five", "b"): {"CQ1": "cannot_be_determined"},
    }
    lines = []
    for (model, item), answers in hard.items():
        for cq in scheme.cq_order:
            answer = answers.get(cq, "yes")
            if model == "one":
                lines.append(_line(schemes, model=model, item=item, scheme="expert_opinion",
                                   cq=cq, answer=answer, masses=FOUR))
                continue
            # sample 0 disagrees with the majority of the three valid samples
            other = "no" if answer != "no" else "yes"
            samples = [other, answer, answer, other]
            for index, given in enumerate(samples):
                lines.append(_line(schemes, model=model, item=item, scheme="expert_opinion",
                                   cq=cq, sample=index, answer=given, masses=FOUR,
                                   finish="length" if index == 3 else "stop"))
    answers, summary = _tables(schemes, lines)
    data = run_report.RunData(
        run_id="r", answers=answers, summary=summary,
        manifest={"config": {"models": ["one", "five"]}}, items=pd.DataFrame(),
        annotations=pd.DataFrame(), schemes=schemes, raw_path=None)
    table = run_report.traversals(data, "diagram").set_index(["entry", "item_id"])
    for (model, item), given in hard.items():
        full = {cq: given.get(cq, "yes") for cq in scheme.cq_order}
        verdict = table.loc[(model, item), "verdict"]
        assert (None if pd.isna(verdict) else verdict) == traverse(scheme, full).verdict
    assert pd.isna(table.loc[("five", "b"), "verdict"])
    assert table.loc[("one", "b"), "verdict"] == "strawman"
