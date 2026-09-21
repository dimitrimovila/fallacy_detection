"""From raw calls to a table.  The pilot version, and no further.

The client never looks at what came back; this is where it is read.  Two files:
``answers.csv``, one row per call, and ``summary.csv``, one row per question and
model, carrying the three probabilities of a yes: from the logprobs, from the
stated confidence and from the frequency over the samples.

Everything here is a second pass over ``raw.jsonl``, which is never rewritten.
Changing the parser costs a re-parse, never a re-run — that separation is the
reason the two live in different packages.

This one stops at what the pilot report needs; the full parser comes later.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator

from ..prompts import CANNOT_BE_DETERMINED, NOT_APPLICABLE, STAGE1, answer_options
from ..schemes import load_all
from ..schemes.loader import SchemeError

INVALID = "invalid"
ALL_OPTIONS = ("yes", "no", "positive", "negative", CANNOT_BE_DETERMINED, NOT_APPLICABLE)

ANSWERS_COLUMNS = [
    "run_id", "item_id", "stage", "scheme_condition", "scheme", "cq_id", "sample_index",
    "model", "model_id", "prompt_version", "cache_key", "from_cache",
    "answer", "confidence", "justification",
    *[f"p_logprob_{option}" for option in ALL_OPTIONS],
    "logprob_partial", "parse_ok", "finish_reason", "latency_s", "error",
]
SUMMARY_COLUMNS = [
    "item_id", "stage", "scheme_condition", "scheme", "cq_id", "model",
    "answer", "p_logprob", "p_verbal", "p_sample",
    "n_samples_ok", "idk", "invalid_count",
]


def options_for(
    row: Mapping[str, Any], schemes: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """What this call was allowed to answer.  Stage one answers with a scheme id."""
    if row.get("stage") == STAGE1:
        from ..labels import scheme_ids

        return tuple(scheme_ids())
    schemes = schemes if schemes is not None else load_all()
    scheme = schemes.get(row.get("scheme"))
    if scheme is None or not row.get("cq_id"):
        return ("yes", "no", CANNOT_BE_DETERMINED, NOT_APPLICABLE)
    return answer_options(scheme, row["cq_id"])


def parse_content(content: str | None) -> dict[str, Any] | None:
    """The JSON object the model was asked for, or None.

    The *last* top-level object in the text, because a reasoning model can write
    a draft object while thinking and the real one only at the end.  Taking the
    span from the first ``{`` to the last ``}`` would glue the draft and the
    answer into something that is not JSON at all.  A fence or a sentence around
    the object is fine; prose with no object in it is ``invalid``.
    """
    if not content:
        return None
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    index = 0
    while True:
        index = content.find("{", index)
        if index < 0:
            return found
        try:
            value, end = decoder.raw_decode(content, index)
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(value, dict):
            found = value
        index = end


def _clean(token: Any) -> str:
    """A token with the JSON punctuation around a value taken off."""
    return str(token).strip().lstrip(":").strip().strip('"').strip()


def answer_position(
    logprobs: Mapping[str, Any] | None, options: Iterable[str], key: str = "answer"
) -> int | None:
    """Index of the token where the value of ``key`` begins, inside the final JSON.

    Not the first answer-looking token.  A reasoning model writes thinking before
    the object, and that text can contain ``yes`` or even a draft object of its
    own.  So the token stream is joined back into text, the *last* ``"key": "``
    in it marks where the real value starts, and the token covering that offset
    (or the first non-empty one after it) is the answer token.  If it does not
    begin an admitted answer there is no position: the answer is invalid and has
    no probability to read.
    """
    if not logprobs:
        return None
    tokens = [str(entry.get("token", "")) for entry in (logprobs.get("content") or [])]
    matches = list(re.finditer('"' + re.escape(key) + r'"\s*:\s*"', "".join(tokens)))
    if not matches:
        return None
    start = matches[-1].end()
    options = list(options)
    offset = 0
    for index, token in enumerate(tokens):
        end = offset + len(token)
        if end > start:
            piece = _clean(token[max(0, start - offset):])
            if piece:
                return index if any(option.startswith(piece) for option in options) else None
        offset = end
    return None


def logprob_masses(
    logprobs: Mapping[str, Any] | None, options: Iterable[str], key: str = "answer"
) -> tuple[dict[str, float], bool]:
    """Probability of each admitted answer, renormalised over the admitted ones.

    Read at :func:`answer_position`, never at the first generated token.
    Alternatives that do not begin an admitted answer are dropped, then what is
    left is renormalised.  An admitted answer that never appears gets zero mass
    and the row is marked ``logprob_partial``: the distribution is real but it is
    missing an arm, and an aggregator that averages it should know.
    """
    options = list(options)
    position = answer_position(logprobs, options, key)
    if position is None:
        return {}, False

    entries = (logprobs.get("content") or [])[position].get("top_logprobs") or []
    mass = {option: 0.0 for option in options}
    for entry in entries:
        token = _clean(entry.get("token", ""))
        if not token:
            continue
        for option in options:
            if option.startswith(token):
                mass[option] += math.exp(float(entry.get("logprob", -math.inf)))
                break

    total = sum(mass.values())
    if total <= 0:
        return {}, True
    normalised = {option: value / total for option, value in mass.items()}
    partial = any(value == 0.0 for value in mass.values())
    return normalised, partial


def parse_row(
    row: Mapping[str, Any], schemes: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """One line of ``raw.jsonl`` turned into one row of ``answers.csv``."""
    raw = row.get("raw_response") or {}
    options = options_for(row, schemes)
    key = "scheme" if row.get("stage") == STAGE1 else "answer"

    parsed = parse_content(raw.get("content"))
    answer, confidence, justification, parse_ok = INVALID, None, None, False
    if (parsed is not None and parsed.get(key) in options
            and validate_against_schema(parsed, row.get("json_schema"))):
        answer = str(parsed[key])
        raw_confidence = parsed.get("confidence")
        confidence = int(raw_confidence) if isinstance(raw_confidence, (int, float)) else None
        justification = parsed.get("justification")
        parse_ok = True

    masses, partial = logprob_masses(raw.get("logprobs"), options, key)
    out: dict[str, Any] = {
        "run_id": row.get("run_id"),
        "item_id": row.get("item_id"),
        "stage": row.get("stage"),
        "scheme_condition": row.get("scheme_condition"),
        "scheme": row.get("scheme"),
        "cq_id": row.get("cq_id") or "",
        "sample_index": row.get("sample_index"),
        "model": row.get("model"),
        "model_id": row.get("model_id"),
        "prompt_version": row.get("prompt_version"),
        "cache_key": row.get("cache_key"),
        "from_cache": bool(row.get("from_cache")),
        "answer": answer,
        "confidence": confidence,
        "justification": justification,
        "logprob_partial": partial,
        "parse_ok": parse_ok,
        "finish_reason": raw.get("finish_reason"),
        "latency_s": row.get("latency_s"),
        "error": row.get("error"),
    }
    for option in ALL_OPTIONS:
        out[f"p_logprob_{option}"] = masses.get(option)
    return out


def validate_against_schema(
    parsed: Mapping[str, Any], schema: Mapping[str, Any] | None
) -> bool:
    """Whether the model's object passes the JSON schema its call was given.

    The schema travels with the call in ``raw.jsonl``.  A line without one has
    nothing to be checked against, so only the allowed answers are checked.
    """
    return schema is None or Draft202012Validator(dict(schema)).is_valid(dict(parsed))


def parse_answers(
    rows: Iterable[Mapping[str, Any]], schemes: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    schemes = schemes if schemes is not None else load_all()
    return pd.DataFrame(
        [parse_row(row, schemes) for row in rows], columns=ANSWERS_COLUMNS
    )


def summarise(
    answers: pd.DataFrame, schemes: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """One row per question and model, with the three probabilities.

    ``p_logprob`` and ``p_sample`` are the probability of the *first* admitted
    answer — ``yes``, or ``positive`` for ad hominem CQ1 — so the three numbers
    always point the same way.  ``p_verbal`` is oriented to match: a confident
    ``no`` is a low probability of yes, not a high one.

    ``na`` gets no ``p_verbal``, the same as ``cannot_be_determined``: reading it
    as a low probability of yes would decide, silently, that na means no.  It
    does not set ``idk`` either — that flag stays exactly ``answer ==
    cannot_be_determined`` — so the two stay distinguishable downstream by
    ``answer`` alone, which is the only thing this parser records about them.
    """
    schemes = schemes if schemes is not None else load_all()
    if answers.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    grouped = answers.groupby(
        ["item_id", "stage", "scheme_condition", "scheme", "cq_id", "model"],
        dropna=False,
    )
    records: list[dict[str, Any]] = []
    for keys, group in grouped:
        item_id, stage, condition, scheme, cq_id, model = keys
        options = options_for(
            {"stage": stage, "scheme": scheme, "cq_id": cq_id}, schemes
        )
        yes_like = options[0]

        group = group.sort_values("sample_index")
        zero = group[group["sample_index"] == 0]
        zero_row = zero.iloc[0] if len(zero) else None
        hard = zero_row["answer"] if zero_row is not None else INVALID

        ok = group[group["parse_ok"]]
        n_ok = len(ok)
        p_sample = (ok["answer"] == yes_like).mean() if n_ok else None

        p_logprob = None
        if zero_row is not None:
            value = zero_row.get(f"p_logprob_{yes_like}")
            p_logprob = None if pd.isna(value) else float(value)

        p_verbal = None
        idk = hard == CANNOT_BE_DETERMINED
        if zero_row is not None and zero_row["parse_ok"] and not idk and hard != NOT_APPLICABLE:
            confidence = zero_row["confidence"]
            if confidence is not None and not pd.isna(confidence):
                fraction = float(confidence) / 100.0
                p_verbal = fraction if hard == yes_like else 1.0 - fraction

        records.append({
            "item_id": item_id, "stage": stage, "scheme_condition": condition,
            "scheme": scheme, "cq_id": cq_id, "model": model,
            "answer": hard,
            "p_logprob": p_logprob,
            "p_verbal": p_verbal,
            "p_sample": None if p_sample is None else float(p_sample),
            "n_samples_ok": n_ok,
            "idk": bool(idk),
            "invalid_count": int((group["answer"] == INVALID).sum()),
        })
    return pd.DataFrame(records, columns=SUMMARY_COLUMNS).sort_values(
        ["model", "item_id", "stage", "cq_id"]
    ).reset_index(drop=True)


def parse_run(run_dir: str | Path, schemes: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Read a run's raw file and write ``answers.csv`` and ``summary.csv`` beside it."""
    from ..client.run import read_raw

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SchemeError(f"no run at {run_dir}")
    schemes = schemes if schemes is not None else load_all()

    answers = parse_answers(read_raw(run_dir), schemes)
    summary = summarise(answers, schemes)
    written = {}
    for name, frame in (("answers.csv", answers), ("summary.csv", summary)):
        path = run_dir / name
        frame.to_csv(path, index=False, lineterminator="\n")
        written[name] = path
    return written
