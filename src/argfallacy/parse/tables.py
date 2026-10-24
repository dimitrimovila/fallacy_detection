"""From raw calls to the two tables every later step reads.

CQ interpretation lives here, including the reads used to plan sequential calls. Two files:
``answers.csv``, one row per call, and ``summary.csv``, one row per question and
model, with the hard answer and the soft answers the aggregators combine: the
masses of every answer from the logprobs, the stated confidence and the shares
over the samples.

Everything here is a second pass over ``raw.jsonl``, which is never rewritten.
Changing the parser costs a re-parse, never a re-run — that separation is the
reason the two live in different packages.

How many samples an entry took decides what its hard answer is.  With one
sample, the hard answer is that sample.  With more, it is the majority over the
valid samples, and every soft answer is a mean or a share over them: the samples
are drawn alike, so sample 0 has no special status.

Stage one answers with a scheme id, a verdict stage with the name its prompt
showed for a verdict; the row carries the terminal id instead, read back through
``labels/fallacies.yaml``, and ``p_label``, the probability of the name chosen,
over the ways of writing it.  At these stages the probabilities of ``summary.csv``
are those of the label chosen.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator

from ..labels import scheme_ids
from ..prompts import (
    CANNOT_BE_DETERMINED,
    JUDGE,
    NOT_APPLICABLE,
    SSOFT,
    STAGE1,
    ZEROSHOT,
    answer_options,
    verdicts,
)
from ..schemes import load_all
from ..schemes.loader import SchemeError, load_vocabulary

INVALID = "invalid"
ALL_OPTIONS = ("yes", "no", "positive", "negative", CANNOT_BE_DETERMINED, NOT_APPLICABLE)
OUTSIDE_ARCS = (CANNOT_BE_DETERMINED, NOT_APPLICABLE)
VERDICT_STAGES = (ZEROSHOT, JUDGE)
"""The stages whose answer is a verdict, a terminal id; their rows are read alike."""
SCHEME_IDS = tuple(scheme_ids())
"""The answers of stage one, from ``labels/schemes.yaml``."""

ANSWERS_COLUMNS = [
    "run_id", "item_id", "stage", "scheme_condition", "scheme", "cq_id", "sample_index",
    "model", "model_id", "prompt_version", "cache_key", "from_cache",
    "answer", "confidence", "justification",
    *[f"p_logprob_{option}" for option in ALL_OPTIONS],
    *[f"p_logprob_{scheme_id}" for scheme_id in SCHEME_IDS],
    "p_label", "p_label_lower_bound",
    "logprob_partial", "parse_ok", "finish_reason", "latency_s", "error",
]
SUMMARY_COLUMNS = [
    "item_id", "stage", "scheme_condition", "scheme", "cq_id", "model",
    "answer", "p_logprob", "p_verbal", "p_sample",
    "n_samples_ok", "idk", "invalid_count", "justification",
    *[f"p_logprob_{option}" for option in ALL_OPTIONS],
    *[f"p_sample_{option}" for option in ALL_OPTIONS],
    "p_label", "p_label_lower_bound",
]


@lru_cache(maxsize=1)
def _vocabulary() -> dict[str, Any]:
    return load_vocabulary(check_schemes=False)


def shown_verdicts(scheme: Any) -> dict[str, str]:
    """Name shown in the zero-shot prompt -> terminal id, for one scheme."""
    return {v.shown: v.terminal for v in verdicts(scheme, _vocabulary())}


def options_for(
    row: Mapping[str, Any], schemes: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """What this call was allowed to answer.

    Stage one answers with a scheme id, a verdict stage with the names its
    prompt showed for the verdicts of the scheme.
    """
    if row.get("stage") == STAGE1:
        return SCHEME_IDS
    schemes = schemes if schemes is not None else load_all()
    scheme = schemes.get(row.get("scheme"))
    if row.get("stage") in VERDICT_STAGES:
        if scheme is None:
            raise SchemeError(f"{row.get('stage')} row of item {row.get('item_id')!r} has no "
                              f"known scheme: {row.get('scheme')!r}")
        return tuple(shown_verdicts(scheme))
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
    start = _value_start(tokens, key)
    if start is None:
        return None
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


def _value_start(tokens: list[str], key: str) -> int | None:
    """Offset, in the joined tokens, where the value of the last ``"key": "`` begins."""
    matches = list(re.finditer('"' + re.escape(key) + r'"\s*:\s*"', "".join(tokens)))
    return matches[-1].end() if matches else None


def _compatible(written: str, options: Iterable[str]) -> set[str]:
    """The labels a value written so far can still become.

    Once the closing quote is written the value is over, and only the label equal to
    it is left.
    """
    if '"' in written:
        value = written[: written.index('"')]
        return {option for option in options if option == value}
    return {option for option in options if option.startswith(written)}


def label_probability(
    logprobs: Mapping[str, Any] | None, options: Iterable[str], chosen: str,
    key: str = "answer",
) -> tuple[float | None, bool]:
    """Probability of the label chosen, over the ways of writing it, and whether it is a bound.

    A model can write the same label split into tokens in different ways, so the
    product along the path it wrote undercounts the label.  Reading the value
    token by token, k is the first position where the text written so far, that
    token included, can only become ``chosen``.  The probability is the product of
    the tokens chosen before k, times the sum, at k, of every alternative in
    ``top_logprobs`` that, after the text written before k, can only become
    ``chosen``, the token chosen included.  After k the grammar admits nothing but
    ``chosen``, so nothing more is multiplied.

    Before k the text is shared by several labels.  An alternative there that is
    another way of writing a prefix of ``chosen`` reaches it by a path this does
    not follow: the probability is then a lower bound, and the flag says so.  None
    where there is no value, the answer is not a label, or the value never singles
    it out.
    """
    options = list(options)
    if chosen not in options or answer_position(logprobs, options, key) is None:
        return None, False
    content = logprobs.get("content") or []
    tokens = [str(entry.get("token", "")) for entry in content]
    start = _value_start(tokens, key)
    total, offset, written, bound = 0.0, 0, "", False
    for entry, token in zip(content, tokens, strict=True):
        token_end = offset + len(token)
        if token_end <= start:
            offset = token_end
            continue
        # the first token of the value may carry what comes before it, the quote say
        lead = token[: max(0, start - offset)]
        piece = token[len(lead):]
        offset = token_end
        alternatives = [
            (str(alt.get("token", ""))[len(lead):], float(alt.get("logprob", -math.inf)))
            for alt in entry.get("top_logprobs") or []
            if str(alt.get("token", "")).startswith(lead) and str(alt.get("token", "")) != lead
        ]
        chosen_logprob = float(entry.get("logprob", -math.inf))
        if _compatible(written + piece, options) == {chosen}:
            mass = sum(math.exp(logprob) for text, logprob in alternatives
                       if _compatible(written + text, options) == {chosen})
            if not any(text == piece for text, _ in alternatives):
                mass += math.exp(chosen_logprob)
            # the server's logprobs are rounded, so a label that takes all the mass can
            # sum to a hair above 1
            return min(1.0, max(0.0, math.exp(total) * mass)), bound
        if any(text != piece and chosen in _compatible(written + text, options)
               for text, _ in alternatives):
            bound = True
        total += chosen_logprob
        written += piece
        if '"' in written:
            break
    return None, bound


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

    p_label, p_label_bound = None, None
    if row.get("stage") in VERDICT_STAGES:
        # the name shown goes back to its terminal; no masses over the first token,
        # since a name may take several tokens
        p_label, p_label_bound = label_probability(raw.get("logprobs"), options, answer, key)
        if parse_ok:
            answer = shown_verdicts((schemes or load_all())[row["scheme"]])[answer]
        masses, partial = {}, False
    else:
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
        "p_label": p_label,
        "p_label_lower_bound": p_label_bound,
        "logprob_partial": partial,
        "parse_ok": parse_ok,
        "finish_reason": raw.get("finish_reason"),
        "latency_s": row.get("latency_s"),
        "error": row.get("error"),
    }
    # the masses are over the answers at stage two, over the scheme ids at stage one
    for option in (*ALL_OPTIONS, *SCHEME_IDS):
        out[f"p_logprob_{option}"] = masses.get(option)
    if row.get("stage") == SSOFT:
        out["path"] = row.get("path") or ""
        out["depth"] = row.get("depth", len(out["path"].split(">")) if out["path"] else 0)
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
    parsed = [parse_row(row, schemes) for row in rows]
    extra = ["path", "depth"] if any(row["stage"] == SSOFT for row in parsed) else []
    return pd.DataFrame(parsed, columns=[*ANSWERS_COLUMNS, *extra])


def valid_samples(group: pd.DataFrame) -> pd.DataFrame:
    """The samples that count: parsed, and not cut off by the token limit."""
    return group[group["parse_ok"].astype(bool) & (group["finish_reason"] != "length")]


def majority(group: pd.DataFrame) -> str:
    """The hard answer of a question asked several times.

    The most frequent answer among the valid samples.  A tie goes to the answer
    with the highest mean confidence over the samples that gave it, and a tie
    that remains to the answer given at the lowest sample index.  No valid
    sample: ``invalid``.
    """
    valid = valid_samples(group)
    if valid.empty:
        return INVALID
    ranked = valid.groupby("answer").agg(
        n=("answer", "size"),
        confidence=("confidence", lambda c: c.astype(float).mean()),
        first=("sample_index", "min"),
    )
    ranked["confidence"] = ranked["confidence"].fillna(-math.inf)
    ranked = ranked.sort_values(["n", "confidence", "first"], ascending=[False, False, True])
    return str(ranked.index[0])


def _number(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _mean(values: Iterable[float | None]) -> float | None:
    numbers = [value for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _oriented(row: pd.Series, yes_like: str) -> float | None:
    """The stated confidence as a probability of the first admitted answer.

    None for an answer outside the arcs: reading ``na`` or ``cannot_be_determined``
    as a low probability of yes would decide, silently, that they mean no.
    """
    confidence = _number(row["confidence"])
    if confidence is None or row["answer"] in (*OUTSIDE_ARCS, INVALID):
        return None
    return confidence / 100.0 if row["answer"] == yes_like else 1.0 - confidence / 100.0


def _answer_probabilities(
    valid: pd.DataFrame, zero_row: pd.Series | None, options: tuple[str, ...], several: bool
) -> dict[str, Any]:
    """Stage two: the masses and shares of every answer, and the three probabilities of yes.

    The three are oriented on the first admitted answer (``yes``, or ``positive`` for
    ad hominem CQ1).  With one sample they are those of sample 0; with several,
    means and shares over the valid samples.
    """
    yes_like = options[0]
    admitted = [option for option in ALL_OPTIONS if option in options]
    columns = [f"p_logprob_{option}" for option in admitted]
    out: dict[str, Any] = {}
    if not several:
        if zero_row is not None:
            out.update({column: _number(zero_row[column]) for column in columns})
            out["p_verbal"] = _oriented(zero_row, yes_like) if zero_row["parse_ok"] else None
        out["p_logprob"] = out.get(f"p_logprob_{yes_like}")
        return out

    with_masses = valid[valid[columns].notna().all(axis=1)]
    if len(with_masses):
        out.update({column: float(with_masses[column].astype(float).mean())
                    for column in columns})
    if len(valid):
        out.update({f"p_sample_{option}": float((valid["answer"] == option).mean())
                    for option in admitted})
    out["p_logprob"] = out.get(f"p_logprob_{yes_like}")
    out["p_verbal"] = _mean(_oriented(row, yes_like) for _, row in valid.iterrows())
    out["p_sample"] = out.get(f"p_sample_{yes_like}")
    return out


def _label_probabilities(
    stage: str, valid: pd.DataFrame, zero_row: pd.Series | None, several: bool, hard: str
) -> dict[str, Any]:
    """Stage one and the verdict stages: the probabilities of the label chosen.

    With several samples, a sample that chose another label counts 0 in
    ``p_label`` and ``p_verbal``; ``p_logprob`` (stage one) is the mean mass of the
    hard answer.  A sample with the hard answer and no ``p_label`` is left out, and
    ``p_label`` stays empty when none of them has one.
    """
    if hard == INVALID or zero_row is None:
        return {}
    verdict = stage in VERDICT_STAGES
    if not several:
        confidence = _number(zero_row["confidence"])
        out = {"p_verbal": None if confidence is None else confidence / 100.0}
        if stage == STAGE1:
            out["p_logprob"] = _number(zero_row[f"p_logprob_{hard}"])
        if verdict and _number(zero_row["p_label"]) is not None:
            out["p_label"] = float(zero_row["p_label"])
            out["p_label_lower_bound"] = bool(zero_row["p_label_lower_bound"])
        return out

    chosen = valid[valid["answer"] == hard]
    others = [0.0] * (len(valid) - len(chosen))
    out = {"p_sample": len(chosen) / len(valid)}
    stated = [_number(c) for c in chosen["confidence"]]
    out["p_verbal"] = _mean([None if c is None else c / 100.0 for c in stated] + others)
    if stage == STAGE1:
        masses = [f"p_logprob_{scheme_id}" for scheme_id in SCHEME_IDS]
        with_masses = valid[valid[masses].notna().all(axis=1)]
        out["p_logprob"] = _mean(_number(v) for v in with_masses[f"p_logprob_{hard}"])
    if verdict:
        used = chosen[chosen["p_label"].notna()]
        if len(used):
            out["p_label"] = _mean([*used["p_label"].astype(float)] + others)
            out["p_label_lower_bound"] = bool(used["p_label_lower_bound"].astype(bool).any())
    return out


def _justification(
    valid: pd.DataFrame, zero_row: pd.Series | None, several: bool, hard: str
) -> str | None:
    """That of sample 0 with one sample; with several, that of the valid sample with the
    lowest index that gave the hard answer."""
    if hard == INVALID:
        return None
    if not several:
        value = None if zero_row is None else zero_row["justification"]
    else:
        chosen = valid[valid["answer"] == hard]
        value = chosen["justification"].iloc[0] if len(chosen) else None
    return None if value is None or pd.isna(value) else str(value)


def summarise(
    answers: pd.DataFrame, schemes: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """One row per question and model: the hard answer and the soft answers.

    At stage two, ``p_logprob_<answer>`` and ``p_sample_<answer>`` give the masses and
    the shares of every admitted answer, and ``p_logprob``, ``p_verbal`` and
    ``p_sample`` the probability of the *first* admitted answer, so the three
    always point the same way: a confident ``no`` is a low probability of yes.  At
    stage one and at the verdict stages the probabilities are those of the label
    chosen, and the verdict stages add ``p_label``.

    An entry that answered once has everything from sample 0.  An entry that
    answered several times has the majority of :func:`majority` as its hard answer
    and every soft answer from its valid samples.

    ``na`` sets no ``idk`` — that flag stays exactly ``answer ==
    cannot_be_determined`` — so the two stay distinguishable by ``answer``.
    """
    schemes = schemes if schemes is not None else load_all()
    answers = answers[answers["stage"] != SSOFT]
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

        group = group.sort_values("sample_index")
        zero = group[group["sample_index"] == 0]
        zero_row = zero.iloc[0] if len(zero) else None
        several = bool((group["sample_index"] > 0).any())

        valid = valid_samples(group)
        if several:
            hard = majority(group)
        else:
            hard = zero_row["answer"] if zero_row is not None else INVALID

        if stage == STAGE1 or stage in VERDICT_STAGES:
            probabilities = _label_probabilities(stage, valid, zero_row, several, hard)
        else:
            probabilities = _answer_probabilities(valid, zero_row, options, several)
        records.append({
            "item_id": item_id, "stage": stage, "scheme_condition": condition,
            "scheme": scheme, "cq_id": cq_id, "model": model,
            "answer": hard,
            "n_samples_ok": len(valid),
            "idk": hard == CANNOT_BE_DETERMINED,
            "invalid_count": int((group["answer"] == INVALID).sum()),
            "justification": _justification(valid, zero_row, several, hard),
            **probabilities,
        })
    return pd.DataFrame(records, columns=SUMMARY_COLUMNS).sort_values(
        ["model", "item_id", "stage", "cq_id"]
    ).reset_index(drop=True)


def summarise_ssoft(
    answers: pd.DataFrame, schemes: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Stage-two probabilities per path, keeping successful retries once per node."""
    from ..prompts import STAGE2

    columns = [*SUMMARY_COLUMNS, "path", "depth"]
    own = answers[answers["stage"] == SSOFT].copy()
    if own.empty:
        return pd.DataFrame(columns=columns)
    keys = ["item_id", "scheme_condition", "scheme", "cq_id", "model", "path", "sample_index"]
    # A successful raw record supersedes failed attempts, including after resume.
    own["_failed"] = own["error"].notna() & (own["error"] != "")
    own = own.sort_values("_failed", ascending=False, kind="stable").drop_duplicates(
        keys, keep="last")
    frames = []
    for (path, depth), group in own.groupby(["path", "depth"], dropna=False, sort=False):
        frame = summarise(group.assign(stage=STAGE2), schemes)
        frame["stage"], frame["path"], frame["depth"] = SSOFT, path, depth
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[columns].sort_values(
        ["model", "item_id", "path", "cq_id"]).reset_index(drop=True)


def parse_run(run_dir: str | Path, schemes: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Read a run's raw file and write ``answers.csv`` and ``summary.csv`` beside it.

    The diagrams are those of the folder the manifest of the run names, so the
    answers are read against the questions they were asked.
    """
    from ..client.run import read_manifest, read_raw, schemes_dir_of

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SchemeError(f"no run at {run_dir}")
    if schemes is None:
        schemes = load_all(schemes_dir_of(read_manifest(run_dir)))

    answers = parse_answers(read_raw(run_dir), schemes)
    summary = summarise(answers, schemes)
    written = {}
    for name, frame in (("answers.csv", answers), ("summary.csv", summary)):
        path = run_dir / name
        frame.to_csv(path, index=False, lineterminator="\n")
        written[name] = path
    if not answers.empty and (answers["stage"] == SSOFT).any():
        path = run_dir / "ssoft.csv"
        summarise_ssoft(answers, schemes).to_csv(path, index=False, lineterminator="\n")
        written["ssoft.csv"] = path
    return written
