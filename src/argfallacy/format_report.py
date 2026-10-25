"""Format checks on parsed verdict calls, without item tables or reference labels."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .client import read_manifest, read_raw
from .parse import parse_content, validate_against_schema
from .parse.tables import keys_in_order
from .prompts import CANNOT_BE_DETERMINED, DQ, NOT_APPLICABLE, ZEROSHOT
from .schemes import SchemeError


def _identity(row: Any) -> tuple:
    return tuple(
        str(row.get(k) or "") for k in ("item_id", "stage", "scheme", "sample_index", "cache_key")
    )


def _present_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and 0 <= float(value) <= 1
    except (ValueError, TypeError):
        return False


def check_format(
    run_dir: str | Path,
    model: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Failure counts per planned call, and informational shares of outside-arc answers."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    if manifest is None:
        raise SchemeError(f"no manifest.json at {run_dir}")
    stages = manifest.get("config", {}).get("stages", [])
    if not stages or any(s not in (DQ, ZEROSHOT) for s in stages):
        raise SchemeError("format check requires a parsed zeroshot or dq run")
    entries = list(manifest.get("models", {}))
    if model is not None:
        if model not in entries:
            raise SchemeError(f"unknown entry {model!r}; manifest has {entries}")
        entries = [model]
    answers_path = run_dir / "answers.csv"
    if not answers_path.is_file():
        raise SchemeError(f"no answers.csv at {run_dir}; parse the run first")
    answers = pd.read_csv(answers_path, keep_default_na=False)
    raw = {entry: {} for entry in entries}
    for row in read_raw(run_dir):
        if row.get("model") not in raw:
            continue
        own = raw[row["model"]]
        identity = _identity(row)
        # A successful response supersedes failed attempts, which remain in raw.jsonl.
        if identity not in own or not row.get("error"):
            response = row.get("raw_response") or {}
            row["raw_response"] = {k: response.get(k) for k in ("content", "finish_reason")}
            own[identity] = row
    parsed: dict[tuple[str, tuple], list[dict]] = {}
    for row in answers.to_dict("records"):
        parsed.setdefault((row["model"], _identity(row)), []).append(row)
    records, shares = [], []
    for entry in entries:
        planned = int(manifest["calls_per_model"][entry]["planned"])
        calls = raw[entry]
        counts = Counter()
        counts["planned calls present"] = abs(planned - len(calls))
        counts["successful response"] = max(0, planned - len(calls))
        cq_counts: dict[tuple[str, str], Counter] = {}
        for identity, row in calls.items():
            raw_response = row.get("raw_response") or {}
            schema = row.get("json_schema") or {}
            properties = schema.get("properties", {})
            content = raw_response.get("content")
            value = parse_content(content)
            rows = parsed.get((entry, identity), [])
            if not row.get("error"):
                rows = [r for r in rows if not r.get("error")]
            expected_cqs = [
                k
                for k in properties
                if k not in ("answer", "verdict", "confidence", "justification")
            ]
            expected = [*expected_cqs, ""] if row.get("stage") == DQ else [""]
            by_cq = {str(r["cq_id"]): r for r in rows}
            counts["successful response"] += bool(row.get("error"))
            counts["parse_ok"] += (
                len(rows) != len(expected)
                or set(by_cq) != set(expected)
                or any(str(r.get("parse_ok")).lower() != "true" for r in rows)
                or value is None
                or not validate_against_schema(value, schema)
            )
            counts["not truncated"] += raw_response.get("finish_reason") == "length"
            counts["keys in schema order"] += not keys_in_order(content, schema)
            if manifest["models"][entry].get("logprobs_available", False):
                counts["verdict p_label"] += not _present_number(by_cq.get("", {}).get("p_label"))
                missing_mass = False
                for cq in expected_cqs:
                    options = properties[cq].get("enum", [])
                    masses = [by_cq.get(cq, {}).get(f"p_logprob_{a}") for a in options]
                    missing_mass |= not masses or any(not _present_number(m) for m in masses)
                counts["CQ logprob masses"] += missing_mass
            for cq in expected_cqs:
                if cq in by_cq:
                    cq_counts.setdefault((row.get("scheme"), cq), Counter())[
                        by_cq[cq]["answer"]
                    ] += 1
        for check in (
            "planned calls present",
            "successful response",
            "parse_ok",
            "not truncated",
            "keys in schema order",
        ):
            records.append(
                {"model": entry, "check": check, "failed": counts[check], "planned": planned}
            )
        if manifest["models"][entry].get("logprobs_available", False):
            for check in ("verdict p_label", "CQ logprob masses"):
                records.append(
                    {"model": entry, "check": check, "failed": counts[check], "planned": planned}
                )
        for (scheme, cq), counter in cq_counts.items():
            total = sum(counter.values())
            shares.append(
                {
                    "model": entry,
                    "scheme": scheme,
                    "cq_id": cq,
                    "n": total,
                    CANNOT_BE_DETERMINED: counter[CANNOT_BE_DETERMINED] / total,
                    NOT_APPLICABLE: counter[NOT_APPLICABLE] / total,
                }
            )
    return (
        pd.DataFrame(records, columns=["model", "check", "failed", "planned"]),
        pd.DataFrame(
            shares, columns=["model", "scheme", "cq_id", "n", CANNOT_BE_DETERMINED, NOT_APPLICABLE]
        ),
    )
