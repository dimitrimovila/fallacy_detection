"""Answers of earlier runs taken over by a new version of the diagrams.

A question of the new version is the same question as one of an earlier version when
its stage-two prompt is the same text apart from the identifier of the question: the
scheme, its variables, the wording, the answers and what each answer means.  The
identifier alone decides nothing.  Such a question is not asked again; ``assemble``
writes the answer table of the new version from the runs that asked each question,
and every row says which run, which CQ id and which prompt version it comes from.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import SCHEMES_ROOT
from ..prompts import STAGE2, render_stage2
from ..schemes import load_all
from ..schemes.loader import SchemeError
from .run import (
    GOLD,
    MANIFEST_NAME,
    RAW_NAME,
    STAGE2_PROMPTS,
    RunConfig,
    questions_asked,
    read_manifest,
    schemes_dir_of,
    schemes_hash,
    schemes_tag,
)

PROMPTS = {"v1": "v2", **STAGE2_PROMPTS}
"""The stage-two prompt version that goes with each folder of ``schemes/``."""

PROBE = {"item_id": "probe", "text": "The text of the argument."}
"""Any text will do: it is the same in the two prompts compared."""

SOURCE_COLUMNS = ["source_run_id", "source_cq_id", "source_prompt_version"]
MODEL_KEYS = ("model_id", "samples", "temperature", "top_p", "top_k", "max_tokens",
              "revision", "tier", "reasoning", "serve_args")
"""What makes two runs of an entry the same calls; display names and limits aside."""


def _without_id(folder: str, scheme: Any, cq_id: str) -> tuple[str, str]:
    rendered = render_stage2(PROBE, scheme, cq_id, PROMPTS[folder])
    marker = f"\n**{cq_id}.** "
    if rendered.text.count(marker) != 1:
        raise SchemeError(f"{scheme.scheme_id} {cq_id}: the prompt does not show the "
                          f"question once as {marker.strip()!r}")
    schema = {k: v for k, v in rendered.json_schema.items() if k != "title"}
    return rendered.text.replace(marker, "\n**CQ.** "), json.dumps(schema, sort_keys=True)


@cache
def _prompts(folder: str) -> dict[tuple[str, str], tuple[str, str]]:
    schemes = load_all(SCHEMES_ROOT / folder)
    return {(scheme_id, cq): _without_id(folder, scheme, scheme.cqs[cq].same_as or cq)
            for scheme_id, scheme in schemes.items() for cq in scheme.cq_order}


def identical_questions(new: str, old: str) -> dict[str, dict[str, str]]:
    """scheme -> CQ of ``schemes/<new>/`` -> the CQ of ``schemes/<old>/`` with its prompt.

    Only the questions with such a match appear; a question that repeats another one of
    its diagram (``same_as``) is never a match itself, and is matched as its first
    question.  Two matches for one question would mean two identical questions in one
    diagram that do not say so, and stop here.
    """
    by_prompt: dict[tuple[str, tuple[str, str]], list[str]] = {}
    repeats = {(scheme_id, cq) for scheme_id, scheme in load_all(SCHEMES_ROOT / old).items()
               for cq in scheme.cq_order if scheme.cqs[cq].same_as is not None}
    for (scheme_id, cq), prompt in _prompts(old).items():
        if (scheme_id, cq) not in repeats:
            by_prompt.setdefault((scheme_id, prompt), []).append(cq)
    out: dict[str, dict[str, str]] = {}
    for (scheme_id, cq), prompt in _prompts(new).items():
        matches = by_prompt.get((scheme_id, prompt), [])
        if len(matches) > 1:
            raise SchemeError(f"{scheme_id} {cq} of {new} has the prompt of {matches} "
                              f"of {old}")
        if matches:
            out.setdefault(scheme_id, {})[cq] = matches[0]
    return out


def questions_to_ask(new: str, old: str) -> dict[str, list[str]]:
    """The CQs of ``schemes/<new>/`` without an identical question in ``schemes/<old>/``."""
    same = identical_questions(new, old)
    return {scheme_id: [cq for cq in scheme.cq_order if cq not in same.get(scheme_id, {})
                        and scheme.cqs[cq].same_as is None]
            for scheme_id, scheme in sorted(load_all(SCHEMES_ROOT / new).items())}


# -------------------------------------------------------------------- assembly
def _summary(run_dir: Path) -> tuple[pd.DataFrame, str]:
    """The stage-two answers of a run: its summary.csv, or its raw.jsonl parsed in memory."""
    from ..parse import parse_answers, summarise
    from .run import read_raw

    path = run_dir / "summary.csv"
    if path.is_file():
        return pd.read_csv(path, dtype={"item_id": str, "cq_id": str}), "summary.csv"
    if not (run_dir / RAW_NAME).is_file():
        raise SchemeError(f"{run_dir.name}: no summary.csv and no {RAW_NAME}")
    schemes = load_all(schemes_dir_of(read_manifest(run_dir)))
    frame = summarise(parse_answers(read_raw(run_dir), schemes), schemes)
    return frame.astype({"item_id": str, "cq_id": str}), f"{RAW_NAME}, parsed in memory"


def _config(manifest: Mapping[str, Any]) -> RunConfig:
    known = {f for f in RunConfig.__dataclass_fields__ if f != "raw"}
    raw = dict(manifest.get("config") or {})
    return RunConfig(raw=raw, **{k: v for k, v in raw.items() if k in known})


def assemble(
    name: str,
    folder: str,
    run_ids: Sequence[str],
    items: pd.DataFrame,
    runs_dir: str | Path,
) -> dict[str, Any]:
    """Write ``runs/<name>/summary.csv`` for the diagrams of ``schemes/<folder>/``.

    For every CQ of those diagrams, every entry of the first run and every item with
    that gold scheme in ``items``, the answer is taken from the first run of
    ``run_ids`` that asked the same question (same prompt apart from the identifier)
    and holds a row for that entry and item.  Every entry of the first run must have
    the same settings in the runs it takes answers from, and every answer must be
    found.  The folder must not exist: nothing earlier is overwritten.
    """
    runs_dir = Path(runs_dir)
    out_dir = runs_dir / name
    if out_dir.exists():
        raise SchemeError(f"{out_dir} already exists; choose another name")
    if not run_ids:
        raise SchemeError("no runs to assemble from")
    schemes = load_all(SCHEMES_ROOT / folder)
    sources = []
    for run_id in run_ids:
        manifest = read_manifest(runs_dir / run_id)
        if manifest is None:
            raise SchemeError(f"no {MANIFEST_NAME} in {runs_dir / run_id}")
        config = _config(manifest)
        if STAGE2 not in config.stages or config.scheme_condition != GOLD:
            raise SchemeError(f"{run_id}: not a stage-two run under the gold scheme")
        run_folder = schemes_dir_of(manifest).name
        if (manifest.get("schemes_version") or {}).get("content_sha256") != schemes_hash(
                SCHEMES_ROOT / run_folder):
            raise SchemeError(f"{run_id}: schemes/{run_folder}/ is not the one it ran on")
        asked = questions_asked(config, load_all(SCHEMES_ROOT / run_folder))
        same = identical_questions(folder, run_folder)
        mapping = {(sid, cq): own for sid, cqs in same.items() for cq, own in cqs.items()
                   if own in asked.get(sid, ())}
        frame, read_from = _summary(runs_dir / run_id)
        frame = frame[(frame["stage"] == STAGE2) & (frame["scheme_condition"] == GOLD)]
        sources.append({"run_id": run_id, "manifest": manifest, "mapping": mapping,
                        "frame": frame, "read_from": read_from,
                        "prompt_version": f"{STAGE2}_{PROMPTS[run_folder]}"})

    entries = list(sources[0]["manifest"]["config"]["models"])
    reference = sources[0]["manifest"]["models"]
    for source in sources:
        for entry in entries:
            own = (source["manifest"].get("models") or {}).get(entry)
            if own is not None and any(own.get(k) != reference[entry].get(k)
                                       for k in MODEL_KEYS):
                raise SchemeError(f"{source['run_id']}: {entry} ran with other settings "
                                  f"than in {sources[0]['run_id']}")

    items = items.astype({"item_id": str})
    frames, used, missing = [], {}, []
    for scheme_id, scheme in sorted(schemes.items()):
        wanted = sorted(items.loc[items["gold_scheme"] == scheme_id, "item_id"])
        for cq in scheme.cq_order:
            for entry in entries:
                left = set(wanted)
                for source in sources:
                    own = source["mapping"].get((scheme_id, cq))
                    if own is None or not left:
                        continue
                    frame = source["frame"]
                    rows = frame[(frame["scheme"] == scheme_id) & (frame["cq_id"] == own)
                                 & (frame["model"] == entry) & frame["item_id"].isin(left)]
                    if rows["item_id"].duplicated().any():
                        raise SchemeError(f"{source['run_id']}: two rows for one item, "
                                          f"{scheme_id} {own} {entry}")
                    if rows.empty:
                        continue
                    left -= set(rows["item_id"])
                    frames.append(rows.assign(
                        cq_id=cq, source_run_id=source["run_id"], source_cq_id=own,
                        source_prompt_version=source["prompt_version"]))
                    used.setdefault(f"{scheme_id} {cq}", {}).setdefault(
                        source["run_id"], own)
                if left:
                    missing.append(f"{scheme_id} {cq} {entry}: {len(left)} items, "
                                   f"for example {sorted(left)[0]}")
    if missing:
        raise SchemeError("no answer in the runs given for: " + "; ".join(missing[:10])
                          + (f"; and {len(missing) - 10} more" if len(missing) > 10 else ""))

    summary = pd.concat(frames, ignore_index=True)
    columns = [c for c in summary.columns if c not in SOURCE_COLUMNS] + SOURCE_COLUMNS
    summary = summary[columns].sort_values(["model", "item_id", "stage", "cq_id"])
    first = sources[0]["manifest"]
    contributing = {run_id for by_run in used.values() for run_id in by_run}
    manifest = {
        "run_id": name,
        "config": {"name": name, "models": entries, "stages": [STAGE2],
                   "scheme_condition": GOLD, "schemes": folder,
                   "prompt_version": PROMPTS[folder]},
        "models": {entry: reference[entry] for entry in entries},
        "prompt_versions": sorted({s["prompt_version"] for s in sources
                                   if s["run_id"] in contributing}),
        "schemes_version": {
            "folder": folder,
            "tag": schemes_tag(SCHEMES_ROOT / folder),
            "files": sorted(p.name for p in (SCHEMES_ROOT / folder).glob("*.yaml")),
            "content_sha256": schemes_hash(SCHEMES_ROOT / folder),
        },
        "generation": first.get("generation"),
        "scheme_condition": GOLD,
        "assembled_at": datetime.now(UTC).isoformat(),
        "assembled_from": [{"run_id": s["run_id"], "read_from": s["read_from"],
                            "prompt_version": s["prompt_version"]} for s in sources],
        "answers_by_cq": used,
        "items": int(len(items)),
        "rows": int(len(summary)),
    }
    out_dir.mkdir(parents=True)
    summary.to_csv(out_dir / "summary.csv", index=False, lineterminator="\n")
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                         encoding="utf-8", newline="\n")
    return manifest
