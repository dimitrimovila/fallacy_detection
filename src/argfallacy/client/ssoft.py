"""The sequential stage: replay recorded answers and ask the next missing level."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..aggregate.burden import BURDEN_DIR, load_burden, routes
from ..aggregate.soft import source_of
from ..aggregate.ssoft import NodeCall, expand, node_masses
from ..parse.tables import parse_row, summarise_ssoft
from ..prompts import SSOFT, render_ssoft
from ..prompts.render import answers_path, schema_path, template_path
from ..schemes import SchemeError, load_all
from .cache import ResponseCache
from .call import dumps
from .run import (
    MANIFEST_NAME,
    RAW_NAME,
    PlannedCall,
    RunConfig,
    _ask_with_retries,
    _first_start,
    _raw_line,
    _rows,
    _write_manifest,
    build_manifest,
    load_models,
    make_run_id,
    max_tokens_for,
    model_spec,
    read_manifest,
    sampling_for,
    select_models,
)


def _call(config, item, scheme, state, entry, spec):
    if not spec.get("enabled", True):
        raise SchemeError(f"model {entry!r} is disabled in serving/models.yaml")
    sampling = sampling_for(entry, spec)
    sampling.pop("samples")
    rendered = render_ssoft(
        item, scheme, state.cq_id, state.history if config.history else False, config.prompt_version
    )
    return PlannedCall(
        model=entry,
        model_id=spec["model_id"],
        item_id=str(item["item_id"]),
        stage=SSOFT,
        scheme=scheme.scheme_id,
        cq_id=state.cq_id,
        sample_index=0,
        prompt=rendered.text,
        prompt_version=rendered.prompt_version,
        json_schema=rendered.json_schema,
        max_tokens=max_tokens_for(spec, config),
        logprobs=bool(spec.get("supports_logprobs", True)),
        revision=spec.get("revision"),
        tier=spec.get("tier"),
        reasoning=spec.get("reasoning") or {},
        serve_args=tuple(str(arg) for arg in spec.get("serve_args") or ()),
        path=state.path if config.history else "",
        **sampling,
    )


def plan_ssoft(config, items, models=None, schemes=None, only=None) -> list[PlannedCall]:
    """First nodes with history; every CQ without history, one sample per node."""
    models = models if models is not None else load_models()
    schemes = schemes if schemes is not None else load_all(config.schemes_dir)
    load_burden(config.schemes, schemes)
    items = list(items)
    calls = []
    for entry in select_models(config, only):
        spec = model_spec(entry, models)
        for item in items:
            scheme = schemes.get(item.get("gold_scheme"))
            if scheme is None:
                continue
            for cq in [scheme.start] if config.history else scheme.cq_order:
                calls.append(_call(config, item, scheme, NodeCall(cq), entry, spec))
    return calls


def _identity(row):
    if isinstance(row, PlannedCall):
        return row.model, row.item_id, row.cq_id, row.path
    return str(row["model"]), str(row["item_id"]), row["cq_id"], row.get("path") or ""


def _update_answers(parsed, observed, schemes):
    if not parsed:
        return
    for row in summarise_ssoft(pd.DataFrame(parsed), schemes).to_dict("records"):
        observed.setdefault((row["model"], str(row["item_id"])), {})[
            (row["cq_id"], row["path"])
        ] = row


def execute_ssoft(
    config: RunConfig,
    items: Iterable[Mapping[str, Any]],
    run_id: str | None,
    runs_dir: str | Path,
    backend: Any,
    cache: ResponseCache | None,
    models: Mapping[str, Any] | None,
    schemes: Mapping[str, Any] | None,
    max_attempts: int,
    sleep: Any,
    only: Sequence[str] | None,
) -> dict[str, Any]:
    """Replay expansion from successful raw records, then execute each missing level.

    A failed node blocks only its descendants during this invocation. Its error
    remains in the raw file and resume retries it; successful nodes are never
    appended twice. No probability is parsed outside the parser package.
    """
    entries = select_models(config, only)
    models = models if models is not None else load_models()
    schemes = schemes if schemes is not None else load_all(config.schemes_dir)
    items = list(items)
    initial = plan_ssoft(config, items, models, schemes)
    table = load_burden(config.schemes, schemes)
    routing = {sid: routes(scheme, table[sid]) for sid, scheme in schemes.items()}
    run_id = run_id or make_run_id(config.name)
    run_dir = Path(runs_dir) / run_id
    raw_path, manifest_path = run_dir / RAW_NAME, run_dir / MANIFEST_NAME
    files = [
        BURDEN_DIR / f"{config.schemes}.yaml",
        template_path(SSOFT, config.prompt_version),
        schema_path(SSOFT, config.prompt_version),
        answers_path("v3"),
    ]
    policy = {
        "history": config.history,
        "threshold": config.threshold,
        "inputs_sha256": hashlib.sha256(
            dumps([(c.model, c.item_id, c.cq_id, c.key()) for c in initial]).encode()
        ).hexdigest(),
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    previous = read_manifest(run_dir)
    if previous is not None and previous.get("ssoft") != policy:
        raise SchemeError("cannot resume ssoft with a different configuration or expansion policy")
    if previous is None and raw_path.exists() and raw_path.stat().st_size:
        raise SchemeError("cannot resume ssoft raw records without their manifest")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(run_id, config, initial, models)
    manifest["config"] = {k: v for k, v in asdict(config).items() if k != "raw"}
    manifest["ssoft"] = policy
    manifest["planning_during_run"] = config.history
    manifest["started_at"] = _first_start(manifest_path) or datetime.now(UTC).isoformat()
    for entry in manifest["models"].values():
        entry["samples"] = 1
    observed, statuses = {}, {}
    discovered = {_identity(c) for c in initial}
    parsed = []
    for row in _rows(raw_path):
        identity = _identity(row)
        discovered.add(identity)
        if not row.get("error"):
            statuses[identity] = "from_cache" if row.get("from_cache") else "executed"
            parsed.append(parse_row(row, schemes))
        elif identity not in statuses:
            statuses[identity] = "failed"
    _update_answers(parsed, observed, schemes)
    del parsed

    def save(finished=False):
        manifest["calls_planned"] = len(discovered)
        for entry, counts in manifest["calls_per_model"].items():
            counts["planned"] = sum(key[0] == entry for key in discovered)
            for kind in ("executed", "from_cache", "failed"):
                counts[kind] = sum(
                    key[0] == entry and status == kind for key, status in statuses.items()
                )
        for kind in ("executed", "from_cache", "failed"):
            manifest[f"calls_{kind}"] = sum(c[kind] for c in manifest["calls_per_model"].values())
        if finished:
            manifest["finished_at"] = datetime.now(UTC).isoformat()
        _write_manifest(manifest_path, manifest)

    save()
    owns_cache = cache is None
    cache = cache or ResponseCache()
    failed = set()
    try:
        with raw_path.open("a", encoding="utf-8", newline="\n") as sink:
            for entry in entries:
                spec = model_spec(entry, models)
                source = source_of({**spec, "samples": 1})
                limit = max(1, int(spec.get("max_concurrency", 1)))
                while True:
                    candidates = []
                    if not config.history:
                        candidates = [c for c in initial if c.model == entry]
                    else:
                        for item in items:
                            scheme = schemes.get(item.get("gold_scheme"))
                            if scheme is None:
                                continue
                            given = observed.get((entry, str(item["item_id"])), {})

                            def provide(state, given=given, scheme=scheme, source=source):
                                row = given.get((state.cq_id, state.path))
                                return (
                                    None
                                    if row is None
                                    else node_masses(scheme, state.cq_id, row, source)
                                )

                            expansion = expand(
                                scheme, provide, routing[scheme.scheme_id], config.threshold
                            )
                            discovered.update(
                                (entry, str(item["item_id"]), state.cq_id, state.path)
                                for state in expansion.visited
                            )
                            candidates.extend(
                                _call(config, item, scheme, state, entry, spec)
                                for state in expansion.frontier
                            )
                    todo = [
                        c
                        for c in candidates
                        if _identity(c) not in failed
                        and statuses.get(_identity(c)) not in ("executed", "from_cache")
                    ]
                    save()
                    if not todo:
                        break

                    def one(call):
                        request = call.request()
                        hit = cache.get(call.key())
                        if hit is not None:
                            return call, hit, True
                        response = _ask_with_retries(
                            request, backend, max_attempts, sleep or time.sleep
                        )
                        cache.put(call.key(), request, response)
                        return call, response, False

                    parsed = []
                    with ThreadPoolExecutor(max_workers=limit) as pool:
                        for call, response, hit in pool.map(one, todo):
                            row = _raw_line(run_id, config, call, response, hit)
                            row.update(
                                path=call.path,
                                depth=(call.path.count(">") + 1 if call.path else 0)
                                if config.history
                                else None,
                            )
                            sink.write(dumps(row) + "\n")
                            sink.flush()
                            identity = _identity(call)
                            if response.error:
                                statuses[identity] = "failed"
                                failed.add(identity)
                            else:
                                statuses[identity] = "from_cache" if hit else "executed"
                                parsed.append(parse_row(row, schemes))
                    _update_answers(parsed, observed, schemes)
                    save()
    finally:
        save(finished=True)
        if owns_cache:
            cache.close()
    return manifest
