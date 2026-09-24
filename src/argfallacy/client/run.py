"""Planning a set of calls, running it, and being able to stop half way.

A run is a folder under ``runs/``: a manifest saying exactly what was asked, and
``raw.jsonl``, append-only, one line per call.  Two rules keep it honest.

*Deterministic order.*  Model, then item, then critical question, then sample.
Calls go out concurrently within the per-model limit, but results are written in
plan order, so two runs of the same plan produce the same file.

*Resume without duplicates.*  ``execute`` reads the ``cache_key`` values already
present in ``raw.jsonl`` and appends only what is missing.  A call already in the
cache still gets its line the first time the run writes one, so the file is a
complete record of the run and not just of the calls that happened to be new.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ..paths import REPO_ROOT
from ..prompts import (
    DEFAULT_VERSION,
    STAGE1,
    STAGE2,
    prompt_version,
    render_stage1,
    render_stage2,
)
from ..schemes import load_all
from ..schemes.loader import SchemeError
from .cache import RUNS_DIR, ResponseCache, cache_key
from .call import RawResponse, Request, ask, dumps

SERVING_DIR = REPO_ROOT / "serving"
MODELS_FILE = SERVING_DIR / "models.yaml"
SCHEMES_DIR = REPO_ROOT / "schemes"

GOLD, PREDICTED = "gold", "predicted"
RAW_NAME = "raw.jsonl"
MANIFEST_NAME = "manifest.json"

RETRYABLE = ("429", "500", "502", "503", "504", "timeout", "connection")


# --------------------------------------------------------------------- models
def load_models(path: str | Path = MODELS_FILE) -> dict[str, dict[str, Any]]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["models"]


def model_spec(name: str, models: Mapping[str, Any] | None = None) -> dict[str, Any]:
    models = models if models is not None else load_models()
    if name not in models:
        raise SchemeError(f"unknown model {name!r}; serving/models.yaml has {sorted(models)}")
    return models[name]


def select_models(config: RunConfig, only: Sequence[str] | None = None) -> list[str]:
    """The entries of the configuration that one invocation runs, in its order.

    All of them by default.  A job that serves one entry per launch of the server
    names that entry; a name the configuration does not have stops everything
    before any call, since the job would otherwise send nothing, or send the calls
    of one entry to the server launched for another.
    """
    if not only:
        return list(config.models)
    unknown = [name for name in only if name not in config.models]
    if unknown:
        raise SchemeError(
            f"{unknown} not among the models of the configuration {config.name!r}: "
            f"{config.models}"
        )
    return [name for name in config.models if name in only]


# --------------------------------------------------------------------- config
@dataclass
class RunConfig:
    name: str
    models: list[str]
    stages: list[str] = field(default_factory=lambda: [STAGE1, STAGE2])
    scheme_condition: str = GOLD
    samples: int = 5
    prompt_version: str = DEFAULT_VERSION
    items: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    predictions_from: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__ if f != "raw"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise SchemeError(f"{Path(path).name}: unknown configuration keys {unknown}")
        return cls(raw=data, **{k: v for k, v in data.items() if k in known})

    def temperature(self, sample_index: int) -> float:
        if sample_index == 0:
            return float(self.generation.get("temperature_sample0", 0.0))
        return float(self.generation.get("temperature_rest", 0.7))

    def max_tokens(self) -> int:
        return int(self.generation.get("max_tokens", 512))


def max_tokens_for(spec: Mapping[str, Any], config: RunConfig) -> int:
    """An entry's own ``max_tokens`` wins over the configuration's.

    The entries that answer with reasoning on write their thinking before the
    JSON and need far more room than the others.  Being a generation parameter
    it travels into the cache key, so the same entry at another limit is a
    different call and not a cached answer reused.
    """
    value = spec.get("max_tokens")
    return config.max_tokens() if value is None else int(value)


# ----------------------------------------------------------------------- plan
@dataclass(frozen=True)
class PlannedCall:
    model: str
    model_id: str
    item_id: str
    stage: str
    scheme: str | None
    cq_id: str | None
    sample_index: int
    prompt: str
    prompt_version: str
    json_schema: dict[str, Any] | None
    temperature: float
    max_tokens: int
    logprobs: bool
    revision: str | None = None
    tier: int | None = None
    reasoning: dict[str, Any] | None = None

    def request(self) -> Request:
        return Request(
            model_id=self.model_id,
            prompt=self.prompt,
            prompt_version=self.prompt_version,
            sample_index=self.sample_index,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.sample_index,
            json_schema=self.json_schema,
            logprobs=self.logprobs,
            revision=self.revision,
            tier=self.tier,
            reasoning=self.reasoning,
        )

    def key(self) -> str:
        return cache_key(self.request())


def plan(
    config: RunConfig,
    items: Iterable[Mapping[str, Any]],
    models: Mapping[str, Any] | None = None,
    schemes: Mapping[str, Any] | None = None,
    predictions: Mapping[str, str] | None = None,
    only: Sequence[str] | None = None,
) -> list[PlannedCall]:
    """Every call the configuration asks for, in the order they will be made.

    ``only`` keeps the named entries of the configuration (see
    :func:`select_models`).

    In the ``predicted`` condition, stage two is planned only for the items whose
    predicted scheme differs from the gold one: where the two agree, the calls
    made in the ``gold`` condition answer this condition too, and the parser
    knows it.
    """
    entries = select_models(config, only)
    models = models if models is not None else load_models()
    schemes = schemes if schemes is not None else load_all()
    items = list(items)
    calls: list[PlannedCall] = []

    for model_name in entries:
        spec = model_spec(model_name, models)
        if not spec.get("enabled", True):
            raise SchemeError(
                f"model {model_name!r} is disabled in serving/models.yaml; "
                f"set `enabled: true` there to plan calls for it"
            )
        identity = {
            "revision": spec.get("revision"),
            "tier": spec.get("tier"),
            "reasoning": spec.get("reasoning") or {},
        }
        logprobs = bool(spec.get("supports_logprobs", True))
        max_tokens = max_tokens_for(spec, config)
        for item in items:
            item_id = str(item["item_id"])
            if STAGE1 in config.stages:
                rendered = render_stage1(item, schemes, config.prompt_version)
                for sample in range(config.samples):
                    calls.append(PlannedCall(
                        model=model_name, model_id=spec["model_id"], item_id=item_id,
                        stage=STAGE1, scheme=None, cq_id=None, sample_index=sample,
                        prompt=rendered.text, prompt_version=rendered.prompt_version,
                        json_schema=rendered.json_schema,
                        temperature=config.temperature(sample),
                        max_tokens=max_tokens, logprobs=logprobs,
                        **identity,
                    ))
            if STAGE2 not in config.stages:
                continue

            scheme_id = _scheme_for(item, config, predictions)
            if scheme_id is None or scheme_id not in schemes:
                continue
            scheme = schemes[scheme_id]
            for cq_id in scheme.cq_order:
                rendered = render_stage2(item, scheme, cq_id, config.prompt_version)
                for sample in range(config.samples):
                    calls.append(PlannedCall(
                        model=model_name, model_id=spec["model_id"], item_id=item_id,
                        stage=STAGE2, scheme=scheme_id, cq_id=cq_id, sample_index=sample,
                        prompt=rendered.text, prompt_version=rendered.prompt_version,
                        json_schema=rendered.json_schema,
                        temperature=config.temperature(sample),
                        max_tokens=max_tokens, logprobs=logprobs,
                        **identity,
                    ))
    return calls


def _scheme_for(
    item: Mapping[str, Any],
    config: RunConfig,
    predictions: Mapping[str, str] | None,
) -> str | None:
    gold = item.get("gold_scheme")
    if config.scheme_condition == GOLD:
        return gold
    if config.scheme_condition != PREDICTED:
        raise SchemeError(
            f"scheme_condition must be {GOLD!r} or {PREDICTED!r}, "
            f"not {config.scheme_condition!r}"
        )
    if predictions is None:
        raise SchemeError(
            "the predicted condition needs stage-one predictions; pass them to plan() "
            "or set predictions_from in the configuration"
        )
    predicted = predictions.get(str(item["item_id"]))
    if predicted is None or predicted == gold:
        # the gold-condition calls already answer this item
        return None
    return predicted


def summarise_plan(
    calls: list[PlannedCall], cache: ResponseCache, models: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """How many calls, how many already answered, how long it would take."""
    models = models if models is not None else load_models()
    per_model: dict[str, dict[str, Any]] = {}
    for call in calls:
        entry = per_model.setdefault(
            call.model,
            {"total": 0, "cached": 0, "missing": 0,
             "max_concurrency": int(model_spec(call.model, models).get("max_concurrency", 1))},
        )
        entry["total"] += 1
        if cache.has(call.key()):
            entry["cached"] += 1
        else:
            entry["missing"] += 1
    for entry in per_model.values():
        # a call costs a couple of seconds; the estimate is a rough order, not a promise
        entry["estimated_minutes"] = round(
            entry["missing"] * 2.0 / max(1, entry["max_concurrency"]) / 60, 1
        )
    return {
        "calls": len(calls),
        "cached": sum(e["cached"] for e in per_model.values()),
        "missing": sum(e["missing"] for e in per_model.values()),
        "per_model": per_model,
    }


# -------------------------------------------------------------------- manifest
def schemes_hash(directory: str | Path = SCHEMES_DIR) -> str:
    """One digest over every scheme YAML, so a changed byte is visible."""
    digest = hashlib.sha256()
    for path in sorted(Path(directory).glob("*.yaml")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def schemes_tag(directory: str | Path = SCHEMES_DIR) -> str:
    """The `version` that every scheme YAML declares; the files must agree."""
    tags = {
        yaml.safe_load(path.read_text(encoding="utf-8")).get("version")
        for path in sorted(Path(directory).glob("*.yaml"))
    }
    if len(tags) != 1 or None in tags:
        raise SchemeError(f"the scheme files must declare one shared version, found {tags}")
    return str(tags.pop())


def make_run_id(config_name: str, when: datetime | None = None) -> str:
    when = when or datetime.now(UTC)
    return f"{when.strftime('%Y%m%dT%H%M%SZ')}_{config_name}"


def build_manifest(
    run_id: str,
    config: RunConfig,
    calls: list[PlannedCall],
    models: Mapping[str, Any],
) -> dict[str, Any]:
    used = {c.model for c in calls} or set(config.models)
    return {
        "run_id": run_id,
        "config": config.raw or {
            "name": config.name, "models": config.models, "stages": config.stages,
            "scheme_condition": config.scheme_condition, "samples": config.samples,
        },
        "models": {
            name: {
                "model_id": model_spec(name, models)["model_id"],
                "display_name": model_spec(name, models).get("display_name", name),
                "logprobs_available": bool(
                    model_spec(name, models).get("supports_logprobs", True)
                ),
                "max_concurrency": int(model_spec(name, models).get("max_concurrency", 1)),
                "max_tokens": max_tokens_for(model_spec(name, models), config),
                "revision": model_spec(name, models).get("revision"),
                "tier": model_spec(name, models).get("tier"),
                "reasoning": model_spec(name, models).get("reasoning") or {},
                "serve_args": model_spec(name, models).get("serve_args"),
            }
            for name in sorted(used)
        },
        "prompt_versions": sorted({c.prompt_version for c in calls}) or [
            prompt_version(STAGE1, config.prompt_version),
            prompt_version(STAGE2, config.prompt_version),
        ],
        "schemes_version": {
            "tag": schemes_tag(),
            "files": sorted(p.name for p in SCHEMES_DIR.glob("*.yaml")),
            "content_sha256": schemes_hash(),
        },
        "generation": {
            "temperature_sample0": config.temperature(0),
            "temperature_rest": config.temperature(1),
            "max_tokens": config.max_tokens(),
        },
        "n_samples": config.samples,
        "scheme_condition": config.scheme_condition,
        "started_at": None,
        "finished_at": None,
        "calls_planned": len(calls),
        "calls_executed": 0,
        "calls_from_cache": 0,
        "calls_failed": 0,
        "calls_per_model": {
            name: {"planned": sum(1 for c in calls if c.model == name),
                   "executed": 0, "from_cache": 0, "failed": 0}
            for name in config.models
        },
    }


# -------------------------------------------------------------------- execute
def _rows(path: Path) -> Iterator[dict[str, Any]]:
    """The lines of a ``raw.jsonl`` that parse; a line cut off by a crash is skipped."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _already_written(path: Path) -> set[str]:
    """Keys of the calls this run has already answered.

    A line that recorded a *failure* does not count: the call still has to be
    made, and its error line stays in the file as the record that it was tried
    and went wrong.  So a retried call leaves two lines with the same key, one
    with ``error`` and one without, which is history rather than duplication.
    """
    return {
        row["cache_key"] for row in _rows(path)
        if not row.get("error") and "cache_key" in row
    }


def _count_calls(manifest: dict[str, Any], path: Path) -> None:
    """Executed, from the cache and failed, over the whole ``raw.jsonl`` of the run.

    Per entry and in total, whichever entries this invocation ran.  A call counts
    once: its answered line says executed or from the cache, and a call that has
    only error lines is failed.  A failure answered on a later try is no longer
    failed; its error line stays in the file as history.
    """
    per_model = manifest["calls_per_model"]
    for counts in per_model.values():
        counts.update(executed=0, from_cache=0, failed=0)
    answered: set[tuple[str, str]] = set()
    tried: set[tuple[str, str]] = set()
    for row in _rows(path):
        call = (str(row.get("model")), str(row.get("cache_key")))
        counts = per_model.setdefault(
            call[0], {"planned": 0, "executed": 0, "from_cache": 0, "failed": 0}
        )
        if row.get("error"):
            tried.add(call)
        elif call not in answered:
            answered.add(call)
            counts["from_cache" if row.get("from_cache") else "executed"] += 1
    for model, _ in tried - answered:
        per_model[model]["failed"] += 1
    for field_name in ("executed", "from_cache", "failed"):
        manifest[f"calls_{field_name}"] = sum(c[field_name] for c in per_model.values())


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )


def _first_start(path: Path) -> str | None:
    """When the first invocation of this run started, if one already wrote a manifest."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("started_at")
    except json.JSONDecodeError:
        return None


def _is_retryable(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in RETRYABLE)


def _ask_with_retries(
    request: Request, backend: Any, attempts: int, sleep: Any
) -> RawResponse:
    """Retry a rate limit or a server error, with growing waits.  Nothing else."""
    response = ask(request, backend)
    tried = 1
    while response.error and _is_retryable(response.error) and tried < attempts:
        sleep(min(2 ** tried, 30))
        response = ask(request, backend)
        tried += 1
    return response


def execute(
    config: RunConfig,
    items: Iterable[Mapping[str, Any]],
    run_id: str | None = None,
    runs_dir: str | Path = RUNS_DIR,
    backend: Any | None = None,
    cache: ResponseCache | None = None,
    models: Mapping[str, Any] | None = None,
    schemes: Mapping[str, Any] | None = None,
    predictions: Mapping[str, str] | None = None,
    max_attempts: int = 3,
    sleep: Any = None,
    only: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the plan, writing the manifest and one line per call.

    ``only`` runs the named entries of the configuration and leaves the others
    alone (see :func:`select_models`).  Several invocations with the same
    ``run_id`` write one run: the manifest always describes the whole
    configuration, with the calls planned for every entry, the counts read from
    the whole ``raw.jsonl``, the start of the first invocation and the end of
    the last.

    Returns the manifest as written.  Raises nothing on a failed call: the
    failure is a line in ``raw.jsonl`` with ``error`` set and no cache entry, so
    the next run tries it again.
    """
    import time

    entries = select_models(config, only)
    sleep = sleep or time.sleep
    models = models if models is not None else load_models()
    schemes = schemes if schemes is not None else load_all()

    calls = plan(config, items, models, schemes, predictions)
    run_id = run_id or make_run_id(config.name)
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / RAW_NAME
    manifest_path = run_dir / MANIFEST_NAME

    manifest = build_manifest(run_id, config, calls, models)
    manifest["started_at"] = _first_start(manifest_path) or datetime.now(UTC).isoformat()
    _count_calls(manifest, raw_path)
    _write_manifest(manifest_path, manifest)

    owns_cache = cache is None
    cache = cache or ResponseCache()
    written = _already_written(raw_path)

    try:
        by_model: dict[str, list[PlannedCall]] = {}
        for call in calls:
            by_model.setdefault(call.model, []).append(call)

        with open(raw_path, "a", encoding="utf-8", newline="\n") as sink:
            for model_name in entries:
                todo = [c for c in by_model.get(model_name, []) if c.key() not in written]
                if not todo:
                    continue
                limit = max(1, int(model_spec(model_name, models).get("max_concurrency", 1)))

                def one(call: PlannedCall) -> tuple[PlannedCall, RawResponse, bool]:
                    request = call.request()
                    key = call.key()
                    hit = cache.get(key)
                    if hit is not None:
                        return call, hit, True
                    response = _ask_with_retries(request, backend, max_attempts, sleep)
                    cache.put(key, request, response)
                    return call, response, False

                # map keeps plan order even though the calls overlap in flight
                with ThreadPoolExecutor(max_workers=limit) as pool:
                    for call, response, hit in pool.map(one, todo):
                        sink.write(dumps(_raw_line(run_id, config, call, response, hit)) + "\n")
                        written.add(call.key())
                        sink.flush()
    finally:
        if owns_cache:
            cache.close()

    manifest["finished_at"] = datetime.now(UTC).isoformat()
    _count_calls(manifest, raw_path)
    _write_manifest(manifest_path, manifest)
    return manifest


def _raw_line(
    run_id: str,
    config: RunConfig,
    call: PlannedCall,
    response: RawResponse,
    hit: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "item_id": call.item_id,
        "stage": call.stage,
        "scheme_condition": config.scheme_condition,
        "scheme": call.scheme,
        "cq_id": call.cq_id or "",
        "sample_index": call.sample_index,
        "model": call.model,
        "model_id": call.model_id,
        "revision": call.revision,
        "tier": call.tier,
        "reasoning": call.reasoning or {},
        "prompt_version": call.prompt_version,
        "json_schema": call.json_schema,
        "params": call.request().generation_params(),
        "cache_key": call.key(),
        "from_cache": hit,
        "raw_response": response.to_dict(),
        "latency_s": response.latency_s,
        "error": response.error,
    }


def read_raw(run_dir: str | Path) -> list[dict[str, Any]]:
    """Every line of a run's ``raw.jsonl``, in the order it was written."""
    path = Path(run_dir)
    if path.is_dir():
        path = path / RAW_NAME
    if not path.is_file():
        raise SchemeError(f"no {RAW_NAME} at {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def allocate(sizes: Mapping[str, int], n: int, minimum: int = 0) -> dict[str, int]:
    """How many items each stratum gets: proportional, at least ``minimum``, summing to ``n``.

    Largest remainder.  Every stratum starts from the floor of its proportional
    quota, raised to ``minimum`` and capped at its own size; the seats still
    missing go to the largest remainders, and any excess is taken back from the
    smallest.  Ties break on the stratum name, so the allocation depends on the
    sizes alone.
    """
    total = sum(sizes.values())
    floor = {s: min(size, minimum) for s, size in sizes.items()}
    if n > total or sum(floor.values()) > n:
        raise SchemeError(
            f"cannot draw {n} items with at least {minimum} per stratum from {dict(sizes)}"
        )
    quota = {s: n * size / total for s, size in sizes.items()}
    alloc = {s: min(sizes[s], max(floor[s], math.floor(quota[s]))) for s in sizes}
    while sum(alloc.values()) < n:
        room = [s for s in sizes if alloc[s] < sizes[s]]
        alloc[max(room, key=lambda s: (quota[s] - alloc[s], s))] += 1
    while sum(alloc.values()) > n:
        room = [s for s in sizes if alloc[s] > floor[s]]
        alloc[min(room, key=lambda s: (quota[s] - alloc[s], s))] -= 1
    return alloc


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _as_records(items: Any) -> list[dict[str, Any]]:
    """Rows as plain dicts with None in every empty cell, whatever came in."""
    rows = items.to_dict("records") if hasattr(items, "to_dict") else [dict(r) for r in items]
    return [{k: (None if _missing(v) else v) for k, v in row.items()} for row in rows]


def select_items(
    config: RunConfig, items: Any, seed: int | None = None
) -> list[dict[str, Any]]:
    """The items the configuration asks for: stratified by scheme, seeded, stable.

    Stratified in proportion to the eligible population with at least
    ``min_per_scheme`` per scheme, through
    :func:`allocate`, then a seeded draw inside each scheme.  Sorting by
    ``item_id`` first is what makes the choice reproducible from ``items.csv``
    alone, without recording a list of ids anywhere.
    """
    import random

    spec = config.items or {}
    records = _as_records(items)
    if spec.get("exclude_none_scheme", True):
        records = [r for r in records if r.get("gold_scheme") not in (None, "", "none")]
    records.sort(key=lambda r: str(r["item_id"]))

    wanted = int(spec.get("n", len(records)))
    if wanted >= len(records):
        return records

    by_scheme: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_scheme.setdefault(str(record["gold_scheme"]), []).append(record)
    counts = allocate({s: len(r) for s, r in by_scheme.items()}, wanted,
                      int(spec.get("min_per_scheme", 0)))

    rng = random.Random(seed if seed is not None else int(spec.get("seed", 0)))
    chosen: list[dict[str, Any]] = []
    for scheme in sorted(by_scheme):
        chosen.extend(rng.sample(by_scheme[scheme], counts[scheme]))
    return sorted(chosen, key=lambda r: str(r["item_id"]))


def load_items(path: str | Path | None = None) -> Any:
    """``data/items.csv``, or an explicit path.  Never a list written in the code."""
    import pandas as pd

    path = Path(path or REPO_ROOT / "data" / "items.csv")
    if not path.is_file():
        raise SchemeError(f"no items table at {path}")
    frame = pd.read_csv(path).astype(object)
    # pandas 3 keeps NaN under where(..., None) unless the column is object: an
    # empty cell must be None, since NaN is truthy and would read as a value
    return frame.where(frame.notna(), None)


def default_runs_dir() -> Path:
    return Path(os.environ.get("ARGFALLACY_RUNS_DIR", RUNS_DIR))
