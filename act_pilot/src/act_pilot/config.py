"""The configuration of the pilot, read from one YAML file.

Relative paths are resolved against the directory of the configuration file, and
``${VAR}`` references are expanded from the environment, so that data prepared elsewhere can
be pointed to without writing a machine path into the repository. Unknown keys are refused:
a misspelt key would otherwise fall back to a default without anyone noticing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_COLUMNS = {
    "item_id": "item_id",
    "text": "text",
    "scheme": "scheme",
    "gold": "gold",
    "source": "source",
}
UNDETERMINED_POLICIES = ("negative", "positive")


class ConfigError(ValueError):
    """Raised when the configuration cannot be used as written."""


@dataclass(frozen=True)
class RoleConfig:
    """One model role: the answerer or the optimizer."""

    name: str
    model: str
    base_url_env: str = "LLM_BASE_URL"
    api_key_env: str = "LLM_API_KEY"
    max_tokens: int = 512
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMConfig:
    answerer: RoleConfig
    optimizer: RoleConfig
    temperature: float = 0.0
    timeout_s: float = 120.0
    max_retries: int = 4
    backoff_s: float = 2.0
    max_workers: int = 8


@dataclass(frozen=True)
class ActConfig:
    steps: int = 10
    examples_per_group: int = 20
    seeds: tuple[int, ...] = (0, 1, 2)
    validation_fraction: float = 0.2
    split_seed: int = 2026
    max_logical_ops: int = 2
    max_rewrite_attempts: int = 3
    specific_term_max_df: float = 0.05
    max_question_chars: int = 300
    max_example_chars: int = 800
    show_error_reasoning: bool = True


@dataclass(frozen=True)
class LexicalConfig:
    max_features: int = 2000
    min_df: int = 2
    ngram_max: int = 2
    cart_depths: tuple[int, ...] = (2, 3, 4, 5)
    rulefit_max_rules: int = 30
    rulefit_n_estimators: int = 100
    rulefit_cv: bool = True
    inner_folds: int = 3
    top_terms: int = 10
    random_state: int = 0


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap: int = 5000
    bootstrap_seed: int = 12345
    ci_level: float = 0.95
    min_items_per_class_in_source: int = 10
    undetermined_policy: str = "negative"
    min_content_validity: int = 2
    lexical: LexicalConfig = field(default_factory=LexicalConfig)


@dataclass(frozen=True)
class Boundary:
    """A binary boundary inside one scheme. The answer "yes" stands for the positive class."""

    id: str
    scheme: str
    positive: str
    negative: str
    seed_question: str
    scheme_text: str
    question_function: str
    positive_description: str
    negative_description: str
    banned_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ControlQuestion:
    id: str
    question: str


@dataclass(frozen=True)
class PilotConfig:
    path: Path
    items: Path
    folds: Path
    columns: dict[str, str]
    manual_rewrites: Path
    content_validity: Path
    output_dir: Path
    cache_dir: Path
    llm: LLMConfig
    act: ActConfig
    evaluation: EvaluationConfig
    boundaries: dict[str, Boundary]
    controls: tuple[ControlQuestion, ...]
    n_outer_folds: int = 5

    def boundary(self, boundary_id: str) -> Boundary:
        try:
            return self.boundaries[boundary_id]
        except KeyError as exc:
            known = ", ".join(sorted(self.boundaries))
            raise ConfigError(f"unknown boundary {boundary_id!r}; known: {known}") from exc


_VAR = re.compile(r"\$\{[^}]*\}")


def _resolve_path(value: Any, base: Path, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}: a path is required")
    expanded = os.path.expandvars(value)
    if _VAR.search(expanded):
        raise ConfigError(f"{where}: unset environment variable in {value!r}")
    path = Path(expanded).expanduser()
    return (path if path.is_absolute() else (base / path)).resolve()


def _section(data: Any, allowed: set[str], where: str, required: set[str] = frozenset()) -> dict:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: a mapping is expected")
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    missing = required - set(data)
    if missing:
        raise ConfigError(f"{where}: missing keys {sorted(missing)}")
    return data


def _dataclass_kwargs(cls: type, data: dict, where: str, tuples: tuple[str, ...] = ()) -> dict:
    allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    data = _section(data, allowed, where)
    out = dict(data)
    for key in tuples:
        if key in out:
            out[key] = tuple(out[key])
    return out


def _role(name: str, data: Any) -> RoleConfig:
    where = f"llm.{name}"
    data = _section(
        data,
        {"model", "base_url_env", "api_key_env", "max_tokens", "extra_body"},
        where,
        required={"model"},
    )
    return RoleConfig(name=name, **data)


def _boundary(boundary_id: str, data: Any) -> Boundary:
    fields = {
        "scheme", "positive", "negative", "seed_question", "scheme_text", "question_function",
        "positive_description", "negative_description", "banned_terms",
    }
    required = fields - {"banned_terms"}
    data = _section(data, fields, f"boundaries.{boundary_id}", required=required)
    clean = {k: (" ".join(str(v).split()) if isinstance(v, str) else v) for k, v in data.items()}
    if "scheme_text" in data:
        clean["scheme_text"] = str(data["scheme_text"]).strip()
    clean["banned_terms"] = tuple(str(t).lower() for t in data.get("banned_terms", ()))
    if clean["positive"] == clean["negative"]:
        raise ConfigError(f"boundaries.{boundary_id}: positive and negative classes coincide")
    return Boundary(id=boundary_id, **clean)


def load_config(path: str | Path) -> PilotConfig:
    """The YAML file at ``path`` is read, checked and turned into a :class:`PilotConfig`."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    top = _section(
        raw,
        {
            "data", "manual_rewrites", "content_validity", "output_dir", "cache_dir", "llm",
            "act", "evaluation", "boundaries", "controls", "n_outer_folds",
        },
        "config",
        required={"data", "llm", "boundaries", "manual_rewrites", "output_dir"},
    )
    base = path.parent

    data = _section(top["data"], {"items", "folds", "columns"}, "data", {"items", "folds"})
    columns = dict(DEFAULT_COLUMNS)
    extra = _section(data.get("columns"), set(DEFAULT_COLUMNS), "data.columns")
    columns.update({k: str(v) for k, v in extra.items()})

    llm_raw = _section(
        top["llm"],
        {"answerer", "optimizer", "temperature", "timeout_s", "max_retries", "backoff_s",
         "max_workers"},
        "llm",
        required={"answerer", "optimizer"},
    )
    llm = LLMConfig(
        answerer=_role("answerer", llm_raw["answerer"]),
        optimizer=_role("optimizer", llm_raw["optimizer"]),
        **{k: v for k, v in llm_raw.items() if k not in {"answerer", "optimizer"}},
    )
    if llm.temperature != 0:
        raise ConfigError("llm.temperature: the pilot is run at temperature 0")

    act = ActConfig(**_dataclass_kwargs(ActConfig, top.get("act") or {}, "act", ("seeds",)))
    if act.steps < 1 or act.examples_per_group < 1:
        raise ConfigError("act: steps and examples_per_group must be positive")
    if not 0 < act.validation_fraction < 1:
        raise ConfigError("act.validation_fraction must lie strictly between 0 and 1")

    ev_raw = dict(top.get("evaluation") or {})
    lexical = LexicalConfig(
        **_dataclass_kwargs(LexicalConfig, ev_raw.pop("lexical", {}) or {}, "evaluation.lexical",
                            ("cart_depths",))
    )
    evaluation = EvaluationConfig(
        lexical=lexical, **_dataclass_kwargs(EvaluationConfig, ev_raw, "evaluation")
    )
    if evaluation.undetermined_policy not in UNDETERMINED_POLICIES:
        raise ConfigError(
            f"evaluation.undetermined_policy must be one of {UNDETERMINED_POLICIES}"
        )

    boundaries_raw = _section(top["boundaries"], set(top["boundaries"] or {}), "boundaries")
    if not boundaries_raw:
        raise ConfigError("boundaries: at least one boundary is required")
    boundaries = {bid: _boundary(bid, spec) for bid, spec in boundaries_raw.items()}

    controls = []
    for i, entry in enumerate(top.get("controls") or []):
        entry = _section(entry, {"id", "question"}, f"controls[{i}]", {"id", "question"})
        controls.append(ControlQuestion(id=str(entry["id"]), question=str(entry["question"])))
    if len({c.id for c in controls}) != len(controls):
        raise ConfigError("controls: ids must be unique")

    output_dir = _resolve_path(top["output_dir"], base, "output_dir")
    cache_dir = (
        _resolve_path(top["cache_dir"], base, "cache_dir")
        if top.get("cache_dir")
        else output_dir / "cache"
    )
    content_validity = (
        _resolve_path(top["content_validity"], base, "content_validity")
        if top.get("content_validity")
        else output_dir / "content_validity.csv"
    )
    return PilotConfig(
        path=path,
        items=_resolve_path(data["items"], base, "data.items"),
        folds=_resolve_path(data["folds"], base, "data.folds"),
        columns=columns,
        manual_rewrites=_resolve_path(top["manual_rewrites"], base, "manual_rewrites"),
        content_validity=content_validity,
        output_dir=output_dir,
        cache_dir=cache_dir,
        llm=llm,
        act=act,
        evaluation=evaluation,
        boundaries=boundaries,
        controls=tuple(controls),
        n_outer_folds=int(top.get("n_outer_folds", 5)),
    )


def load_manual_rewrites(path: Path, boundaries: dict[str, Boundary]) -> dict[str, list[str]]:
    """The hand-written rewrites are read, one list per boundary id."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: a mapping from boundary id to a list of questions is expected")
    unknown = set(raw) - set(boundaries)
    if unknown:
        raise ConfigError(f"{path}: unknown boundaries {sorted(unknown)}")
    out: dict[str, list[str]] = {}
    for bid in boundaries:
        questions = raw.get(bid) or []
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            raise ConfigError(f"{path}: {bid} must be a list of strings")
        out[bid] = [" ".join(q.split()) for q in questions]
    return out
