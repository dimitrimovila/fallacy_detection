"""Filling the templates in ``prompts/``.

The division of labour is the point of this module.  The words of a critical
question come from the scheme YAML, character for character: the renderer only
places them.  Everything around them — the role, the instructions, what each
answer means, the output format — is ours, lives in ``prompts/`` and carries a
version that ends up in the manifest of every run.

Substitution is deliberately dumb: ``{{name}}`` and nothing else.  No template
library, no expressions, no conditionals.  A placeholder left unfilled raises
rather than shipping a prompt with a hole in it, and a value nobody asked for
raises too, because a silently ignored field means the template and the caller
have drifted apart.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..paths import REPO_ROOT
from ..schemes.loader import Scheme, SchemeError

PROMPTS_DIR = REPO_ROOT / "prompts"
SCHEMAS_DIR = PROMPTS_DIR / "schemas"

STAGE1 = "stage1"
STAGE2 = "stage2"
DEFAULT_VERSION = "v1"

CANNOT_BE_DETERMINED = "cannot_be_determined"
"""The third option.  Enrico's thesis prints it; this is it made operational."""

NOT_APPLICABLE = "na"
"""The fourth option, added the same way: never in a diagram's answer_space.

Distinct from ``cannot_be_determined``: the question has no object in the text
at all, as opposed to having one whose state cannot be read off the text.
"""

RESERVED_OPTIONS = (CANNOT_BE_DETERMINED, NOT_APPLICABLE)
"""What the code appends to every CQ.  A diagram answer may never be one of these."""

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")



@dataclass(frozen=True)
class RenderedPrompt:
    """A prompt ready to send, with everything needed to reproduce it."""

    text: str
    stage: str
    prompt_version: str
    scheme_id: str | None
    cq_id: str | None
    answer_options: tuple[str, ...]
    json_schema: dict[str, Any]


def template_path(stage: str, version: str = DEFAULT_VERSION) -> Path:
    return PROMPTS_DIR / f"{stage}_{version}.md"


def schema_path(stage: str, version: str = DEFAULT_VERSION) -> Path:
    return SCHEMAS_DIR / f"{stage}_{version}.json"


def prompt_version(stage: str, version: str = DEFAULT_VERSION) -> str:
    """The string that names this template, in the prompt and in the manifest."""
    return f"{stage}_{version}"


def load_template(stage: str, version: str = DEFAULT_VERSION) -> str:
    path = template_path(stage, version)
    if not path.is_file():
        raise SchemeError(f"no template at {path}")
    return path.read_text(encoding="utf-8")


def answers_path(version: str = DEFAULT_VERSION) -> Path:
    return PROMPTS_DIR / f"{STAGE2}_{version}_answers.yaml"


def load_answer_meanings(version: str = DEFAULT_VERSION) -> dict[str, Any]:
    """What each stage-two answer means: prompt configuration, versioned with the template."""
    path = answers_path(version)
    if not path.is_file():
        raise SchemeError(f"no answer definitions at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = prompt_version(STAGE2, version)
    if data.get("prompt_version") != expected:
        raise SchemeError(
            f"{path.name}: prompt_version {data.get('prompt_version')!r}, expected {expected!r}"
        )
    return data


def load_json_schema(stage: str, version: str = DEFAULT_VERSION) -> dict[str, Any]:
    path = schema_path(stage, version)
    if not path.is_file():
        raise SchemeError(f"no JSON schema at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fill(template: str, values: Mapping[str, str], where: str = "template") -> str:
    """Replace every ``{{name}}``.  A missing or unused name is an error."""
    wanted = set(_PLACEHOLDER.findall(template))
    given = set(values)
    missing = sorted(wanted - given)
    unused = sorted(given - wanted)
    if missing:
        raise SchemeError(f"{where}: no value for {missing}")
    if unused:
        raise SchemeError(f"{where}: values nobody asked for: {unused}")
    for name in wanted:
        value = values[name]
        if value is None:
            raise SchemeError(f"{where}: {name!r} is None")
        template = template.replace("{{" + name + "}}", str(value))
    return template


def answer_options(scheme: Scheme, cq_id: str) -> tuple[str, ...]:
    """The answers this critical question admits, plus the two the code adds.

    Taken from the ``answer_space`` the diagram draws, so ad hominem CQ1 gets
    ``positive`` and ``negative`` without anything here knowing that it is
    special.  ``cannot_be_determined`` and ``na`` are never drawn on an arc; if
    a diagram ever used one of those two words as a real answer it would
    collide with the option the code appends, so that is checked here instead
    of being left to silently duplicate.
    """
    real = scheme.answer_space(cq_id)
    collision = set(real) & set(RESERVED_OPTIONS)
    if collision:
        raise SchemeError(
            f"{scheme.scheme_id} {cq_id}: the diagram draws {sorted(collision)} as a real "
            "answer, reserved for the options the prompt adds"
        )
    return (*real, *RESERVED_OPTIONS)


def answer_schema(
    scheme: Scheme, cq_id: str, version: str = DEFAULT_VERSION
) -> dict[str, Any]:
    """The stage-two JSON schema with ``answer`` narrowed to this question."""
    schema = load_json_schema(STAGE2, version)
    schema = json.loads(json.dumps(schema))  # a copy; the file is read many times
    schema["properties"]["answer"]["enum"] = list(answer_options(scheme, cq_id))
    schema["title"] = f"{schema['title']}_{scheme.scheme_id}_{cq_id}"
    return schema


def _variables_block(scheme: Scheme) -> str:
    if not scheme.variables:
        return "(the scheme uses no lettered variables)"
    return "\n".join(f"* **{name}**: {meaning}"
                     for name, meaning in scheme.variables.items())


def _scheme_catalogue(schemes: Mapping[str, Scheme]) -> str:
    parts: list[str] = []
    for scheme_id in sorted(schemes):
        scheme = schemes[scheme_id]
        parts.append(
            f"### {scheme_id}\n\n"
            f"*{scheme.name}.* {scheme.schema}\n\n"
            f"Identifying question: {scheme.identification_question}"
        )
    return "\n\n".join(parts)


def listed_options(
    scheme: Scheme, cq_id: str, meanings: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """The real alternatives of a question, in prompt order, each with its meaning.

    A question with its own entry under ``per_cq`` lists its options in the order
    written there, which is the order the question names them; the set must be
    the one the diagram admits.  Every other question gets ``default``, in the
    order of its ``answer_space``.
    """
    real = scheme.answer_space(cq_id)
    own = (meanings.get("per_cq") or {}).get(scheme.scheme_id, {}).get(cq_id)
    if own is None:
        default = meanings.get("default") or {}
        missing = [o for o in real if o not in default]
        if missing:
            raise SchemeError(f"{scheme.scheme_id} {cq_id}: no definition for {missing}")
        return [(o, default[o]) for o in real]
    if set(own) != set(real):
        raise SchemeError(
            f"{scheme.scheme_id} {cq_id}: the answer definitions list {sorted(own)}, "
            f"the diagram admits {sorted(real)}"
        )
    return list(own.items())


def _answer_options_block(scheme: Scheme, cq_id: str, meanings: Mapping[str, Any]) -> str:
    """The admitted answers with what each one means.

    The ``note`` field of a critical question is **not** printed here. Those notes
    are editorial commentary on the encoding — ad hominem CQ1's note names
    ``appeal_to_authority`` and ``expert_opinion CQ2`` — and putting our label
    taxonomy inside a stage-two prompt would tell the model where its answer leads.
    The definitions come from the prompt configuration instead, and say what the
    question asks, never where the answer leads.
    """
    listed = listed_options(scheme, cq_id, meanings)
    joined = " or ".join(f"`{o}`" for o, _ in listed)
    lines = [f"* `{option}`: {meaning}" for option, meaning in listed]
    third = meanings["cannot_be_determined"].format(options=joined)
    lines.append(f"* `{CANNOT_BE_DETERMINED}`: {third}")
    lines.append(f"* `{NOT_APPLICABLE}`: {meanings['na']}")
    return "\n".join(lines)


def render_stage1(
    item: Mapping[str, Any],
    schemes: Mapping[str, Scheme] | None = None,
    version: str = DEFAULT_VERSION,
) -> RenderedPrompt:
    """The stage-one prompt for one item: which scheme, or none."""
    from ..schemes import load_all

    schemes = schemes if schemes is not None else load_all()
    text = str(item.get("text") or "").strip()
    if not text:
        raise SchemeError(f"item {item.get('item_id')!r} has no text")

    filled = fill(
        load_template(STAGE1, version),
        {
            "text": text,
            "scheme_catalogue": _scheme_catalogue(schemes),
            "prompt_version": prompt_version(STAGE1, version),
        },
        where=f"{STAGE1}_{version}.md",
    )
    return RenderedPrompt(
        text=filled,
        stage=STAGE1,
        prompt_version=prompt_version(STAGE1, version),
        scheme_id=None,
        cq_id=None,
        answer_options=(),
        json_schema=load_json_schema(STAGE1, version),
    )


def render_stage2(
    item: Mapping[str, Any],
    scheme: Scheme,
    cq_id: str,
    version: str = DEFAULT_VERSION,
) -> RenderedPrompt:
    """The stage-two prompt: one item, one scheme, one critical question."""
    if cq_id not in scheme.cqs:
        raise SchemeError(f"{scheme.scheme_id}: unknown CQ {cq_id!r}")
    text = str(item.get("text") or "").strip()
    if not text:
        raise SchemeError(f"item {item.get('item_id')!r} has no text")

    options = answer_options(scheme, cq_id)
    filled = fill(
        load_template(STAGE2, version),
        {
            "scheme_name": scheme.name,
            "schema": scheme.schema,
            "variables": _variables_block(scheme),
            "text": text,
            "cq_id": cq_id,
            "cq_text": scheme.cqs[cq_id].text,
            "answer_options": _answer_options_block(
                scheme, cq_id, load_answer_meanings(version)
            ),
            "prompt_version": prompt_version(STAGE2, version),
        },
        where=f"{STAGE2}_{version}.md",
    )
    return RenderedPrompt(
        text=filled,
        stage=STAGE2,
        prompt_version=prompt_version(STAGE2, version),
        scheme_id=scheme.scheme_id,
        cq_id=cq_id,
        answer_options=options,
        json_schema=answer_schema(scheme, cq_id, version),
    )
