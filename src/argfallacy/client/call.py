"""One call to one model, and the answer exactly as it arrived.

The client saves and does not interpret.  Nothing
here reads the JSON the model produced, decides whether it is valid, or picks a
probability out of the logprobs: that is the parser's job, on a later pass over
the file this writes.  Keeping the two apart is what lets the parser change
without re-running a single call.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..schemes.loader import SchemeError

BASE_URL_VAR = "LLM_BASE_URL"
API_KEY_VAR = "LLM_API_KEY"
PLACEHOLDER = "PLACEHOLDER"


@dataclass(frozen=True)
class Request:
    """Everything that decides what comes back, and so everything in the cache key."""

    model_id: str
    prompt: str
    prompt_version: str
    sample_index: int
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = None
    json_schema: dict[str, Any] | None = None
    logprobs: bool = True
    top_logprobs: int = 20
    system: str | None = None
    revision: str | None = None
    tier: int | None = None
    reasoning: dict[str, Any] | None = None

    def messages(self) -> list[dict[str, str]]:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.prompt})
        return messages

    def generation_params(self) -> dict[str, Any]:
        """The parameters that go into the cache key, in a stable order."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs if self.logprobs else None,
            "response_format": bool(self.json_schema),
        }


@dataclass
class RawResponse:
    """What came back, untouched, plus when and how long it took."""

    content: str | None = None
    logprobs: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None
    latency_s: float = 0.0
    requested_at: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawResponse:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class _OpenAIBackend:
    """The real thing.  Isolated so a test can put something else in its place."""

    base_url: str
    api_key: str
    client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def create(self, **kwargs: Any) -> dict[str, Any]:
        response = self.client.chat.completions.create(**kwargs)
        return response.model_dump()


def _completion_backend(base_url: str, api_key: str) -> Any:
    """Seam for the tests.  ``tests/fakes/llm.py`` replaces this function."""
    return _OpenAIBackend(base_url=base_url, api_key=api_key)


def endpoint() -> tuple[str, str]:
    """Where to call and with what key, from the environment only."""
    base_url = os.environ.get(BASE_URL_VAR)
    api_key = os.environ.get(API_KEY_VAR)
    if not base_url:
        raise SchemeError(
            f"{BASE_URL_VAR} is not set; see serving/README.md. It belongs in the .env, "
            f"never in the code."
        )
    return base_url, api_key or "not-checked-by-vllm"


def response_format(request: Request) -> dict[str, Any] | None:
    if not request.json_schema:
        return None
    name = request.json_schema.get("title", "response").replace(".", "_")
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": request.json_schema, "strict": True},
    }


def ask(request: Request, backend: Any | None = None) -> RawResponse:
    """Make the call and keep the answer whole.

    An unpinned revision is refused here, before anything goes out.  Without a
    Hugging Face commit the weights behind a model id can change between the
    pilot and the full run, and nothing downstream would notice.

    The reasoning settings of the model travel with the request: template
    arguments go in ``extra_body``, which is how vLLM receives them through the
    OpenAI client, and ``reasoning_effort`` is a parameter of the call itself.
    """
    if request.revision is not None and PLACEHOLDER in request.revision:
        raise SchemeError(
            f"model {request.model_id!r} has no pinned revision (still "
            f"{request.revision!r}): put the Hugging Face commit in "
            f"serving/models.yaml before running anything"
        )
    if backend is None:
        backend = _completion_backend(*endpoint())

    kwargs: dict[str, Any] = {
        "model": request.model_id,
        "messages": request.messages(),
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    if request.seed is not None:
        kwargs["seed"] = request.seed
    fmt = response_format(request)
    if fmt:
        kwargs["response_format"] = fmt
    if request.logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = request.top_logprobs
    reasoning = request.reasoning or {}
    if reasoning.get("chat_template_kwargs"):
        kwargs["extra_body"] = {
            "chat_template_kwargs": dict(reasoning["chat_template_kwargs"])
        }
    if reasoning.get("reasoning_effort"):
        kwargs["reasoning_effort"] = reasoning["reasoning_effort"]

    started = time.perf_counter()
    requested_at = datetime.now(UTC).isoformat()
    try:
        payload = backend.create(**kwargs)
    except Exception as failure:  # noqa: BLE001 - the reason is data, not control flow
        return RawResponse(
            error=f"{type(failure).__name__}: {failure}",
            latency_s=time.perf_counter() - started,
            requested_at=requested_at,
        )

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return RawResponse(
        content=message.get("content"),
        logprobs=choice.get("logprobs"),
        usage=payload.get("usage"),
        finish_reason=choice.get("finish_reason"),
        raw=payload,
        latency_s=time.perf_counter() - started,
        requested_at=requested_at,
    )


def dumps(value: Any) -> str:
    """Stable JSON: the cache key depends on it being byte-identical."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
