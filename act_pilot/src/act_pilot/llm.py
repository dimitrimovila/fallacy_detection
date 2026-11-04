"""The client for OpenAI-compatible chat endpoints (vLLM), with a disk cache and a call log.

Three layers are kept apart so that the tests can replace the network:

* a *backend* sends one chat request and returns the text of the reply
  (:class:`OpenAICompatibleBackend` for vLLM, a fake in the tests);
* :class:`DiskCache` stores every reply under a key built from the model, the prompt and the
  text (plus the generation parameters), one JSON file per key, written atomically;
* :class:`LLMClient` puts them together for one role (answerer or optimizer): it looks in
  the cache, retries transient failures with exponential backoff, and appends one line per
  call to a JSONL log, cache hits included.

Requests are sent at temperature 0. Nothing is ever evicted from the cache: a run that is
interrupted resumes where it stopped, and a repeated question costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .answers import Prompt
from .config import LLMConfig, RoleConfig


class TransientLLMError(RuntimeError):
    """A failure that is worth retrying: timeouts, connection errors, HTTP 429 and 5xx."""


class PermanentLLMError(RuntimeError):
    """A failure that a retry would not fix, such as HTTP 400 or a malformed reply."""


class ChatBackend(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any],
    ) -> str: ...


@dataclass
class OpenAICompatibleBackend:
    """A minimal client of ``POST {base_url}/chat/completions``, as served by vLLM."""

    base_url: str
    api_key: str = "not-checked-by-vllm"
    timeout_s: float = 120.0
    opener: Callable[..., Any] = field(default=urllib.request.urlopen, repr=False)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any],
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra_body,
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with self.opener(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                raise TransientLLMError(f"HTTP {exc.code}") from exc
            raise PermanentLLMError(f"HTTP {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise TransientLLMError(str(exc)) from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentLLMError(f"unexpected reply: {str(body)[:200]}") from exc
        return content or ""


def backend_from_env(role: RoleConfig, timeout_s: float) -> OpenAICompatibleBackend:
    """The backend of a role, with the endpoint and key taken from the environment."""
    base_url = os.environ.get(role.base_url_env)
    if not base_url:
        raise PermanentLLMError(
            f"{role.base_url_env} is not set: it must hold the URL of the server of the "
            f"{role.name}, for example http://localhost:8000/v1"
        )
    api_key = os.environ.get(role.api_key_env) or "not-checked-by-vllm"
    return OpenAICompatibleBackend(base_url=base_url, api_key=api_key, timeout_s=timeout_s)


def cache_key(
    *,
    model: str,
    prompt: Prompt,
    text: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict[str, Any],
) -> str:
    """The cache key: a SHA-256 over the model, the prompt, the text and the parameters."""
    material = json.dumps(
        {
            "model": model,
            "system": prompt.system,
            "user_template": prompt.user_template,
            "prompt_version": prompt.version,
            "text": text,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class DiskCache:
    """One JSON file per key under ``root/<first two hex digits>/<key>.json``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        except (OSError, ValueError, KeyError):
            return None

    def put(self, key: str, record: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)


class CallLog:
    """An append-only JSONL file with one record per call, shared by threads."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


@dataclass
class CallStats:
    requests: int = 0
    cache_hits: int = 0
    backend_calls: int = 0
    retries: int = 0
    failures: int = 0


@dataclass(frozen=True)
class Request:
    prompt: Prompt
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """The client of one role. ``ask`` returns the raw reply text."""

    def __init__(
        self,
        role: RoleConfig,
        backend: ChatBackend,
        cache: DiskCache,
        log: CallLog,
        *,
        temperature: float = 0.0,
        max_retries: int = 4,
        backoff_s: float = 2.0,
        max_workers: int = 8,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.role = role
        self.backend = backend
        self.cache = cache
        self.log = log
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.max_workers = max_workers
        self.sleep = sleep
        self.stats = CallStats()
        self._lock = threading.Lock()

    def _count(self, **deltas: int) -> None:
        with self._lock:
            for name, delta in deltas.items():
                setattr(self.stats, name, getattr(self.stats, name) + delta)

    def ask(self, prompt: Prompt, text: str, meta: dict[str, Any] | None = None) -> str:
        meta = dict(meta or {})
        key = cache_key(
            model=self.role.model,
            prompt=prompt,
            text=text,
            temperature=self.temperature,
            max_tokens=self.role.max_tokens,
            extra_body=self.role.extra_body,
        )
        self._count(requests=1)
        started = time.monotonic()
        cached = self.cache.get(key)
        if cached is not None:
            self._count(cache_hits=1)
            self._log(key, meta, cached, cache_hit=True, attempts=0, started=started)
            return cached
        messages = prompt.render(text)
        attempt, error = 0, None
        while True:
            attempt += 1
            try:
                self._count(backend_calls=1)
                response = self.backend.complete(
                    model=self.role.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.role.max_tokens,
                    extra_body=self.role.extra_body,
                )
                break
            except TransientLLMError as exc:
                error = str(exc)
                if attempt > self.max_retries:
                    self._count(failures=1)
                    self._log(key, meta, None, cache_hit=False, attempts=attempt,
                              started=started, error=error)
                    raise
                self._count(retries=1)
                self.sleep(self.backoff_s * (2 ** (attempt - 1)))
            except PermanentLLMError as exc:
                self._count(failures=1)
                self._log(key, meta, None, cache_hit=False, attempts=attempt,
                          started=started, error=str(exc))
                raise
        self.cache.put(
            key,
            {
                "model": self.role.model,
                "prompt_version": prompt.version,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.role.max_tokens,
                "response": response,
            },
        )
        self._log(key, meta, response, cache_hit=False, attempts=attempt, started=started,
                  error=error)
        return response

    def ask_many(self, requests: Sequence[Request]) -> list[str]:
        """Several requests are sent in parallel threads; the order of the replies is kept."""
        if self.max_workers <= 1 or len(requests) <= 1:
            return [self.ask(r.prompt, r.text, r.meta) for r in requests]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda r: self.ask(r.prompt, r.text, r.meta), requests))

    def _log(
        self,
        key: str,
        meta: dict[str, Any],
        response: str | None,
        *,
        cache_hit: bool,
        attempts: int,
        started: float,
        error: str | None = None,
    ) -> None:
        self.log.write(
            {
                "time": datetime.now(UTC).isoformat(timespec="seconds"),
                "role": self.role.name,
                "model": self.role.model,
                "key": key,
                "cache_hit": cache_hit,
                "attempts": attempts,
                "latency_s": round(time.monotonic() - started, 3),
                "error": error,
                **{f"meta_{k}": v for k, v in meta.items()},
                "response": response,
            }
        )


def make_clients(
    llm: LLMConfig,
    cache_dir: Path,
    log_path: Path,
    backends: dict[str, ChatBackend] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[LLMClient, LLMClient]:
    """The answerer and optimizer clients. ``backends`` replaces the network in the tests."""
    cache = DiskCache(cache_dir)
    log = CallLog(log_path)
    clients = []
    for role in (llm.answerer, llm.optimizer):
        backend = (backends or {}).get(role.name) or backend_from_env(role, llm.timeout_s)
        clients.append(
            LLMClient(
                role,
                backend,
                cache,
                log,
                temperature=llm.temperature,
                max_retries=llm.max_retries,
                backoff_s=llm.backoff_s,
                max_workers=llm.max_workers,
                sleep=sleep,
            )
        )
    return clients[0], clients[1]


def total_stats(clients: Iterable[LLMClient]) -> dict[str, dict[str, int]]:
    return {c.role.name: vars(c.stats).copy() for c in clients}
