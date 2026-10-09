"""A stand-in for a served model.  No network, ever: tests never call a real model.

It answers as a function of the prompt and the sample index, so a test can ask
the same question twice and get the same thing, and can ask for five samples and
get five that differ the way real ones do.

The logprob block is shaped like vLLM's: one entry per generated token, whose
text concatenates to exactly the reasoning, if any, and the content, with
alternatives at the token where the answer value begins, and ``bytes`` on every
token and alternative, as the OpenAI client returns them.  Two options reproduce
what a real run will meet:

* ``partial_logprobs`` drops one admitted answer out of the alternatives, so the
  parser has to renormalise over what is left and mark the row;
* ``split_labels`` writes an answer in two tokens, its first two characters and
  the rest, and offers the whole answer as an alternative to the first: the same
  label split in two ways, as a model writes a verdict name;
* ``reasoning`` thinks before the JSON, as a server with a reasoning parser
  answers: the thinking goes in the ``reasoning`` field of the message and the
  content is the JSON alone, while the logprob block keeps the tokens of both.
  The thinking holds a *decoy*: a draft object whose answer token carries its
  own, different alternatives.  A client that reads the first answer-looking
  token instead of the one inside the final JSON gets the decoy's probabilities,
  and the test catches it.

The counter is the reason this exists.  Several acceptance tests are about *not*
calling the model — the cache, the resume — and the only way to check that is to
count how many times the backend was actually reached.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_ANSWERS = ("yes", "no", "cannot_be_determined")
FIRST_TOKEN_OF = {"cannot_be_determined": "cannot"}

REAL_WEIGHT = 0.6
"""Probability the fake gives its own answer at the real answer token."""
DECOY_WEIGHT = 0.9
"""Probability the decoy in the reasoning gives to a different answer."""
SPLIT_SHORT, SPLIT_WHOLE, SPLIT_REST = 0.5, 0.2, 0.9
"""With ``split_labels``: the first two characters of the answer, the whole answer as an
alternative to them, and the rest after the two characters."""


class FakeError(RuntimeError):
    """What the fake raises when a test asks it to fail."""


def _digest(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def _options_and_key(response_format: dict[str, Any] | None) -> tuple[tuple[str, ...], str]:
    schema = (response_format or {}).get("json_schema", {}).get("schema", {})
    properties = schema.get("properties", {})
    for key in ("answer", "scheme"):
        if key in properties and "enum" in properties[key]:
            return tuple(properties[key]["enum"]), key
    return DEFAULT_ANSWERS, "answer"


def first_token(value: str) -> str:
    """How the fake tokenises the start of an answer value."""
    return FIRST_TOKEN_OF.get(value, value)


def _token(text: str, logprob: float = -0.001, alternatives: list | None = None) -> dict:
    return {"token": text, "bytes": list(text.encode("utf-8")), "logprob": logprob,
            "top_logprobs": alternatives or []}


def _alternatives(options: tuple[str, ...], favourite: str, weight: float,
                  drop_last: bool) -> list[dict[str, Any]]:
    tokens = [first_token(o) for o in options]
    if drop_last and len(tokens) > 1:
        tokens = tokens[:-1]  # one admitted answer never shows up
    rest = (1.0 - weight) / max(1, len(tokens) - 1)
    weights = {t: (weight if t == first_token(favourite) else rest) for t in tokens}
    total = sum(weights.values())
    out = [{"token": t, "bytes": list(t.encode("utf-8")), "logprob": math.log(w / total)}
           for t, w in weights.items()]
    # noise, not an answer
    out.append({"token": " the", "bytes": list(b" the"), "logprob": math.log(1e-6)})
    return out


@dataclass
class FakeBackend:
    """Deterministic answers, a countable number of calls, optional failures.

    ``fail_after`` makes the next call raise once the counter passes it, which is
    how the resume test interrupts an execution half way.  ``last_kwargs`` keeps
    what the client sent, so a test can check the reasoning settings went out.
    """

    calls: int = 0
    fail_after: int | None = None
    partial_logprobs: bool = False
    split_labels: bool = False
    reasoning: bool = False
    invalid_on: tuple[str, ...] = ()
    latency_s: float = 0.0
    prompts_seen: list[str] = field(default_factory=list)
    last_kwargs: dict[str, Any] = field(default_factory=dict)

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        **rest: Any,
    ) -> dict[str, Any]:
        self.last_kwargs = {"model": model, "temperature": temperature, "seed": seed,
                            "max_tokens": max_tokens, "logprobs": logprobs,
                            "top_logprobs": top_logprobs, **rest}
        if self.fail_after is not None and self.calls >= self.fail_after:
            self.calls += 1
            raise FakeError(f"the fake backend was told to fail after {self.fail_after} calls")

        prompt = "\n".join(m["content"] for m in messages)
        self.calls += 1
        self.prompts_seen.append(prompt)
        if self.latency_s:
            time.sleep(self.latency_s)

        options, key = _options_and_key(response_format)
        seed_value = _digest(prompt, model, str(int(seed or 0)))
        answer = options[seed_value % len(options)]
        confidence = 50 + (seed_value % 50)

        thinking = self._reasoning_tokens(options, key, answer) if self.reasoning else []
        if any(marker in prompt for marker in self.invalid_on):
            answer_tokens = [_token("I am not going to answer that.", -0.1)]
        else:
            body = json.dumps({key: answer, "confidence": confidence,
                               "justification": "Because the text says so."})
            answer_tokens = self._json_tokens(body, key, answer, options)
        tokens = thinking + answer_tokens

        message = {"role": "assistant",
                   "content": "".join(t["token"] for t in answer_tokens)}
        if thinking:
            message["reasoning"] = "".join(t["token"] for t in thinking)
        return {
            "id": f"fake-{seed_value:08x}",
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "stop",
                "logprobs": {"content": tokens} if logprobs else None,
            }],
            "usage": {"prompt_tokens": len(prompt.split()),
                      "completion_tokens": len(tokens),
                      "total_tokens": len(prompt.split()) + len(tokens)},
        }

    def _json_tokens(self, body: str, key: str, answer: str,
                     options: tuple[str, ...]) -> list[dict[str, Any]]:
        """The object as ``{"``, ``answer``, ``": "``, ``yes``, then the rest."""
        match = re.search(re.escape(f'"{key}": "'), body)
        start = match.end()
        value_end = body.index('"', start)
        if self.split_labels and len(answer) > 2:
            return self._split_tokens(body, key, answer, options, start, value_end)
        first = first_token(answer)
        pieces = [
            _token(body[:2]),
            _token(body[2 : 2 + len(key)]),
            _token(body[2 + len(key) : start]),
            _token(first, math.log(REAL_WEIGHT),
                   _alternatives(options, answer, REAL_WEIGHT, self.partial_logprobs)),
        ]
        if value_end > start + len(first):
            pieces.append(_token(body[start + len(first) : value_end]))
        pieces.append(_token(body[value_end:]))
        return pieces

    def _split_tokens(self, body: str, key: str, answer: str, options: tuple[str, ...],
                      start: int, value_end: int) -> list[dict[str, Any]]:
        """The answer as its first two characters and the rest, the whole as an alternative."""
        others = [first_token(o) for o in options if o != answer]
        rest = (1.0 - SPLIT_SHORT - SPLIT_WHOLE) / max(1, len(others))
        alternatives = [_token(answer[:2], math.log(SPLIT_SHORT)),
                        _token(answer, math.log(SPLIT_WHOLE))]
        alternatives += [_token(o, math.log(rest)) for o in others]
        return [
            _token(body[:2]),
            _token(body[2 : 2 + len(key)]),
            _token(body[2 + len(key) : start]),
            _token(answer[:2], math.log(SPLIT_SHORT), alternatives),
            _token(answer[2:], math.log(SPLIT_REST)),
            _token(body[value_end:]),
        ]

    def _reasoning_tokens(self, options: tuple[str, ...], key: str,
                          answer: str) -> list[dict[str, Any]]:
        """Thinking text holding a draft object whose answer is *not* the final one."""
        decoy = next((o for o in options if o != answer), answer)
        return [
            _token("<think>Let me reason. A first draft would be ", -0.3),
            _token("{" + f'"{key}": "', -0.2),
            _token(first_token(decoy), math.log(DECOY_WEIGHT),
                   _alternatives(options, decoy, DECOY_WEIGHT, False)),
            _token('"} but reading again, the text says otherwise.</think>' + "\n", -0.4),
        ]


def install(monkeypatch, backend: FakeBackend) -> FakeBackend:
    """Point the client at the fake for the duration of a test."""
    from argfallacy.client import call

    monkeypatch.setattr(call, "_completion_backend", lambda base_url, api_key: backend)
    return backend
