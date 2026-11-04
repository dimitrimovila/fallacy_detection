import io
import json
import urllib.error

import pytest

from act_pilot.answers import answer_prompt
from act_pilot.config import RoleConfig
from act_pilot.llm import (
    CallLog,
    DiskCache,
    LLMClient,
    OpenAICompatibleBackend,
    PermanentLLMError,
    Request,
    TransientLLMError,
    cache_key,
)
from fakes import FakeAnswerer, FlakyBackend


def _client(tmp_path, backend, **kwargs):
    role = RoleConfig(name="answerer", model="m1", max_tokens=64)
    return LLMClient(role, backend, DiskCache(tmp_path / "cache"), CallLog(tmp_path / "log.jsonl"),
                     sleep=kwargs.pop("sleep", lambda s: None), **kwargs)


def test_cache_hit_skips_backend(tmp_path):
    backend = FakeAnswerer()
    client = _client(tmp_path, backend)
    prompt = answer_prompt("Is it popular?")
    first = client.ask(prompt, "Everyone likes it.")
    second = client.ask(prompt, "Everyone likes it.")
    assert first == second
    assert len(backend.calls) == 1
    assert client.stats.cache_hits == 1 and client.stats.backend_calls == 1
    lines = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [x["cache_hit"] for x in lines] == [False, True]
    assert lines[0]["response"] == first


def test_cache_persists_across_clients(tmp_path):
    prompt = answer_prompt("Is it popular?")
    _client(tmp_path, FakeAnswerer()).ask(prompt, "Text A")
    backend = FakeAnswerer()
    _client(tmp_path, backend).ask(prompt, "Text A")
    assert backend.calls == []


def test_cache_key_depends_on_model_prompt_and_text():
    base = dict(model="m", prompt=answer_prompt("Q?"), text="t", temperature=0.0,
                max_tokens=10, extra_body={})
    key = cache_key(**base)
    assert key == cache_key(**base)
    assert key != cache_key(**{**base, "model": "m2"})
    assert key != cache_key(**{**base, "prompt": answer_prompt("Other?")})
    assert key != cache_key(**{**base, "text": "u"})
    assert key != cache_key(**{**base, "extra_body": {"x": 1}})


def test_retry_then_success(tmp_path):
    backend = FlakyBackend([TransientLLMError("503"), TransientLLMError("timeout")])
    waits = []
    client = _client(tmp_path, backend, max_retries=3, backoff_s=2.0, sleep=waits.append)
    assert client.ask(answer_prompt("Q?"), "t").endswith("Answer: yes")
    assert backend.calls == 3
    assert waits == [2.0, 4.0]
    assert client.stats.retries == 2


def test_retries_exhausted(tmp_path):
    backend = FlakyBackend([TransientLLMError("503")] * 5)
    client = _client(tmp_path, backend, max_retries=2)
    with pytest.raises(TransientLLMError):
        client.ask(answer_prompt("Q?"), "t")
    assert backend.calls == 3
    assert client.stats.failures == 1


def test_permanent_error_is_not_retried(tmp_path):
    backend = FlakyBackend([PermanentLLMError("400")])
    client = _client(tmp_path, backend, max_retries=4)
    with pytest.raises(PermanentLLMError):
        client.ask(answer_prompt("Q?"), "t")
    assert backend.calls == 1


def test_ask_many_keeps_order(tmp_path):
    client = _client(tmp_path, FakeAnswerer(lambda q, t: "yes" if t.endswith("1") else "no"),
                     max_workers=4)
    prompt = answer_prompt("Q?")
    texts = [f"text {i % 2}" for i in range(20)]
    replies = client.ask_many([Request(prompt, t) for t in texts])
    assert [r.endswith("yes") for r in replies] == [t.endswith("1") for t in texts]


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_openai_backend_payload_and_errors():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["auth"] = request.get_header("Authorization")
        reply = {"choices": [{"message": {"content": "Answer: no"}}]}
        return _Response(json.dumps(reply).encode())

    backend = OpenAICompatibleBackend("http://h:8000/v1/", api_key="k", opener=opener)
    out = backend.complete(model="m", messages=[{"role": "user", "content": "x"}],
                           temperature=0.0, max_tokens=5,
                           extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    assert out == "Answer: no"
    assert seen["url"] == "http://h:8000/v1/chat/completions"
    assert seen["body"]["temperature"] == 0.0
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["auth"] == "Bearer k"

    def http_error(code):
        def opener(request, timeout):
            raise urllib.error.HTTPError(request.full_url, code, "err", {}, None)
        return opener

    with pytest.raises(TransientLLMError):
        OpenAICompatibleBackend("http://h", opener=http_error(503)).complete(
            model="m", messages=[], temperature=0, max_tokens=1, extra_body={})
    with pytest.raises(PermanentLLMError):
        OpenAICompatibleBackend("http://h", opener=http_error(400)).complete(
            model="m", messages=[], temperature=0, max_tokens=1, extra_body={})


def test_backend_from_env_requires_url(monkeypatch):
    from act_pilot.llm import backend_from_env

    monkeypatch.delenv("NO_SUCH_URL", raising=False)
    role = RoleConfig(name="optimizer", model="m", base_url_env="NO_SUCH_URL")
    with pytest.raises(PermanentLLMError, match="NO_SUCH_URL"):
        backend_from_env(role, 10)
