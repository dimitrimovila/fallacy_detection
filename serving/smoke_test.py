#!/usr/bin/env python3
"""The first command to run against a newly served model.

Sends one stage-two call with structured output and logprobs and prints what
came back: the answer, the token where the answer value starts *inside the final
JSON* and its position, the alternatives at that token, and the latency.

Five checks matter more than the rest:

* whether the prompt reached the model.  A chat template that drops the content
  of the message leaves the model a prompt of a few tokens, and it answers
  something all the same; fewer than a quarter of the tokens expected at four
  characters per token means the prompt was not read;
* whether logprobs arrive at all.  That decides ``supports_logprobs`` in
  ``serving/models.yaml`` and whether ``p_logprob`` will exist for that model;
* whether the token read is the answer's.  A reasoning model writes thinking
  before the object, and that text can contain ``yes`` or a whole draft object.
  The script says where a naive reader — first answer-looking token — would have
  landed, and warns if that is somewhere else;
* on an entry that reasons, whether thinking actually happened and whether the
  answer survived it.  No reasoning, neither in ``usage`` nor in the reasoning
  field of the message, means the server is not passing the reasoning settings
  through, and a ``length`` finish means the object was cut off before it closed;
* whether the JSON has the one layout the server should impose.  The same
  question goes out three more times at temperature 1, and every content (after
  the reasoning parser, on an entry that reasons) should begin with
  ``{"answer": "``.  Any other spacing means the server is not compacting the
  JSON, and the context before the answer token changes from call to call.  That
  matters where the soft answer is the logprobs, an entry with one sample, and
  there it stops the entry.  An entry with several samples has the frequency as
  its soft answer, and with gpt oss vLLM 0.30.0 hands the schema to xgrammar as a
  structural tag that ``disable_any_whitespace`` does not reach: there a
  different layout is a warning, with the prefixes seen.

The exit code is zero only if the entry can be run: a job stops on anything else.
Besides a refusal, an error or an answer token missing from the JSON, that means
a prompt the server did not read, no reasoning (unless the entry has
``reasoning_optional``) or a cut-off answer on an entry that reasons, a JSON
that does not begin with ``{"answer": "`` on an entry with one sample, and no
logprobs on an entry whose ``supports_logprobs`` is true.

    python serving/smoke_test.py --model qwen3_8_27b        # a key of serving/models.yaml
    python serving/smoke_test.py --model qwen3_8_27b_think  # same weights, thinking on
    python serving/smoke_test.py --model fake               # the fake backend, no network
    python serving/smoke_test.py --model fake --fake-reasoning

Endpoint and key come from ``LLM_BASE_URL`` and ``LLM_API_KEY`` in the ``.env``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argfallacy.client import Request, ask, model_spec, sampling_for  # noqa: E402
from argfallacy.client.call import endpoint  # noqa: E402
from argfallacy.env import load_env  # noqa: E402
from argfallacy.parse import answer_position, logprob_masses, parse_content  # noqa: E402
from argfallacy.prompts import render_stage2  # noqa: E402
from argfallacy.schemes import SchemeError, load_all  # noqa: E402

PROBE_ITEM = {
    "item_id": "smoke-test",
    "text": (
        "Professor Vance has a doctorate in economics, so her claim that the new "
        "tariff will raise prices must be right."
    ),
}
PROBE_SCHEME = "expert_opinion"
PROBE_CQ = "CQ1"
ANSWER_KEY = "answer"

FAKE = "fake"
FAKE_SPEC = {
    "model_id": "fake-model", "display_name": "Fake backend", "revision": "fake",
    "tier": 0, "reasoning": {}, "samples": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
}
THINK_END = "</think>"
DEFAULT_MAX_TOKENS = 256
CHARS_PER_TOKEN = 4
"""A rough size of a token: enough to tell a prompt read whole from one that was dropped."""
PROMPT_FLOOR = 0.25
"""The share of the expected prompt tokens below which the prompt was not read."""
LAYOUT = '{"answer": "'
"""How every stage-two content must begin once the server compacts the JSON."""
LAYOUT_CALLS = 3
LAYOUT_TEMPERATURE = 1.0


def build_backend(model_name: str, fake_reasoning: bool):
    """The fake for ``--model fake``, the real endpoint otherwise."""
    if model_name == FAKE:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from fakes.llm import FakeBackend

        print("Fake backend: no network calls.")
        if fake_reasoning:
            print("With reasoning text and a draft answer before the JSON.")
        print()
        return FakeBackend(reasoning=fake_reasoning)
    base_url, _ = endpoint()
    print(f"Endpoint: {base_url}\n")
    return None


def thinking_on(spec) -> bool:
    """Whether this entry reasons before answering.

    ``enable_thinking`` decides where the entry sets it.  Where it does not, as for
    K2 Horizon and gpt oss, whose reasoning does not switch off, ``is_reasoning``
    decides.  Not ``is_reasoning`` alone: the entries with ``enable_thinking: false``
    are reasoning models too, and they must not reason.
    """
    kwargs = (spec.get("reasoning") or {}).get("chat_template_kwargs") or {}
    return bool(kwargs.get("enable_thinking", spec.get("is_reasoning", False)))


def reasoning_length(response) -> tuple[int, str]:
    """How much thinking there was, and where the number comes from.

    Two sources, in order: ``usage``, where vLLM counts the reasoning tokens, and
    the reasoning field of the message (``reasoning_content`` or ``reasoning``),
    where the reasoning parser puts the thinking.  ``usage`` can say zero while
    the field is full, as it did for K2 Horizon, so its zero is not the answer.
    The readings after those are for a server without a parser, where the
    thinking stays in the content before the think marker.  Zero from all of them
    is the answer the caller needs: the entry reasons and did not.
    """
    details = (response.usage or {}).get("completion_tokens_details") or {}
    counted = int(details.get("reasoning_tokens") or 0)
    if counted:
        return counted, "tokens, from usage"

    message = ((response.raw or {}).get("choices") or [{}])[0].get("message") or {}
    for name in ("reasoning_content", "reasoning"):
        if message.get(name):
            return len(str(message[name])), f"characters, from the {name} of the message"

    tokens = (response.logprobs or {}).get("content") or []
    for index, entry in enumerate(tokens):
        if THINK_END in str(entry.get("token", "")):
            return index + 1, f"tokens up to {THINK_END}, from the logprobs"
    if THINK_END in (response.content or ""):
        head = (response.content or "").split(THINK_END)[0]
        return len(head), f"characters before {THINK_END}, from the content"
    return 0, "no reasoning in usage, in the message or in the content"


def layout_holds(request: Request, backend, blocking: bool) -> bool:
    """The same question at temperature 1, three times: does every content begin the same way?

    Seeds 1 to 3, so that the three calls can differ.  An error counts as a failure.
    Not ``blocking``, a different layout is reported and the answer is True.
    """
    holds = True
    for seed in range(1, LAYOUT_CALLS + 1):
        response = ask(replace(request, temperature=LAYOUT_TEMPERATURE, seed=seed,
                               sample_index=seed), backend=backend)
        head = (response.content or "")[:len(LAYOUT) + 8]
        ok = not response.error and (response.content or "").startswith(LAYOUT)
        holds = holds and ok
        print(f"Layout {seed}  : {'ok' if ok else 'NO'}  {response.error or repr(head)}")
    if not holds and blocking:
        print(f"WARNING: not every content begins with {LAYOUT!r}. The server must be launched "
              f"with the `--structured-outputs-config` of the entry's `serve_args`.")
    elif not holds:
        print(f"WARNING, not blocking: not every content begins with {LAYOUT!r}. This entry "
              f"takes several samples, and its soft answer is their frequency, not the logprobs.")
    return holds or not blocking


def naive_position(logprobs, options) -> int | None:
    """Where a reader that takes the first answer-looking token would land."""
    for index, entry in enumerate(logprobs.get("content") or []):
        token = str(entry.get("token", "")).strip().strip('"').strip()
        if token and entry.get("top_logprobs") and any(o.startswith(token) for o in options):
            return index
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=FAKE, help="a key of serving/models.yaml, or 'fake'")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="default: the entry's max_tokens, or 256")
    parser.add_argument("--fake-reasoning", action="store_true",
                        help="with --model fake: thinking text before the JSON")
    args = parser.parse_args(argv)

    load_env()
    scheme = load_all()[PROBE_SCHEME]
    rendered = render_stage2(PROBE_ITEM, scheme, PROBE_CQ)
    spec = dict(FAKE_SPEC) if args.model == FAKE else model_spec(args.model)
    if args.model == FAKE and args.fake_reasoning:
        spec["reasoning"] = {"chat_template_kwargs": {"enable_thinking": True}}
    try:
        sampling = sampling_for(args.model, spec)
    except SchemeError as refusal:
        print(f"REFUSED: {refusal}")
        return 2
    # an entry with its own max_tokens knows better than a default meant for the
    # entries that answer without thinking first
    max_tokens = args.max_tokens or spec.get("max_tokens") or DEFAULT_MAX_TOKENS

    print(f"Model     : {spec.get('display_name', args.model)}  ({spec['model_id']})")
    print(f"Revision  : {spec.get('revision')}   tier: {spec.get('tier')}")
    print(f"Reasoning : {json.dumps(spec.get('reasoning') or {})}")
    print(f"Sampling  : temperature {sampling['temperature']}, top_p {sampling['top_p']}, "
          f"top_k {sampling['top_k']}")
    print(f"Question  : {PROBE_SCHEME} {PROBE_CQ}")
    print(f"Options   : {', '.join(rendered.answer_options)}")
    print("-" * 72)

    request = Request(
        model_id=spec["model_id"],
        prompt=rendered.text,
        prompt_version=rendered.prompt_version,
        sample_index=0,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        max_tokens=max_tokens,
        seed=0,
        json_schema=rendered.json_schema,
        logprobs=True,
        top_logprobs=20,
        revision=spec.get("revision"),
        tier=spec.get("tier"),
        reasoning=spec.get("reasoning") or {},
    )
    try:
        backend = build_backend(args.model, args.fake_reasoning)
        response = ask(request, backend=backend)
    except SchemeError as refusal:
        print(f"REFUSED: {refusal}")
        return 2

    if response.error:
        print(f"ERROR: {response.error}")
        return 1

    print(f"Response  : {response.content}")
    print(f"Latency   : {response.latency_s:.2f} s")
    print(f"Finish    : {response.finish_reason}")
    if response.usage:
        print(f"Tokens    : {json.dumps(response.usage)}")
    unusable = False
    prompt_tokens = (response.usage or {}).get("prompt_tokens")
    expected = len(rendered.text) / CHARS_PER_TOKEN
    if prompt_tokens is None:
        print("Prompt    : no prompt_tokens in usage, not checked")
    else:
        print(f"Prompt    : {prompt_tokens} tokens read, about {expected:.0f} expected "
              f"from {len(rendered.text)} characters")
        if prompt_tokens < PROMPT_FLOOR * expected:
            unusable = True
            print("WARNING: the server read far fewer tokens than the prompt holds: the chat "
                  "template dropped the content of the message. Check the content format "
                  "of the template (`--chat-template-content-format` in the `serve_args`).")
    if thinking_on(spec):
        count, source = reasoning_length(response)
        print(f"Thinking  : {count} {source}")
        if count == 0 and spec.get("reasoning_optional"):
            print("WARNING, not blocking: no reasoning. The entry may answer without it "
                  "(`reasoning_optional`), and the run measures how often it does.")
        elif count == 0:
            unusable = True
            print("WARNING: this entry reasons and there was no reasoning. "
                  "Check that the server passes the reasoning settings of the entry to the "
                  "template and that it was launched with the `serve_args` of the entry.")
        if response.finish_reason == "length":
            unusable = True
            print(f"WARNING: response cut off at {max_tokens} tokens "
                  f"(finish_reason = length): the reasoning used up the room for the "
                  f"JSON. Raise the entry's `max_tokens` in serving/models.yaml.")
    print()
    if not layout_holds(request, backend, blocking=sampling["samples"] == 1):
        unusable = True
    print()

    if not response.logprobs:
        print("LOGPROBS NOT AVAILABLE.")
        print("Set `supports_logprobs: false` for this model in serving/models.yaml.")
        print("`p_logprob` will stay empty and the manifest will say so.")
        return 1 if unusable or spec.get("supports_logprobs", True) else 0

    content = response.logprobs.get("content") or []
    position = answer_position(response.logprobs, rendered.answer_options, ANSWER_KEY)
    if position is None:
        print("Logprobs block present but no answer token inside the final JSON.")
        return 1

    token = str(content[position].get("token"))
    parsed = parse_content(response.content) or {}
    answer = str(parsed.get(ANSWER_KEY, ""))
    agrees = bool(answer) and answer.startswith(token.strip().strip('"').strip())
    print(f"Answer token: {token!r} at position {position} of {len(content)} "
          f"(inside the final JSON, after \"{ANSWER_KEY}\")")
    print(f"Consistent with the answer in the JSON ({answer!r}): {'yes' if agrees else 'NO'}")

    naive = naive_position(response.logprobs, rendered.answer_options)
    if naive is not None and naive != position:
        print(f"WARNING: the first token that looks like an answer is at position {naive} "
              f"({str(content[naive].get('token'))!r}), before the JSON. Reading that one "
              f"would have given the probabilities of the reasoning, not of the answer.")
    print()

    entries = content[position].get("top_logprobs") or []
    print(f"Alternatives at position {position} (first token of `{ANSWER_KEY}`):")
    print(f"  {'token':<24}{'logprob':>12}{'p':>10}")
    print("  " + "-" * 44)
    for entry in sorted(entries, key=lambda e: -e.get("logprob", -math.inf))[:20]:
        logprob = float(entry.get("logprob", -math.inf))
        print(f"  {str(entry.get('token'))!r:<24}{logprob:>12.4f}{math.exp(logprob):>10.4f}")

    masses, partial = logprob_masses(response.logprobs, rendered.answer_options, ANSWER_KEY)
    print("\nRenormalised over the admitted answers:")
    for option in rendered.answer_options:
        print(f"  {option:<24}{masses.get(option, 0.0):>10.4f}")
    if partial:
        print("\nWARNING: an admitted answer does not appear among the top_logprobs.")
        print("The parser marks such rows `logprob_partial`.")
    return 0 if agrees and not unusable else 1


if __name__ == "__main__":
    sys.exit(main())
