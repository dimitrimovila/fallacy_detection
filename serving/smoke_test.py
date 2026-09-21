#!/usr/bin/env python3
"""The first command to run against a newly served model.

Sends one stage-two call with structured output and logprobs and prints what
came back: the answer, the token where the answer value starts *inside the final
JSON* and its position, the alternatives at that token, and the latency.

Two checks matter more than the rest:

* whether logprobs arrive at all.  That decides ``supports_logprobs`` in
  ``serving/models.yaml`` and whether ``p_logprob`` will exist for that model;
* whether the token read is the answer's.  A reasoning model writes thinking
  before the object, and that text can contain ``yes`` or a whole draft object.
  The script says where a naive reader — first answer-looking token — would have
  landed, and warns if that is somewhere else.

    python serving/smoke_test.py --model qwen3_6_27b   # a key of serving/models.yaml
    python serving/smoke_test.py --model fake          # the fake backend, no network
    python serving/smoke_test.py --model fake --fake-reasoning

Endpoint and key come from ``LLM_BASE_URL`` and ``LLM_API_KEY`` in the ``.env``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argfallacy.client import Request, ask, model_spec  # noqa: E402
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
    "model_id": "fake-model", "display_name": "Backend finto", "revision": "fake",
    "tier": 0, "reasoning": {},
}


def build_backend(model_name: str, fake_reasoning: bool):
    """The fake for ``--model fake``, the real endpoint otherwise."""
    if model_name == FAKE:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from fakes.llm import FakeBackend

        print("Backend finto: nessuna chiamata di rete.")
        if fake_reasoning:
            print("Con testo di ragionamento e una bozza di risposta prima del JSON.")
        print()
        return FakeBackend(reasoning=fake_reasoning)
    base_url, _ = endpoint()
    print(f"Endpoint: {base_url}\n")
    return None


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
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--fake-reasoning", action="store_true",
                        help="with --model fake: thinking text before the JSON")
    args = parser.parse_args(argv)

    load_env()
    scheme = load_all()[PROBE_SCHEME]
    rendered = render_stage2(PROBE_ITEM, scheme, PROBE_CQ)
    spec = FAKE_SPEC if args.model == FAKE else model_spec(args.model)

    print(f"Modello   : {spec.get('display_name', args.model)}  ({spec['model_id']})")
    print(f"Revisione : {spec.get('revision')}   fascia: {spec.get('tier')}")
    print(f"Reasoning : {json.dumps(spec.get('reasoning') or {})}")
    print(f"Domanda   : {PROBE_SCHEME} {PROBE_CQ}")
    print(f"Opzioni   : {', '.join(rendered.answer_options)}")
    print("-" * 72)

    request = Request(
        model_id=spec["model_id"],
        prompt=rendered.text,
        prompt_version=rendered.prompt_version,
        sample_index=0,
        temperature=0.0,
        max_tokens=args.max_tokens,
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
        print(f"RIFIUTATO: {refusal}")
        return 2

    if response.error:
        print(f"ERRORE: {response.error}")
        return 1

    print(f"Risposta  : {response.content}")
    print(f"Latenza   : {response.latency_s:.2f} s")
    print(f"Fine      : {response.finish_reason}")
    if response.usage:
        print(f"Token     : {json.dumps(response.usage)}")
    print()

    if not response.logprobs:
        print("LOGPROB NON DISPONIBILI.")
        print("Metti `supports_logprobs: false` per questo modello in serving/models.yaml.")
        print("`p_logprob` restera' vuota e il manifest lo dichiarera'.")
        return 0

    content = response.logprobs.get("content") or []
    position = answer_position(response.logprobs, rendered.answer_options, ANSWER_KEY)
    if position is None:
        print("Blocco logprobs presente ma nessun token di risposta dentro il JSON finale.")
        return 1

    token = str(content[position].get("token"))
    parsed = parse_content(response.content) or {}
    answer = str(parsed.get(ANSWER_KEY, ""))
    agrees = bool(answer) and answer.startswith(token.strip().strip('"').strip())
    print(f"Token della risposta: {token!r} alla posizione {position} di {len(content)} "
          f"(dentro il JSON finale, dopo \"{ANSWER_KEY}\")")
    print(f"Coerente con la risposta del JSON ({answer!r}): {'si' if agrees else 'NO'}")

    naive = naive_position(response.logprobs, rendered.answer_options)
    if naive is not None and naive != position:
        print(f"ATTENZIONE: il primo token che sembra una risposta e' alla posizione {naive} "
              f"({str(content[naive].get('token'))!r}), prima del JSON. Leggere quello "
              f"avrebbe dato le probabilita' del ragionamento, non della risposta.")
    print()

    entries = content[position].get("top_logprobs") or []
    print(f"Alternative alla posizione {position} (primo token di `{ANSWER_KEY}`):")
    print(f"  {'token':<24}{'logprob':>12}{'p':>10}")
    print("  " + "-" * 44)
    for entry in sorted(entries, key=lambda e: -e.get("logprob", -math.inf))[:20]:
        logprob = float(entry.get("logprob", -math.inf))
        print(f"  {str(entry.get('token'))!r:<24}{logprob:>12.4f}{math.exp(logprob):>10.4f}")

    masses, partial = logprob_masses(response.logprobs, rendered.answer_options, ANSWER_KEY)
    print("\nRinormalizzato sulle risposte ammesse:")
    for option in rendered.answer_options:
        print(f"  {option:<24}{masses.get(option, 0.0):>10.4f}")
    if partial:
        print("\nATTENZIONE: una risposta ammessa non compare fra i top_logprobs.")
        print("Le righe cosi' vengono marcate `logprob_partial` dal parser.")
    return 0 if agrees else 1


if __name__ == "__main__":
    sys.exit(main())
