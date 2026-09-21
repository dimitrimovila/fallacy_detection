"""The prompts: rendering, answer options, response format, versions.

The point of most of these is one rule: the words of a critical question come
from the YAML, and the renderer only places them.
"""

from __future__ import annotations

import pytest

from argfallacy.prompts import (
    CANNOT_BE_DETERMINED,
    NOT_APPLICABLE,
    answer_options,
    load_answer_meanings,
    render_stage1,
    render_stage2,
)
from argfallacy.schemes import SchemeError

ITEM = {"item_id": "0123456789abcdef", "text": "Everyone I know buys it, so it must be good."}


def every_cq(schemes):
    for scheme_id in sorted(schemes):
        for cq_id in schemes[scheme_id].cq_order:
            yield schemes[scheme_id], cq_id


def test_every_cq_text_appears_verbatim(schemes):
    for scheme, cq_id in every_cq(schemes):
        prompt = render_stage2(ITEM, scheme, cq_id)
        assert scheme.cqs[cq_id].text in prompt.text, f"{scheme.scheme_id} {cq_id}"


def test_every_cq_renders_without_a_missing_field(schemes):
    rendered = [render_stage2(ITEM, s, c).text for s, c in every_cq(schemes)]
    assert len(rendered) == sum(len(s.cqs) for s in schemes.values()) == 48
    assert all("{{" not in text for text in rendered)


def test_ad_hominem_cq1_has_its_own_options(schemes):
    assert answer_options(schemes["ad_hominem"], "CQ1") == (
        "positive", "negative", CANNOT_BE_DETERMINED, NOT_APPLICABLE
    )


def test_every_other_cq_is_yes_no(schemes):
    for scheme, cq_id in every_cq(schemes):
        if (scheme.scheme_id, cq_id) == ("ad_hominem", "CQ1"):
            continue
        assert answer_options(scheme, cq_id) == (
            "yes", "no", CANNOT_BE_DETERMINED, NOT_APPLICABLE
        )


def _answers_block(prompt: str) -> str:
    return prompt.split("## The answers you may give")[1].split("## What to do")[0]


def test_the_answer_definitions_name_no_fallacy(schemes, vocab):
    import re

    names = set()
    for group in ("terminals", "coarse_labels", "out_of_scope"):
        for fallacy_id, spec in (vocab.get(group) or {}).items():
            names.add(fallacy_id.replace("_", " "))
            if isinstance(spec, dict) and spec.get("label"):
                names.add(spec["label"].lower())
    for scheme, cq_id in every_cq(schemes):
        block = _answers_block(render_stage2(ITEM, scheme, cq_id).text).lower()
        found = [n for n in names if re.search(rf"\b{re.escape(n)}\b", block)]
        assert not found, f"{scheme.scheme_id} {cq_id}: {found}"


def test_every_own_definition_matches_its_diagram(schemes):
    meanings = load_answer_meanings()
    assert meanings["per_cq"], "ad hominem CQ1 must have its own definitions"
    for scheme_id, cqs in meanings["per_cq"].items():
        assert scheme_id in schemes, scheme_id
        for cq_id, options in cqs.items():
            assert set(options) == set(schemes[scheme_id].answer_space(cq_id))


def test_no_prompt_built_from_items_csv_says_nan(schemes):
    from argfallacy.client import RunConfig, load_items, select_items
    from argfallacy.paths import REPO_ROOT

    try:
        frame = load_items()
    except SchemeError:
        pytest.skip("no data/items.csv")
    config = RunConfig.load(REPO_ROOT / "configs" / "pilot.yaml")
    for item in select_items(config, frame):
        scheme = schemes[item["gold_scheme"]]
        for prompt in (render_stage1(item, schemes).text,
                       render_stage2(item, scheme, scheme.cq_order[0]).text):
            assert "nan" not in prompt.split()
