"""tests/validate_schemes.py still passes, now from pytest.

The script lives next to the tests and still prints a readable report when run
on its own (``python tests/validate_schemes.py``); what pytest owns is the
pass/fail half of it.  ``tests/`` is on ``sys.path`` (see conftest), so the
import below finds it there.
"""

from __future__ import annotations

import validate_schemes
from argfallacy.paths import SCHEMES_ROOT
from argfallacy.schemes import load_all


def test_snapshot_unchanged(schemes, vocab):
    assert validate_schemes.check(schemes, vocab) == []


def test_every_scheme_is_in_the_snapshot(schemes):
    assert set(schemes) == set(validate_schemes.SNAPSHOT)


def test_v2_snapshot_unchanged(vocab):
    assert validate_schemes.check(
        load_all(SCHEMES_ROOT / "v2"), vocab, validate_schemes.SNAPSHOT_V2
    ) == []


def test_v2_declares_one_version():
    from argfallacy.client.run import schemes_tag

    assert schemes_tag(SCHEMES_ROOT / "v2") == "2.0"
    assert schemes_tag() == "1.1"


def test_v3_and_v3b_snapshots_unchanged(vocab):
    from argfallacy.client.run import schemes_tag

    for folder, tag in (("v3", "2.1"), ("v3b", "2.1b")):
        assert validate_schemes.check(
            load_all(SCHEMES_ROOT / folder), vocab, validate_schemes.SNAPSHOT_V3
        ) == []
        assert schemes_tag(SCHEMES_ROOT / folder) == tag


def test_v4_snapshot_unchanged(vocab):
    from argfallacy.client.run import schemes_tag

    assert validate_schemes.check(
        load_all(SCHEMES_ROOT / "v4"), vocab, validate_schemes.SNAPSHOT_V4
    ) == []
    assert schemes_tag(SCHEMES_ROOT / "v4") == "2.1"
