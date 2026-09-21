"""tests/validate_schemes.py still passes, now from pytest.

The script lives next to the tests and still prints a readable report when run
on its own (``python tests/validate_schemes.py``); what pytest owns is the
pass/fail half of it.  ``tests/`` is on ``sys.path`` (see conftest), so the
import below finds it there.
"""

from __future__ import annotations

import validate_schemes


def test_snapshot_unchanged(schemes, vocab):
    assert validate_schemes.check(schemes, vocab) == []


def test_every_scheme_is_in_the_snapshot(schemes):
    assert set(schemes) == set(validate_schemes.SNAPSHOT)
