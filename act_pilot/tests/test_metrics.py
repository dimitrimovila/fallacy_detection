import math

import pytest

from act_pilot.metrics import (
    balanced_accuracy,
    gini,
    mcnemar_exact,
    paired_bootstrap_ba,
    recalls,
    weighted_gini,
)


def test_gini():
    assert gini([]) == 0.0
    assert gini([1, 1, 1]) == 0.0
    assert gini([0, 1]) == pytest.approx(0.5)


def test_weighted_gini():
    y = [1, 1, 0, 0]
    assert weighted_gini(y, [1, 1, 0, 0]) == 0.0
    assert weighted_gini(y, [1, 0, 1, 0]) == pytest.approx(0.5)
    assert weighted_gini(y, [1, 1, 1, 1]) == pytest.approx(0.5)


def test_balanced_accuracy_and_recalls():
    y = [1, 1, 1, 0]
    p = [1, 0, 1, 0]
    assert recalls(y, p) == (pytest.approx(2 / 3), 1.0)
    assert balanced_accuracy(y, p) == pytest.approx((2 / 3 + 1) / 2)
    assert math.isnan(balanced_accuracy([1, 1], [1, 0]))
    assert balanced_accuracy(y, [1, 1, 1, 1]) == pytest.approx(0.5)


def test_bootstrap_identical_predictions():
    y = [1] * 20 + [0] * 20
    p = [1, 0] * 20
    d = paired_bootstrap_ba(y, p, p, n_boot=300, seed=1)
    assert d.value == 0 and d.low == 0 and d.high == 0
    assert not d.excludes_zero_in_favour


def test_bootstrap_detects_clear_improvement():
    y = [1] * 50 + [0] * 50
    good = y[:]
    bad = [1] * 100
    d = paired_bootstrap_ba(y, good, bad, n_boot=1000, seed=3)
    assert d.value == pytest.approx(0.5)
    assert d.low > 0.3 and d.excludes_zero_in_favour
    reverse = paired_bootstrap_ba(y, bad, good, n_boot=1000, seed=3)
    assert reverse.high < 0


def test_bootstrap_is_reproducible():
    y = [1] * 15 + [0] * 25
    a = [1, 0, 0, 1] * 10
    b = [0, 1, 0, 0] * 10
    assert paired_bootstrap_ba(y, a, b, n_boot=200, seed=9) == paired_bootstrap_ba(
        y, a, b, n_boot=200, seed=9)


def test_mcnemar():
    y = [1, 1, 1, 1, 0, 0]
    a = [1, 1, 1, 1, 0, 0]
    b = [0, 0, 0, 1, 0, 1]
    m = mcnemar_exact(y, a, b)
    assert (m.b, m.c) == (4, 0)
    assert m.p_value == pytest.approx(0.125)
    assert mcnemar_exact(y, a, a).p_value == 1.0
