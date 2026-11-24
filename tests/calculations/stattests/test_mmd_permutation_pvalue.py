import numpy as np
import pytest

from evidently.legacy.calculations.stattests.mmd_stattest import mmd_pval


def test_mmd_monte_carlo_pvalue_is_never_zero_for_separated_samples():
    reference = np.zeros((25, 1))
    current = np.ones((25, 1))
    state = np.random.get_state()
    try:
        np.random.seed(0)
        p_value, _ = mmd_pval(reference, current)
    finally:
        np.random.set_state(state)
    # With 100 random permutations the observed labeling is one additional
    # possible outcome, so the Monte Carlo p-value is at least 1/101.
    assert p_value == pytest.approx(1 / 101)


def test_mmd_monte_carlo_pvalue_is_one_for_equal_constant_samples():
    sample = np.ones((5, 1))
    state = np.random.get_state()
    try:
        p_value, _ = mmd_pval(sample, sample)
    finally:
        np.random.set_state(state)
    assert p_value == 1.0
