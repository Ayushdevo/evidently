import numpy as np
import pytest

from evidently.legacy.calculations.stattests.mmd_stattest import mmd_pval


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("invalid_in_reference", [False, True])
def test_mmd_rejects_nonfinite_observations(invalid, invalid_in_reference):
    good = np.array([0.0, 1.0, 2.0]).reshape(-1, 1)
    bad = np.array([0.0, invalid, 2.0]).reshape(-1, 1)
    reference, current = (bad, good) if invalid_in_reference else (good, bad)
    with pytest.raises(ValueError, match="finite observations"):
        mmd_pval(reference, current)
