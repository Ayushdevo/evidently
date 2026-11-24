import numpy as np
import pytest

from evidently.legacy.calculations.stattests.mmd_stattest import mmd_pval


@pytest.mark.parametrize("reference_size,current_size", [(0, 2), (1, 2), (2, 0), (2, 1)])
def test_mmd_requires_two_observations_per_sample(reference_size, current_size):
    reference = np.ones((reference_size, 1))
    current = np.ones((current_size, 1))
    with pytest.raises(ValueError, match="at least two observations in each sample"):
        mmd_pval(reference, current)
