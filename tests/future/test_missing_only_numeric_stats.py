import pandas as pd
import pytest

from evidently import Dataset


@pytest.mark.parametrize("values", [[], [None, None], [float("nan"), None]])
@pytest.mark.filterwarnings("error:invalid value encountered in scalar divide:RuntimeWarning")
def test_dataset_infinite_share_is_zero_with_no_numeric_observations(values):
    data = pd.DataFrame({"value": pd.Series(values, dtype="float64")})
    stats = Dataset.from_pandas(data).stats().column_stats["value"].numerical_stats
    assert stats is not None
    assert stats.infinite.count == 0
    assert stats.infinite.share == 0.0
