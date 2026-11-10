import pandas as pd
import pytest

from evidently import Dataset


@pytest.mark.parametrize("dtype", ["float64", "object", "category"])
def test_missing_value_stats_count_rows_including_missing(dtype):
    data = pd.DataFrame({"feature": pd.Series([None, 1, None, 2], dtype=dtype)})
    missing = Dataset.from_pandas(data).stats().column_stats["feature"].general_stats.missing_values
    assert missing.count == 2
    assert missing.share == 0.5


def test_empty_dataset_has_zero_missing_value_share():
    data = pd.DataFrame({"feature": pd.Series(dtype="object")})
    missing = Dataset.from_pandas(data).stats().column_stats["feature"].general_stats.missing_values
    assert missing.count == 0
    assert missing.share == 0.0
