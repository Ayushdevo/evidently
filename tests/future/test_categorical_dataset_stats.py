import pandas as pd
import pytest

from evidently import Dataset


@pytest.mark.parametrize(
    ("values", "expected"),
    [(["a", "b", "b"], "b"), (["b", "a", "b", "a"], "a"), ([], None), ([None, None], None)],
)
def test_categorical_stats_use_observed_counts(values, expected):
    column = pd.Categorical(values, categories=["a", "b", "unused"])
    stats = Dataset.from_pandas(pd.DataFrame({"category": column})).stats().column_stats["category"].categorical_stats
    assert stats is not None
    assert stats.unique_count == len(stats.label_stats)
    assert set(stats.label_stats) == set(values) - {None}
    if expected is None:
        assert stats.most_common is None
    else:
        assert stats.most_common is not None
        label, label_stats = stats.most_common
        assert label == expected
        assert label_stats.count.count == values.count(expected)
        assert label_stats.count.share == values.count(expected) / len(values)
