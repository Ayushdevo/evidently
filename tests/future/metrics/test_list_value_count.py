import numpy as np
import pandas as pd
import pytest

from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently.metrics import InListValueCount
from evidently.metrics import OutListValueCount


@pytest.mark.parametrize("metric_type,expected_count", [(InListValueCount, 2), (OutListValueCount, 1)])
@pytest.mark.parametrize(
    "data,values",
    [
        (["a", "a", "b", None], ["a", "a"]),
        ([1, 1, 2, None], [1, 1]),
        ([True, True, False, None], [True, True]),
        (["a", "a", "b", None], ["a", "a", "absent", np.nan]),
    ],
)
def test_list_value_counts_ignore_duplicate_filters(metric_type, expected_count, data, values):
    definition = DataDefinition(categorical_columns=["value"])
    current = Dataset.from_pandas(pd.DataFrame({"value": data}), data_definition=definition)
    reference = Dataset.from_pandas(pd.DataFrame({"value": data}), data_definition=definition)
    metric = metric_type(column="value", values=values)
    result = Report([metric]).run(current, reference)
    current_result = result._context.get_metric_result(metric)
    reference_result = result._context.get_reference_metric_result(metric.get_fingerprint())
    for value in (current_result, reference_result):
        assert value.count.value == expected_count
        assert value.share.value == pytest.approx(expected_count / 3)


@pytest.mark.parametrize("metric_type,expected_count", [(InListValueCount, 0), (OutListValueCount, 3)])
def test_list_value_counts_empty_filter(metric_type, expected_count):
    dataset = Dataset.from_pandas(pd.DataFrame({"value": ["a", "a", "b", None]}))
    metric = metric_type(column="value", values=[])
    value = Report([metric]).run(dataset)._context.get_metric_result(metric)
    assert value.count.value == expected_count
    assert value.share.value == expected_count / 3
