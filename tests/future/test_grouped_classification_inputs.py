import pandas as pd

from evidently import BinaryClassification
from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently.metrics import Accuracy
from evidently.metrics.group_by import GroupBy


def test_grouped_accuracy_uses_group_inputs_and_separate_legacy_results():
    definition = DataDefinition(classification=[BinaryClassification(target="label", prediction_labels="prediction")])
    current = Dataset.from_pandas(
        pd.DataFrame({"label": [0, 1, 0, 1], "prediction": [0, 1, 1, 0], "group": ["A", "A", "B", "B"]}),
        data_definition=definition,
    )
    reference = Dataset.from_pandas(
        pd.DataFrame({"label": [0, 1, 0, 1], "prediction": [1, 0, 0, 1], "group": ["A", "A", "B", "B"]}),
        data_definition=definition,
    )
    accuracy = Accuracy()
    grouped = GroupBy(accuracy, column_name="group")
    result = Report([accuracy, grouped]).run(current, reference)

    assert result.metric_results[accuracy.metric_id].value == 0.5
    for label, expected in (("A", 1.0), ("B", 0.0)):
        metric_id = grouped.label_metric(label).metric_id
        assert result.metric_results[metric_id].value == expected
        assert result._context.get_reference_metric_result(metric_id).value == 1.0 - expected
