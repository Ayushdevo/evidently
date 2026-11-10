import pandas as pd

from evidently import BinaryClassification
from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently.metrics import PrecisionByLabel


def test_precision_by_label_uses_the_selected_classification_labels():
    definition = DataDefinition(
        classification=[
            BinaryClassification(target="a", prediction_labels="a_pred", labels={0: "cold", 1: "hot"}),
            BinaryClassification(
                name="other", target="b", prediction_labels="b_pred", pos_label=3, labels={2: "no", 3: "yes"}
            ),
        ]
    )
    dataset = Dataset.from_pandas(
        pd.DataFrame({"a": [0, 1, 0, 1], "a_pred": [0, 1, 0, 1], "b": [2, 3, 2, 3], "b_pred": [2, 3, 2, 3]}),
        data_definition=definition,
    )
    primary = PrecisionByLabel()
    other = PrecisionByLabel(classification_name="other")
    results = Report([primary, other]).run(dataset).metric_results
    assert {key: val.value for key, val in results[primary.metric_id].values.items()} == {"cold": 1.0, "hot": 1.0}
    assert {key: val.value for key, val in results[other.metric_id].values.items()} == {"no": 1.0, "yes": 1.0}
