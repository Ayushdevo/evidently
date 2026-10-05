import pandas as pd
import pytest

from evidently.legacy.metrics import ClassificationQualityByClass
from evidently.legacy.metrics.classification_performance.quality_by_class_metric import (
    ClassificationQualityByClassRenderer,
)
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report


@pytest.mark.parametrize("target_names", [None, {0: "zero", 1: "one", 2: "two", 3: "three"}])
@pytest.mark.parametrize("current_labels,reference_labels", [([0, 1], [0, 1, 2, 3]), ([2, 3], [0, 1])])
def test_reference_heatmap_uses_reference_class_labels(target_names, current_labels, reference_labels):
    metric = ClassificationQualityByClass()
    report = Report(metrics=[metric])
    report.run(
        current_data=pd.DataFrame({"target": current_labels, "prediction": current_labels}),
        reference_data=pd.DataFrame({"target": reference_labels, "prediction": reference_labels}),
        column_mapping=ColumnMapping(target_names=target_names),
    )

    widget = ClassificationQualityByClassRenderer().render_html(metric)[1]
    current_trace, reference_trace = widget.params["data"]
    for trace, labels in [(current_trace, current_labels), (reference_trace, reference_labels)]:
        expected_names = [target_names[label] if target_names is not None else str(label) for label in labels]
        assert trace["x"] == expected_names
        assert trace["z"].shape[1] == len(trace["x"])
