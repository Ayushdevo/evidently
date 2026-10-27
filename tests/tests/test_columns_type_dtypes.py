from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evidently.legacy.metrics import DatasetSummaryMetric
from evidently.legacy.tests.base_test import TestStatus
from evidently.legacy.tests.data_integrity_tests import TestColumnsType
from evidently.legacy.tests.data_integrity_tests import _get_dtype_type


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (int, np.dtype(int).type),
        (float, np.float64),
        (str, pd.api.types.pandas_dtype(str).type),
        (bool, np.bool_),
        (object, np.object_),
        ("int64", np.int64),
        ("float", np.float64),
        ("str", pd.api.types.pandas_dtype("str").type),
        ("bool", np.bool_),
        ("object", np.object_),
        (np.int32, np.int32),
        (np.dtype("int64"), np.int64),
        ("datetime64[ns]", np.datetime64),
        ("category", pd.CategoricalDtype.type),
        (pd.Int64Dtype(), np.int64),
        ("Int64", np.int64),
        (pd.StringDtype(), pd.StringDtype().type),
        ("string", pd.StringDtype().type),
        ("number", np.number),
        (np.number, np.number),
        ("datetime", np.datetime64),
        ("timedelta", np.timedelta64),
        ("datetimetz", pd.DatetimeTZDtype.type),
        ("datetime64tz", pd.DatetimeTZDtype.type),
        (pd.DatetimeTZDtype(tz="UTC"), pd.DatetimeTZDtype.type),
    ],
)
def test_column_dtype_scalar_types(dtype: Any, expected: type) -> None:
    assert _get_dtype_type(dtype) is expected


@pytest.mark.parametrize("from_reference", [False, True])
def test_columns_type_compares_nullable_and_datetime_dtypes(from_reference: bool) -> None:
    actual = {"count": pd.Int64Dtype(), "when": np.dtype("datetime64[ns]")}
    expected = {"count": "number", "when": "datetime"}
    result = SimpleNamespace(
        current=SimpleNamespace(columns_type=actual),
        reference=SimpleNamespace(columns_type=expected) if from_reference else None,
    )
    check = TestColumnsType(columns_type=None if from_reference else expected)
    with patch.object(DatasetSummaryMetric, "get_result", return_value=result):
        assert check.check().status == TestStatus.SUCCESS


def test_columns_type_still_reports_mismatches() -> None:
    result = SimpleNamespace(current=SimpleNamespace(columns_type={"value": np.dtype("int64")}))
    check = TestColumnsType(columns_type={"value": "float64"})
    with patch.object(DatasetSummaryMetric, "get_result", return_value=result):
        assert check.check().status == TestStatus.FAIL


def test_invalid_column_dtype_still_raises() -> None:
    with pytest.raises(TypeError):
        _get_dtype_type("not-a-dtype")
