import math

import pandas as pd
import pytest

from anomaly_detector.statistics import (
    iqr_bounds,
    median_absolute_deviation,
    modified_zscore,
    percentile_threshold,
    rolling_mean_std,
    zscore,
)


def test_zscore_formula():
    assert zscore(15, 10, 2) == 2.5


def test_zscore_zero_variance_is_safe():
    assert zscore(15, 10, 0) == 0.0


def test_mad_formula():
    assert median_absolute_deviation([1, 2, 2, 4, 6]) == 1.0


def test_mad_empty_is_safe():
    assert median_absolute_deviation([]) == 0.0


def test_modified_zscore_formula():
    assert modified_zscore(12, 10, 2) == pytest.approx(0.6745)


def test_modified_zscore_zero_mad_is_safe():
    assert modified_zscore(12, 10, 0) == 0.0


def test_iqr_bounds_formula():
    lower, upper = iqr_bounds([1, 2, 3, 4, 5], multiplier=1.5)
    assert lower == pytest.approx(-1.0)
    assert upper == pytest.approx(7.0)


def test_percentile_threshold():
    assert percentile_threshold([1, 2, 3, 4], 0.5) == pytest.approx(2.5)


def test_percentile_rejects_invalid_probability():
    with pytest.raises(ValueError):
        percentile_threshold([1, 2], 1.2)


def test_rolling_statistics_exclude_current_row():
    output = rolling_mean_std(pd.Series([10.0, 20.0, 1000.0]), window=2, min_periods=1)
    assert math.isnan(output.loc[0, "rolling_mean"])
    assert output.loc[2, "rolling_mean"] == 15.0
