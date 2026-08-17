"""Small, dependency-light statistical building blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-9


def zscore(value: float, mean: float, std: float) -> float:
    if any(pd.isna(x) for x in (value, mean, std)) or abs(std) < EPSILON:
        return 0.0
    return float((value - mean) / std)


def median_absolute_deviation(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    median = np.median(arr)
    return float(np.median(np.abs(arr - median)))


def modified_zscore(value: float, median: float, mad: float) -> float:
    if any(pd.isna(x) for x in (value, median, mad)) or abs(mad) < EPSILON:
        return 0.0
    return float(0.6745 * (value - median) / mad)


def iqr_bounds(values, multiplier: float = 1.5) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def percentile_threshold(values, percentile: float = 0.99) -> float:
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1")
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, percentile)) if arr.size else float("nan")


def rolling_mean_std(series: pd.Series, window: int, min_periods: int = 2) -> pd.DataFrame:
    shifted = series.shift(1)
    return pd.DataFrame(
        {
            "rolling_mean": shifted.rolling(window, min_periods=min_periods).mean(),
            "rolling_std": shifted.rolling(window, min_periods=min_periods).std(ddof=0),
        },
        index=series.index,
    )


def robust_statistics(values) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: 0.0 for k in ("mean", "std", "median", "mad", "q1", "q3", "p99")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "mad": median_absolute_deviation(arr),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
        "p99": float(np.quantile(arr, 0.99)),
    }
