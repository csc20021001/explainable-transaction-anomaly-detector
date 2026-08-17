import numpy as np
import pandas as pd

from anomaly_detector.features import build_features, chronological_split, normalize_features


def test_first_transaction_has_no_history(base_transactions):
    features = build_features(base_transactions)
    assert features.iloc[0]["history_count"] == 0
    assert features.iloc[0]["amount_zscore"] == 0


def test_hourly_transaction_count_uses_rolling_window(base_transactions):
    base_transactions.loc[1, "timestamp"] = base_transactions.loc[0, "timestamp"] + pd.Timedelta(minutes=20)
    features = build_features(base_transactions)
    second = features[features.transaction_id.eq("TX-1")].iloc[0]
    assert second["hourly_transaction_count"] == 2


def test_daily_transaction_amount_includes_current(base_transactions):
    features = build_features(base_transactions)
    second = features[features.transaction_id.eq("TX-1")].iloc[0]
    assert second["daily_transaction_amount"] == 201


def test_minutes_since_incoming(base_transactions):
    features = build_features(base_transactions)
    withdrawal = features[features.transaction_id.eq("TX-4")].iloc[0]
    assert withdrawal["minutes_since_incoming"] == 60


def test_new_device_flag(base_transactions):
    base_transactions.loc[6, "device_id"] = "DEV-NEW"
    features = build_features(base_transactions)
    assert features.loc[features.transaction_id.eq("TX-6"), "is_new_device"].iloc[0] == 1


def test_country_change_flag(base_transactions):
    base_transactions.loc[6, "ip_country"] = "US"
    features = build_features(base_transactions)
    assert features.loc[features.transaction_id.eq("TX-6"), "country_change"].iloc[0] == 1


def test_dormancy_days(base_transactions):
    base_transactions.loc[7, "timestamp"] = base_transactions.loc[6, "timestamp"] + pd.Timedelta(days=50)
    features = build_features(base_transactions)
    assert features.loc[features.transaction_id.eq("TX-7"), "dormancy_days"].iloc[0] == 50


def test_identical_values_do_not_create_infinite_statistics(base_transactions):
    base_transactions["amount_usd"] = 100.0
    features = build_features(base_transactions)
    assert np.isfinite(features["amount_zscore"]).all()
    assert np.isfinite(features["amount_modified_zscore"]).all()
    assert np.isfinite(features["amount_rolling_zscore"]).all()


def test_rolling_amount_baseline_excludes_current(base_transactions):
    base_transactions.loc[7, "amount_usd"] = 100_000
    features = build_features(base_transactions)
    last = features[features.transaction_id.eq("TX-7")].iloc[0]
    assert last["rolling_mean_amount"] < 200


def test_chronological_split_has_all_partitions():
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=10, tz="UTC"))
    assert chronological_split(timestamps).value_counts().to_dict() == {"train": 6, "validation": 2, "test": 2}


def test_normalization_handles_constant_feature(base_transactions):
    features = build_features(base_transactions)
    features["country_change"] = 0
    normalized = normalize_features(features, ["country_change"])
    assert normalized["country_change_normalized"].eq(0).all()


def test_empty_input_is_supported(base_transactions):
    empty = build_features(base_transactions.iloc[0:0])
    assert empty.empty
    assert "amount_zscore" in empty.columns
