import json

import pandas as pd
import pytest

from anomaly_detector.config import get_profile
from anomaly_detector.detector import detect, score_transaction
from anomaly_detector.evaluation import binary_metrics, evaluate_profiles
from anomaly_detector.features import build_features


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        get_profile("reckless")


def test_normal_transaction_has_no_reason(base_transactions):
    row = build_features(base_transactions).iloc[0]
    alert = score_transaction(row)
    assert alert["reason_codes"] == []
    assert alert["anomaly_score"] == 0


def test_large_transaction_has_statistical_reason(base_transactions):
    extra = base_transactions.iloc[[-1]].copy()
    extra["transaction_id"] = "TX-LARGE"
    extra["timestamp"] = pd.Timestamp("2025-01-03T00:00:00Z")
    extra["amount_usd"] = 10_000
    features = build_features(pd.concat([base_transactions, extra], ignore_index=True))
    alert = score_transaction(features[features.transaction_id.eq("TX-LARGE")].iloc[0])
    assert "AMOUNT_STATISTICAL_OUTLIER" in alert["reason_codes"]
    assert "modified_zscore" in alert["explanation"]


def test_new_device_withdrawal_reason(base_transactions):
    base_transactions.loc[7, "transaction_type"] = "withdrawal"
    base_transactions.loc[7, "device_id"] = "DEV-NEW"
    alert = score_transaction(build_features(base_transactions).iloc[-1])
    assert "NEW_DEVICE_WITHDRAWAL" in alert["reason_codes"]


def test_first_transaction_does_not_trigger_new_device(base_transactions):
    alert = score_transaction(build_features(base_transactions.iloc[[0]]).iloc[0])
    assert "NEW_DEVICE_WITHDRAWAL" not in alert["reason_codes"]


def test_unusual_night_hour_reason(base_transactions):
    extra = base_transactions.iloc[[-1]].copy()
    extra["transaction_id"] = "TX-NIGHT"
    extra["timestamp"] = pd.Timestamp("2025-01-03T02:00:00Z")
    features = build_features(pd.concat([base_transactions, extra], ignore_index=True))
    alert = score_transaction(features[features.transaction_id.eq("TX-NIGHT")].iloc[0])
    assert "UNUSUAL_TRANSACTION_HOUR" in alert["reason_codes"]


def test_every_alert_contains_reason_and_evidence(base_transactions):
    base_transactions.loc[7, "transaction_type"] = "withdrawal"
    base_transactions.loc[7, "device_id"] = "DEV-NEW"
    result = detect(build_features(base_transactions))
    assert all(alert["reason_codes"] for alert in result.alerts)
    assert all("evidence_by_reason" in alert["explanation"] for alert in result.alerts)


def test_binary_metrics_known_confusion_matrix():
    metrics = binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["confusion_matrix"] == {"true_negative": 1, "false_positive": 1, "false_negative": 1, "true_positive": 1}


def test_binary_metrics_empty_positive_denominators():
    metrics = binary_metrics([0, 0], [0, 0])
    assert metrics["precision"] == 0
    assert metrics["recall"] == 0
    assert metrics["false_negative_rate"] == 0


def test_binary_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        binary_metrics([0], [0, 1])


def test_evaluation_compares_three_profiles(base_transactions):
    base_transactions.loc[7, "ground_truth_label"] = 1
    base_transactions.loc[7, "anomaly_type"] = "new_device_withdrawal"
    base_transactions.loc[7, "device_id"] = "DEV-X"
    base_transactions.loc[7, "transaction_type"] = "withdrawal"
    features = build_features(base_transactions)
    report, comparison, cases = evaluate_profiles(features, evaluation_split="test")
    assert set(comparison["profile"]) == {"conservative", "balanced", "sensitive"}
    assert "data_leakage_assessment" in report["evaluation_policy"]
    assert set(["false_positives", "false_negatives", "balanced"]).issubset(cases)


def test_reason_codes_column_is_json(base_transactions):
    scored = detect(build_features(base_transactions)).scored
    assert all(isinstance(json.loads(raw), list) for raw in scored["reason_codes"])
