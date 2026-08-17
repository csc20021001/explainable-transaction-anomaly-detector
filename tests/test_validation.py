import pandas as pd
import pytest

from anomaly_detector.validation import DataValidationError, validate_and_clean


def test_missing_required_column_fails(base_transactions):
    with pytest.raises(DataValidationError):
        validate_and_clean(base_transactions.drop(columns="asset"))


def test_duplicate_transaction_id_removed(base_transactions):
    duplicated = pd.concat([base_transactions, base_transactions.iloc[[0]]], ignore_index=True)
    result = validate_and_clean(duplicated)
    assert len(result.data) == len(base_transactions)
    assert result.report["duplicate_transaction_ids_removed"] == 1


def test_missing_categorical_value_becomes_unknown(base_transactions):
    base_transactions.loc[0, "device_id"] = None
    result = validate_and_clean(base_transactions)
    assert result.data.iloc[0]["device_id"] == "UNKNOWN"


def test_missing_amount_is_imputed(base_transactions):
    base_transactions.loc[0, "amount_usd"] = None
    result = validate_and_clean(base_transactions)
    assert result.data["amount_usd"].isna().sum() == 0


def test_invalid_timestamp_row_is_removed(base_transactions):
    base_transactions["timestamp"] = base_transactions["timestamp"].astype(object)
    base_transactions.loc[0, "timestamp"] = "not-a-time"
    result = validate_and_clean(base_transactions)
    assert len(result.data) == len(base_transactions) - 1
    assert result.report["invalid_timestamp_rows_removed"] == 1


def test_negative_amount_is_clipped(base_transactions):
    base_transactions.loc[0, "amount_usd"] = -10
    result = validate_and_clean(base_transactions)
    assert result.data["amount_usd"].min() == 0


def test_unknown_transaction_type_is_normalized(base_transactions):
    base_transactions.loc[0, "transaction_type"] = "mystery"
    result = validate_and_clean(base_transactions)
    assert result.data.iloc[0]["transaction_type"] == "trade"


def test_outliers_are_retained(base_transactions):
    base_transactions.loc[0, "amount_usd"] = 1_000_000
    result = validate_and_clean(base_transactions)
    assert result.data["amount_usd"].max() == 1_000_000
