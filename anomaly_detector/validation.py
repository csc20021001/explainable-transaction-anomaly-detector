"""Input schema validation, cleaning, and audit reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "transaction_id",
    "account_id",
    "timestamp",
    "asset",
    "amount",
    "amount_usd",
    "transaction_type",
    "source_wallet",
    "destination_wallet",
    "device_id",
    "ip_country",
    "account_age_days",
    "ground_truth_label",
]

CATEGORICAL_COLUMNS = [
    "asset",
    "transaction_type",
    "source_wallet",
    "destination_wallet",
    "device_id",
    "ip_country",
]
NUMERIC_COLUMNS = ["amount", "amount_usd", "account_age_days", "ground_truth_label"]
VALID_TRANSACTION_TYPES = {"deposit", "withdrawal", "transfer_in", "transfer_out", "trade"}


class DataValidationError(ValueError):
    """Raised when input cannot be safely normalized."""


@dataclass
class ValidationResult:
    data: pd.DataFrame
    report: dict[str, Any]


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def validate_and_clean(df: pd.DataFrame) -> ValidationResult:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")

    clean = df.copy()
    input_rows = len(clean)
    missing_before = clean[REQUIRED_COLUMNS].isna().sum().astype(int).to_dict()
    duplicate_transaction_ids = int(clean.duplicated("transaction_id", keep="first").sum())
    exact_duplicates = int(clean.duplicated(keep="first").sum())

    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce", utc=True, format="mixed")
    invalid_timestamp_rows = int(clean["timestamp"].isna().sum())
    essential_missing = clean[["transaction_id", "account_id", "timestamp"]].isna().any(axis=1)
    dropped_essential_rows = int(essential_missing.sum())
    clean = clean.loc[~essential_missing].copy()
    clean = clean.drop_duplicates("transaction_id", keep="first")

    for column in CATEGORICAL_COLUMNS:
        clean[column] = clean[column].fillna("UNKNOWN").astype(str).str.strip()
        clean.loc[clean[column].eq(""), column] = "UNKNOWN"

    invalid_numeric: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        original_non_null = clean[column].notna()
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        invalid_numeric[column] = int((original_non_null & clean[column].isna()).sum())

    for column in ("amount", "amount_usd"):
        median = clean[column].median()
        clean[column] = clean[column].fillna(0.0 if pd.isna(median) else median).clip(lower=0.0)
    clean["account_age_days"] = clean["account_age_days"].fillna(0).clip(lower=0).astype(int)
    clean["ground_truth_label"] = clean["ground_truth_label"].fillna(0).clip(0, 1).astype(int)
    clean["transaction_type"] = clean["transaction_type"].str.lower()
    unknown_transaction_types = int((~clean["transaction_type"].isin(VALID_TRANSACTION_TYPES)).sum())
    clean.loc[~clean["transaction_type"].isin(VALID_TRANSACTION_TYPES), "transaction_type"] = "trade"
    if "anomaly_type" not in clean.columns:
        clean["anomaly_type"] = np.where(clean["ground_truth_label"].eq(1), "unspecified", "normal")
    clean["anomaly_type"] = clean["anomaly_type"].fillna("normal").astype(str)
    clean = clean.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)

    if len(clean):
        q1, q3 = clean["amount_usd"].quantile([0.25, 0.75])
        upper = float(q3 + 1.5 * (q3 - q1))
        outlier_count = int((clean["amount_usd"] > upper).sum())
    else:
        upper, outlier_count = 0.0, 0

    report: dict[str, Any] = {
        "status": "valid" if not dropped_essential_rows else "valid_with_dropped_rows",
        "input_rows": input_rows,
        "output_rows": len(clean),
        "missing_values_before": missing_before,
        "duplicate_transaction_ids_removed": duplicate_transaction_ids,
        "exact_duplicate_rows_observed": exact_duplicates,
        "invalid_timestamp_rows_removed": invalid_timestamp_rows,
        "essential_rows_removed": dropped_essential_rows,
        "invalid_numeric_values": invalid_numeric,
        "unknown_transaction_types_normalized": unknown_transaction_types,
        "amount_usd_iqr_upper_bound": upper,
        "amount_usd_outliers_for_inspection": outlier_count,
        "ground_truth_positive_rows": int(clean["ground_truth_label"].sum()),
        "note": "Outliers are retained for anomaly analysis; they are not treated as data errors.",
    }
    return ValidationResult(clean, report)
