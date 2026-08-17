"""Point-in-time feature engineering for transaction anomaly detection."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .statistics import modified_zscore, zscore


INBOUND_TYPES = {"deposit", "transfer_in"}
OUTBOUND_TYPES = {"withdrawal", "transfer_out"}
NUMERIC_FEATURES = [
    "hourly_transaction_count",
    "daily_transaction_amount",
    "historical_mean_amount",
    "historical_std_amount",
    "rolling_mean_amount",
    "rolling_std_amount",
    "amount_deviation_from_mean",
    "minutes_since_incoming",
    "unique_counterparties_24h",
    "new_wallet_ratio",
    "night_transaction_ratio",
    "new_device_ratio",
    "country_change",
    "dormancy_days",
    "inbound_outbound_ratio_24h",
    "transaction_velocity",
    "asset_conversion_count_24h",
]


def _counterparty(row: pd.Series) -> str:
    if row["transaction_type"] in OUTBOUND_TYPES:
        return str(row["destination_wallet"])
    return str(row["source_wallet"])


def _historical_stats(amounts: list[float]) -> tuple[float, float, float, float, float, float, float]:
    if not amounts:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    values = np.asarray(amounts, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    q1, q3 = (float(x) for x in np.quantile(values, [0.25, 0.75]))
    p99 = float(np.quantile(values, 0.99))
    return mean, std, median, mad, q1, q3, p99


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build historical-only features. Each row uses no later transaction."""
    if df.empty:
        empty = df.copy()
        for column in NUMERIC_FEATURES + ["amount_zscore", "amount_modified_zscore", "amount_rolling_zscore"]:
            empty[column] = pd.Series(dtype=float)
        return empty

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work = work.sort_values(["account_id", "timestamp", "transaction_id"], kind="stable").copy()
    result_rows: list[dict] = []

    for _, group in work.groupby("account_id", sort=False):
        history_amounts: list[float] = []
        seen_wallets: set[str] = set()
        seen_devices: set[str] = set()
        seen_countries: set[str] = set()
        recent: deque[tuple[pd.Timestamp, float, str, str, str]] = deque()
        last_timestamp: pd.Timestamp | None = None
        last_incoming_timestamp: pd.Timestamp | None = None
        last_asset: str | None = None
        new_wallet_flags: list[int] = []
        night_flags: list[int] = []
        new_device_flags: list[int] = []

        for _, row in group.iterrows():
            record = row.to_dict()
            ts = row["timestamp"]
            amount = float(row["amount_usd"])
            tx_type = row["transaction_type"]
            counterparty = _counterparty(row)
            cutoff_24h = ts - pd.Timedelta(hours=24)
            cutoff_1h = ts - pd.Timedelta(hours=1)
            while recent and recent[0][0] < cutoff_24h:
                recent.popleft()

            mean, std, median, mad, q1, q3, p99 = _historical_stats(history_amounts)
            rolling_mean, rolling_std, _, _, _, _, _ = _historical_stats(history_amounts[-30:])
            history_count = len(history_amounts)
            hour_rows = [item for item in recent if item[0] >= cutoff_1h]
            recent_counterparties = {item[2] for item in recent}
            inbound = sum(item[1] for item in recent if item[3] in INBOUND_TYPES)
            outbound = sum(item[1] for item in recent if item[3] in OUTBOUND_TYPES)
            assets = [item[4] for item in recent] + [str(row["asset"])]
            conversions = sum(a != b for a, b in zip(assets, assets[1:]))
            is_night = int(ts.hour < 6 or ts.hour >= 22)
            is_new_wallet = int(counterparty not in seen_wallets)
            is_new_device = int(str(row["device_id"]) not in seen_devices)
            country_change = int(bool(seen_countries) and str(row["ip_country"]) not in seen_countries)

            record.update(
                {
                    "counterparty": counterparty,
                    "history_count": history_count,
                    "hourly_transaction_count": len(hour_rows) + 1,
                    "daily_transaction_amount": sum(item[1] for item in recent) + amount,
                    "historical_mean_amount": mean,
                    "historical_std_amount": std,
                    "rolling_mean_amount": rolling_mean,
                    "rolling_std_amount": rolling_std,
                    "historical_median_amount": median,
                    "historical_mad_amount": mad,
                    "historical_q1_amount": q1,
                    "historical_q3_amount": q3,
                    "historical_p99_amount": p99,
                    "amount_deviation_from_mean": amount - mean if history_count else 0.0,
                    "amount_ratio_to_mean": amount / mean if mean > 0 else 1.0,
                    "amount_zscore": zscore(amount, mean, std),
                    "amount_modified_zscore": modified_zscore(amount, median, mad),
                    "amount_rolling_zscore": zscore(amount, rolling_mean, rolling_std),
                    "minutes_since_incoming": (
                        (ts - last_incoming_timestamp).total_seconds() / 60
                        if last_incoming_timestamp is not None
                        else np.nan
                    ),
                    "unique_counterparties_24h": len(recent_counterparties | {counterparty}),
                    "is_new_wallet": is_new_wallet,
                    "new_wallet_ratio": (sum(new_wallet_flags) + is_new_wallet) / (history_count + 1),
                    "is_night_transaction": is_night,
                    "historical_night_ratio": sum(night_flags) / history_count if history_count else 0.0,
                    "night_transaction_ratio": (sum(night_flags) + is_night) / (history_count + 1),
                    "is_new_device": is_new_device,
                    "new_device_ratio": (sum(new_device_flags) + is_new_device) / (history_count + 1),
                    "country_change": country_change,
                    "dormancy_days": (
                        (ts - last_timestamp).total_seconds() / 86400 if last_timestamp is not None else float(row["account_age_days"])
                    ),
                    "inbound_outbound_ratio_24h": (inbound + (amount if tx_type in INBOUND_TYPES else 0.0) + 1.0)
                    / (outbound + (amount if tx_type in OUTBOUND_TYPES else 0.0) + 1.0),
                    "transaction_velocity": len(hour_rows) + 1,
                    "asset_conversion_count_24h": conversions,
                    "is_asset_change": int(last_asset is not None and str(row["asset"]) != last_asset),
                    "threshold_band_flag": int(tx_type in OUTBOUND_TYPES and 9000 <= amount < 10000),
                }
            )
            result_rows.append(record)

            history_amounts.append(amount)
            seen_wallets.add(counterparty)
            seen_devices.add(str(row["device_id"]))
            seen_countries.add(str(row["ip_country"]))
            new_wallet_flags.append(is_new_wallet)
            night_flags.append(is_night)
            new_device_flags.append(is_new_device)
            recent.append((ts, amount, counterparty, tx_type, str(row["asset"])))
            last_timestamp = ts
            if tx_type in INBOUND_TYPES:
                last_incoming_timestamp = ts
            last_asset = str(row["asset"])

    features = pd.DataFrame(result_rows)
    features = features.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)
    # A point-in-time count of sub-threshold withdrawals in the preceding 24 hours.
    features["threshold_band_count_24h"] = 0
    for _, indices in features.groupby("account_id").groups.items():
        ordered = features.loc[indices].sort_values("timestamp")
        count_series = ordered.set_index("timestamp")["threshold_band_flag"].rolling("24h").sum().astype(int)
        features.loc[ordered.index, "threshold_band_count_24h"] = count_series.to_numpy()
    features["data_split"] = chronological_split(features["timestamp"])
    return features


def chronological_split(timestamps: pd.Series, train_fraction: float = 0.60, validation_fraction: float = 0.20) -> pd.Series:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave a non-empty test fraction")
    if len(timestamps) == 0:
        return pd.Series(dtype=str, index=timestamps.index)
    order = timestamps.rank(method="first", pct=True)
    return pd.Series(
        np.select(
            [order <= train_fraction, order <= train_fraction + validation_fraction],
            ["train", "validation"],
            default="test",
        ),
        index=timestamps.index,
    )


def normalize_features(features: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    result = features.copy()
    columns = columns or [column for column in NUMERIC_FEATURES if column in result.columns]
    train = result[result.get("data_split", "train") == "train"]
    for column in columns:
        center = float(train[column].median()) if len(train) else 0.0
        q1, q3 = train[column].quantile([0.25, 0.75]) if len(train) else (0.0, 0.0)
        scale = float(q3 - q1)
        result[f"{column}_normalized"] = (result[column] - center) / (scale if scale > 1e-9 else 1.0)
    return result


def feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in NUMERIC_FEATURES + ["amount_zscore", "amount_modified_zscore", "amount_rolling_zscore"] if c in features]
    rows = []
    for column in columns:
        series = pd.to_numeric(features[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "count": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "mean": float(series.mean()) if series.notna().any() else 0.0,
                "std": float(series.std(ddof=0)) if series.notna().any() else 0.0,
                "min": float(series.min()) if series.notna().any() else 0.0,
                "median": float(series.median()) if series.notna().any() else 0.0,
                "p95": float(series.quantile(0.95)) if series.notna().any() else 0.0,
                "max": float(series.max()) if series.notna().any() else 0.0,
            }
        )
    return pd.DataFrame(rows)
