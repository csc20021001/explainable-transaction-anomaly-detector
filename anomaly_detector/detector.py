"""Explainable rule engine backed by transparent statistical evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ThresholdProfile, get_profile


@dataclass
class DetectionResult:
    scored: pd.DataFrame
    alerts: list[dict[str, Any]]


def _clean_number(value: Any, digits: int = 4) -> float | int | None:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    number = float(value)
    return int(number) if number.is_integer() else round(number, digits)


def _add_reason(reasons: list[str], evidence: dict[str, Any], code: str, **values: Any) -> None:
    reasons.append(code)
    evidence[code] = {key: _clean_number(value) if isinstance(value, (int, float, np.number)) else value for key, value in values.items()}


def score_transaction(row: pd.Series, profile: ThresholdProfile | str = "balanced") -> dict[str, Any]:
    if isinstance(profile, str):
        profile = get_profile(profile)
    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    history = int(row.get("history_count", 0))
    enough_history = history >= profile.minimum_history
    amount = float(row.get("amount_usd", 0.0))
    modified_z = float(row.get("amount_modified_zscore", 0.0))
    standard_z = float(row.get("amount_zscore", 0.0))
    rolling_z = float(row.get("amount_rolling_zscore", 0.0))
    q1 = float(row.get("historical_q1_amount", 0.0))
    q3 = float(row.get("historical_q3_amount", 0.0))
    iqr_upper = q3 + profile.amount_iqr_multiplier * (q3 - q1)
    historical_p99 = float(row.get("historical_p99_amount", 0.0))

    amount_methods = {
        "zscore": enough_history and standard_z >= profile.amount_zscore,
        "modified_zscore": enough_history and modified_z >= profile.amount_modified_zscore,
        "mad": enough_history and modified_z >= profile.amount_modified_zscore,
        "iqr": enough_history and q3 > 0 and amount > iqr_upper,
        "percentile": enough_history and historical_p99 > 0 and amount > historical_p99,
        "rolling_mean_std": enough_history and rolling_z >= profile.amount_zscore,
    }
    if sum(amount_methods.values()) >= 2:
        _add_reason(
            reasons,
            evidence,
            "AMOUNT_STATISTICAL_OUTLIER",
            transaction_amount=amount,
            account_mean_amount=row.get("historical_mean_amount", 0.0),
            account_median_amount=row.get("historical_median_amount", 0.0),
            zscore=standard_z,
            modified_zscore=modified_z,
            rolling_zscore=rolling_z,
            rolling_mean=row.get("rolling_mean_amount", 0.0),
            rolling_std=row.get("rolling_std_amount", 0.0),
            mad=row.get("historical_mad_amount", 0.0),
            iqr_upper_bound=iqr_upper,
            historical_p99=historical_p99,
            methods_triggered=", ".join(k for k, v in amount_methods.items() if v),
        )

    velocity = int(row.get("transaction_velocity", 1))
    if velocity >= profile.velocity_per_hour:
        _add_reason(reasons, evidence, "ABNORMAL_TRANSACTION_VELOCITY", transactions_last_hour=velocity, threshold=profile.velocity_per_hour)

    minutes_since_incoming = row.get("minutes_since_incoming", np.nan)
    if (
        row.get("transaction_type") in {"withdrawal", "transfer_out"}
        and pd.notna(minutes_since_incoming)
        and 0 <= float(minutes_since_incoming) <= profile.pass_through_minutes
    ):
        _add_reason(
            reasons,
            evidence,
            "RAPID_PASS_THROUGH",
            minutes_since_incoming=minutes_since_incoming,
            threshold_minutes=profile.pass_through_minutes,
        )

    dormancy = float(row.get("dormancy_days", 0.0))
    if history > 0 and dormancy >= profile.dormant_days:
        _add_reason(reasons, evidence, "DORMANT_ACCOUNT_ACTIVATION", dormant_days=dormancy, threshold_days=profile.dormant_days)

    if enough_history and int(row.get("country_change", 0)) == 1:
        _add_reason(reasons, evidence, "GEOGRAPHIC_LOGIN_ANOMALY", observed_country=row.get("ip_country", "UNKNOWN"), prior_country_unseen=True)

    if history > 0 and row.get("transaction_type") in {"withdrawal", "transfer_out"} and int(row.get("is_new_device", 0)) == 1:
        _add_reason(reasons, evidence, "NEW_DEVICE_WITHDRAWAL", device_id=row.get("device_id", "UNKNOWN"), new_device=True)

    counterparties = int(row.get("unique_counterparties_24h", 0))
    if counterparties >= profile.counterparty_count_24h:
        _add_reason(reasons, evidence, "HIGH_COUNTERPARTY_DIVERSITY", unique_counterparties_24h=counterparties, threshold=profile.counterparty_count_24h)

    amount_ratio = float(row.get("amount_ratio_to_mean", 1.0))
    if enough_history and amount_ratio >= profile.behavior_amount_ratio:
        _add_reason(reasons, evidence, "SUDDEN_BEHAVIOR_CHANGE", amount_ratio_to_historical_mean=amount_ratio, threshold=profile.behavior_amount_ratio)

    threshold_count = int(row.get("threshold_band_count_24h", 0))
    if threshold_count >= profile.threshold_avoidance_count:
        _add_reason(
            reasons,
            evidence,
            "REPEATED_THRESHOLD_AVOIDANCE",
            sub_threshold_withdrawals_24h=threshold_count,
            amount_band_usd="[9000, 10000)",
            threshold=profile.threshold_avoidance_count,
        )

    historical_night_ratio = float(row.get("historical_night_ratio", 0.0))
    if enough_history and int(row.get("is_night_transaction", 0)) == 1 and historical_night_ratio <= profile.unusual_hour_max_history_ratio:
        _add_reason(
            reasons,
            evidence,
            "UNUSUAL_TRANSACTION_HOUR",
            transaction_hour=pd.Timestamp(row["timestamp"]).hour,
            historical_night_ratio=historical_night_ratio,
            maximum_expected_ratio=profile.unusual_hour_max_history_ratio,
        )

    # Independent, capped contributions make the score auditable; it is not a crime probability.
    strength = 0.0
    strength += min(max(modified_z, standard_z, 0.0) / 6.0, 1.5) if "AMOUNT_STATISTICAL_OUTLIER" in reasons else 0.0
    strength += min(velocity / max(profile.velocity_per_hour, 1), 1.5) if "ABNORMAL_TRANSACTION_VELOCITY" in reasons else 0.0
    strength += 0.9 * sum(
        code in reasons
        for code in (
            "RAPID_PASS_THROUGH",
            "DORMANT_ACCOUNT_ACTIVATION",
            "GEOGRAPHIC_LOGIN_ANOMALY",
            "NEW_DEVICE_WITHDRAWAL",
            "HIGH_COUNTERPARTY_DIVERSITY",
            "SUDDEN_BEHAVIOR_CHANGE",
            "REPEATED_THRESHOLD_AVOIDANCE",
            "UNUSUAL_TRANSACTION_HOUR",
        )
    )
    score = 0.0 if not reasons else min(0.99, 1.0 - math.exp(-strength / 2.2))
    severity = "none" if not reasons else "low" if score < 0.45 else "medium" if score < 0.70 else "high"
    return {
        "transaction_id": str(row["transaction_id"]),
        "account_id": str(row["account_id"]),
        "anomaly_score": round(score, 4),
        "severity": severity,
        "reason_codes": reasons,
        "explanation": {
            "transaction_amount": _clean_number(amount),
            "account_median_amount": _clean_number(row.get("historical_median_amount", 0.0)),
            "modified_zscore": _clean_number(modified_z),
            "evidence_by_reason": evidence,
            "score_interpretation": "Rule-strength ranking score; not a probability of crime.",
        },
    }


def detect(features: pd.DataFrame, profile: ThresholdProfile | str = "balanced") -> DetectionResult:
    if isinstance(profile, str):
        profile = get_profile(profile)
    scored = features.copy()
    alerts: list[dict[str, Any]] = []
    scores: list[float] = []
    predicted: list[int] = []
    severities: list[str] = []
    reason_json: list[str] = []
    explanations: list[str] = []
    for _, row in scored.iterrows():
        alert = score_transaction(row, profile)
        is_alert = int(bool(alert["reason_codes"]))
        if is_alert:
            alerts.append(alert)
        scores.append(alert["anomaly_score"])
        predicted.append(is_alert)
        severities.append(alert["severity"])
        reason_json.append(json.dumps(alert["reason_codes"]))
        explanations.append(json.dumps(alert["explanation"], ensure_ascii=False))
    scored["anomaly_score"] = scores
    scored["predicted_label"] = predicted
    scored["severity"] = severities
    scored["reason_codes"] = reason_json
    scored["explanation"] = explanations
    scored["threshold_profile"] = profile.name
    return DetectionResult(scored=scored, alerts=alerts)
