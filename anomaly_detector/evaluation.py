"""Metrics, threshold comparison, and error-case analysis."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import PROFILES
from .detector import detect


def _statistical_method_comparison(scored: pd.DataFrame) -> list[dict[str, Any]]:
    """Compare each amount method independently on the same held-out rows."""
    profile = PROFILES["balanced"]
    iqr_upper = scored["historical_q3_amount"] + profile.amount_iqr_multiplier * (
        scored["historical_q3_amount"] - scored["historical_q1_amount"]
    )
    history_ok = scored["history_count"] >= profile.minimum_history
    method_flags = {
        "z_score": history_ok & (scored["amount_zscore"] >= profile.amount_zscore),
        "modified_z_score": history_ok & (scored["amount_modified_zscore"] >= profile.amount_modified_zscore),
        "median_absolute_deviation": history_ok
        & (
            (scored["amount_usd"] - scored["historical_median_amount"])
            > (profile.amount_modified_zscore / 0.6745) * scored["historical_mad_amount"]
        ),
        "interquartile_range": history_ok & (scored["historical_q3_amount"] > 0) & (scored["amount_usd"] > iqr_upper),
        "percentile_threshold": history_ok
        & (scored["historical_p99_amount"] > 0)
        & (scored["amount_usd"] > scored["historical_p99_amount"]),
        "rolling_mean_std": history_ok & (scored["amount_rolling_zscore"] >= profile.amount_zscore),
    }
    rows = []
    for method, flag in method_flags.items():
        metrics = binary_metrics(scored["ground_truth_label"], flag.astype(int))
        rows.append(
            {
                "method": method,
                "alerts": int(flag.sum()),
                **{key: value for key, value in metrics.items() if key not in {"confusion_matrix", "support"}},
            }
        )
    return rows


def binary_metrics(y_true, y_pred) -> dict[str, float | int | dict[str, int]]:
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    tp = int(((truth == 1) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 6) if fp + tn else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 6) if fn + tp else 0.0,
        "support": int(len(truth)),
        "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
    }


def _reason_precision(scored: pd.DataFrame) -> list[dict[str, Any]]:
    exploded = []
    for _, row in scored[scored["predicted_label"].eq(1)].iterrows():
        for reason in json.loads(row["reason_codes"]):
            exploded.append((reason, int(row["ground_truth_label"])))
    if not exploded:
        return []
    frame = pd.DataFrame(exploded, columns=["reason_code", "ground_truth_label"])
    return [
        {
            "reason_code": reason,
            "alerts": int(len(group)),
            "true_positives": int(group["ground_truth_label"].sum()),
            "false_positives": int((group["ground_truth_label"] == 0).sum()),
            "precision": round(float(group["ground_truth_label"].mean()), 6),
        }
        for reason, group in frame.groupby("reason_code")
    ]


def _pattern_detection(scored: pd.DataFrame) -> list[dict[str, Any]]:
    positives = scored[scored["ground_truth_label"].eq(1)]
    rows = []
    for pattern, group in positives.groupby("anomaly_type"):
        rows.append(
            {
                "anomaly_type": pattern,
                "ground_truth_count": int(len(group)),
                "detected_count": int(group["predicted_label"].sum()),
                "detection_rate": round(float(group["predicted_label"].mean()), 6),
            }
        )
    return sorted(rows, key=lambda item: item["detection_rate"])


def _feature_usefulness(scored: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = [
        "amount_modified_zscore",
        "transaction_velocity",
        "minutes_since_incoming",
        "dormancy_days",
        "country_change",
        "is_new_device",
        "unique_counterparties_24h",
        "amount_ratio_to_mean",
        "threshold_band_count_24h",
        "is_night_transaction",
    ]
    rows = []
    truth = scored["ground_truth_label"].astype(float)
    for feature in candidates:
        if feature not in scored:
            continue
        values = pd.to_numeric(scored[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        normal_std = float(values[truth.eq(0)].std(ddof=0))
        delta = float(values[truth.eq(1)].mean() - values[truth.eq(0)].mean())
        effect = abs(delta) / (normal_std if normal_std > 1e-9 else 1.0)
        corr = float(values.corr(truth)) if values.nunique() > 1 and truth.nunique() > 1 else 0.0
        rows.append({"feature": feature, "standardized_mean_difference": round(effect, 6), "absolute_correlation": round(abs(corr), 6)})
    return sorted(rows, key=lambda item: item["standardized_mean_difference"], reverse=True)


def evaluate_profiles(features: pd.DataFrame, evaluation_split: str = "test") -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    subset = features[features["data_split"].eq(evaluation_split)].copy() if "data_split" in features else features.copy()
    if subset.empty:
        raise ValueError(f"No rows available for evaluation split {evaluation_split!r}")
    comparison_rows = []
    scored_by_profile: dict[str, pd.DataFrame] = {}
    profile_reports: dict[str, Any] = {}
    for name in ("conservative", "balanced", "sensitive"):
        scored = detect(subset, PROFILES[name]).scored
        scored_by_profile[name] = scored
        metrics = binary_metrics(scored["ground_truth_label"], scored["predicted_label"])
        comparison_rows.append({"profile": name, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
        profile_reports[name] = {
            "thresholds": PROFILES[name].to_dict(),
            "metrics": metrics,
            "rule_performance": _reason_precision(scored),
            "detection_rate_by_anomaly_type": _pattern_detection(scored),
        }
    comparison = pd.DataFrame(comparison_rows)
    balanced = scored_by_profile["balanced"]
    fp = balanced[(balanced["ground_truth_label"] == 0) & (balanced["predicted_label"] == 1)].copy()
    fn = balanced[(balanced["ground_truth_label"] == 1) & (balanced["predicted_label"] == 0)].copy()
    balanced_rules = profile_reports["balanced"]["rule_performance"]
    balanced_patterns = profile_reports["balanced"]["detection_rate_by_anomaly_type"]
    highest_precision = max(balanced_rules, key=lambda x: (x["precision"], x["alerts"]), default=None)
    fp_prone = max(balanced_rules, key=lambda x: x["false_positives"], default=None)
    most_missed = min(balanced_patterns, key=lambda x: x["detection_rate"], default=None)
    recall = comparison.set_index("profile")["recall"]
    report = {
        "evaluation_policy": {
            "split": evaluation_split,
            "threshold_selection": "Profiles are fixed in configuration and are not tuned on test labels.",
            "data_leakage_assessment": "No known label or future-event leakage: features use current/prior events only; chronological split is applied; labels are used only for evaluation.",
        },
        "profiles": profile_reports,
        "statistical_method_comparison": _statistical_method_comparison(balanced),
        "analysis_answers": {
            "highest_precision_rule": highest_precision,
            "most_false_positive_prone_rule": fp_prone,
            "most_missed_anomaly_type": most_missed,
            "recall_gain_balanced_to_sensitive": round(float(recall.get("sensitive", 0) - recall.get("balanced", 0)), 6),
            "recall_gain_conservative_to_sensitive": round(float(recall.get("sensitive", 0) - recall.get("conservative", 0)), 6),
            "data_leakage": "No known leakage under the point-in-time feature and chronological evaluation design.",
            "most_useful_features": _feature_usefulness(balanced)[:5],
        },
    }
    return report, comparison, {"false_positives": fp, "false_negatives": fn, **scored_by_profile}
