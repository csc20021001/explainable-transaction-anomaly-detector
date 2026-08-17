"""Write machine-readable outputs, a risk-analyst report, and independent figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _save_figure(output_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(output_dir / name, dpi=160, bbox_inches="tight")
    plt.close()


def create_visualizations(
    features: pd.DataFrame,
    scored: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: str | Path,
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    plt.figure(figsize=(8, 5))
    upper = features["amount_usd"].quantile(0.99)
    features["amount_usd"].clip(upper=upper).hist(bins=45, color="#3b82f6", edgecolor="white")
    plt.title("Transaction Amount Distribution (capped at p99 for display)")
    plt.xlabel("Amount (USD)")
    plt.ylabel("Transactions")
    _save_figure(output_dir, "transaction_amount_distribution.png")
    files.append("transaction_amount_distribution.png")

    plt.figure(figsize=(8, 5))
    scored["anomaly_score"].hist(bins=30, color="#f59e0b", edgecolor="white")
    plt.title("Risk Score Distribution")
    plt.xlabel("Explainable anomaly score")
    plt.ylabel("Transactions")
    _save_figure(output_dir, "risk_score_distribution.png")
    files.append("risk_score_distribution.png")

    plt.figure(figsize=(8, 5))
    x = np.arange(len(comparison))
    plt.plot(x, comparison["precision"], marker="o", label="Precision")
    plt.plot(x, comparison["recall"], marker="o", label="Recall")
    plt.xticks(x, comparison["profile"].str.title())
    plt.ylim(0, 1.05)
    plt.title("Precision / Recall by Threshold Profile")
    plt.ylabel("Metric")
    plt.legend()
    _save_figure(output_dir, "precision_recall_by_threshold.png")
    files.append("precision_recall_by_threshold.png")

    tn = int(((scored.ground_truth_label == 0) & (scored.predicted_label == 0)).sum())
    fp = int(((scored.ground_truth_label == 0) & (scored.predicted_label == 1)).sum())
    fn = int(((scored.ground_truth_label == 1) & (scored.predicted_label == 0)).sum())
    tp = int(((scored.ground_truth_label == 1) & (scored.predicted_label == 1)).sum())
    matrix = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Predicted normal", "Predicted alert"])
    plt.yticks([0, 1], ["Actual normal", "Actual anomaly"])
    for (i, j), value in np.ndenumerate(matrix):
        plt.text(j, i, str(value), ha="center", va="center", color="black")
    plt.title("Confusion Matrix — Balanced")
    _save_figure(output_dir, "confusion_matrix.png")
    files.append("confusion_matrix.png")

    plt.figure(figsize=(9, 5))
    daily = scored.assign(date=pd.to_datetime(scored["timestamp"]).dt.date).groupby("date")["predicted_label"].sum()
    daily.plot(kind="bar", color="#dc2626")
    plt.title("Daily Alert Count — Balanced")
    plt.xlabel("Date")
    plt.ylabel("Alerts")
    plt.xticks(rotation=45, ha="right")
    _save_figure(output_dir, "daily_alert_count.png")
    files.append("daily_alert_count.png")

    plt.figure(figsize=(9, 5))
    false_positives = scored[(scored.ground_truth_label == 0) & (scored.predicted_label == 1)]
    reasons: list[str] = []
    for raw in false_positives["reason_codes"]:
        reasons.extend(json.loads(raw))
    counts = pd.Series(reasons).value_counts().sort_values() if reasons else pd.Series({"No false positives": 0})
    counts.plot(kind="barh", color="#ef4444")
    plt.title("False Positive Reasons — Balanced")
    plt.xlabel("Occurrences")
    _save_figure(output_dir, "false_positive_reasons.png")
    files.append("false_positive_reasons.png")

    plt.figure(figsize=(9, 5))
    anomaly_rows = features[features["ground_truth_label"].eq(1)]
    if len(anomaly_rows):
        chosen = anomaly_rows.iloc[0]
        account = chosen["account_id"]
        account_rows = features[features["account_id"].eq(account)].sort_values("timestamp")
        plt.plot(pd.to_datetime(account_rows["timestamp"]), account_rows["amount_usd"], marker="o", linewidth=1, label="Amount USD")
        plt.axvline(pd.Timestamp(chosen["timestamp"]), color="red", linestyle="--", label="First labelled anomaly")
        plt.title(f"Account Behavior Before and After Anomaly — {account}")
        plt.xlabel("Timestamp")
        plt.ylabel("Amount (USD)")
        plt.legend()
        plt.xticks(rotation=30, ha="right")
    else:
        plt.text(0.5, 0.5, "No labelled anomaly available", ha="center")
        plt.title("Account Behavior Before and After Anomaly")
    _save_figure(output_dir, "account_behavior_before_after_anomaly.png")
    files.append("account_behavior_before_after_anomaly.png")
    return files


def generate_markdown_report(report: dict[str, Any], comparison: pd.DataFrame, output_path: str | Path) -> None:
    answers = report["analysis_answers"]
    hp = answers.get("highest_precision_rule") or {}
    fp = answers.get("most_false_positive_prone_rule") or {}
    missed = answers.get("most_missed_anomaly_type") or {}
    useful = answers.get("most_useful_features") or []
    columns = ["profile", "precision", "recall", "f1_score", "false_positive_rate", "false_negative_rate"]
    table_rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, item in comparison[columns].iterrows():
        table_rows.append("| " + " | ".join(str(item[column]) for column in columns) + " |")
    table = "\n".join(table_rows)
    feature_lines = "\n".join(
        f"- `{item['feature']}` — standardized mean difference {item['standardized_mean_difference']:.3f}"
        for item in useful
    ) or "- No feature ranking available."
    method_rows = ["| method | alerts | precision | recall | f1_score |", "| --- | ---: | ---: | ---: | ---: |"]
    for item in report.get("statistical_method_comparison", []):
        method_rows.append(
            f"| {item['method']} | {item['alerts']} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1_score']:.3f} |"
        )
    method_table = "\n".join(method_rows)
    text = f"""# Explainable Transaction Anomaly Detection Report

## Scope and interpretation

This report evaluates transparent statistical alerts on anonymous synthetic transactions. An alert indicates behavior worth reviewing; it does **not** establish criminal activity. A qualified analyst should review context before any action.

## Threshold comparison

{table}

Lowering the profile from Balanced to Sensitive changed recall by **{answers['recall_gain_balanced_to_sensitive']:.3f}**. Lowering it from Conservative to Sensitive changed recall by **{answers['recall_gain_conservative_to_sensitive']:.3f}**.

## Questions answered

- **Which rule has the highest precision?** `{hp.get('reason_code', 'N/A')}` at {hp.get('precision', 0):.3f} precision across {hp.get('alerts', 0)} alerts.
- **Which rule is most prone to false positives?** `{fp.get('reason_code', 'N/A')}` produced {fp.get('false_positives', 0)} false-positive reason occurrences.
- **Which anomaly is missed most often?** `{missed.get('anomaly_type', 'N/A')}` had a {missed.get('detection_rate', 0):.3f} detection rate.
- **Is there data leakage?** {answers['data_leakage']}

## Most useful features

The following descriptive ranking compares labelled anomalies with normal test rows; it is not model feature importance:

{feature_lines}

## Statistical method comparison

Each amount method below is evaluated independently on the same test rows. Because these methods target amount outliers rather than every anomaly family, their overall recall is expected to be lower than the combined rule engine.

{method_table}

## False positives and false negatives

Detailed cases are exported to `false_positive_cases.csv` and `false_negative_cases.csv`. False positives commonly arise when legitimate travel, a device replacement, or a burst of activity resembles an alert rule. False negatives are expected for subtle patterns that require external context or a longer history.

## Methodology controls

- Account features are computed point-in-time from current and prior events only.
- Train, validation, and test are chronological.
- Threshold profiles are fixed configuration and are not tuned using test labels.
- Z-score, modified Z-score/MAD, IQR, percentile, and rolling mean/standard deviation evidence are retained in explanations.

## Human review

Analysts should validate wallet context, customer history, operational events, and data quality. Do not treat a statistical alert or score as a conclusion about intent or illegality.
"""
    Path(output_path).write_text(text, encoding="utf-8")
