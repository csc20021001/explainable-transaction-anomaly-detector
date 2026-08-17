# Explainable Transaction Anomaly Detector

A production-shaped portfolio project for detecting transactions that deviate from an account's historical behavior. It uses transparent statistical rules, point-in-time features, reproducible synthetic cryptocurrency transactions, and labelled evaluation data.

> An alert is a signal for review, not evidence of crime. This project does not infer intent or make legal conclusions.

## Why this project exists

Risk analysts need to understand *why* a transaction was surfaced. A single opaque model score is difficult to challenge, audit, or improve. This package therefore prioritizes:

- statistical evidence that can be reproduced by an analyst;
- human-readable reason codes and transaction-level explanations;
- chronological, point-in-time evaluation controls;
- explicit false-positive and false-negative analysis;
- three fixed operating points that expose the precision/recall trade-off.

The project deliberately does not use real private transactions. The bundled dataset is anonymous, deterministic, and entirely synthetic.

## Data flow

```text
CSV input
  -> schema validation and cleaning audit
  -> point-in-time account history and rolling windows
  -> robust normalization fitted on the chronological train split
  -> explainable statistical rules
  -> Conservative / Balanced / Sensitive alerts
  -> held-out test metrics, error cases, figures, and analyst report
```

The split is chronological: 60% train, 20% validation, and 20% test. Features for a row use the current event and/or earlier events only. Ground-truth labels are never inputs to a rule or feature.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Demo commands

Each command is independently runnable and writes to `artifacts/` by default:

```bash
python -m anomaly_detector validate-data
python -m anomaly_detector build-features
python -m anomaly_detector detect
python -m anomaly_detector evaluate
python -m anomaly_detector generate-report
```

Use a CSV exported from `crypto-transaction-risk-pipeline`:

```bash
python -m anomaly_detector --input path/to/transactions.csv --output artifacts/imported generate-report
```

Regenerate the bundled sample deterministically:

```bash
python -m anomaly_detector --input data/sample_transactions.csv simulate-data --seed 42
```

## Input contract and validation

Required columns:

- `transaction_id`, `account_id`, `timestamp`, `asset`
- `amount`, `amount_usd`, `transaction_type`
- `source_wallet`, `destination_wallet`, `device_id`, `ip_country`
- `account_age_days`, `ground_truth_label`

An optional `anomaly_type` supports pattern-level evaluation. Timestamps are normalized to UTC. Rows missing a transaction ID, account ID, or valid timestamp are rejected; duplicate transaction IDs are removed; missing categoricals become `UNKNOWN`; invalid amounts are median-imputed and negative values are clipped to zero. Statistical outliers are retained for inspection rather than silently deleted. Every decision is recorded in `validation_report.json`.

## Feature list

All behavioral features are calculated per account:

| Feature | Interpretation |
|---|---|
| `hourly_transaction_count` | Transactions in the trailing hour |
| `daily_transaction_amount` | USD volume in the trailing 24 hours |
| `historical_mean_amount` | Mean of strictly prior transaction amounts |
| `historical_std_amount` | Population standard deviation of prior amounts |
| `amount_deviation_from_mean` | Current amount minus historical mean |
| `minutes_since_incoming` | Time between the latest incoming transaction and this event |
| `unique_counterparties_24h` | Distinct counterparties in the trailing 24 hours |
| `new_wallet_ratio` | Expanding share of first-seen counterparties |
| `night_transaction_ratio` | Expanding share of transactions from 22:00–05:59 UTC |
| `new_device_ratio` | Expanding share of first-seen device events |
| `country_change` | Country not previously observed for the account |
| `dormancy_days` | Days since the preceding transaction |
| `inbound_outbound_ratio_24h` | Trailing inbound USD divided by outbound USD |
| `transaction_velocity` | Auditable alias for trailing hourly event count |
| `asset_conversion_count_24h` | Asset changes within the trailing window |
| `threshold_band_count_24h` | Withdrawals from USD 9,000 to below 10,000 in 24 hours |

Robust-normalized versions use the training median and IQR. Zero-IQR features use a safe scale of one.

## Statistical methods

The amount rule compares six interpretable signals:

1. **Z-score** — distance from the prior account mean in standard deviations.
2. **Modified Z-score** — robust distance from the prior median, scaled by MAD.
3. **Median Absolute Deviation (MAD)** — dispersion resistant to large historical values.
4. **Interquartile Range (IQR)** — flags values beyond a configurable upper fence.
5. **Percentile threshold** — compares the event with the account's historical p99.
6. **Rolling mean and standard deviation** — represented by the point-in-time historical mean/std evidence.

The amount alert requires at least two statistical methods to agree. Constant-value and small-history accounts return finite, conservative statistics rather than dividing by zero. No black-box model is required: the goal is auditability, stable evidence, and straightforward analyst challenge. Isolation Forest could be added as a benchmark, but should not replace these explanations.

## Detected anomaly patterns and reason codes

| Pattern | Primary reason code |
|---|---|
| Unusually large transaction | `AMOUNT_STATISTICAL_OUTLIER` |
| Abnormal transaction velocity | `ABNORMAL_TRANSACTION_VELOCITY` |
| Rapid pass-through | `RAPID_PASS_THROUGH` |
| Dormant account activation | `DORMANT_ACCOUNT_ACTIVATION` |
| Geographic login anomaly | `GEOGRAPHIC_LOGIN_ANOMALY` |
| New device withdrawal | `NEW_DEVICE_WITHDRAWAL` |
| High counterparty diversity | `HIGH_COUNTERPARTY_DIVERSITY` |
| Sudden behavior change | `SUDDEN_BEHAVIOR_CHANGE` |
| Repeated threshold avoidance | `REPEATED_THRESHOLD_AVOIDANCE` |
| Unusual transaction hour | `UNUSUAL_TRANSACTION_HOUR` |

Every emitted alert includes a non-empty `reason_codes` list and numerical evidence. For example:

```json
{
  "transaction_id": "TX-0001824",
  "account_id": "ACC-00001",
  "anomaly_score": 0.87,
  "severity": "high",
  "reason_codes": ["AMOUNT_STATISTICAL_OUTLIER", "NEW_DEVICE_WITHDRAWAL"],
  "explanation": {
    "transaction_amount": 9800,
    "account_median_amount": 240,
    "modified_zscore": 9.4,
    "evidence_by_reason": {}
  }
}
```

The anomaly score is an auditable ranking of capped rule strengths. It is **not** a probability of fraud or crime.

## Threshold comparison

Thresholds are fixed in `anomaly_detector/config.py`; they are not tuned on test labels.

| Profile | Design goal | Amount modified Z | Velocity/hour | Dormancy days | Counterparties/24h |
|---|---|---:|---:|---:|---:|
| Conservative | Fewer alerts, higher expected precision | 5.0 | 10 | 90 | 14 |
| Balanced | Default analyst queue | 3.5 | 7 | 45 | 10 |
| Sensitive | Wider coverage, more review work | 2.5 | 5 | 21 | 7 |

Seed-42 sample results on the untouched chronological test split:

| Profile | Precision | Recall | F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| Conservative | 0.882 | 0.488 | 0.628 | 0.048 | 0.512 |
| Balanced | 0.920 | 0.893 | 0.906 | 0.057 | 0.107 |
| Sensitive | 0.888 | 0.988 | 0.935 | 0.093 | 0.012 |

Lowering the sample from Balanced to Sensitive adds 0.095 recall. These figures verify the pipeline on synthetic labels; they are not estimates of production performance. Re-run `evaluate` for imported data.

## Evaluation metrics

On the chronological test split the package calculates:

- precision, recall, and F1 score;
- false-positive rate and false-negative rate;
- TN/FP/FN/TP confusion matrix;
- detection rate for each labelled anomaly pattern;
- precision and false-positive counts by reason code.

`evaluation_report.json` answers which rule has the highest precision, which rule produces the most false positives, which anomaly type is missed most often, whether leakage is known, and which features show the largest labelled-vs-normal standardized difference.

## False-positive cases

Potential legitimate causes include customer travel, device replacement, payday bursts, treasury consolidation, or a new counterparty campaign. The report does not hide these cases: the complete Balanced-profile set is written to `false_positive_cases.csv`, including reason codes and evidence.

For example, the sample flags normal `TX-0000957` because USD 1,194.50 is a statistical outlier relative to that account, and `TX-0001198` because a withdrawal uses a first-seen device. Both remain false positives under the synthetic ground truth and demonstrate why contextual review matters.

## False-negative cases

Subtle behavior changes can look normal statistically, particularly for accounts with short histories or cases requiring external wallet intelligence. Missed labelled cases are written to `false_negative_cases.csv` and summarized by anomaly type. Lowering thresholds can improve recall, at the cost of additional false-positive review.

In the sample, the earliest events in velocity and counterparty-diversity bursts can be missed because the rolling count has not yet crossed the Balanced threshold. Unusual-hour activity is the lowest-recall pattern at 0.667; two labelled events occur after prior overnight behavior has already raised the account's historical night ratio.

## Generated outputs

`generate-report` creates:

- `evaluation_report.json` and `evaluation_report.csv`
- `false_positive_cases.csv` and `false_negative_cases.csv`
- `feature_summary.csv` and `threshold_comparison.csv`
- `analysis_report.md`
- `balanced_alerts.json`, alert/scored transaction CSVs, and validation artifacts
- seven standalone Matplotlib figures (no subplots): amount distribution, score distribution, precision/recall, confusion matrix, daily alerts, false-positive reasons, and account behavior around an anomaly

## Testing

```bash
pytest
```

The suite contains more than 20 tests covering formulas, MAD/zero variance, rolling windows, missing values, duplicate handling, threshold logic, reason codes, evaluation metrics, empty inputs, small-history accounts, no-history accounts, identical numeric values, and simulator reproducibility.

## Known limitations

- Synthetic labels are useful for verification but cannot represent all operational behavior.
- Country and device changes lack authentication context; wallet identity is intentionally anonymous.
- Fixed USD threshold-avoidance bands may differ across organizations and jurisdictions.
- Account-only history misses network-level relationships among wallets.
- Very new accounts have insufficient history, so some rules intentionally remain conservative.
- The score is not calibrated as a probability.

## Human review is essential

An alert should start a review, not end one. Analysts should validate data quality, customer context, wallet provenance, expected business activity, and applicable policy before deciding whether escalation is warranted.

## Future improvements

- add entity-graph and shared-device features;
- add delayed-label monitoring and threshold drift reports;
- benchmark Isolation Forest without removing rule explanations;
- introduce account peer groups and asset-specific seasonality;
- add rule-level analyst feedback and calibrated queue prioritization;
- add static type checking, package publishing, and a lightweight dashboard.

## 中文摘要

本專案以匿名模擬虛擬貨幣交易資料，建立可解釋、可驗證的異常偵測流程。系統會執行資料驗證、缺值與重複值處理、帳戶歷史與滾動視窗特徵、六種基本統計方法、三組固定門檻、逐筆 reason code 與統計證據、測試集 Precision／Recall／F1、False Positive／False Negative 分析，以及獨立圖表與 Markdown 報告。所有特徵只使用當下及過去資料，測試標籤不參與門檻調整。警示僅代表需要人工複核的偏離行為，不能被解讀為犯罪或不法行為。
