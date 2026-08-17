"""Command-line entry points for each pipeline stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .detector import detect
from .evaluation import evaluate_profiles
from .features import build_features, feature_summary, normalize_features
from .reporting import create_visualizations, generate_markdown_report, write_json
from .simulator import write_synthetic_csv
from .validation import load_csv, validate_and_clean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "sample_transactions.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts"


def _ensure_sample(path: Path) -> None:
    if path == DEFAULT_INPUT and not path.exists():
        write_synthetic_csv(path)


def _validated(input_path: Path, output_dir: Path) -> pd.DataFrame:
    _ensure_sample(input_path)
    result = validate_and_clean(load_csv(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(output_dir / "cleaned_transactions.csv", index=False)
    write_json(output_dir / "validation_report.json", result.report)
    return result.data


def _features(input_path: Path, output_dir: Path) -> pd.DataFrame:
    clean = _validated(input_path, output_dir)
    features = normalize_features(build_features(clean))
    features.to_csv(output_dir / "transaction_features.csv", index=False)
    feature_summary(features).to_csv(output_dir / "feature_summary.csv", index=False)
    return features


def _detections(input_path: Path, output_dir: Path, profile: str = "balanced") -> pd.DataFrame:
    features = _features(input_path, output_dir)
    result = detect(features, profile)
    result.scored.to_csv(output_dir / f"{profile}_scored_transactions.csv", index=False)
    write_json(output_dir / f"{profile}_alerts.json", result.alerts)
    result.scored[result.scored["predicted_label"].eq(1)].to_csv(output_dir / f"{profile}_alerts.csv", index=False)
    return result.scored


def _evaluation(input_path: Path, output_dir: Path):
    features = _features(input_path, output_dir)
    report, comparison, cases = evaluate_profiles(features)
    write_json(output_dir / "evaluation_report.json", report)
    comparison.to_csv(output_dir / "evaluation_report.csv", index=False)
    comparison.to_csv(output_dir / "threshold_comparison.csv", index=False)
    cases["false_positives"].to_csv(output_dir / "false_positive_cases.csv", index=False)
    cases["false_negatives"].to_csv(output_dir / "false_negative_cases.csv", index=False)
    balanced = cases["balanced"]
    balanced.to_csv(output_dir / "balanced_scored_transactions.csv", index=False)
    balanced[balanced["predicted_label"].eq(1)].to_csv(output_dir / "balanced_alerts.csv", index=False)
    alerts = []
    for _, row in balanced[balanced["predicted_label"].eq(1)].iterrows():
        alerts.append(
            {
                "transaction_id": row["transaction_id"],
                "account_id": row["account_id"],
                "anomaly_score": float(row["anomaly_score"]),
                "severity": row["severity"],
                "reason_codes": json.loads(row["reason_codes"]),
                "explanation": json.loads(row["explanation"]),
            }
        )
    write_json(output_dir / "balanced_alerts.json", alerts)
    return features, report, comparison, cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m anomaly_detector", description="Explainable transaction anomaly detector")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV input (default: anonymous sample data)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Artifact output directory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-data", help="Validate, clean, and audit the input CSV")
    subparsers.add_parser("build-features", help="Create point-in-time account and rolling features")
    detect_parser = subparsers.add_parser("detect", help="Generate explainable transaction alerts")
    detect_parser.add_argument("--profile", choices=["conservative", "balanced", "sensitive"], default="balanced")
    subparsers.add_parser("evaluate", help="Compare profiles on the chronological test split")
    subparsers.add_parser("generate-report", help="Run the pipeline and generate all reports and figures")
    simulate = subparsers.add_parser("simulate-data", help="Regenerate the anonymous labelled sample CSV")
    simulate.add_argument("--accounts", type=int, default=60)
    simulate.add_argument("--normal-per-account", type=int, default=30)
    simulate.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path: Path = args.input
    output_dir: Path = args.output
    if args.command == "simulate-data":
        frame = write_synthetic_csv(input_path, n_accounts=args.accounts, normal_per_account=args.normal_per_account, seed=args.seed)
        print(f"Wrote {len(frame):,} anonymous synthetic transactions to {input_path}")
    elif args.command == "validate-data":
        frame = _validated(input_path, output_dir)
        print(f"Validated {len(frame):,} rows; report: {output_dir / 'validation_report.json'}")
    elif args.command == "build-features":
        frame = _features(input_path, output_dir)
        print(f"Built {len(frame):,} feature rows; output: {output_dir / 'transaction_features.csv'}")
    elif args.command == "detect":
        scored = _detections(input_path, output_dir, args.profile)
        print(f"Generated {int(scored['predicted_label'].sum()):,} {args.profile} alerts in {output_dir}")
    elif args.command == "evaluate":
        _, _, comparison, _ = _evaluation(input_path, output_dir)
        print(comparison[["profile", "precision", "recall", "f1_score"]].to_string(index=False))
    elif args.command == "generate-report":
        features, report, comparison, cases = _evaluation(input_path, output_dir)
        figures = create_visualizations(features, cases["balanced"], comparison, output_dir)
        generate_markdown_report(report, comparison, output_dir / "analysis_report.md")
        print(f"Generated complete report, {len(figures)} figures, and evaluation artifacts in {output_dir}")
    return 0
