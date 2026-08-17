"""Generate anonymous, reproducible transactions with labelled anomaly patterns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ASSET_PRICES = {"BTC": 60000.0, "ETH": 3000.0, "USDT": 1.0, "USDC": 1.0}
COUNTRIES = ["TW", "JP", "SG", "US", "DE"]


def generate_synthetic_transactions(
    n_accounts: int = 60,
    normal_per_account: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Return synthetic data only; no real identities or wallet addresses are used."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    rows: list[dict] = []
    serial = 1
    profiles: dict[str, dict] = {}

    def add(
        account: str,
        timestamp: pd.Timestamp,
        amount_usd: float,
        tx_type: str = "trade",
        asset: str = "USDT",
        device: str | None = None,
        country: str | None = None,
        source: str | None = None,
        destination: str | None = None,
        label: int = 0,
        anomaly_type: str = "normal",
    ) -> None:
        nonlocal serial
        profile = profiles[account]
        rows.append(
            {
                "transaction_id": f"TX-{serial:07d}",
                "account_id": account,
                "timestamp": timestamp.isoformat(),
                "asset": asset,
                "amount": round(amount_usd / ASSET_PRICES[asset], 8),
                "amount_usd": round(float(amount_usd), 2),
                "transaction_type": tx_type,
                "source_wallet": source or ("EXTERNAL-IN" if tx_type in {"deposit", "transfer_in"} else f"WALLET-{account}"),
                "destination_wallet": destination or (f"WALLET-{account}" if tx_type in {"deposit", "transfer_in"} else "EXTERNAL-OUT"),
                "device_id": device or profile["devices"][0],
                "ip_country": country or profile["country"],
                "account_age_days": profile["age"],
                "ground_truth_label": int(label),
                "anomaly_type": anomaly_type,
            }
        )
        serial += 1

    for index in range(n_accounts):
        account = f"ACC-{index + 1:05d}"
        country = COUNTRIES[index % len(COUNTRIES)]
        profiles[account] = {
            "country": country,
            "devices": [f"DEV-{index + 1:05d}-A", f"DEV-{index + 1:05d}-B"],
            "age": int(rng.integers(120, 1600)),
            "median": float(rng.uniform(180, 900)),
        }
        # Dormant cohort stops early so its eventual activation has a genuine historical gap.
        max_day = 20 if index in range(18, 24) else 70
        day_offsets = np.sort(rng.uniform(0, max_day, normal_per_account))
        for offset in day_offsets:
            hour = int(np.clip(rng.normal(14, 4), 6, 21))
            ts = start + pd.Timedelta(days=float(offset))
            ts = ts.normalize() + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
            amount_usd = float(rng.lognormal(np.log(profiles[account]["median"]), 0.48))
            tx_type = rng.choice(["deposit", "withdrawal", "transfer_in", "transfer_out", "trade"], p=[0.18, 0.18, 0.14, 0.14, 0.36])
            asset = rng.choice(list(ASSET_PRICES), p=[0.18, 0.22, 0.40, 0.20])
            device = profiles[account]["devices"][int(rng.random() < 0.12)]
            party = f"CP-{account}-{int(rng.integers(1, 7)):02d}"
            add(
                account,
                ts,
                amount_usd,
                tx_type=tx_type,
                asset=asset,
                device=device,
                source=party if tx_type in {"deposit", "transfer_in"} else None,
                destination=party if tx_type in {"withdrawal", "transfer_out"} else None,
            )

    anomaly_day = start + pd.Timedelta(days=82)
    # Six examples for each anomaly family, placed after the training/validation history.
    for pattern_index in range(6):
        minute = pattern_index * 70
        account = f"ACC-{pattern_index + 1:05d}"
        base = profiles[account]["median"]
        add(account, anomaly_day + pd.Timedelta(minutes=minute), base * 18, "withdrawal", label=1, anomaly_type="unusually_large_transaction")

        account = f"ACC-{pattern_index + 7:05d}"
        base_ts = anomaly_day + pd.Timedelta(days=1, minutes=minute)
        for burst in range(8):
            add(account, base_ts + pd.Timedelta(minutes=burst * 3), profiles[account]["median"] * 0.8, "trade", label=1, anomaly_type="abnormal_transaction_velocity")

        account = f"ACC-{pattern_index + 13:05d}"
        base_ts = anomaly_day + pd.Timedelta(days=2, minutes=minute)
        pass_amount = profiles[account]["median"] * 2
        add(account, base_ts, pass_amount, "deposit", source=f"PASS-IN-{pattern_index}")
        add(account, base_ts + pd.Timedelta(minutes=4), pass_amount * 0.96, "transfer_out", destination=f"PASS-OUT-{pattern_index}", label=1, anomaly_type="rapid_pass_through")

        account = f"ACC-{pattern_index + 19:05d}"
        add(account, anomaly_day + pd.Timedelta(days=3, minutes=minute), profiles[account]["median"] * 3, "withdrawal", label=1, anomaly_type="dormant_account_activation")

        account = f"ACC-{pattern_index + 25:05d}"
        foreign = next(c for c in COUNTRIES if c != profiles[account]["country"])
        add(account, anomaly_day + pd.Timedelta(days=4, minutes=minute), profiles[account]["median"], "trade", country=foreign, label=1, anomaly_type="geographic_login_anomaly")

        account = f"ACC-{pattern_index + 31:05d}"
        add(account, anomaly_day + pd.Timedelta(days=5, minutes=minute), profiles[account]["median"] * 1.4, "withdrawal", device=f"DEV-{account}-NEW", label=1, anomaly_type="new_device_withdrawal")

        account = f"ACC-{pattern_index + 37:05d}"
        base_ts = anomaly_day + pd.Timedelta(days=6, minutes=minute)
        for party_index in range(10):
            add(account, base_ts + pd.Timedelta(minutes=party_index * 8), profiles[account]["median"] * 0.45, "transfer_out", destination=f"DIVERSE-{pattern_index}-{party_index}", label=1, anomaly_type="high_counterparty_diversity")

        account = f"ACC-{pattern_index + 43:05d}"
        add(account, anomaly_day + pd.Timedelta(days=7, minutes=minute), profiles[account]["median"] * 7, "trade", label=1, anomaly_type="sudden_behavior_change")

        account = f"ACC-{pattern_index + 49:05d}"
        base_ts = anomaly_day + pd.Timedelta(days=8, minutes=minute)
        for avoidance in range(3):
            add(account, base_ts + pd.Timedelta(hours=avoidance * 2), 9500 - avoidance * 20, "withdrawal", destination=f"THRESH-{pattern_index}-{avoidance}", label=1, anomaly_type="repeated_threshold_avoidance")

        account = f"ACC-{pattern_index + 55:05d}"
        add(account, anomaly_day + pd.Timedelta(days=9, hours=2, minutes=minute), profiles[account]["median"], "trade", label=1, anomaly_type="unusual_transaction_hour")

    frame = pd.DataFrame(rows).sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)
    return frame


def write_synthetic_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    frame = generate_synthetic_transactions(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
