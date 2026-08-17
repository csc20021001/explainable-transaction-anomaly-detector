from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def base_transactions() -> pd.DataFrame:
    rows = []
    for index in range(8):
        tx_type = "deposit" if index == 3 else "withdrawal" if index == 4 else "trade"
        rows.append(
            {
                "transaction_id": f"TX-{index}",
                "account_id": "ACC-1",
                "timestamp": pd.Timestamp("2025-01-01T10:00:00Z") + pd.Timedelta(hours=index),
                "asset": "USDT",
                "amount": 100 + index,
                "amount_usd": 100 + index,
                "transaction_type": tx_type,
                "source_wallet": f"SRC-{index % 2}",
                "destination_wallet": f"DST-{index % 3}",
                "device_id": "DEV-A",
                "ip_country": "TW",
                "account_age_days": 400,
                "ground_truth_label": 0,
                "anomaly_type": "normal",
            }
        )
    return pd.DataFrame(rows)
