"""Configuration and threshold profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ThresholdProfile:
    name: str
    amount_zscore: float
    amount_modified_zscore: float
    amount_iqr_multiplier: float
    amount_percentile: float
    velocity_per_hour: int
    pass_through_minutes: float
    dormant_days: float
    counterparty_count_24h: int
    behavior_amount_ratio: float
    threshold_avoidance_count: int
    unusual_hour_max_history_ratio: float
    minimum_history: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


PROFILES = {
    "conservative": ThresholdProfile(
        "conservative", 4.0, 5.0, 3.0, 0.995, 10, 5.0, 90.0, 14, 8.0, 4, 0.05, 8
    ),
    "balanced": ThresholdProfile(
        "balanced", 3.0, 3.5, 2.0, 0.990, 7, 15.0, 45.0, 10, 5.0, 3, 0.12, 5
    ),
    "sensitive": ThresholdProfile(
        "sensitive", 2.0, 2.5, 1.5, 0.975, 5, 30.0, 21.0, 7, 3.0, 2, 0.22, 3
    ),
}


def get_profile(name: str) -> ThresholdProfile:
    try:
        return PROFILES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown profile {name!r}; choose from {sorted(PROFILES)}") from exc

