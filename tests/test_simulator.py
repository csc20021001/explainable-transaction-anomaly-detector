from anomaly_detector.simulator import generate_synthetic_transactions
from anomaly_detector.validation import REQUIRED_COLUMNS, validate_and_clean


def test_simulator_has_required_schema():
    frame = generate_synthetic_transactions(n_accounts=60, normal_per_account=3, seed=1)
    assert set(REQUIRED_COLUMNS).issubset(frame.columns)


def test_simulator_is_reproducible():
    left = generate_synthetic_transactions(n_accounts=60, normal_per_account=2, seed=7)
    right = generate_synthetic_transactions(n_accounts=60, normal_per_account=2, seed=7)
    assert left.equals(right)


def test_simulator_contains_all_ten_anomaly_types():
    frame = generate_synthetic_transactions(n_accounts=60, normal_per_account=2, seed=7)
    patterns = set(frame.loc[frame.ground_truth_label.eq(1), "anomaly_type"])
    assert len(patterns) == 10


def test_simulator_passes_validation():
    frame = generate_synthetic_transactions(n_accounts=60, normal_per_account=2, seed=7)
    result = validate_and_clean(frame)
    assert result.report["status"] == "valid"
