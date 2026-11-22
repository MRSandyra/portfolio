import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lol_analytics.config import TEN_MINUTES_MS, features_available_at
from lol_analytics.monitoring import (
    bootstrap_calibration_intervals,
    build_model_input_contracts,
    calibration_metrics,
    feature_drift_report,
    validate_model_input_contract,
)


def _valid_early_frame(rows: int = 6) -> pd.DataFrame:
    features = features_available_at(TEN_MINUTES_MS)
    frame = pd.DataFrame({feature: np.arange(rows, dtype=float) for feature in features})
    for feature in features:
        if feature.startswith("first_"):
            frame[feature] = [-1, 0, 1, -1, 0, 1][:rows]
    frame.insert(0, "win", [0, 1, 0, 1, 0, 1][:rows])
    frame.insert(0, "matchId", [f"m-{index}" for index in range(rows)])
    return frame


def test_calibration_metrics_have_known_values_and_bootstrap_is_deterministic() -> None:
    truth = np.array([0, 0, 1, 1])
    perfect = np.array([0.0, 0.0, 1.0, 1.0])
    metrics = calibration_metrics(truth, perfect, bins=2)

    assert metrics["brier_score"] == 0
    assert metrics["expected_calibration_error"] == 0
    assert metrics["log_loss"] == pytest.approx(0, abs=1e-12)

    first = bootstrap_calibration_intervals(truth, perfect, bins=2, iterations=100)
    second = bootstrap_calibration_intervals(truth, perfect, bins=2, iterations=100)
    assert first == second


def test_calibration_rejects_invalid_probabilities_or_single_class() -> None:
    with pytest.raises(ValueError, match="both target classes"):
        calibration_metrics([1, 1], [0.5, 0.6])
    with pytest.raises(ValueError, match="finite and lie"):
        calibration_metrics([0, 1], [0.2, np.inf])


def test_runtime_contract_rejects_cross_horizon_nonfinite_and_bad_side_values() -> None:
    frame = _valid_early_frame()
    assert validate_model_input_contract(frame, horizon_minutes=10)["passed"] is True

    cross_horizon = frame.assign(gold_diff_20=0.0)
    with pytest.raises(ValueError, match="another horizon"):
        validate_model_input_contract(cross_horizon, horizon_minutes=10)

    nonfinite = frame.copy()
    nonfinite.loc[0, "gold_diff_10"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_model_input_contract(nonfinite, horizon_minutes=10)

    invalid_side = frame.copy()
    invalid_side.loc[0, "first_blood_side_10"] = 2
    with pytest.raises(ValueError, match="outside -1, 0, 1"):
        validate_model_input_contract(invalid_side, horizon_minutes=10)

    with pytest.raises(ValueError, match="Unregistered model-input columns"):
        validate_model_input_contract(frame.assign(duration_min=30), horizon_minutes=10)

    with pytest.raises(ValueError, match="at least one row"):
        validate_model_input_contract(frame.iloc[0:0], horizon_minutes=10)


def test_runtime_contract_requires_explicit_all_missing_exception() -> None:
    frame = _valid_early_frame()
    frame["first_dragon_side_10"] = np.nan
    with pytest.raises(ValueError, match="entirely missing"):
        validate_model_input_contract(frame, horizon_minutes=10)

    report = validate_model_input_contract(
        frame,
        horizon_minutes=10,
        allowed_all_missing=["first_dragon_side_10"],
    )
    assert report["observed_all_missing"] == ["first_dragon_side_10"]


def test_drift_report_distinguishes_identical_and_shifted_distributions() -> None:
    reference = pd.DataFrame({"feature": np.arange(100, dtype=float)})
    identical = feature_drift_report(reference, reference.copy(), ["feature"])
    assert identical["severity_counts"] == {"low": 1, "medium": 0, "high": 0}
    assert identical["features"]["feature"]["population_stability_index"] == 0

    shifted = pd.DataFrame({"feature": np.arange(100, dtype=float) + 1_000})
    report = feature_drift_report(reference, shifted, ["feature"])
    assert report["severity_counts"]["high"] == 1
    assert report["features"]["feature"][
        "current_out_of_reference_range_fraction"
    ] == 1


def test_committed_contract_matches_feature_registry() -> None:
    committed = json.loads(
        Path("data/contracts/model-input-v1.json").read_text(encoding="utf-8")
    )
    assert committed == build_model_input_contracts()
