import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lol_analytics.calibration import (
    apply_platt_calibrator,
    fit_platt_calibrator,
    paired_calibration_delta_intervals,
    probability_logit,
    run_calibration_phase,
    temporal_cross_fitted_platt,
)


POLICY_PATH = Path("data/manifests/cohort-policy-v1.json")


def _temporal_predictions(*, rows_per_fold: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for fold in (1, 2, 3):
        latent = rng.uniform(-2.5, 2.5, size=rows_per_fold)
        raw_probability = 1 / (1 + np.exp(-latent))
        true_probability = 1 / (1 + np.exp(-(2.0 * latent - 0.5)))
        truth = rng.binomial(1, true_probability)
        for index in range(rows_per_fold):
            rows.append(
                {
                    "matchId": f"fold-{fold}-{index}",
                    "game_start_timestamp_ms": fold * 1_000_000 + index,
                    "win": int(truth[index]),
                    "win_probability": float(raw_probability[index]),
                    "outer_fold": fold,
                    "boundary_offset": 0,
                }
            )
    return pd.DataFrame(rows)


def test_probability_logit_and_platt_application_are_monotonic() -> None:
    probability = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    logits = probability_logit(probability)
    assert np.isfinite(logits).all()
    assert np.all(np.diff(logits) > 0)

    truth = np.array([0, 0, 0, 1, 1])
    calibrator = fit_platt_calibrator(probability, truth)
    calibrated = apply_platt_calibrator(calibrator, probability)
    assert np.all(np.diff(calibrated) > 0)
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_temporal_cross_fit_uses_only_past_folds_and_improves_miscalibration() -> None:
    summary, predictions, _ = temporal_cross_fitted_platt(
        _temporal_predictions(), bins=10
    )

    assert summary["base_repeat_rows"] == 900
    assert summary["cross_fitted_evaluation_rows"] == 600
    assert summary["evaluated_folds"] == [2, 3]
    assert predictions["matchId"].is_unique
    assert summary["acceptance_policy"]["accepted"] is True
    assert summary["delta_after_minus_before"]["brier_score"] < 0
    assert summary["delta_after_minus_before"]["log_loss"] < 0
    for fold in summary["folds"]:
        assert fold["calibrator_training_end_ms"] < fold["evaluation_start_ms"]


def test_temporal_cross_fit_rejects_duplicate_or_non_temporal_evidence() -> None:
    predictions = _temporal_predictions(rows_per_fold=20)
    predictions.loc[1, "matchId"] = predictions.loc[0, "matchId"]
    with pytest.raises(ValueError, match="unique by matchId"):
        temporal_cross_fitted_platt(predictions)

    predictions = _temporal_predictions(rows_per_fold=20)
    predictions.loc[predictions["outer_fold"].eq(2), "game_start_timestamp_ms"] -= 2_000_000
    with pytest.raises(ValueError, match="overlaps"):
        temporal_cross_fitted_platt(predictions)


def test_paired_calibration_bootstrap_is_deterministic() -> None:
    _, predictions, _ = temporal_cross_fitted_platt(_temporal_predictions(), bins=5)
    first = paired_calibration_delta_intervals(
        predictions, bins=5, iterations=100
    )
    second = paired_calibration_delta_intervals(
        predictions, bins=5, iterations=100
    )
    assert first == second
    assert first["metrics"]["brier_score"]["estimate"] < 0


def test_calibration_phase_rejects_diagnostic_cohort_before_reading_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="does not allow fit_calibrator"):
        run_calibration_phase(
            "data/raw/events_asia_train_105.parquet",
            POLICY_PATH,
            tmp_path / "unused-robustness.json",
            tmp_path / "unused-final-model.json",
            tmp_path / "output",
            tmp_path / "manifest.json",
            bootstrap_iterations=100,
        )


def test_committed_phase_two_decision_keeps_base_probabilities() -> None:
    manifest = json.loads(
        Path("data/manifests/calibration-v1.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "calibration_rejected_keep_base_probabilities"
    assert manifest["method_policy"]["external_diagnostic_labels_used"] is False
    assert manifest["method_policy"]["future_holdout_accessed"] is False
    for key in ("early_10m", "mid_20m"):
        model = manifest["models"][key]
        assert model["decision"] == "rejected_keep_base_probability"
        assert model["calibrator_artifact"] is None
        evaluation = model["temporal_cross_fitted_evaluation"]
        assert evaluation["delta_after_minus_before"]["brier_score"] > 0
        assert evaluation["delta_after_minus_before"]["log_loss"] > 0
