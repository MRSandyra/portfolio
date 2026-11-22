"""Temporal cross-fitted probability calibration on development evidence only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .final_results import authorize_pre_release_tuning
from .config import RANDOM_STATE
from .monitoring import calibration_metrics
from .quality import sha256_file, write_manifest

PLATT_CLIP = 1e-6
PLATT_C = 1_000_000.0
REQUIRED_PREDICTION_COLUMNS = frozenset(
    {
        "matchId",
        "game_start_timestamp_ms",
        "win",
        "win_probability",
        "outer_fold",
        "boundary_offset",
    }
)


def probability_logit(probability: Iterable[float]) -> np.ndarray:
    scores = np.asarray(list(probability), dtype=float)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("probability must be a non-empty one-dimensional array")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probability must be finite and lie in [0, 1]")
    clipped = np.clip(scores, PLATT_CLIP, 1 - PLATT_CLIP)
    return np.log(clipped / (1 - clipped))


def fit_platt_calibrator(
    probability: Iterable[float],
    y_true: Iterable[int],
) -> LogisticRegression:
    truth = np.asarray(list(y_true), dtype=int)
    logits = probability_logit(probability)
    if truth.shape != logits.shape or set(np.unique(truth)) != {0, 1}:
        raise ValueError("Platt fitting requires aligned predictions with both classes")
    calibrator = LogisticRegression(
        C=PLATT_C,
        solver="lbfgs",
        max_iter=2_000,
        random_state=RANDOM_STATE,
    )
    calibrator.fit(logits.reshape(-1, 1), truth)
    if float(calibrator.coef_[0, 0]) <= 0:
        raise ValueError("Platt calibrator must preserve probability ranking")
    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression,
    probability: Iterable[float],
) -> np.ndarray:
    logits = probability_logit(probability)
    calibrated = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
    if not np.isfinite(calibrated).all() or np.any((calibrated < 0) | (calibrated > 1)):
        raise ValueError("Calibrator produced invalid probabilities")
    return calibrated


def _validate_base_repeat(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"Calibration predictions are missing columns: {missing}")
    base = predictions.loc[predictions["boundary_offset"].eq(0)].copy()
    if base.empty:
        raise ValueError("Calibration evidence has no base repeat")
    if base["matchId"].duplicated().any():
        raise ValueError("Base-repeat calibration evidence must be unique by matchId")
    if base["win"].isna().any() or set(base["win"].unique()) != {0, 1}:
        raise ValueError("Base-repeat calibration evidence requires both target classes")
    if base["game_start_timestamp_ms"].isna().any():
        raise ValueError("Calibration evidence requires complete absolute timestamps")
    probability_logit(base["win_probability"])
    folds = sorted(int(value) for value in base["outer_fold"].unique())
    if len(folds) < 3 or folds != list(range(folds[0], folds[-1] + 1)):
        raise ValueError("Calibration requires at least three contiguous temporal folds")
    return base.sort_values(
        ["outer_fold", "game_start_timestamp_ms", "matchId"], kind="stable"
    ).reset_index(drop=True)


def calibration_slope_intercept(
    y_true: Iterable[int], probability: Iterable[float]
) -> dict[str, float]:
    """Estimate diagnostic calibration intercept and slope on supplied predictions."""
    model = fit_platt_calibrator(probability, y_true)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def temporal_cross_fitted_platt(
    predictions: pd.DataFrame,
    *,
    bins: int = 10,
) -> tuple[dict[str, Any], pd.DataFrame, LogisticRegression]:
    """Fit on past OOF folds, evaluate on later folds, then fit all-OOF artifact."""
    base = _validate_base_repeat(predictions)
    folds = sorted(int(value) for value in base["outer_fold"].unique())
    evaluated: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in folds[1:]:
        train = base.loc[base["outer_fold"].lt(fold)]
        test = base.loc[base["outer_fold"].eq(fold)]
        if train.empty or test.empty:
            raise ValueError("Temporal calibration fold is empty")
        train_end = int(train["game_start_timestamp_ms"].max())
        test_start = int(test["game_start_timestamp_ms"].min())
        if train_end >= test_start:
            raise ValueError("Temporal calibrator training overlaps its evaluation fold")
        calibrator = fit_platt_calibrator(train["win_probability"], train["win"])
        calibrated = apply_platt_calibrator(calibrator, test["win_probability"])
        fold_frame = test[
            ["matchId", "game_start_timestamp_ms", "win", "outer_fold"]
        ].copy()
        fold_frame["raw_probability"] = test["win_probability"].to_numpy()
        fold_frame["calibrated_probability"] = calibrated
        evaluated.append(fold_frame)
        fold_reports.append(
            {
                "evaluation_outer_fold": fold,
                "calibrator_training_rows": len(train),
                "evaluation_rows": len(test),
                "calibrator_training_end_ms": train_end,
                "evaluation_start_ms": test_start,
                "platt_intercept": float(calibrator.intercept_[0]),
                "platt_slope": float(calibrator.coef_[0, 0]),
            }
        )
    cross_fitted = pd.concat(evaluated, ignore_index=True)
    if cross_fitted["matchId"].duplicated().any():
        raise AssertionError("A match appears in multiple calibration evaluation folds")
    before = calibration_metrics(
        cross_fitted["win"], cross_fitted["raw_probability"], bins=bins
    )
    after = calibration_metrics(
        cross_fitted["win"], cross_fitted["calibrated_probability"], bins=bins
    )
    deltas = {
        name: float(after[name] - before[name])
        for name in ("brier_score", "log_loss", "expected_calibration_error")
    }
    accepted = deltas["brier_score"] <= 0 and deltas["log_loss"] <= 0
    final_calibrator = fit_platt_calibrator(
        base["win_probability"], base["win"]
    )
    summary = {
        "method": "platt_logistic_on_probability_logit",
        "method_selection": "fixed_before_future_holdout_release",
        "base_repeat_rows": len(base),
        "cross_fitted_evaluation_rows": len(cross_fitted),
        "first_fold_reserved_for_initial_calibrator_training": folds[0],
        "evaluated_folds": folds[1:],
        "folds": fold_reports,
        "before": before,
        "after": after,
        "delta_after_minus_before": deltas,
        "diagnostic_calibration": {
            "before": calibration_slope_intercept(
                cross_fitted["win"], cross_fitted["raw_probability"]
            ),
            "after": calibration_slope_intercept(
                cross_fitted["win"], cross_fitted["calibrated_probability"]
            ),
        },
        "acceptance_policy": {
            "rule": "Accept only when temporal cross-fitted Brier score and log loss both do not worsen.",
            "brier_non_worsening": deltas["brier_score"] <= 0,
            "log_loss_non_worsening": deltas["log_loss"] <= 0,
            "accepted": accepted,
        },
        "all_oof_platt_parameters": {
            "intercept": float(final_calibrator.intercept_[0]),
            "slope": float(final_calibrator.coef_[0, 0]),
        },
    }
    return summary, cross_fitted, final_calibrator


def paired_calibration_delta_intervals(
    predictions: pd.DataFrame,
    *,
    bins: int = 10,
    iterations: int = 2_000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    truth = predictions["win"].to_numpy(dtype=int)
    raw = predictions["raw_probability"].to_numpy(dtype=float)
    calibrated = predictions["calibrated_probability"].to_numpy(dtype=float)
    negative = np.flatnonzero(truth == 0)
    positive = np.flatnonzero(truth == 1)
    rng = np.random.default_rng(random_state)
    names = ("brier_score", "log_loss", "expected_calibration_error")
    samples = {name: np.empty(iterations, dtype=float) for name in names}
    for iteration in range(iterations):
        selected = np.concatenate(
            (
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            )
        )
        before = calibration_metrics(truth[selected], raw[selected], bins=bins)
        after = calibration_metrics(truth[selected], calibrated[selected], bins=bins)
        for name in names:
            samples[name][iteration] = after[name] - before[name]
    before = calibration_metrics(truth, raw, bins=bins)
    after = calibration_metrics(truth, calibrated, bins=bins)
    alpha = (1 - confidence) / 2
    return {
        "confidence": confidence,
        "iterations": iterations,
        "delta_definition": "calibrated_minus_raw; negative favors calibration",
        "metrics": {
            name: {
                "estimate": float(after[name] - before[name]),
                "lower": float(np.quantile(samples[name], alpha)),
                "upper": float(np.quantile(samples[name], 1 - alpha)),
                "bootstrap_probability_improved": float(np.mean(samples[name] < 0)),
            }
            for name in names
        },
    }


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Phase 2 temporal calibration report",
        "",
        "Platt scaling was evaluated with temporal cross-fitting on development base-repeat "
        "OOF predictions only. External diagnostic cohorts and the frozen future holdout "
        "were not read.",
        "",
        "| Model | Evaluation rows | Raw Brier | Calibrated Brier | Raw log loss | Calibrated log loss | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key in ("early_10m", "mid_20m"):
        result = manifest["models"][key]
        evaluation = result["temporal_cross_fitted_evaluation"]
        lines.append(
            f"| {key} | {evaluation['cross_fitted_evaluation_rows']} | "
            f"{evaluation['before']['brier_score']:.4f} | "
            f"{evaluation['after']['brier_score']:.4f} | "
            f"{evaluation['before']['log_loss']:.4f} | "
            f"{evaluation['after']['log_loss']:.4f} | "
            f"{'accept' if evaluation['acceptance_policy']['accepted'] else 'reject'} |"
        )
    lines.extend(
        [
            "",
            "The fixed acceptance rule requires both Brier score and log loss to be "
            "non-worsening. A rejected calibrator is not packaged or carried into the "
            "future-holdout release workflow; the base model probability remains canonical.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_calibration_phase(
    development_input: str | Path,
    cohort_policy_path: str | Path,
    robustness_summary_path: str | Path,
    final_model_manifest_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    bins: int = 10,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    """Evaluate and conditionally package calibrators without external data access."""
    development = Path(development_input)
    policy_path = Path(cohort_policy_path)
    robustness_path = Path(robustness_summary_path)
    final_manifest_path = Path(final_model_manifest_path)
    destination = Path(output_dir)
    manifest_destination = Path(manifest_path)
    cohort_key = authorize_pre_release_tuning(
        policy_path, development, "fit_calibrator"
    )

    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    if final_manifest.get("status") != "final_models_fitted_unreleased":
        raise ValueError("Final model manifest has an unexpected status")
    if final_manifest["training"]["file"]["sha256"] != sha256_file(development):
        raise ValueError("Development input differs from the final-model training evidence")
    if final_manifest["selection_evidence"]["file"]["sha256"] != sha256_file(
        robustness_path
    ):
        raise ValueError("Robustness summary differs from final selection evidence")
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    if robustness.get("frozen_future_holdout_accessed") is not False:
        raise ValueError("Calibration evidence accessed the frozen future holdout")
    if Path(robustness.get("input", "")).resolve() != development.resolve():
        raise ValueError("Robustness evidence does not belong to the development cohort")

    destination.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    any_accepted = False
    for horizon, key in ((10, "early_10m"), (20, "mid_20m")):
        prediction_path = robustness_path.parent / robustness["models"][key][
            "predictions_path"
        ]
        predictions = pd.read_csv(prediction_path)
        evaluation, cross_fitted, calibrator = temporal_cross_fitted_platt(
            predictions, bins=bins
        )
        intervals = paired_calibration_delta_intervals(
            cross_fitted,
            bins=bins,
            iterations=bootstrap_iterations,
            random_state=RANDOM_STATE + horizon,
        )
        predictions_name = f"cross_fitted_calibration_predictions_{horizon}m.csv"
        cross_fitted.to_csv(destination / predictions_name, index=False)
        accepted = bool(evaluation["acceptance_policy"]["accepted"])
        artifact: dict[str, Any] | None = None
        if accepted:
            artifact_path = destination / f"platt_calibrator_{horizon}m.joblib"
            joblib.dump(
                {
                    "method": "platt_logistic_on_probability_logit",
                    "calibrator": calibrator,
                    "horizon_minutes": horizon,
                    "development_sha256": sha256_file(development),
                    "robustness_summary_sha256": sha256_file(robustness_path),
                    "prediction_evidence_sha256": sha256_file(prediction_path),
                },
                artifact_path,
                compress=3,
            )
            artifact = {
                "path": artifact_path.as_posix(),
                "bytes": artifact_path.stat().st_size,
                "sha256": sha256_file(artifact_path),
            }
            any_accepted = True
        models[key] = {
            "horizon_minutes": horizon,
            "prediction_evidence": {
                "path": prediction_path.as_posix(),
                "bytes": prediction_path.stat().st_size,
                "sha256": sha256_file(prediction_path),
                "base_repeat_only": True,
            },
            "temporal_cross_fitted_evaluation": evaluation,
            "paired_bootstrap_delta_intervals": intervals,
            "decision": "accepted" if accepted else "rejected_keep_base_probability",
            "calibrator_artifact": artifact,
            "cross_fitted_predictions_path": predictions_name,
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "calibrators_fitted_unreleased"
            if any_accepted
            else "calibration_rejected_keep_base_probabilities"
        ),
        "method_policy": {
            "method": "Platt scaling on clipped probability logits",
            "method_was_fixed_before_future_holdout_release": True,
            "acceptance_rule": (
                "Accept per horizon only if temporal cross-fitted Brier score and log loss "
                "both do not worsen."
            ),
            "external_diagnostic_labels_used": False,
            "future_holdout_accessed": False,
            "future_holdout_labels_read": False,
            "future_holdout_scored": False,
        },
        "development": {
            "cohort_key": cohort_key,
            "file": {
                "path": development.as_posix(),
                "bytes": development.stat().st_size,
                "sha256": sha256_file(development),
            },
        },
        "evidence": {
            "cohort_policy_sha256": sha256_file(policy_path),
            "robustness_summary_sha256": sha256_file(robustness_path),
            "final_model_manifest_sha256": sha256_file(final_manifest_path),
        },
        "models": models,
    }
    if manifest_destination.exists():
        existing = json.loads(manifest_destination.read_text(encoding="utf-8"))
        if existing != manifest:
            raise FileExistsError(
                "Refusing to replace frozen calibration evidence with changed results"
            )
    else:
        write_manifest(manifest_destination, manifest)
    _write_report(destination / "calibration_report.md", manifest)
    return manifest
