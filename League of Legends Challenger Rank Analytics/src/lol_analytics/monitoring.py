"""Calibration, drift, and runtime data-contract audits."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import (
    FEATURE_REGISTRY,
    RANDOM_STATE,
    TEN_MINUTES_MS,
    TWENTY_MINUTES_MS,
    features_available_at,
)
from .cohort_policy import authorize_cohort_file_operation
from .data import read_events
from .features import build_match_features, eligible_at_horizon
from .quality import sha256_file, write_manifest
from .rank_era_validation import (
    UNAVAILABLE_FEATURES,
    adapt_high_diamond_10m,
    load_final_early_bundle,
)

DEFAULT_CALIBRATION_BINS = 10
DRIFT_THRESHOLDS = {
    "psi_medium": 0.10,
    "psi_high": 0.25,
    "ks_medium": 0.10,
    "ks_high": 0.20,
    "missing_rate_delta_medium": 0.10,
    "missing_rate_delta_high": 0.25,
    "out_of_reference_range_medium": 0.05,
    "out_of_reference_range_high": 0.10,
}


def _validated_probability_inputs(
    y_true: Iterable[int], probability: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(probability), dtype=float)
    if truth.ndim != 1 or truth.shape != scores.shape or not len(truth):
        raise ValueError("truth and probability must be non-empty one-dimensional arrays")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("Calibration metrics require both target classes")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    return truth, scores


def calibration_metrics(
    y_true: Iterable[int],
    probability: Iterable[float],
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, Any]:
    """Return proper scoring rules and an equal-width reliability table."""
    truth, scores = _validated_probability_inputs(y_true, probability)
    if bins < 2 or bins > 100:
        raise ValueError("bins must be between 2 and 100")
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    brier = float(np.mean((scores - truth) ** 2))
    log_loss = float(
        -np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))
    )
    bin_index = np.minimum((scores * bins).astype(int), bins - 1)
    reliability: list[dict[str, Any]] = []
    weighted_errors: list[float] = []
    absolute_errors: list[float] = []
    for index in range(bins):
        selected = bin_index == index
        count = int(selected.sum())
        if count:
            mean_probability = float(scores[selected].mean())
            observed_rate = float(truth[selected].mean())
            absolute_error = abs(mean_probability - observed_rate)
            weighted_errors.append(absolute_error * count / len(truth))
            absolute_errors.append(absolute_error)
        else:
            mean_probability = None
            observed_rate = None
        reliability.append(
            {
                "bin": index + 1,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "mean_probability": mean_probability,
                "observed_positive_rate": observed_rate,
            }
        )
    return {
        "samples": len(truth),
        "bins": bins,
        "brier_score": brier,
        "log_loss": log_loss,
        "expected_calibration_error": float(sum(weighted_errors)),
        "maximum_calibration_error": float(max(absolute_errors)),
        "reliability": reliability,
    }


def bootstrap_calibration_intervals(
    y_true: Iterable[int],
    probability: Iterable[float],
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
    iterations: int = 2_000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Stratified bootstrap intervals for aggregate calibration metrics."""
    truth, scores = _validated_probability_inputs(y_true, probability)
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
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
        metrics = calibration_metrics(truth[selected], scores[selected], bins=bins)
        for name in names:
            samples[name][iteration] = metrics[name]
    point = calibration_metrics(truth, scores, bins=bins)
    alpha = (1 - confidence) / 2
    return {
        "confidence": confidence,
        "iterations": iterations,
        "metrics": {
            name: {
                "estimate": point[name],
                "lower": float(np.quantile(samples[name], alpha)),
                "upper": float(np.quantile(samples[name], 1 - alpha)),
            }
            for name in names
        },
    }


def build_model_input_contracts() -> dict[str, Any]:
    """Build the versioned runtime contract directly from the feature registry."""
    models: dict[str, Any] = {}
    for horizon, name in ((10, "early_10m"), (20, "mid_20m")):
        features = features_available_at(horizon * 60_000)
        models[name] = {
            "horizon_minutes": horizon,
            "required_features": features,
            "rules": {
                "registered_features_from_other_horizons": "reject",
                "unregistered_columns_except_declared_metadata": "reject",
                "non_numeric_values": "reject",
                "infinite_values": "reject",
                "all_missing_feature": "reject_unless_source_exception_is_declared",
                "partial_missing_values": "allow_training_fitted_imputer",
                "side_indicator_allowed_values": [-1, 0, 1],
            },
            "features": {
                feature: {
                    "available_at_ms": FEATURE_REGISTRY[feature].available_at_ms,
                    "description": FEATURE_REGISTRY[feature].description,
                    "dtype": "number",
                    "nullable": True,
                    **(
                        {"allowed_values": [-1, 0, 1]}
                        if feature.startswith("first_")
                        else {}
                    ),
                }
                for feature in features
            },
        }
    return {
        "schema_version": 1,
        "contract_name": "lol_model_input_features",
        "metadata_columns_allowed": ["matchId", "win"],
        "models": models,
    }


def write_model_input_contracts(path: str | Path) -> dict[str, Any]:
    payload = build_model_input_contracts()
    write_manifest(path, payload)
    return payload


def validate_model_input_contract(
    frame: pd.DataFrame,
    *,
    horizon_minutes: int,
    allowed_all_missing: Iterable[str] = (),
) -> dict[str, Any]:
    """Hard-fail incompatible model input before preprocessing or scoring."""
    if horizon_minutes not in {10, 20}:
        raise ValueError("horizon_minutes must be 10 or 20")
    if frame.empty:
        raise ValueError("Model input must contain at least one row")
    required = features_available_at(horizon_minutes * 60_000)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Required model features are missing: {missing}")
    unexpected_registered = sorted(
        set(frame.columns).intersection(FEATURE_REGISTRY).difference(required)
    )
    if unexpected_registered:
        raise ValueError(
            f"Registered features from another horizon are not allowed: {unexpected_registered}"
        )
    allowed_columns = set(required).union(("matchId", "win"))
    unexpected_columns = sorted(set(frame.columns).difference(allowed_columns))
    if unexpected_columns:
        raise ValueError(f"Unregistered model-input columns are not allowed: {unexpected_columns}")
    allowed_missing = set(allowed_all_missing)
    unknown_exceptions = sorted(allowed_missing.difference(required))
    if unknown_exceptions:
        raise ValueError(f"Unknown all-missing exceptions: {unknown_exceptions}")
    all_missing: list[str] = []
    partial_missing: dict[str, float] = {}
    for feature in required:
        values = frame[feature]
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"Model feature must be numeric: {feature}")
        numeric = values.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(numeric).any():
            raise ValueError(f"Model feature contains an infinite value: {feature}")
        missing_fraction = float(np.isnan(numeric).mean())
        if missing_fraction == 1:
            all_missing.append(feature)
            if feature not in allowed_missing:
                raise ValueError(f"Model feature is entirely missing: {feature}")
        elif missing_fraction:
            partial_missing[feature] = missing_fraction
        if feature.startswith("first_"):
            observed = values.dropna()
            if not observed.isin((-1, 0, 1)).all():
                raise ValueError(f"Side indicator has values outside -1, 0, 1: {feature}")
    if "matchId" in frame and frame["matchId"].duplicated().any():
        raise ValueError("matchId must be unique at model-input grain")
    if "win" in frame:
        target = frame["win"].dropna()
        if not target.isin((0, 1)).all():
            raise ValueError("win must be binary when present")
    return {
        "passed": True,
        "horizon_minutes": horizon_minutes,
        "rows": len(frame),
        "required_features": required,
        "allowed_all_missing": sorted(allowed_missing),
        "observed_all_missing": sorted(all_missing),
        "partial_missing_fraction": partial_missing,
        "unexpected_registered_features": [],
    }


def _ks_statistic(reference: np.ndarray, current: np.ndarray) -> float | None:
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if not len(reference) or not len(current):
        return None
    support = np.sort(np.unique(np.concatenate((reference, current))))
    reference_cdf = np.searchsorted(np.sort(reference), support, side="right") / len(reference)
    current_cdf = np.searchsorted(np.sort(current), support, side="right") / len(current)
    return float(np.max(np.abs(reference_cdf - current_cdf)))


def _population_stability_index(reference: np.ndarray, current: np.ndarray) -> float:
    reference_missing = float(np.isnan(reference).mean())
    current_missing = float(np.isnan(current).mean())
    observed_reference = reference[np.isfinite(reference)]
    observed_current = current[np.isfinite(current)]
    if not len(observed_reference):
        raise ValueError("Reference drift feature must have at least one finite value")
    quantiles = np.unique(np.quantile(observed_reference, np.linspace(0, 1, 11)))
    if len(quantiles) == 1:
        boundaries = np.array([-np.inf, np.inf])
    else:
        boundaries = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))
    reference_histogram = np.histogram(observed_reference, bins=boundaries)[0].astype(float)
    current_histogram = np.histogram(observed_current, bins=boundaries)[0].astype(float)
    reference_proportions = np.append(
        reference_histogram / len(reference), reference_missing
    )
    current_proportions = np.append(current_histogram / len(current), current_missing)
    epsilon = 1e-6
    reference_proportions = np.clip(reference_proportions, epsilon, None)
    current_proportions = np.clip(current_proportions, epsilon, None)
    return float(
        np.sum(
            (current_proportions - reference_proportions)
            * np.log(current_proportions / reference_proportions)
        )
    )


def _drift_severity(
    *, psi: float, ks: float | None, missing_delta: float, out_of_range: float
) -> str:
    high = (
        psi >= DRIFT_THRESHOLDS["psi_high"]
        or (ks is not None and ks >= DRIFT_THRESHOLDS["ks_high"])
        or missing_delta >= DRIFT_THRESHOLDS["missing_rate_delta_high"]
        or out_of_range >= DRIFT_THRESHOLDS["out_of_reference_range_high"]
    )
    if high:
        return "high"
    medium = (
        psi >= DRIFT_THRESHOLDS["psi_medium"]
        or (ks is not None and ks >= DRIFT_THRESHOLDS["ks_medium"])
        or missing_delta >= DRIFT_THRESHOLDS["missing_rate_delta_medium"]
        or out_of_range >= DRIFT_THRESHOLDS["out_of_reference_range_medium"]
    )
    return "medium" if medium else "low"


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Iterable[str],
) -> dict[str, Any]:
    """Compare distributions using declared heuristic monitoring thresholds."""
    rows: dict[str, Any] = {}
    for feature in features:
        if feature not in reference or feature not in current:
            raise ValueError(f"Drift input is missing feature: {feature}")
        reference_values = reference[feature].to_numpy(dtype=float, na_value=np.nan)
        current_values = current[feature].to_numpy(dtype=float, na_value=np.nan)
        finite_reference = reference_values[np.isfinite(reference_values)]
        finite_current = current_values[np.isfinite(current_values)]
        if not len(finite_reference):
            raise ValueError(f"Reference drift feature is entirely missing: {feature}")
        reference_missing = float(np.isnan(reference_values).mean())
        current_missing = float(np.isnan(current_values).mean())
        missing_delta = abs(current_missing - reference_missing)
        if len(finite_current):
            out_of_range = float(
                np.mean(
                    (finite_current < finite_reference.min())
                    | (finite_current > finite_reference.max())
                )
            )
        else:
            out_of_range = 0.0
        psi = _population_stability_index(reference_values, current_values)
        ks = _ks_statistic(reference_values, current_values)
        severity = _drift_severity(
            psi=psi,
            ks=ks,
            missing_delta=missing_delta,
            out_of_range=out_of_range,
        )
        rows[feature] = {
            "severity": severity,
            "population_stability_index": psi,
            "ks_statistic": ks,
            "reference_missing_rate": reference_missing,
            "current_missing_rate": current_missing,
            "absolute_missing_rate_delta": missing_delta,
            "current_out_of_reference_range_fraction": out_of_range,
            "reference_min": float(finite_reference.min()),
            "reference_max": float(finite_reference.max()),
            "current_min": float(finite_current.min()) if len(finite_current) else None,
            "current_max": float(finite_current.max()) if len(finite_current) else None,
        }
    counts = {
        severity: sum(row["severity"] == severity for row in rows.values())
        for severity in ("low", "medium", "high")
    }
    return {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "thresholds": DRIFT_THRESHOLDS,
        "severity_counts": counts,
        "features": rows,
    }


def _load_base_repeat_predictions(
    robustness_summary_path: Path,
    horizon_minutes: int,
) -> pd.DataFrame:
    summary = json.loads(robustness_summary_path.read_text(encoding="utf-8"))
    key = "early_10m" if horizon_minutes == 10 else "mid_20m"
    if summary.get("frozen_future_holdout_accessed") is not False:
        raise ValueError("Robustness evidence accessed the frozen future holdout")
    if summary["models"][key]["horizon_minutes"] != horizon_minutes:
        raise ValueError(f"Robustness horizon mismatch for {key}")
    prediction_path = robustness_summary_path.parent / summary["models"][key][
        "predictions_path"
    ]
    predictions = pd.read_csv(prediction_path)
    required = {"matchId", "win", "win_probability", "boundary_offset"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Robustness predictions are missing columns: {missing}")
    base = predictions.loc[predictions["boundary_offset"].eq(0)].copy()
    if base.empty or base["matchId"].duplicated().any():
        raise ValueError("Base-repeat predictions must be non-empty and unique by matchId")
    return base


def _write_report(path: Path, result: dict[str, Any]) -> None:
    external = result["calibration"]["external_early_10m"]["metrics"]
    development = result["calibration"]["development_base_repeat"]
    drift = result["drift"]["early_10m_external_vs_training"]
    lines = [
        "# Calibration, drift, and data-contract audit",
        "",
        "This is a diagnostic audit. It does not fit or deploy a calibrator, refit a model, "
        "or access the frozen future holdout.",
        "",
        "## Calibration",
        "",
        f"- External early 10m Brier score: {external['brier_score']:.4f}",
        f"- External early 10m ECE: {external['expected_calibration_error']:.4f}",
        f"- Development early 10m ECE: "
        f"{development['early_10m']['metrics']['expected_calibration_error']:.4f}",
        f"- Development mid 20m ECE: "
        f"{development['mid_20m']['metrics']['expected_calibration_error']:.4f}",
        "",
        "The development values use only the repeated-nested base-repeat out-of-fold "
        "predictions. The external value is for the frozen final early-game model. These "
        "are different estimands and are not a paired horizon comparison.",
        "",
        "## Drift",
        "",
        f"- Low/medium/high feature counts: {drift['severity_counts']['low']}/"
        f"{drift['severity_counts']['medium']}/{drift['severity_counts']['high']}",
        f"- Prediction-distribution severity: "
        f"{result['drift']['early_10m_prediction_distribution']['features']['win_probability']['severity']}",
        "",
        "Drift levels use declared heuristic thresholds and are alerts, not statistical "
        "proof of a causal rank or season effect.",
        "",
        "## Guards",
        "",
        "- Early 10m training input contract: passed",
        "- Mid 20m training input contract: passed",
        "- Early 10m external input contract: passed with three declared all-missing source exceptions",
        "- Final mid 20m model loaded or scored: no",
        "- Frozen future holdout accessed or scored: no",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_monitoring_audit(
    training_input: str | Path,
    external_input: str | Path,
    external_source_manifest_path: str | Path,
    final_model_manifest_path: str | Path,
    robustness_summary_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    cohort_policy_path: str | Path | None = None,
    bins: int = DEFAULT_CALIBRATION_BINS,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    """Audit monitoring readiness without reading or scoring the future holdout."""
    training_path = Path(training_input)
    external_path = Path(external_input)
    external_manifest_path = Path(external_source_manifest_path)
    final_manifest_path = Path(final_model_manifest_path)
    robustness_path = Path(robustness_summary_path)
    contract_file = Path(contract_path)
    destination = Path(output_dir)

    if cohort_policy_path is not None:
        authorize_cohort_file_operation(
            cohort_policy_path, training_path, "monitoring_reference"
        )
        authorize_cohort_file_operation(
            cohort_policy_path, external_path, "monitoring_current"
        )

    contracts = json.loads(contract_file.read_text(encoding="utf-8"))
    if contracts != build_model_input_contracts():
        raise ValueError("Committed model-input contract differs from the code registry")
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(training_path) != final_manifest["training"]["file"]["sha256"]:
        raise ValueError("Training input checksum differs from the final-model manifest")
    if sha256_file(robustness_path) != final_manifest["selection_evidence"]["file"][
        "sha256"
    ]:
        raise ValueError("Robustness summary checksum differs from final selection evidence")
    external_manifest = json.loads(external_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(external_path) != external_manifest["file"]["sha256"]:
        raise ValueError("External input checksum differs from its source manifest")

    match_features = build_match_features(read_events(training_path))
    training_early = eligible_at_horizon(match_features, 10)
    training_mid = eligible_at_horizon(match_features, 20)
    early_features = features_available_at(TEN_MINUTES_MS)
    mid_features = features_available_at(TWENTY_MINUTES_MS)
    training_contract = validate_model_input_contract(
        training_early[["matchId", "win", *early_features]], horizon_minutes=10
    )
    training_mid_contract = validate_model_input_contract(
        training_mid[["matchId", "win", *mid_features]], horizon_minutes=20
    )
    external = adapt_high_diamond_10m(pd.read_csv(external_path))
    external_contract = validate_model_input_contract(
        external,
        horizon_minutes=10,
        allowed_all_missing=UNAVAILABLE_FEATURES,
    )

    _, bundle, artifact_path = load_final_early_bundle(final_manifest_path)
    training_probability = bundle["model"].predict_proba(training_early[early_features])[:, 1]
    external_probability = bundle["model"].predict_proba(external[early_features])[:, 1]
    external_calibration = calibration_metrics(
        external["win"], external_probability, bins=bins
    )
    external_interval = bootstrap_calibration_intervals(
        external["win"],
        external_probability,
        bins=bins,
        iterations=bootstrap_iterations,
    )

    development: dict[str, Any] = {}
    for horizon, key in ((10, "early_10m"), (20, "mid_20m")):
        predictions = _load_base_repeat_predictions(robustness_path, horizon)
        development[key] = {
            "prediction_source": "repeated_nested_base_repeat_out_of_fold",
            "rows": len(predictions),
            "metrics": calibration_metrics(
                predictions["win"], predictions["win_probability"], bins=bins
            ),
            "bootstrap_intervals": bootstrap_calibration_intervals(
                predictions["win"],
                predictions["win_probability"],
                bins=bins,
                iterations=bootstrap_iterations,
                random_state=RANDOM_STATE + horizon,
            ),
        }

    feature_drift = feature_drift_report(training_early, external, early_features)
    prediction_drift = feature_drift_report(
        pd.DataFrame({"win_probability": training_probability}),
        pd.DataFrame({"win_probability": external_probability}),
        ["win_probability"],
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "audit_design": "calibration_drift_and_runtime_contract_without_holdout_release",
        "inputs": {
            "training_sha256": sha256_file(training_path),
            "external_sha256": sha256_file(external_path),
            "final_early_model_sha256": sha256_file(artifact_path),
            "robustness_summary_sha256": sha256_file(robustness_path),
            "contract_sha256": sha256_file(contract_file),
        },
        "data_contracts": {
            "contract_file": contract_file.as_posix(),
            "training_early_10m": training_contract,
            "training_mid_20m": training_mid_contract,
            "external_early_10m": external_contract,
        },
        "calibration": {
            "policy": (
                "Diagnostic only. No post-hoc calibrator is fitted or deployed from "
                "external evaluation labels."
            ),
            "bins": bins,
            "external_early_10m": {
                "model": "frozen_final_early_10m",
                "metrics": external_calibration,
                "bootstrap_intervals": external_interval,
            },
            "development_base_repeat": development,
        },
        "drift": {
            "interpretation": (
                "Heuristic alerts for combined rank, era, and schema shift; not causal tests."
            ),
            "early_10m_external_vs_training": feature_drift,
            "early_10m_prediction_distribution": prediction_drift,
            "training_prediction_labels_used": False,
        },
        "evaluation_guards": {
            "model_refit": False,
            "post_hoc_calibrator_fitted": False,
            "mid_20m_final_model_loaded": False,
            "mid_20m_external_model_scored": False,
            "frozen_future_holdout_accessed": False,
            "frozen_future_holdout_scored": False,
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_manifest(destination / "monitoring_audit.json", result)
    _write_report(destination / "monitoring_audit.md", result)
    return result
