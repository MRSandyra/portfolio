"""External-region validation without fitting on the external cohort."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import features_available_at
from .cohort_policy import authorize_cohort_file_operation
from .data import read_events
from .features import build_match_features, eligible_at_horizon
from .leakage import assert_leakage_safe
from .modeling import train_and_evaluate
from .reporting import write_external_validation_report, write_json
from .validation import bootstrap_auc_interval, paired_auc_delta_interval


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("unknown").value_counts().items()
    }


def _cohort_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    starts = pd.to_datetime(frame["game_start_timestamp_ms"], unit="ms", utc=True)
    return {
        "matches": len(frame),
        "regions": _distribution(frame, "region"),
        "patches": _distribution(frame, "patch"),
        "time_range_utc": {
            "start": starts.min().isoformat(),
            "end": starts.max().isoformat(),
        },
    }


def _external_predictions(result, external: pd.DataFrame) -> pd.DataFrame:
    context_columns = [
        column for column in ["matchId", "region", "patch"] if column in external
    ]
    context = external[context_columns].drop_duplicates("matchId")
    return result.test_predictions.merge(
        context,
        on="matchId",
        how="left",
        validate="one_to_one",
    )


def evaluate_external_features(
    training_features: pd.DataFrame,
    external_features: pd.DataFrame,
    output_dir: str | Path,
    *,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    """Fit on the training cohort and evaluate once on a disjoint external cohort."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    overlap = sorted(
        set(training_features["matchId"]).intersection(external_features["matchId"])
    )
    if overlap:
        raise ValueError(
            "Training and external cohorts must have disjoint matchId values; "
            f"found {len(overlap)} overlap(s): {overlap[:5]}"
        )

    training_regions = set(training_features["region"].dropna().astype(str))
    external_regions = set(external_features["region"].dropna().astype(str))
    region_overlap = sorted(training_regions.intersection(external_regions))
    if region_overlap:
        raise ValueError(
            "External-region validation requires disjoint region values; "
            f"found overlap: {region_overlap}"
        )

    models: dict[str, Any] = {}
    for horizon_minutes in (10, 20):
        training = eligible_at_horizon(training_features, horizon_minutes)
        external = eligible_at_horizon(external_features, horizon_minutes)
        features = features_available_at(horizon_minutes * 60_000)
        training_audit = assert_leakage_safe(
            training, features, horizon_minutes * 60_000
        )
        external_audit = assert_leakage_safe(
            external, features, horizon_minutes * 60_000
        )
        result = train_and_evaluate(training, external, features)
        interval = bootstrap_auc_interval(
            result.test_predictions["win"].to_numpy(),
            result.test_predictions["win_probability"].to_numpy(),
            iterations=bootstrap_iterations,
        )
        model_name = f"external_model_{horizon_minutes}m.joblib"
        predictions_name = f"external_predictions_{horizon_minutes}m.csv"
        joblib.dump(
            {
                "model": result.model,
                "features": features,
                "horizon_minutes": horizon_minutes,
                "fit_cohort": "training_only",
            },
            destination / model_name,
        )
        _external_predictions(result, external).to_csv(
            destination / predictions_name, index=False
        )
        models[f"early_{horizon_minutes}m" if horizon_minutes == 10 else "mid_20m"] = {
            "horizon_minutes": horizon_minutes,
            "training_matches": len(training),
            "external_matches": len(external),
            "training_excluded_before_horizon": len(training_features) - len(training),
            "external_excluded_before_horizon": len(external_features) - len(external),
            "features": features,
            "training_leakage_audit": training_audit.to_dict(),
            "external_leakage_audit": external_audit.to_dict(),
            "external_metrics": result.metrics,
            "external_roc_auc_interval": interval,
            "model_path": model_name,
            "predictions_path": predictions_name,
        }

    paired_training = eligible_at_horizon(training_features, 20)
    paired_external = eligible_at_horizon(external_features, 20)
    early = train_and_evaluate(
        paired_training,
        paired_external,
        features_available_at(10 * 60_000),
    )
    mid = train_and_evaluate(
        paired_training,
        paired_external,
        features_available_at(20 * 60_000),
    )
    truth = early.test_predictions["win"].to_numpy()
    paired = early.test_predictions[
        ["matchId", "game_start_timestamp_ms", "win"]
    ].copy()
    paired["early_10m_probability"] = early.test_predictions[
        "win_probability"
    ].to_numpy()
    paired["mid_20m_probability"] = mid.test_predictions["win_probability"].to_numpy()
    paired_name = "external_paired_predictions.csv"
    paired.to_csv(destination / paired_name, index=False)

    summary: dict[str, Any] = {
        "validation_design": "fit_training_region_evaluate_disjoint_external_region",
        "training": _cohort_metadata(training_features),
        "external": _cohort_metadata(external_features),
        "match_id_overlap": 0,
        "region_overlap": [],
        "models": models,
        "paired_comparison": {
            "cohort": "external_matches_eligible_at_20m",
            "training_matches": len(paired_training),
            "external_matches": len(paired_external),
            "early_10m_roc_auc": early.metrics["roc_auc"],
            "mid_20m_roc_auc": mid.metrics["roc_auc"],
            "delta_interval": paired_auc_delta_interval(
                truth,
                early.test_predictions["win_probability"].to_numpy(),
                mid.test_predictions["win_probability"].to_numpy(),
                iterations=bootstrap_iterations,
            ),
            "predictions_path": paired_name,
        },
    }
    write_json(destination / "external_validation.json", summary)
    write_external_validation_report(destination / "external_validation.md", summary)
    return summary


def run_external_validation(
    training_input: str | Path,
    external_input: str | Path,
    output_dir: str | Path,
    *,
    cohort_policy_path: str | Path | None = None,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    """Build features independently, then run external-region validation."""
    if cohort_policy_path is not None:
        authorize_cohort_file_operation(
            cohort_policy_path, training_input, "fit_base_model"
        )
        authorize_cohort_file_operation(
            cohort_policy_path, external_input, "diagnostic_evaluation"
        )
    training_events = read_events(training_input)
    external_events = read_events(external_input)
    training_features = build_match_features(training_events)
    external_features = build_match_features(external_events)
    summary = evaluate_external_features(
        training_features,
        external_features,
        output_dir,
        bootstrap_iterations=bootstrap_iterations,
    )
    summary["training_input"] = Path(training_input).as_posix()
    summary["external_input"] = Path(external_input).as_posix()
    destination = Path(output_dir)
    write_json(destination / "external_validation.json", summary)
    write_external_validation_report(destination / "external_validation.md", summary)
    return summary
