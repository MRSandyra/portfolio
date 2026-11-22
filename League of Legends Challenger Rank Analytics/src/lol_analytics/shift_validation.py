"""Controlled same-region and cross-region shift validation."""

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
from .modeling import evaluate_model, fit_model
from .reporting import write_json, write_shift_validation_report
from .validation import bootstrap_auc_interval, independent_auc_delta_interval


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("unknown").value_counts().items()
    }


def _metadata(frame: pd.DataFrame) -> dict[str, Any]:
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


def _shared_support(named_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    patch_sets = [set(frame["patch"].astype(str)) for frame in named_frames.values()]
    shared_patches = sorted(set.intersection(*patch_sets))
    if not shared_patches:
        raise ValueError("Controlled shift validation requires shared patch support")
    starts = [int(frame["game_start_timestamp_ms"].min()) for frame in named_frames.values()]
    ends = [int(frame["game_start_timestamp_ms"].max()) for frame in named_frames.values()]
    overlap_start = max(starts)
    overlap_end = min(ends)
    if overlap_start > overlap_end:
        raise ValueError("Controlled shift validation requires overlapping time ranges")
    return {
        "shared_patches": shared_patches,
        "overlap_time_range_utc": {
            "start": pd.to_datetime(overlap_start, unit="ms", utc=True).isoformat(),
            "end": pd.to_datetime(overlap_end, unit="ms", utc=True).isoformat(),
        },
    }


def _assert_disjoint_ids(named_frames: dict[str, pd.DataFrame]) -> None:
    names = list(named_frames)
    for index, first_name in enumerate(names):
        first_ids = set(named_frames[first_name]["matchId"])
        for second_name in names[index + 1 :]:
            overlap = sorted(first_ids.intersection(named_frames[second_name]["matchId"]))
            if overlap:
                raise ValueError(
                    f"{first_name} and {second_name} must have disjoint matchId values; "
                    f"found {len(overlap)} overlap(s): {overlap[:5]}"
                )


def _predictions_with_context(result, frame: pd.DataFrame) -> pd.DataFrame:
    context = frame[["matchId", "region", "patch"]].drop_duplicates("matchId")
    return result.test_predictions.merge(
        context, on="matchId", how="left", validate="one_to_one"
    )


def evaluate_shift_features(
    training_features: pd.DataFrame,
    same_region_features: pd.DataFrame,
    cross_region_features: pd.DataFrame,
    output_dir: str | Path,
    *,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    """Fit once per horizon and evaluate same- and cross-region test cohorts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frames = {
        "training": training_features,
        "same_region": same_region_features,
        "cross_region": cross_region_features,
    }
    _assert_disjoint_ids(frames)
    support = _shared_support(frames)

    training_regions = set(training_features["region"].dropna().astype(str))
    same_regions = set(same_region_features["region"].dropna().astype(str))
    cross_regions = set(cross_region_features["region"].dropna().astype(str))
    if not same_regions or not same_regions.issubset(training_regions):
        raise ValueError("same_region cohort must use only training-region labels")
    if training_regions.intersection(cross_regions):
        raise ValueError("cross_region cohort must not use a training-region label")

    models: dict[str, Any] = {}
    for horizon_minutes in (10, 20):
        features = features_available_at(horizon_minutes * 60_000)
        training = eligible_at_horizon(training_features, horizon_minutes)
        same = eligible_at_horizon(same_region_features, horizon_minutes)
        cross = eligible_at_horizon(cross_region_features, horizon_minutes)
        audits = {
            name: assert_leakage_safe(frame, features, horizon_minutes * 60_000).to_dict()
            for name, frame in {
                "training": training,
                "same_region": same,
                "cross_region": cross,
            }.items()
        }
        model = fit_model(training, features)
        same_result = evaluate_model(model, same, features)
        cross_result = evaluate_model(model, cross, features)
        same_interval = bootstrap_auc_interval(
            same_result.test_predictions["win"].to_numpy(),
            same_result.test_predictions["win_probability"].to_numpy(),
            iterations=bootstrap_iterations,
        )
        cross_interval = bootstrap_auc_interval(
            cross_result.test_predictions["win"].to_numpy(),
            cross_result.test_predictions["win_probability"].to_numpy(),
            iterations=bootstrap_iterations,
        )
        model_name = f"shift_model_{horizon_minutes}m.joblib"
        same_name = f"same_region_predictions_{horizon_minutes}m.csv"
        cross_name = f"cross_region_predictions_{horizon_minutes}m.csv"
        joblib.dump(
            {
                "model": model,
                "features": features,
                "horizon_minutes": horizon_minutes,
                "fit_cohort": "training_only",
            },
            destination / model_name,
        )
        _predictions_with_context(same_result, same).to_csv(
            destination / same_name, index=False
        )
        _predictions_with_context(cross_result, cross).to_csv(
            destination / cross_name, index=False
        )
        model_key = "early_10m" if horizon_minutes == 10 else "mid_20m"
        models[model_key] = {
            "horizon_minutes": horizon_minutes,
            "features": features,
            "fit_count": 1,
            "training_matches": len(training),
            "same_region_matches": len(same),
            "cross_region_matches": len(cross),
            "leakage_audits": audits,
            "same_region_metrics": same_result.metrics,
            "same_region_roc_auc_interval": same_interval,
            "cross_region_metrics": cross_result.metrics,
            "cross_region_roc_auc_interval": cross_interval,
            "cross_minus_same_auc_interval": independent_auc_delta_interval(
                same_result.test_predictions["win"].to_numpy(),
                same_result.test_predictions["win_probability"].to_numpy(),
                cross_result.test_predictions["win"].to_numpy(),
                cross_result.test_predictions["win_probability"].to_numpy(),
                iterations=bootstrap_iterations,
            ),
            "model_path": model_name,
            "same_region_predictions_path": same_name,
            "cross_region_predictions_path": cross_name,
        }

    summary: dict[str, Any] = {
        "validation_design": "one_fit_per_horizon_same_vs_cross_region_test",
        "training": _metadata(training_features),
        "same_region": _metadata(same_region_features),
        "cross_region": _metadata(cross_region_features),
        "controlled_support": support,
        "match_id_overlap": 0,
        "models": models,
    }
    write_json(destination / "shift_validation.json", summary)
    write_shift_validation_report(destination / "shift_validation.md", summary)
    return summary


def run_shift_validation(
    training_input: str | Path,
    same_region_input: str | Path,
    cross_region_input: str | Path,
    output_dir: str | Path,
    *,
    cohort_policy_path: str | Path | None = None,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    if cohort_policy_path is not None:
        authorize_cohort_file_operation(
            cohort_policy_path, training_input, "fit_base_model"
        )
        authorize_cohort_file_operation(
            cohort_policy_path, same_region_input, "diagnostic_evaluation"
        )
        authorize_cohort_file_operation(
            cohort_policy_path, cross_region_input, "diagnostic_evaluation"
        )
    frames = [
        build_match_features(read_events(path))
        for path in (training_input, same_region_input, cross_region_input)
    ]
    summary = evaluate_shift_features(
        *frames,
        output_dir,
        bootstrap_iterations=bootstrap_iterations,
    )
    summary["inputs"] = {
        "training": Path(training_input).as_posix(),
        "same_region": Path(same_region_input).as_posix(),
        "cross_region": Path(cross_region_input).as_posix(),
    }
    destination = Path(output_dir)
    write_json(destination / "shift_validation.json", summary)
    write_shift_validation_report(destination / "shift_validation.md", summary)
    return summary
