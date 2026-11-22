"""Rolling development backtest and one-time holdout release gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .cohort_policy import (
    audit_cohort_policy,
    authorize_cohort_file_operation,
)
from .config import features_available_at
from .data import read_events
from .features import build_match_features, eligible_at_horizon
from .inference import load_inference_bundle, predict_inference_bundle
from .monitoring import (
    build_model_input_contracts,
    calibration_metrics,
    feature_drift_report,
)
from .quality import sha256_file, write_manifest

DEFAULT_ROLLING_WINDOWS = 5
FORBIDDEN_CI_RELEASE_TOKENS = (
    "events_europe_future_500",
    "europe-future-500",
    "release-future-holdout",
    "release-holdout",
)


def chronological_windows(
    frame: pd.DataFrame,
    *,
    windows: int = DEFAULT_ROLLING_WINDOWS,
    order_column: str = "game_start_timestamp_ms",
) -> list[pd.DataFrame]:
    """Split unique absolute timestamps into contiguous non-overlapping windows."""
    if windows < 3:
        raise ValueError("windows must be at least 3")
    if order_column not in frame or frame[order_column].isna().any():
        raise ValueError(f"A complete {order_column} column is required")
    if "matchId" not in frame or frame["matchId"].duplicated().any():
        raise ValueError("Rolling input requires unique matchId values")
    ordered = frame.sort_values([order_column, "matchId"], kind="stable").reset_index(
        drop=True
    )
    unique_times = ordered[order_column].drop_duplicates().sort_values().tolist()
    if len(unique_times) < windows:
        raise ValueError("Not enough unique timestamps for rolling windows")
    base_size, remainder = divmod(len(unique_times), windows)
    sizes = [base_size + int(index < remainder) for index in range(windows)]
    partitions: list[pd.DataFrame] = []
    start = 0
    for size in sizes:
        selected_times = set(unique_times[start : start + size])
        partition = ordered.loc[ordered[order_column].isin(selected_times)].copy()
        partitions.append(partition.reset_index(drop=True))
        start += size
    if sum(len(partition) for partition in partitions) != len(ordered):
        raise AssertionError("Rolling windows do not cover every row exactly once")
    for previous, current in zip(partitions, partitions[1:], strict=False):
        if previous[order_column].max() >= current[order_column].min():
            raise AssertionError("Rolling windows overlap or are not chronological")
    return partitions


def ci_excludes_holdout_release(ci_path: str | Path) -> dict[str, Any]:
    path = Path(ci_path)
    text = path.read_text(encoding="utf-8").lower()
    found = [token for token in FORBIDDEN_CI_RELEASE_TOKENS if token in text]
    if found:
        raise ValueError(f"CI must not release or reference the future holdout: {found}")
    return {
        "passed": True,
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "forbidden_tokens_found": [],
    }


def _window_metadata(frame: pd.DataFrame, index: int) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["game_start_timestamp_ms"], unit="ms", utc=True)
    return {
        "window": index,
        "rows": len(frame),
        "start_utc": timestamps.min().isoformat(),
        "end_utc": timestamps.max().isoformat(),
    }


def _load_base_oof_predictions(
    robustness: dict[str, Any], robustness_path: Path, key: str
) -> pd.DataFrame:
    prediction_path = robustness_path.parent / robustness["models"][key][
        "predictions_path"
    ]
    predictions = pd.read_csv(prediction_path)
    required = {
        "matchId",
        "game_start_timestamp_ms",
        "win",
        "win_probability",
        "outer_fold",
        "boundary_offset",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Rolling OOF predictions are missing columns: {missing}")
    base = predictions.loc[predictions["boundary_offset"].eq(0)].copy()
    if base.empty or base["matchId"].duplicated().any():
        raise ValueError("Base-repeat OOF predictions must be non-empty and unique")
    return base


def run_rolling_backtest(
    match_features: pd.DataFrame,
    inference_manifest: dict[str, Any],
    robustness: dict[str, Any],
    robustness_path: Path,
    *,
    windows: int = DEFAULT_ROLLING_WINDOWS,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Backtest feature/output drift and OOF calibration without holdout data."""
    models: dict[str, Any] = {}
    for horizon, key in ((10, "early_10m"), (20, "mid_20m")):
        cohort = eligible_at_horizon(match_features, horizon)
        features = features_available_at(horizon * 60_000)
        partitions = chronological_windows(cohort, windows=windows)
        artifact = inference_manifest["bundles"][key]["artifact"]
        bundle = load_inference_bundle(
            artifact["path"], expected_sha256=artifact["sha256"]
        )
        prediction_windows: list[pd.DataFrame] = []
        for partition in partitions:
            model_input = partition[["matchId", *features]].copy()
            prediction_windows.append(
                predict_inference_bundle(bundle, model_input)
            )

        reference_features = partitions[0]
        reference_probability = prediction_windows[0][["served_probability"]]
        comparisons: list[dict[str, Any]] = []
        for index in range(1, len(partitions)):
            feature_drift = feature_drift_report(
                reference_features, partitions[index], features
            )
            prediction_drift = feature_drift_report(
                reference_probability,
                prediction_windows[index][["served_probability"]],
                ["served_probability"],
            )
            comparisons.append(
                {
                    "current_window": _window_metadata(partitions[index], index + 1),
                    "feature_drift": feature_drift,
                    "prediction_drift": prediction_drift,
                }
            )

        oof = _load_base_oof_predictions(robustness, robustness_path, key)
        oof_metrics: list[dict[str, Any]] = []
        for fold, fold_predictions in oof.groupby("outer_fold", sort=True):
            if set(fold_predictions["win"].unique()) != {0, 1}:
                raise ValueError(f"OOF calibration fold lacks both classes: {key} fold {fold}")
            oof_metrics.append(
                {
                    "outer_fold": int(fold),
                    "rows": len(fold_predictions),
                    "start_utc": pd.to_datetime(
                        fold_predictions["game_start_timestamp_ms"].min(),
                        unit="ms",
                        utc=True,
                    ).isoformat(),
                    "end_utc": pd.to_datetime(
                        fold_predictions["game_start_timestamp_ms"].max(),
                        unit="ms",
                        utc=True,
                    ).isoformat(),
                    "metrics": calibration_metrics(
                        fold_predictions["win"],
                        fold_predictions["win_probability"],
                        bins=calibration_bins,
                    ),
                }
            )
        models[key] = {
            "horizon_minutes": horizon,
            "eligible_development_matches": len(cohort),
            "rolling_windows": windows,
            "feature_reference_window": _window_metadata(partitions[0], 1),
            "comparisons": comparisons,
            "prediction_drift_labels_used": False,
            "prediction_performance_computed_on_final_model": False,
            "calibration_source": "base_repeat_temporal_out_of_fold_predictions",
            "oof_calibration_by_outer_fold": oof_metrics,
        }
    return {
        "schema_version": 1,
        "design": "development_rolling_drift_and_oof_calibration_backtest",
        "future_holdout_rows_parsed": False,
        "future_holdout_labels_read": False,
        "models": models,
    }


def _validate_holdout_manifest(holdout: dict[str, Any]) -> dict[str, bool]:
    evaluation = holdout.get("evaluation_policy", {})
    separation = holdout.get("separation", {})
    return {
        "holdout_status_frozen_unscored": holdout.get("status") == "frozen_unscored",
        "holdout_model_not_fitted": evaluation.get("model_fitted") is False,
        "holdout_predictions_not_written": evaluation.get("predictions_written") is False,
        "holdout_metrics_not_computed": evaluation.get("performance_metrics_computed")
        is False,
        "holdout_development_overlap_zero": int(separation.get("match_id_overlap", -1))
        == 0,
        "holdout_strictly_future": separation.get("strictly_future") is True,
        "holdout_prior_diagnostic_overlap_zero": all(
            int(record.get("match_id_overlap", -1)) == 0
            for record in separation.get("prior_evaluated_cohorts", [])
        ),
    }


def _write_report(path: Path, readiness: dict[str, Any], backtest: dict[str, Any]) -> None:
    lines = [
        "# Phase 4 release-readiness report",
        "",
        f"- Status: `{readiness['status']}`",
        f"- Passed gates: {sum(readiness['gates'].values())}/{len(readiness['gates'])}",
        "- Future holdout row parsing: no",
        "- Future holdout label access: no",
        "- Future holdout scoring: no",
        "- Explicit Phase 5 release command required: yes",
        "",
        "Rolling drift alerts are diagnostics and do not automatically block release. The "
        "gate verifies evidence integrity and protocol isolation; Phase 5 will measure the "
        "actual combined future-time, patch, and region shift once.",
        "",
        "| Model | Development matches | Rolling windows | Feature drift L/M/H | Output drift L/M/H | OOF calibration folds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("early_10m", "mid_20m"):
        model = backtest["models"][key]
        feature_counts = {"low": 0, "medium": 0, "high": 0}
        output_counts = {"low": 0, "medium": 0, "high": 0}
        for comparison in model["comparisons"]:
            for severity in feature_counts:
                feature_counts[severity] += comparison["feature_drift"][
                    "severity_counts"
                ][severity]
                output_counts[severity] += comparison["prediction_drift"][
                    "severity_counts"
                ][severity]
        lines.append(
            f"| {key} | {model['eligible_development_matches']} | "
            f"{model['rolling_windows']} | "
            f"{feature_counts['low']}/{feature_counts['medium']}/{feature_counts['high']} | "
            f"{output_counts['low']}/{output_counts['medium']}/{output_counts['high']} | "
            f"{len(model['oof_calibration_by_outer_fold'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_release_readiness(
    development_input: str | Path,
    future_holdout_input: str | Path,
    cohort_policy_path: str | Path,
    future_holdout_manifest_path: str | Path,
    final_model_manifest_path: str | Path,
    calibration_manifest_path: str | Path,
    inference_manifest_path: str | Path,
    contract_path: str | Path,
    robustness_summary_path: str | Path,
    ci_path: str | Path,
    output_dir: str | Path,
    readiness_manifest_path: str | Path,
    *,
    windows: int = DEFAULT_ROLLING_WINDOWS,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Build a release gate without parsing or scoring the future holdout."""
    development = Path(development_input)
    future = Path(future_holdout_input)
    policy_path = Path(cohort_policy_path)
    holdout_manifest_path = Path(future_holdout_manifest_path)
    final_path = Path(final_model_manifest_path)
    calibration_path = Path(calibration_manifest_path)
    inference_path = Path(inference_manifest_path)
    contract_file = Path(contract_path)
    robustness_path = Path(robustness_summary_path)
    destination = Path(output_dir)
    readiness_path = Path(readiness_manifest_path)

    development_key = authorize_cohort_file_operation(
        policy_path, development, "monitoring_reference"
    )
    future_key = authorize_cohort_file_operation(
        policy_path, future, "integrity_check"
    )
    if development_key != "development_americas_2000" or future_key != "future_europe_500":
        raise ValueError("Release-readiness cohorts do not have the expected frozen roles")
    policy_audit = audit_cohort_policy(policy_path, verify_raw_files=False)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_manifest_path.read_text(encoding="utf-8"))
    final_manifest = json.loads(final_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    contracts = json.loads(contract_file.read_text(encoding="utf-8"))
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    ci_audit = ci_excludes_holdout_release(ci_path)

    if contracts != build_model_input_contracts():
        raise ValueError("Runtime contracts differ from the feature registry")
    if policy["cohorts"][future_key]["manifest"]["sha256"] != sha256_file(
        holdout_manifest_path
    ):
        raise ValueError("Future holdout manifest differs from the cohort policy")
    if policy["cohorts"][future_key]["raw_file"] != holdout["holdout"]["file"]:
        raise ValueError("Future holdout file evidence differs across manifests")
    if robustness.get("frozen_future_holdout_accessed") is not False:
        raise ValueError("Robustness evidence accessed the future holdout")
    if final_manifest["selection_evidence"]["file"]["sha256"] != sha256_file(
        robustness_path
    ):
        raise ValueError("Robustness summary differs from final-model evidence")
    if inference["evidence"]["cohort_policy_sha256"] != sha256_file(policy_path):
        raise ValueError("Inference bundles belong to a different cohort policy")
    if inference["evidence"]["calibration_manifest_sha256"] != sha256_file(
        calibration_path
    ):
        raise ValueError("Inference bundles belong to different calibration evidence")
    if inference["evidence"]["final_model_manifest_sha256"] != sha256_file(
        final_path
    ):
        raise ValueError("Inference bundles belong to different final models")
    if inference["evidence"]["runtime_contract_sha256"] != sha256_file(contract_file):
        raise ValueError("Inference bundles belong to different runtime contracts")
    if calibration["evidence"]["cohort_policy_sha256"] != sha256_file(policy_path):
        raise ValueError("Calibration evidence belongs to a different cohort policy")
    if calibration["evidence"]["final_model_manifest_sha256"] != sha256_file(
        final_path
    ):
        raise ValueError("Calibration evidence belongs to different final models")
    if calibration["evidence"]["robustness_summary_sha256"] != sha256_file(
        robustness_path
    ):
        raise ValueError("Calibration evidence belongs to a different robustness run")
    if holdout["holdout"]["file"]["sha256"] != sha256_file(future):
        raise ValueError("Future holdout checksum differs from its frozen manifest")

    match_features = build_match_features(read_events(development))
    backtest = run_rolling_backtest(
        match_features,
        inference,
        robustness,
        robustness_path,
        windows=windows,
        calibration_bins=calibration_bins,
    )
    destination.mkdir(parents=True, exist_ok=True)
    backtest_path = destination / "rolling_drift_backtest.json"
    write_manifest(backtest_path, backtest)

    gates: dict[str, bool] = {
        "cohort_policy_integrity": policy_audit["passed"] is True,
        "development_role_frozen": development_key == "development_americas_2000",
        "future_role_frozen": future_key == "future_europe_500",
        "future_release_count_zero": policy["policy"]["future_holdout_release_count"]
        == 0,
        "future_final_evaluation_still_blocked": "final_evaluation"
        not in policy["cohorts"][future_key]["allowed_operations"],
        "runtime_contracts_match_registry": contracts == build_model_input_contracts(),
        "final_models_unreleased": final_manifest.get("status")
        == "final_models_fitted_unreleased",
        "final_models_never_scored_future": final_manifest["evaluation_policy"][
            "frozen_future_holdout_scored"
        ]
        is False,
        "calibration_decision_complete": calibration.get("status")
        in {
            "calibration_rejected_keep_base_probabilities",
            "calibrators_fitted_unreleased",
        },
        "calibration_never_used_external_labels": calibration["method_policy"][
            "external_diagnostic_labels_used"
        ]
        is False,
        "calibration_never_accessed_future": calibration["method_policy"][
            "future_holdout_accessed"
        ]
        is False,
        "inference_bundles_unreleased": inference.get("status")
        in {
            "inference_bundles_base_probability_unreleased",
            "inference_bundles_calibrated_unreleased",
        },
        "inference_never_accessed_future": inference["evaluation_guards"][
            "future_holdout_accessed"
        ]
        is False,
        "rolling_backtest_complete": set(backtest["models"])
        == {"early_10m", "mid_20m"},
        "ci_excludes_holdout_release": ci_audit["passed"] is True,
    }
    gates.update(_validate_holdout_manifest(holdout))
    if not all(gates.values()):
        failed = sorted(name for name, passed in gates.items() if not passed)
        raise ValueError(f"Release-readiness gates failed: {failed}")

    readiness: dict[str, Any] = {
        "schema_version": 1,
        "status": "ready_for_one_time_holdout_release",
        "gates": gates,
        "evidence": {
            "cohort_policy_sha256": sha256_file(policy_path),
            "future_holdout_manifest_sha256": sha256_file(holdout_manifest_path),
            "final_model_manifest_sha256": sha256_file(final_path),
            "calibration_manifest_sha256": sha256_file(calibration_path),
            "inference_manifest_sha256": sha256_file(inference_path),
            "runtime_contract_sha256": sha256_file(contract_file),
            "robustness_summary_sha256": sha256_file(robustness_path),
            "ci_sha256": ci_audit["sha256"],
            "rolling_backtest": {
                "path": backtest_path.as_posix(),
                "bytes": backtest_path.stat().st_size,
                "sha256": sha256_file(backtest_path),
            },
        },
        "future_holdout": {
            "cohort_key": future_key,
            "file_path": holdout["holdout"]["file"]["path"],
            "file_sha256": holdout["holdout"]["file"]["sha256"],
            "matches": holdout["holdout"]["matches"],
            "eligible_10m": holdout["holdout"]["horizon_coverage"]["10m"][
                "eligible_matches"
            ],
            "eligible_20m": holdout["holdout"]["horizon_coverage"]["20m"][
                "eligible_matches"
            ],
            "integrity_verification": "checksum_bytes_only",
            "rows_parsed": False,
            "labels_read": False,
            "predictions_written": False,
            "metrics_computed": False,
        },
        "release_protocol": {
            "explicit_phase5_command_required": True,
            "current_policy_allows_final_evaluation": False,
            "release_must_be_one_time": True,
            "post_release_model_or_calibrator_tuning_allowed": False,
        },
    }
    if readiness_path.exists():
        existing = json.loads(readiness_path.read_text(encoding="utf-8"))
        if existing != readiness:
            raise FileExistsError(
                "Refusing to replace release-readiness evidence with changed inputs"
            )
    else:
        write_manifest(readiness_path, readiness)
    _write_report(destination / "release_readiness_report.md", readiness, backtest)
    return readiness
