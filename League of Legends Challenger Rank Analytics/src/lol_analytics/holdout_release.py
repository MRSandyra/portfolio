"""Explicit one-time evaluation of the frozen future holdout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .cohort_policy import authorize_cohort_file_operation
from .data import read_events
from .features import build_match_features, eligible_at_horizon
from .inference import load_inference_bundle, predict_inference_bundle
from .metrics import classification_metrics
from .monitoring import calibration_metrics
from .quality import sha256_file, write_manifest
from .validation import bootstrap_auc_interval, paired_auc_delta_interval

RELEASE_MANIFEST_NAME = "future-holdout-release-v1.json"
RELEASE_STARTED_NAME = "future-holdout-release-started-v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _return_existing_release(
    release_path: Path,
    holdout: Path,
    policy_path: Path,
    readiness_path: Path,
    holdout_manifest_path: Path,
    inference_path: Path,
    destination: Path,
) -> dict[str, Any]:
    """Verify immutable non-holdout evidence without reading the raw holdout again."""
    existing = _load_json(release_path)
    if existing.get("status") != "future_holdout_evaluated_once":
        raise FileExistsError("Existing release ledger has an unexpected status")
    if existing.get("release_count") != 1:
        raise ValueError("Existing release ledger does not record exactly one release")
    if Path(existing["cohort"]["file"]["path"]).resolve() != holdout.resolve():
        raise ValueError("Requested holdout path differs from the completed release")
    if existing["authorization"]["output_dir"] != destination.as_posix():
        raise ValueError("Requested output differs from the completed release")
    evidence = existing["authorization"]["evidence"]
    sources = (
        (policy_path, "cohort_policy_sha256"),
        (readiness_path, "release_readiness_sha256"),
        (holdout_manifest_path, "future_holdout_manifest_sha256"),
        (inference_path, "inference_manifest_sha256"),
    )
    for source, evidence_key in sources:
        if not source.is_file() or sha256_file(source) != evidence[evidence_key]:
            raise ValueError(f"Completed release evidence changed: {evidence_key}")
    return existing


def _require_pre_release_evidence(
    holdout: Path,
    policy_path: Path,
    readiness_path: Path,
    holdout_manifest_path: Path,
    inference_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy = _load_json(policy_path)
    readiness = _load_json(readiness_path)
    holdout_manifest = _load_json(holdout_manifest_path)
    inference_manifest = _load_json(inference_manifest_path)

    if readiness.get("status") != "ready_for_one_time_holdout_release":
        raise ValueError("Phase 4 readiness status does not authorize release")
    if not readiness.get("gates") or not all(readiness["gates"].values()):
        raise ValueError("Phase 4 readiness gates are incomplete")
    protocol = readiness.get("release_protocol", {})
    if protocol.get("explicit_phase5_command_required") is not True:
        raise ValueError("Readiness evidence does not require an explicit Phase 5 release")
    if protocol.get("release_must_be_one_time") is not True:
        raise ValueError("Readiness evidence does not enforce one-time release")
    if protocol.get("post_release_model_or_calibrator_tuning_allowed") is not False:
        raise ValueError("Readiness evidence permits forbidden post-release tuning")

    if readiness["evidence"]["cohort_policy_sha256"] != sha256_file(policy_path):
        raise ValueError("Cohort policy differs from readiness evidence")
    if readiness["evidence"]["future_holdout_manifest_sha256"] != sha256_file(
        holdout_manifest_path
    ):
        raise ValueError("Future holdout manifest differs from readiness evidence")
    if readiness["evidence"]["inference_manifest_sha256"] != sha256_file(
        inference_manifest_path
    ):
        raise ValueError("Inference manifest differs from readiness evidence")

    policy_state = policy.get("policy", {})
    if policy_state.get("future_holdout_release_count") != 0:
        raise ValueError("Pre-release cohort policy must have release count zero")
    future = policy["cohorts"].get("future_europe_500", {})
    if future.get("role") != "future_holdout":
        raise ValueError("Registered future cohort role changed")
    if future.get("allowed_operations") != ["integrity_check"]:
        raise ValueError("Pre-release future cohort permissions changed")
    if readiness["future_holdout"].get("cohort_key") != "future_europe_500":
        raise ValueError("Readiness evidence names a different future cohort")
    for field in ("rows_parsed", "labels_read", "predictions_written", "metrics_computed"):
        if readiness["future_holdout"].get(field) is not False:
            raise ValueError(f"Readiness evidence already records holdout {field}")

    if holdout_manifest.get("status") != "frozen_unscored":
        raise ValueError("Future holdout is not frozen_unscored before release")
    evaluation = holdout_manifest.get("evaluation_policy", {})
    for field in ("model_fitted", "predictions_written", "performance_metrics_computed"):
        if evaluation.get(field) is not False:
            raise ValueError(f"Future manifest already records {field}")
    registered = holdout_manifest["holdout"]["file"]
    if Path(registered["path"]).resolve() != holdout.resolve():
        raise ValueError("Holdout path differs from the frozen manifest")
    if registered["sha256"] != readiness["future_holdout"]["file_sha256"]:
        raise ValueError("Holdout checksum differs across frozen evidence")

    if inference_manifest.get("status") not in {
        "inference_bundles_base_probability_unreleased",
        "inference_bundles_calibrated_unreleased",
    }:
        raise ValueError("Inference bundles do not have an unreleased status")
    guards = inference_manifest.get("evaluation_guards", {})
    for field in ("model_refit", "calibrator_refit", "future_holdout_accessed", "future_holdout_scored"):
        if guards.get(field) is not False:
            raise ValueError(f"Inference evidence violates release guard: {field}")

    cohort_key = authorize_cohort_file_operation(
        policy_path, holdout, "integrity_check"
    )
    if cohort_key != "future_europe_500":
        raise ValueError("Integrity authorization returned a different future cohort")
    return policy, readiness, holdout_manifest, inference_manifest


def _evaluate_model(
    match_features: pd.DataFrame,
    inference_manifest: dict[str, Any],
    inference_manifest_path: Path,
    output_dir: Path,
    *,
    horizon: int,
    key: str,
    bootstrap_iterations: int,
    calibration_bins: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    eligible = eligible_at_horizon(match_features, horizon)
    record = inference_manifest["bundles"][key]
    if record.get("horizon_minutes") != horizon:
        raise ValueError(f"Inference horizon changed for {key}")
    artifact = Path(record["artifact"]["path"])
    if not artifact.is_absolute() and not artifact.exists():
        artifact = inference_manifest_path.parent.parent.parent / artifact
    bundle = load_inference_bundle(
        artifact, expected_sha256=record["artifact"]["sha256"]
    )
    model_input = eligible[["matchId", "win", *bundle["features"]]].copy()
    predictions = predict_inference_bundle(bundle, model_input)
    if predictions["matchId"].duplicated().any() or len(predictions) != len(eligible):
        raise AssertionError(f"Prediction identity coverage changed for {key}")
    truth = predictions["win"].to_numpy()
    probability = predictions["served_probability"].to_numpy()
    metrics = classification_metrics(truth, probability)
    auc_interval = bootstrap_auc_interval(
        truth, probability, iterations=bootstrap_iterations
    )
    calibration = calibration_metrics(
        truth, probability, bins=calibration_bins
    )
    prediction_path = output_dir / f"holdout_predictions_{horizon}m.csv"
    predictions.to_csv(prediction_path, index=False)
    return (
        {
            "horizon_minutes": horizon,
            "eligible_holdout_matches": len(eligible),
            "features": bundle["features"],
            "probability_transform": bundle["probability_transform"],
            "metrics": metrics,
            "roc_auc_interval": auc_interval,
            "calibration_metrics": calibration,
            "prediction_artifact": _artifact_record(prediction_path),
            "model_refit": False,
            "calibrator_refit": False,
        },
        predictions,
    )


def _paired_comparison(
    early: pd.DataFrame,
    mid: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    paired = mid[["matchId", "win", "served_probability"]].rename(
        columns={"served_probability": "mid_20m_probability"}
    )
    paired = paired.merge(
        early[["matchId", "win", "served_probability"]].rename(
            columns={
                "win": "early_win",
                "served_probability": "early_10m_probability",
            }
        ),
        on="matchId",
        how="left",
        validate="one_to_one",
    )
    if paired["early_10m_probability"].isna().any():
        raise ValueError("A 20-minute-eligible match lacks an early prediction")
    if not paired["win"].equals(paired["early_win"]):
        raise ValueError("Paired horizon labels differ")
    paired = paired.drop(columns="early_win")
    truth = paired["win"].to_numpy()
    early_probability = paired["early_10m_probability"].to_numpy()
    mid_probability = paired["mid_20m_probability"].to_numpy()
    early_metrics = classification_metrics(truth, early_probability)
    mid_metrics = classification_metrics(truth, mid_probability)
    interval = paired_auc_delta_interval(
        truth,
        early_probability,
        mid_probability,
        iterations=bootstrap_iterations,
    )
    path = output_dir / "holdout_paired_predictions.csv"
    paired.to_csv(path, index=False)
    return {
        "cohort": "future_holdout_matches_eligible_at_20m",
        "matches": len(paired),
        "early_10m_roc_auc": early_metrics["roc_auc"],
        "mid_20m_roc_auc": mid_metrics["roc_auc"],
        "auc_delta_interval": interval,
        "prediction_artifact": _artifact_record(path),
    }


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    early = manifest["models"]["early_10m"]
    mid = manifest["models"]["mid_20m"]
    paired = manifest["paired_comparison"]
    lines = [
        "# Phase 5 one-time future-holdout evaluation",
        "",
        "The frozen holdout was released exactly once. Models and probability transforms "
        "were loaded from checksum-bound Phase 3 bundles without refitting.",
        "",
        "| Model | Horizon | N | ROC-AUC | 95% bootstrap interval | Brier | Log loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, record in (("Early game", early), ("Mid game", mid)):
        interval = record["roc_auc_interval"]
        calibration = record["calibration_metrics"]
        lines.append(
            f"| {label} | {record['horizon_minutes']} min | "
            f"{record['eligible_holdout_matches']} | {record['metrics']['roc_auc']:.4f} | "
            f"[{interval['lower']:.4f}, {interval['upper']:.4f}] | "
            f"{calibration['brier_score']:.4f} | {calibration['log_loss']:.4f} |"
        )
    delta = paired["auc_delta_interval"]
    lines.extend(
        [
            "",
            f"On the same {paired['matches']} matches eligible at 20 minutes, early "
            f"ROC-AUC was {paired['early_10m_roc_auc']:.4f} and mid ROC-AUC was "
            f"{paired['mid_20m_roc_auc']:.4f}. Paired mid-minus-early delta was "
            f"{delta['estimate_mid_minus_early']:+.4f} "
            f"[{delta['lower']:+.4f}, {delta['upper']:+.4f}].",
            "",
            "The mid-game result is not an early-game prediction claim. The holdout changes "
            "time, patch, and region together, so those effects cannot be separated.",
            "No post-release model, feature, calibration, or threshold tuning is allowed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def release_future_holdout(
    future_holdout_input: str | Path,
    cohort_policy_path: str | Path,
    readiness_manifest_path: str | Path,
    future_holdout_manifest_path: str | Path,
    inference_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    confirm_one_time_release: bool,
    bootstrap_iterations: int = 2_000,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Score the frozen holdout once, then make subsequent calls read-only no-ops."""
    if not confirm_one_time_release:
        raise PermissionError("Phase 5 requires --confirm-one-time-release")
    if bootstrap_iterations < 100:
        raise ValueError("bootstrap_iterations must be at least 100")
    if calibration_bins < 2 or calibration_bins > 100:
        raise ValueError("calibration_bins must be between 2 and 100")

    holdout = Path(future_holdout_input)
    policy_path = Path(cohort_policy_path)
    readiness_path = Path(readiness_manifest_path)
    holdout_manifest_path = Path(future_holdout_manifest_path)
    inference_path = Path(inference_manifest_path)
    destination = Path(output_dir)
    release_path = policy_path.parent / RELEASE_MANIFEST_NAME
    started_path = policy_path.parent / RELEASE_STARTED_NAME

    if release_path.exists():
        return _return_existing_release(
            release_path,
            holdout,
            policy_path,
            readiness_path,
            holdout_manifest_path,
            inference_path,
            destination,
        )
    if started_path.exists():
        raise FileExistsError(
            "A Phase 5 release was already started but did not produce a final ledger; "
            "refusing a second holdout read"
        )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Release output directory must be absent or empty")

    policy, readiness, holdout_manifest, inference_manifest = (
        _require_pre_release_evidence(
            holdout,
            policy_path,
            readiness_path,
            holdout_manifest_path,
            inference_path,
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    authorization = {
        "schema_version": 1,
        "status": "future_holdout_release_started",
        "authorization": "explicit_phase5_one_time_override",
        "authorized_operation": "final_evaluation",
        "cohort_key": "future_europe_500",
        "pre_release_count": policy["policy"]["future_holdout_release_count"],
        "output_dir": destination.as_posix(),
        "evidence": {
            "cohort_policy_sha256": sha256_file(policy_path),
            "release_readiness_sha256": sha256_file(readiness_path),
            "future_holdout_manifest_sha256": sha256_file(holdout_manifest_path),
            "inference_manifest_sha256": sha256_file(inference_path),
            "future_holdout_sha256": holdout_manifest["holdout"]["file"]["sha256"],
        },
    }
    write_manifest(started_path, authorization)

    match_features = build_match_features(read_events(holdout))
    if len(match_features) != holdout_manifest["holdout"]["matches"]:
        raise ValueError("Parsed holdout match count differs from frozen evidence")

    models: dict[str, Any] = {}
    predictions: dict[str, pd.DataFrame] = {}
    for horizon, key in ((10, "early_10m"), (20, "mid_20m")):
        models[key], predictions[key] = _evaluate_model(
            match_features,
            inference_manifest,
            inference_path,
            destination,
            horizon=horizon,
            key=key,
            bootstrap_iterations=bootstrap_iterations,
            calibration_bins=calibration_bins,
        )
        expected = readiness["future_holdout"][f"eligible_{horizon}m"]
        if models[key]["eligible_holdout_matches"] != expected:
            raise ValueError(f"Eligible holdout count changed for {key}")

    paired = _paired_comparison(
        predictions["early_10m"],
        predictions["mid_20m"],
        destination,
        bootstrap_iterations=bootstrap_iterations,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "future_holdout_evaluated_once",
        "release_count": 1,
        "authorization": authorization,
        "cohort": {
            "key": "future_europe_500",
            "role": "future_holdout",
            "matches": len(match_features),
            "file": holdout_manifest["holdout"]["file"],
            "strictly_future": holdout_manifest["separation"]["strictly_future"],
            "development_match_id_overlap": holdout_manifest["separation"][
                "match_id_overlap"
            ],
        },
        "models": models,
        "paired_comparison": paired,
        "release_policy": {
            "model_refit": False,
            "calibrator_refit": False,
            "feature_selection_after_release": False,
            "threshold_tuning_after_release": False,
            "post_release_tuning_allowed": False,
            "additional_holdout_evaluation_allowed": False,
            "interpretation": (
                "Combined future-time, patch, and external-region evaluation; effects "
                "cannot be attributed separately."
            ),
        },
    }
    report_path = destination / "future_holdout_report.md"
    _write_report(report_path, manifest)
    manifest["report_artifact"] = _artifact_record(report_path)
    write_manifest(release_path, manifest)
    return manifest
