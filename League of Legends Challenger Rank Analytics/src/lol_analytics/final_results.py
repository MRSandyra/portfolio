"""Freeze the consumed holdout result without reading or scoring it again."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .quality import sha256_file, write_manifest

POST_RELEASE_TUNING_OPERATIONS = frozenset(
    {
        "model_selection",
        "fit_base_model",
        "fit_calibrator",
        "feature_selection",
        "threshold_tuning",
    }
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_artifact(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{label} size differs from its frozen record")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} checksum differs from its frozen record")
    return path


def _source_hashes(
    release_path: Path,
    started_path: Path,
    inference_path: Path,
    calibration_path: Path,
    final_model_path: Path,
) -> dict[str, str]:
    return {
        "holdout_release_manifest_sha256": sha256_file(release_path),
        "holdout_release_started_manifest_sha256": sha256_file(started_path),
        "inference_manifest_sha256": sha256_file(inference_path),
        "calibration_manifest_sha256": sha256_file(calibration_path),
        "final_model_manifest_sha256": sha256_file(final_model_path),
    }


def assert_post_release_tuning_not_allowed(
    cohort_policy_path: str | Path,
    operation: str,
) -> None:
    """Reject real-cohort tuning after the final holdout has been consumed."""
    if operation not in POST_RELEASE_TUNING_OPERATIONS:
        raise ValueError(f"Operation is not a declared tuning operation: {operation}")
    final_path = Path(cohort_policy_path).parent / "final-results-v1.json"
    if not final_path.exists():
        return
    final = _load_json(final_path)
    if final.get("status") != "final_holdout_consumed" or final.get(
        "release_count"
    ) != 1:
        raise ValueError("Final-results state is inconsistent or incomplete")
    if final.get("holdout", {}).get("consumed") is not True:
        raise ValueError("Final-results manifest does not mark the holdout consumed")
    policy = final.get("immutability_policy", {})
    expected_guard = {
        "model_selection": "post_release_feature_selection_allowed",
        "fit_base_model": "post_release_model_tuning_allowed",
        "fit_calibrator": "post_release_calibrator_tuning_allowed",
        "feature_selection": "post_release_feature_selection_allowed",
        "threshold_tuning": "post_release_threshold_tuning_allowed",
    }[operation]
    if policy.get(expected_guard) is not False:
        raise ValueError(f"Final-results tuning guard changed: {expected_guard}")
    raise PermissionError(
        f"Holdout is final_holdout_consumed; post-release {operation} is forbidden"
    )


def authorize_pre_release_tuning(
    cohort_policy_path: str | Path,
    cohort_file: str | Path,
    operation: str,
) -> str:
    """Check the cohort role and consumed state before reading a tuning cohort."""
    from .cohort_policy import authorize_cohort_file_operation

    cohort_key = authorize_cohort_file_operation(
        cohort_policy_path,
        cohort_file,
        operation,
        verify_checksum=False,
    )
    assert_post_release_tuning_not_allowed(cohort_policy_path, operation)
    authorize_cohort_file_operation(
        cohort_policy_path,
        cohort_file,
        operation,
        verify_checksum=True,
    )
    return cohort_key


def freeze_final_results(
    holdout_release_manifest_path: str | Path,
    holdout_release_started_manifest_path: str | Path,
    inference_manifest_path: str | Path,
    calibration_manifest_path: str | Path,
    final_model_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write one immutable final-result manifest from Phase 5 evidence."""
    release_path = Path(holdout_release_manifest_path)
    started_path = Path(holdout_release_started_manifest_path)
    inference_path = Path(inference_manifest_path)
    calibration_path = Path(calibration_manifest_path)
    final_model_path = Path(final_model_manifest_path)
    destination = Path(output_path)
    evidence_hashes = _source_hashes(
        release_path,
        started_path,
        inference_path,
        calibration_path,
        final_model_path,
    )

    if destination.exists():
        existing = _load_json(destination)
        if existing.get("status") != "final_holdout_consumed":
            raise FileExistsError("Existing final-results manifest has an unexpected status")
        if existing.get("evidence", {}).get("source_manifests") != evidence_hashes:
            raise FileExistsError(
                "Refusing to replace final results with changed source evidence"
            )
        return existing

    release = _load_json(release_path)
    started = _load_json(started_path)
    inference = _load_json(inference_path)
    calibration = _load_json(calibration_path)
    final_models = _load_json(final_model_path)

    if release.get("status") != "future_holdout_evaluated_once":
        raise ValueError("Phase 5 release is not complete")
    if release.get("release_count") != 1:
        raise ValueError("Phase 5 must record exactly one holdout release")
    if release.get("authorization") != started:
        raise ValueError("Release-started ledger differs from final authorization")
    if release["authorization"]["evidence"]["inference_manifest_sha256"] != sha256_file(
        inference_path
    ):
        raise ValueError("Release ledger belongs to a different inference manifest")
    if release["authorization"]["evidence"]["future_holdout_sha256"] != release[
        "cohort"
    ]["file"]["sha256"]:
        raise ValueError("Released holdout checksum differs across Phase 5 evidence")

    policy = release.get("release_policy", {})
    required_false = (
        "model_refit",
        "calibrator_refit",
        "feature_selection_after_release",
        "threshold_tuning_after_release",
        "post_release_tuning_allowed",
        "additional_holdout_evaluation_allowed",
    )
    for field in required_false:
        if policy.get(field) is not False:
            raise ValueError(f"Phase 5 immutability guard changed: {field}")

    if inference["evidence"]["calibration_manifest_sha256"] != sha256_file(
        calibration_path
    ):
        raise ValueError("Inference bundles belong to different calibration evidence")
    if inference["evidence"]["final_model_manifest_sha256"] != sha256_file(
        final_model_path
    ):
        raise ValueError("Inference bundles belong to different final models")

    models: dict[str, Any] = {}
    for horizon, key in ((10, "early_10m"), (20, "mid_20m")):
        released = release["models"][key]
        inference_record = inference["bundles"][key]
        final_record = final_models["models"][key]
        calibration_record = calibration["models"][key]
        if released.get("horizon_minutes") != horizon:
            raise ValueError(f"Released horizon changed for {key}")
        if released.get("model_refit") is not False:
            raise ValueError(f"Phase 5 refit the model for {key}")
        if released.get("calibrator_refit") is not False:
            raise ValueError(f"Phase 5 refit the calibrator for {key}")
        if released["probability_transform"]["decision"] != calibration_record[
            "decision"
        ]:
            raise ValueError(f"Calibration decision changed for {key}")

        _verify_artifact(final_record["artifact"], f"{key} base model")
        _verify_artifact(inference_record["artifact"], f"{key} inference bundle")
        _verify_artifact(released["prediction_artifact"], f"{key} predictions")
        calibrator_artifact = calibration_record.get("calibrator_artifact")
        if calibrator_artifact is not None:
            _verify_artifact(calibrator_artifact, f"{key} calibrator")

        models[key] = {
            "horizon_minutes": horizon,
            "eligible_holdout_matches": released["eligible_holdout_matches"],
            "base_model_artifact": final_record["artifact"],
            "inference_bundle_artifact": inference_record["artifact"],
            "calibration": {
                "decision": calibration_record["decision"],
                "calibrator_artifact": calibrator_artifact,
                "calibration_manifest_sha256": sha256_file(calibration_path),
            },
            "probability_transform": released["probability_transform"],
            "metrics": released["metrics"],
            "roc_auc_interval": released["roc_auc_interval"],
            "calibration_metrics": released["calibration_metrics"],
            "prediction_artifact": released["prediction_artifact"],
        }

    _verify_artifact(
        release["paired_comparison"]["prediction_artifact"],
        "paired predictions",
    )
    _verify_artifact(release["report_artifact"], "Phase 5 report")
    evaluated_at = datetime.fromtimestamp(
        release_path.stat().st_mtime, tz=UTC
    ).isoformat()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "final_holdout_consumed",
        "release_count": 1,
        "evaluated_at_utc": evaluated_at,
        "evaluation_time_source": "phase5_release_manifest_file_mtime_at_freeze",
        "holdout": {
            **release["cohort"],
            "consumed": True,
            "available_for_future_selection_or_tuning": False,
        },
        "models": models,
        "paired_comparison": release["paired_comparison"],
        "generated_report_artifact": release["report_artifact"],
        "evidence": {
            "source_manifests": evidence_hashes,
            "release_authorization": release["authorization"],
        },
        "immutability_policy": {
            "raw_holdout_read_in_phase6": False,
            "predictions_recomputed_in_phase6": False,
            "generated_artifacts_verified_by_checksum": True,
            "additional_holdout_evaluation_allowed": False,
            "post_release_model_tuning_allowed": False,
            "post_release_calibrator_tuning_allowed": False,
            "post_release_feature_selection_allowed": False,
            "post_release_threshold_tuning_allowed": False,
        },
        "interpretation": release["release_policy"]["interpretation"],
    }
    write_manifest(destination, payload)
    return payload


def audit_final_results(
    final_results_path: str | Path,
    holdout_release_manifest_path: str | Path,
    holdout_release_started_manifest_path: str | Path,
    inference_manifest_path: str | Path,
    calibration_manifest_path: str | Path,
    final_model_manifest_path: str | Path,
    *,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    """Audit committed final evidence, optionally hashing ignored local artifacts."""
    final_path = Path(final_results_path)
    release_path = Path(holdout_release_manifest_path)
    started_path = Path(holdout_release_started_manifest_path)
    inference_path = Path(inference_manifest_path)
    calibration_path = Path(calibration_manifest_path)
    final_model_path = Path(final_model_manifest_path)
    final = _load_json(final_path)
    release = _load_json(release_path)
    started = _load_json(started_path)
    inference = _load_json(inference_path)
    calibration = _load_json(calibration_path)
    final_models = _load_json(final_model_path)

    expected_sources = _source_hashes(
        release_path,
        started_path,
        inference_path,
        calibration_path,
        final_model_path,
    )
    if final.get("status") != "final_holdout_consumed":
        raise ValueError("Final result is not final_holdout_consumed")
    if final.get("release_count") != 1 or release.get("release_count") != 1:
        raise ValueError("Final evidence must record exactly one release")
    if final.get("evidence", {}).get("source_manifests") != expected_sources:
        raise ValueError("Final-results source hashes changed")
    if release.get("authorization") != started:
        raise ValueError("Phase 5 started and completed authorization differ")
    if final["holdout"]["file"] != release["cohort"]["file"]:
        raise ValueError("Final and Phase 5 holdout records differ")
    if final["holdout"].get("consumed") is not True:
        raise ValueError("Final holdout is not marked consumed")
    if final["holdout"].get("available_for_future_selection_or_tuning") is not False:
        raise ValueError("Consumed holdout remains available for tuning")

    verified_artifacts = 0
    for horizon, key in ((10, "early_10m"), (20, "mid_20m")):
        frozen = final["models"][key]
        released = release["models"][key]
        if frozen.get("horizon_minutes") != horizon:
            raise ValueError(f"Final horizon changed for {key}")
        if frozen["eligible_holdout_matches"] != released["eligible_holdout_matches"]:
            raise ValueError(f"Final eligible count differs from Phase 5 for {key}")
        if frozen["metrics"] != released["metrics"]:
            raise ValueError(f"Final metrics differ from Phase 5 for {key}")
        if frozen["roc_auc_interval"] != released["roc_auc_interval"]:
            raise ValueError(f"Final AUC interval differs from Phase 5 for {key}")
        if frozen["calibration_metrics"] != released["calibration_metrics"]:
            raise ValueError(f"Final calibration metrics differ from Phase 5 for {key}")
        if frozen["probability_transform"] != released["probability_transform"]:
            raise ValueError(f"Final probability transform differs from Phase 5 for {key}")
        if frozen["prediction_artifact"] != released["prediction_artifact"]:
            raise ValueError(f"Final prediction record differs from Phase 5 for {key}")
        if frozen["base_model_artifact"] != final_models["models"][key]["artifact"]:
            raise ValueError(f"Final base-model record changed for {key}")
        if frozen["inference_bundle_artifact"] != inference["bundles"][key][
            "artifact"
        ]:
            raise ValueError(f"Final inference-bundle record changed for {key}")
        if frozen["calibration"]["decision"] != calibration["models"][key][
            "decision"
        ]:
            raise ValueError(f"Final calibration decision changed for {key}")
        if frozen["calibration"]["calibrator_artifact"] != calibration["models"][
            key
        ].get("calibrator_artifact"):
            raise ValueError(f"Final calibrator artifact changed for {key}")
        if verify_artifacts:
            for label, record in (
                (f"{key} base model", frozen["base_model_artifact"]),
                (f"{key} inference bundle", frozen["inference_bundle_artifact"]),
                (f"{key} predictions", frozen["prediction_artifact"]),
            ):
                _verify_artifact(record, label)
                verified_artifacts += 1

    if final["paired_comparison"] != release["paired_comparison"]:
        raise ValueError("Final paired comparison differs from Phase 5")
    if final["generated_report_artifact"] != release["report_artifact"]:
        raise ValueError("Final report record differs from Phase 5")
    if verify_artifacts:
        _verify_artifact(
            final["paired_comparison"]["prediction_artifact"],
            "paired predictions",
        )
        _verify_artifact(final["generated_report_artifact"], "Phase 5 report")
        verified_artifacts += 2

    immutability = final.get("immutability_policy", {})
    if immutability.get("generated_artifacts_verified_by_checksum") is not True:
        raise ValueError("Final artifact-verification flag changed")
    for field in (
        "additional_holdout_evaluation_allowed",
        "post_release_model_tuning_allowed",
        "post_release_calibrator_tuning_allowed",
        "post_release_feature_selection_allowed",
        "post_release_threshold_tuning_allowed",
        "predictions_recomputed_in_phase6",
        "raw_holdout_read_in_phase6",
    ):
        if immutability.get(field) is not False:
            raise ValueError(f"Final immutability guard changed: {field}")
    return {
        "passed": True,
        "status": final["status"],
        "release_count": final["release_count"],
        "models_verified": 2,
        "source_manifest_hashes_verified": len(expected_sources),
        "artifact_verification_requested": verify_artifacts,
        "artifacts_verified": verified_artifacts,
        "raw_holdout_accessed": False,
        "predictions_recomputed": False,
    }
