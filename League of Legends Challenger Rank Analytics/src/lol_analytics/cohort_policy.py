"""Frozen cohort roles and operation authorization for the case study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .quality import sha256_file, write_manifest

POLICY_STATUS = "frozen_pre_calibration"

COHORT_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "key": "development_americas_2000",
        "manifest": "americas-2000.json",
        "role": "development",
        "purpose": "Feature engineering, temporal validation, model selection, final fitting, and OOF calibration evidence.",
        "identity_namespace": "riot_match_id",
        "horizons_minutes": [10, 20],
    },
    {
        "key": "diagnostic_americas_test_105",
        "manifest": "americas-test-105.json",
        "role": "diagnostic_external",
        "purpose": "Same-region diagnostic evaluation only.",
        "identity_namespace": "riot_match_id",
        "horizons_minutes": [10, 20],
    },
    {
        "key": "diagnostic_asia_105",
        "manifest": "asia-train-105.json",
        "role": "diagnostic_external",
        "purpose": "Same-period cross-region diagnostic evaluation only.",
        "identity_namespace": "riot_match_id",
        "horizons_minutes": [10, 20],
    },
    {
        "key": "diagnostic_europe_test_300",
        "manifest": "europe-test-300.json",
        "role": "diagnostic_external",
        "purpose": "Cross-region, time, and patch diagnostic evaluation only.",
        "identity_namespace": "riot_match_id",
        "horizons_minutes": [10, 20],
    },
    {
        "key": "diagnostic_diamond_historical_9879",
        "manifest": "high-diamond-10m.json",
        "role": "diagnostic_external",
        "purpose": "Cross-rank and historical-era stress test only.",
        "identity_namespace": "kaggle_game_id_prefixed_by_adapter",
        "horizons_minutes": [10],
    },
    {
        "key": "future_europe_500",
        "manifest": "europe-future-500.json",
        "role": "future_holdout",
        "purpose": "One-time final evaluation only after an explicit release phase.",
        "identity_namespace": "riot_match_id",
        "horizons_minutes": [10, 20],
    },
)

ROLE_OPERATIONS = {
    "development": [
        "integrity_check",
        "feature_engineering",
        "temporal_validation",
        "model_selection",
        "fit_base_model",
        "fit_calibrator",
        "monitoring_reference",
    ],
    "diagnostic_external": [
        "integrity_check",
        "diagnostic_evaluation",
        "monitoring_current",
    ],
    "future_holdout": ["integrity_check"],
}

KNOWN_OPERATIONS = frozenset(
    {
        operation
        for operations in ROLE_OPERATIONS.values()
        for operation in operations
    }
    | {"final_evaluation"}
)


def _event_manifest_record(manifest: dict[str, Any]) -> tuple[dict[str, Any], int, str]:
    if manifest.get("passed") is not True:
        raise ValueError("Event cohort manifest must have passed all quality gates")
    return manifest["file"], int(manifest["matches"]), "matches"


def _extract_record(definition: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    key = definition["key"]
    if key == "future_europe_500":
        if manifest.get("status") != "frozen_unscored":
            raise ValueError("Future holdout must remain frozen_unscored")
        evaluation = manifest.get("evaluation_policy", {})
        for field in ("model_fitted", "performance_metrics_computed", "predictions_written"):
            if evaluation.get(field) is not False:
                raise ValueError(f"Future holdout policy must keep {field}=false")
        file_record = manifest["holdout"]["file"]
        units = int(manifest["holdout"]["matches"])
        unit_name = "matches"
    elif key == "diagnostic_diamond_historical_9879":
        if manifest.get("quality", {}).get("schema_and_redundancy_checks_passed") is not True:
            raise ValueError("Diamond diagnostic schema audit did not pass")
        if manifest.get("quality", {}).get("unique_game_ids") is not True:
            raise ValueError("Diamond diagnostic game IDs are not unique")
        file_record = manifest["file"]
        units = int(file_record["rows"])
        unit_name = "rows"
    else:
        file_record, units, unit_name = _event_manifest_record(manifest)
    return {
        "role": definition["role"],
        "purpose": definition["purpose"],
        "identity_namespace": definition["identity_namespace"],
        "horizons_minutes": definition["horizons_minutes"],
        "allowed_operations": ROLE_OPERATIONS[definition["role"]],
        "selection_allowed": definition["role"] == "development",
        "base_model_fit_allowed": definition["role"] == "development",
        "calibrator_fit_allowed": definition["role"] == "development",
        "units": units,
        "unit_name": unit_name,
        "raw_file": {
            "path": file_record["path"],
            "bytes": int(file_record["bytes"]),
            "sha256": file_record["sha256"],
        },
    }


def _validate_future_separation(
    future_manifest: dict[str, Any], records: dict[str, Any]
) -> dict[str, Any]:
    separation = future_manifest.get("separation", {})
    if separation.get("strictly_future") is not True:
        raise ValueError("Future holdout is not strictly later than development")
    if int(separation.get("match_id_overlap", -1)) != 0:
        raise ValueError("Future holdout overlaps the development cohort")
    development_hash = records["development_americas_2000"]["raw_file"]["sha256"]
    if future_manifest["training_reference"]["file"]["sha256"] != development_hash:
        raise ValueError("Future holdout training reference does not match development")

    expected_prior = {
        records["diagnostic_americas_test_105"]["raw_file"]["sha256"],
        records["diagnostic_asia_105"]["raw_file"]["sha256"],
        records["diagnostic_europe_test_300"]["raw_file"]["sha256"],
    }
    observed_prior: set[str] = set()
    for prior in separation.get("prior_evaluated_cohorts", []):
        if int(prior.get("match_id_overlap", -1)) != 0:
            raise ValueError("Future holdout overlaps a prior diagnostic cohort")
        observed_prior.add(prior["file"]["sha256"])
    if observed_prior != expected_prior:
        raise ValueError("Future holdout prior-cohort evidence is incomplete or changed")
    return {
        "development_match_id_overlap": 0,
        "prior_riot_diagnostic_match_id_overlaps": {
            "diagnostic_americas_test_105": 0,
            "diagnostic_asia_105": 0,
            "diagnostic_europe_test_300": 0,
        },
        "diamond_identity_separation": (
            "Adapter prefixes Kaggle game IDs, placing them outside the Riot match-ID namespace."
        ),
        "strictly_future": True,
    }


def build_cohort_policy(
    manifest_dir: str | Path = Path("data/manifests"),
) -> dict[str, Any]:
    """Build a deterministic policy from committed cohort manifests only."""
    directory = Path(manifest_dir)
    records: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for definition in COHORT_DEFINITIONS:
        manifest_path = directory / definition["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        loaded[definition["key"]] = manifest
        record = _extract_record(definition, manifest)
        record["manifest"] = {
            "path": manifest_path.as_posix(),
            "sha256": sha256_file(manifest_path),
        }
        records[definition["key"]] = record

    separation = _validate_future_separation(
        loaded["future_europe_500"], records
    )
    raw_hashes = [record["raw_file"]["sha256"] for record in records.values()]
    if len(raw_hashes) != len(set(raw_hashes)):
        raise ValueError("Cohort policy contains duplicate raw-file checksums")
    return {
        "schema_version": 1,
        "status": POLICY_STATUS,
        "policy": {
            "only_development_may_fit_or_select": True,
            "diagnostic_cohorts_are_evaluation_only": True,
            "future_holdout_release_count": 0,
            "future_holdout_rows_parsed": False,
            "future_holdout_labels_read": False,
            "future_holdout_predictions_written": False,
            "future_holdout_metrics_computed": False,
            "future_holdout_allowed_before_release": ["integrity_check"],
            "future_holdout_release_requires": (
                "A separate explicit Phase 5 release command after calibration and release gates."
            ),
        },
        "cohorts": records,
        "identity_separation_evidence": separation,
    }


def freeze_cohort_policy(
    output_path: str | Path,
    *,
    manifest_dir: str | Path = Path("data/manifests"),
) -> dict[str, Any]:
    """Write once or verify an identical existing frozen cohort policy."""
    destination = Path(output_path)
    payload = build_cohort_policy(manifest_dir)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                "Refusing to replace the frozen cohort policy with changed roles or evidence"
            )
        return existing
    write_manifest(destination, payload)
    return payload


def audit_cohort_policy(
    policy_path: str | Path,
    *,
    verify_raw_files: bool = False,
) -> dict[str, Any]:
    """Verify policy evidence; raw verification hashes bytes without parsing rows."""
    path = Path(policy_path)
    policy = json.loads(path.read_text(encoding="utf-8"))
    manifest_directories = {
        Path(record["manifest"]["path"]).parent
        for record in policy.get("cohorts", {}).values()
    }
    if len(manifest_directories) != 1:
        raise ValueError("Cohort manifests must share one registered directory")
    manifest_directory = next(iter(manifest_directories))
    if policy != build_cohort_policy(manifest_directory):
        raise ValueError("Frozen cohort policy differs from current committed manifests")
    raw_verified: list[str] = []
    if verify_raw_files:
        for key, record in policy["cohorts"].items():
            raw_path = Path(record["raw_file"]["path"])
            if not raw_path.is_file():
                raise FileNotFoundError(f"Cohort raw file is missing: {raw_path}")
            if raw_path.stat().st_size != record["raw_file"]["bytes"]:
                raise ValueError(f"Cohort raw-file size changed: {key}")
            if sha256_file(raw_path) != record["raw_file"]["sha256"]:
                raise ValueError(f"Cohort raw-file checksum changed: {key}")
            raw_verified.append(key)
    return {
        "passed": True,
        "policy_path": path.as_posix(),
        "policy_sha256": sha256_file(path),
        "manifest_checksums_verified": len(policy["cohorts"]),
        "raw_file_checksums_verified": len(raw_verified),
        "raw_verified_cohorts": raw_verified,
        "future_holdout_integrity_mode": (
            "checksum_bytes_only_no_row_parsing" if verify_raw_files else "manifest_only"
        ),
        "future_holdout_rows_parsed": False,
        "future_holdout_labels_read": False,
        "future_holdout_scored": False,
    }


def assert_cohort_operation_allowed(
    policy: dict[str, Any], cohort_key: str, operation: str
) -> None:
    """Reject role misuse before any cohort file is opened."""
    if operation not in KNOWN_OPERATIONS:
        raise ValueError(f"Unknown cohort operation: {operation}")
    try:
        cohort = policy["cohorts"][cohort_key]
    except KeyError as error:
        raise ValueError(f"Unknown cohort key: {cohort_key}") from error
    if operation not in cohort["allowed_operations"]:
        raise PermissionError(
            f"Cohort {cohort_key} with role {cohort['role']} does not allow {operation}"
        )


def authorize_cohort_file_operation(
    policy_path: str | Path,
    cohort_file: str | Path,
    operation: str,
    *,
    verify_checksum: bool = True,
) -> str:
    """Authorize by role, then verify the registered file before returning its key."""
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    requested = Path(cohort_file).resolve()
    matches = [
        key
        for key, record in policy["cohorts"].items()
        if Path(record["raw_file"]["path"]).resolve() == requested
    ]
    if len(matches) != 1:
        raise ValueError("Input file is not uniquely registered in the cohort policy")
    cohort_key = matches[0]
    assert_cohort_operation_allowed(policy, cohort_key, operation)
    if verify_checksum:
        record = policy["cohorts"][cohort_key]["raw_file"]
        if not requested.is_file():
            raise FileNotFoundError(f"Registered cohort file is missing: {requested}")
        if requested.stat().st_size != record["bytes"]:
            raise ValueError(f"Registered cohort file size changed: {cohort_key}")
        if sha256_file(requested) != record["sha256"]:
            raise ValueError(f"Registered cohort file checksum changed: {cohort_key}")
    return cohort_key
