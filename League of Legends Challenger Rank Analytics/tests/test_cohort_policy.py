import json
import shutil
from pathlib import Path

import pytest

from lol_analytics.cohort_policy import (
    COHORT_DEFINITIONS,
    assert_cohort_operation_allowed,
    audit_cohort_policy,
    authorize_cohort_file_operation,
    build_cohort_policy,
    freeze_cohort_policy,
)
from lol_analytics.external_validation import run_external_validation
from lol_analytics.monitoring import run_monitoring_audit
from lol_analytics.nested_validation import run_nested_validation
from lol_analytics.rank_era_validation import validate_rank_era


POLICY_PATH = Path("data/manifests/cohort-policy-v1.json")


def test_committed_cohort_policy_matches_source_manifests() -> None:
    committed = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert committed == build_cohort_policy()

    audit = audit_cohort_policy(POLICY_PATH)
    assert audit["passed"] is True
    assert audit["manifest_checksums_verified"] == 6
    assert audit["raw_file_checksums_verified"] == 0
    assert audit["future_holdout_rows_parsed"] is False


def test_roles_authorize_only_declared_operations() -> None:
    policy = build_cohort_policy()

    assert_cohort_operation_allowed(
        policy, "development_americas_2000", "fit_base_model"
    )
    assert_cohort_operation_allowed(
        policy, "development_americas_2000", "fit_calibrator"
    )
    assert_cohort_operation_allowed(
        policy, "diagnostic_asia_105", "diagnostic_evaluation"
    )
    with pytest.raises(PermissionError, match="does not allow fit_base_model"):
        assert_cohort_operation_allowed(
            policy, "diagnostic_asia_105", "fit_base_model"
        )
    with pytest.raises(PermissionError, match="does not allow fit_calibrator"):
        assert_cohort_operation_allowed(
            policy, "diagnostic_diamond_historical_9879", "fit_calibrator"
        )
    with pytest.raises(PermissionError, match="does not allow model_selection"):
        assert_cohort_operation_allowed(
            policy, "diagnostic_europe_test_300", "model_selection"
        )


def test_future_holdout_allows_only_integrity_before_release() -> None:
    policy = build_cohort_policy()
    assert_cohort_operation_allowed(policy, "future_europe_500", "integrity_check")
    for forbidden in (
        "fit_base_model",
        "fit_calibrator",
        "model_selection",
        "diagnostic_evaluation",
        "final_evaluation",
    ):
        with pytest.raises(PermissionError, match=f"does not allow {forbidden}"):
            assert_cohort_operation_allowed(policy, "future_europe_500", forbidden)


def test_file_authorization_is_checksum_bound_to_registered_role() -> None:
    development = Path("data/raw/events_americas_2000.parquet")
    diagnostic = Path("data/raw/events_europe_test_300.parquet")

    assert (
        authorize_cohort_file_operation(
            POLICY_PATH, development, "fit_base_model", verify_checksum=False
        )
        == "development_americas_2000"
    )
    with pytest.raises(PermissionError, match="does not allow fit_base_model"):
        authorize_cohort_file_operation(POLICY_PATH, diagnostic, "fit_base_model")


def test_allowed_operation_rejects_changed_raw_checksum(tmp_path: Path) -> None:
    raw = tmp_path / "development.parquet"
    raw.write_bytes(b"changed-development-data")
    policy = build_cohort_policy()
    record = policy["cohorts"]["development_americas_2000"]["raw_file"]
    record["path"] = raw.as_posix()
    record["bytes"] = raw.stat().st_size
    record["sha256"] = "0" * 64
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum changed"):
        authorize_cohort_file_operation(policy_path, raw, "fit_base_model")


def test_frozen_policy_refuses_changed_manifest_evidence(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    for definition in COHORT_DEFINITIONS:
        shutil.copy2(
            Path("data/manifests") / definition["manifest"],
            manifest_dir / definition["manifest"],
        )
    policy_path = tmp_path / "policy.json"
    freeze_cohort_policy(policy_path, manifest_dir=manifest_dir)

    changed = manifest_dir / "americas-test-105.json"
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload["generated_at_utc"] = "changed-after-policy-freeze"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        freeze_cohort_policy(policy_path, manifest_dir=manifest_dir)


def test_identity_evidence_is_bound_and_zero_overlap() -> None:
    policy = build_cohort_policy()
    evidence = policy["identity_separation_evidence"]
    assert evidence["development_match_id_overlap"] == 0
    assert set(evidence["prior_riot_diagnostic_match_id_overlaps"].values()) == {0}
    assert evidence["strictly_future"] is True
    assert policy["policy"]["future_holdout_release_count"] == 0
    assert policy["policy"]["future_holdout_labels_read"] is False


def test_workflows_enforce_roles_before_reading_forbidden_cohorts(tmp_path: Path) -> None:
    diagnostic = Path("data/raw/events_asia_train_105.parquet")
    development = Path("data/raw/events_americas_2000.parquet")
    future = Path("data/raw/events_europe_future_500.parquet")

    with pytest.raises(PermissionError, match="does not allow model_selection"):
        run_nested_validation(
            diagnostic,
            tmp_path / "nested",
            cohort_policy_path=POLICY_PATH,
            bootstrap_iterations=100,
        )
    with pytest.raises(PermissionError, match="does not allow fit_base_model"):
        run_external_validation(
            diagnostic,
            development,
            tmp_path / "external",
            cohort_policy_path=POLICY_PATH,
            bootstrap_iterations=100,
        )
    with pytest.raises(PermissionError, match="does not allow diagnostic_evaluation"):
        validate_rank_era(
            future,
            tmp_path / "unused-source.json",
            tmp_path / "unused-model.json",
            tmp_path / "rank",
            cohort_policy_path=POLICY_PATH,
            bootstrap_iterations=100,
        )
    with pytest.raises(PermissionError, match="does not allow monitoring_reference"):
        run_monitoring_audit(
            future,
            development,
            tmp_path / "unused-source.json",
            tmp_path / "unused-model.json",
            tmp_path / "unused-robustness.json",
            tmp_path / "unused-contract.json",
            tmp_path / "monitoring",
            cohort_policy_path=POLICY_PATH,
            bootstrap_iterations=100,
        )
