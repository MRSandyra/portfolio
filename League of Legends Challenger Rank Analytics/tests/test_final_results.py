import json
from pathlib import Path

import pytest

from lol_analytics.final_results import (
    _verify_artifact,
    audit_final_results,
    assert_post_release_tuning_not_allowed,
    authorize_pre_release_tuning,
    freeze_final_results,
)
from lol_analytics.quality import sha256_file


def test_artifact_verification_rejects_changed_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    record = {
        "path": artifact.as_posix(),
        "bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }
    assert _verify_artifact(record, "test artifact") == artifact

    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size differs|checksum differs"):
        _verify_artifact(record, "test artifact")


def test_existing_final_results_are_write_once_and_evidence_bound(
    tmp_path: Path,
) -> None:
    sources = [tmp_path / f"source-{index}.json" for index in range(5)]
    for index, source in enumerate(sources):
        source.write_text(str(index), encoding="utf-8")
    hashes = {
        "holdout_release_manifest_sha256": sha256_file(sources[0]),
        "holdout_release_started_manifest_sha256": sha256_file(sources[1]),
        "inference_manifest_sha256": sha256_file(sources[2]),
        "calibration_manifest_sha256": sha256_file(sources[3]),
        "final_model_manifest_sha256": sha256_file(sources[4]),
    }
    expected = {
        "status": "final_holdout_consumed",
        "evidence": {"source_manifests": hashes},
    }
    output = tmp_path / "final.json"
    output.write_text(json.dumps(expected), encoding="utf-8")

    assert freeze_final_results(*sources, output) == expected
    sources[4].write_text("changed", encoding="utf-8")
    with pytest.raises(FileExistsError, match="changed source evidence"):
        freeze_final_results(*sources, output)


@pytest.mark.parametrize(
    "operation",
    ("model_selection", "fit_base_model", "fit_calibrator", "feature_selection", "threshold_tuning"),
)
def test_consumed_holdout_rejects_every_declared_tuning_operation(
    operation: str,
) -> None:
    with pytest.raises(PermissionError, match="post-release"):
        assert_post_release_tuning_not_allowed(
            "data/manifests/cohort-policy-v1.json", operation
        )


def test_real_development_tuning_is_blocked_before_raw_file_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_checksum(*args: object, **kwargs: object) -> str:
        raise AssertionError("Consumed-state guard must run before raw checksum access")

    monkeypatch.setattr("lol_analytics.cohort_policy.sha256_file", forbidden_checksum)
    with pytest.raises(PermissionError, match="final_holdout_consumed"):
        authorize_pre_release_tuning(
            "data/manifests/cohort-policy-v1.json",
            "data/raw/events_americas_2000.parquet",
            "fit_base_model",
        )


def test_ci_checks_lock_and_sample_without_releasing_holdout() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8").lower()
    assert "uv lock --check" in workflow
    assert "lol-analytics reproduce" in workflow
    assert "lol-analytics audit-final-results" in workflow
    assert "data/sample/events.jsonl" in workflow
    assert "release-future-holdout" not in workflow
    assert "events_europe_future_500" not in workflow


def test_manifest_only_final_audit_does_not_require_ignored_artifacts() -> None:
    result = audit_final_results(
        "data/manifests/final-results-v1.json",
        "data/manifests/future-holdout-release-v1.json",
        "data/manifests/future-holdout-release-started-v1.json",
        "data/manifests/inference-bundles-v1.json",
        "data/manifests/calibration-v1.json",
        "data/manifests/final-models-2000.json",
    )
    assert result["passed"] is True
    assert result["artifact_verification_requested"] is False
    assert result["artifacts_verified"] == 0
    assert result["raw_holdout_accessed"] is False
    assert result["predictions_recomputed"] is False


def test_final_audit_rejects_changed_frozen_metric(tmp_path: Path) -> None:
    final = json.loads(
        Path("data/manifests/final-results-v1.json").read_text(encoding="utf-8")
    )
    final["models"]["early_10m"]["metrics"]["roc_auc"] = 0.901
    changed = tmp_path / "changed-final.json"
    changed.write_text(json.dumps(final), encoding="utf-8")

    with pytest.raises(ValueError, match="metrics differ"):
        audit_final_results(
            changed,
            "data/manifests/future-holdout-release-v1.json",
            "data/manifests/future-holdout-release-started-v1.json",
            "data/manifests/inference-bundles-v1.json",
            "data/manifests/calibration-v1.json",
            "data/manifests/final-models-2000.json",
        )


def test_committed_final_results_match_consumed_release() -> None:
    final = json.loads(
        Path("data/manifests/final-results-v1.json").read_text(encoding="utf-8")
    )
    release = json.loads(
        Path("data/manifests/future-holdout-release-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert final["status"] == "final_holdout_consumed"
    assert final["release_count"] == 1
    assert final["holdout"]["consumed"] is True
    assert final["holdout"]["available_for_future_selection_or_tuning"] is False
    assert final["models"]["early_10m"]["horizon_minutes"] == 10
    assert final["models"]["mid_20m"]["horizon_minutes"] == 20
    for key in ("early_10m", "mid_20m"):
        assert final["models"][key]["metrics"] == release["models"][key]["metrics"]
        assert final["models"][key]["calibration"]["calibrator_artifact"] is None
    assert all(value is False for key, value in final["immutability_policy"].items() if key != "generated_artifacts_verified_by_checksum")
    assert final["immutability_policy"]["generated_artifacts_verified_by_checksum"] is True
