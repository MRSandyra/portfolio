import json
from pathlib import Path

import pandas as pd
import pytest

from lol_analytics.holdout_release import (
    RELEASE_MANIFEST_NAME,
    RELEASE_STARTED_NAME,
    _paired_comparison,
    release_future_holdout,
)
from lol_analytics.quality import sha256_file


def _release_call(tmp_path: Path, *, confirm: bool) -> dict[str, object]:
    return release_future_holdout(
        tmp_path / "future.parquet",
        tmp_path / "cohort-policy.json",
        tmp_path / "readiness.json",
        tmp_path / "future-manifest.json",
        tmp_path / "inference.json",
        tmp_path / "output",
        confirm_one_time_release=confirm,
        bootstrap_iterations=100,
    )


def test_release_requires_explicit_confirmation_before_file_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="confirm-one-time-release"):
        _release_call(tmp_path, confirm=False)
    assert not (tmp_path / RELEASE_STARTED_NAME).exists()


def test_completed_release_is_an_idempotent_read_only_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "cohort-policy.json": "cohort",
        "readiness.json": "readiness",
        "future-manifest.json": "future",
        "inference.json": "inference",
    }.items():
        (tmp_path / name).write_text(value, encoding="utf-8")
    expected = {
        "schema_version": 1,
        "status": "future_holdout_evaluated_once",
        "release_count": 1,
        "authorization": {
            "output_dir": (tmp_path / "output").as_posix(),
            "evidence": {
                "cohort_policy_sha256": sha256_file(tmp_path / "cohort-policy.json"),
                "release_readiness_sha256": sha256_file(tmp_path / "readiness.json"),
                "future_holdout_manifest_sha256": sha256_file(
                    tmp_path / "future-manifest.json"
                ),
                "inference_manifest_sha256": sha256_file(tmp_path / "inference.json"),
            },
        },
        "cohort": {"file": {"path": (tmp_path / "future.parquet").as_posix()}},
    }
    (tmp_path / RELEASE_MANIFEST_NAME).write_text(
        json.dumps(expected), encoding="utf-8"
    )

    def forbidden_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("A completed release must not read the holdout again")

    monkeypatch.setattr("lol_analytics.holdout_release.read_events", forbidden_read)
    assert _release_call(tmp_path, confirm=True) == expected


def test_incomplete_release_marker_blocks_second_attempt(tmp_path: Path) -> None:
    (tmp_path / RELEASE_STARTED_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing a second holdout read"):
        _release_call(tmp_path, confirm=True)


def test_paired_comparison_uses_identical_twenty_minute_rows(tmp_path: Path) -> None:
    early = pd.DataFrame(
        {
            "matchId": ["a", "b", "c", "d", "extra"],
            "win": [0, 0, 1, 1, 1],
            "served_probability": [0.1, 0.8, 0.2, 0.9, 0.6],
        }
    )
    mid = pd.DataFrame(
        {
            "matchId": ["a", "b", "c", "d"],
            "win": [0, 0, 1, 1],
            "served_probability": [0.05, 0.2, 0.8, 0.95],
        }
    )
    result = _paired_comparison(
        early, mid, tmp_path, bootstrap_iterations=100
    )

    assert result["matches"] == 4
    assert result["mid_20m_roc_auc"] == 1.0
    assert result["early_10m_roc_auc"] < result["mid_20m_roc_auc"]
    saved = pd.read_csv(tmp_path / "holdout_paired_predictions.csv")
    assert saved["matchId"].tolist() == mid["matchId"].tolist()


def test_committed_release_ledger_preserves_horizon_semantics() -> None:
    manifest = json.loads(
        Path("data/manifests/future-holdout-release-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "future_holdout_evaluated_once"
    assert manifest["release_count"] == 1
    assert manifest["models"]["early_10m"]["horizon_minutes"] == 10
    assert manifest["models"]["mid_20m"]["horizon_minutes"] == 20
    assert manifest["release_policy"] == {
        "additional_holdout_evaluation_allowed": False,
        "calibrator_refit": False,
        "feature_selection_after_release": False,
        "interpretation": (
            "Combined future-time, patch, and external-region evaluation; effects "
            "cannot be attributed separately."
        ),
        "model_refit": False,
        "post_release_tuning_allowed": False,
        "threshold_tuning_after_release": False,
    }
