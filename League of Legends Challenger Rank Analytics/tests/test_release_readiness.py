import json
from pathlib import Path

import pandas as pd
import pytest

from lol_analytics.release_readiness import (
    build_release_readiness,
    chronological_windows,
    ci_excludes_holdout_release,
)


POLICY_PATH = Path("data/manifests/cohort-policy-v1.json")


def _ordered_frame(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "matchId": [f"match-{index}" for index in range(rows)],
            "game_start_timestamp_ms": [1_700_000_000_000 + index for index in range(rows)],
        }
    )


def test_chronological_windows_cover_rows_once_without_time_overlap() -> None:
    frame = _ordered_frame()
    partitions = chronological_windows(frame.sample(frac=1, random_state=7), windows=5)

    assert len(partitions) == 5
    assert sum(len(partition) for partition in partitions) == len(frame)
    assert set(pd.concat(partitions)["matchId"]) == set(frame["matchId"])
    for previous, current in zip(partitions, partitions[1:], strict=False):
        assert previous["game_start_timestamp_ms"].max() < current[
            "game_start_timestamp_ms"
        ].min()


def test_chronological_windows_keep_timestamp_ties_together() -> None:
    frame = _ordered_frame()
    frame.loc[[4, 5], "game_start_timestamp_ms"] = frame.loc[
        4, "game_start_timestamp_ms"
    ]
    partitions = chronological_windows(frame, windows=5)

    tied_membership = [
        set(partition["matchId"]).intersection({"match-4", "match-5"})
        for partition in partitions
    ]
    assert {"match-4", "match-5"} in tied_membership


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="game_start_timestamp_ms"), "complete"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "unique matchId"),
    ],
)
def test_chronological_windows_reject_invalid_identity_or_time(
    mutation: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        chronological_windows(mutation(_ordered_frame()), windows=5)  # type: ignore[operator]


def test_ci_guard_rejects_holdout_release_reference(tmp_path: Path) -> None:
    clean = tmp_path / "clean.yml"
    clean.write_text("run: pytest\n", encoding="utf-8")
    assert ci_excludes_holdout_release(clean)["passed"] is True

    forbidden = tmp_path / "forbidden.yml"
    forbidden.write_text("run: release-future-holdout\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not release"):
        ci_excludes_holdout_release(forbidden)


def test_phase_four_rejects_diagnostic_development_before_other_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="does not allow monitoring_reference"):
        build_release_readiness(
            "data/raw/events_asia_train_105.parquet",
            "data/raw/events_europe_future_500.parquet",
            POLICY_PATH,
            tmp_path / "unused-holdout.json",
            tmp_path / "unused-final.json",
            tmp_path / "unused-calibration.json",
            tmp_path / "unused-inference.json",
            tmp_path / "unused-contract.json",
            tmp_path / "unused-robustness.json",
            tmp_path / "unused-ci.yml",
            tmp_path / "output",
            tmp_path / "manifest.json",
        )


def test_committed_phase_four_manifest_keeps_holdout_unread_and_unscored() -> None:
    manifest = json.loads(
        Path("data/manifests/release-readiness-v1.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "ready_for_one_time_holdout_release"
    assert all(manifest["gates"].values())
    assert manifest["future_holdout"]["integrity_verification"] == (
        "checksum_bytes_only"
    )
    for field in ("rows_parsed", "labels_read", "predictions_written", "metrics_computed"):
        assert manifest["future_holdout"][field] is False
    assert manifest["release_protocol"] == {
        "current_policy_allows_final_evaluation": False,
        "explicit_phase5_command_required": True,
        "post_release_model_or_calibrator_tuning_allowed": False,
        "release_must_be_one_time": True,
    }
