import pandas as pd
import pytest

from lol_analytics.data import canonicalize_events
from lol_analytics.future_holdout import (
    build_future_holdout_manifest,
    freeze_future_holdout,
)


def _events(match_id: str, start_ms: int, region: str = "americas") -> pd.DataFrame:
    rows = []
    for event_id, (timestamp, event_type, gold) in enumerate(
        [(0, "START", 500), (600_000, "SNAPSHOT", 5_000), (1_300_000, "GAME_END", 12_000)]
    ):
        rows.append(
            {
                "matchId": match_id,
                "eventId": event_id,
                "gameStartTimestamp": start_ms,
                "timestamp": timestamp,
                "type": event_type,
                "winningTeam": 100 if event_type == "GAME_END" else None,
                "gameVersion": "15.9.1",
                "platformId": "NA1",
                "region": region,
                "totalGold_1": gold,
            }
        )
    return canonicalize_events(pd.DataFrame(rows))


def test_future_holdout_is_strictly_later_and_unscored() -> None:
    training = _events("train", 1_742_000_000_000)
    holdout = _events("future", 1_743_000_000_000)
    manifest = build_future_holdout_manifest(training, holdout, expected_matches=1)

    assert manifest["status"] == "frozen_unscored"
    assert manifest["separation"]["strictly_future"] is True
    assert manifest["separation"]["match_id_overlap"] == 0
    assert manifest["evaluation_policy"]["model_fitted"] is False
    assert manifest["evaluation_policy"]["predictions_written"] is False
    assert manifest["evaluation_policy"]["performance_metrics_computed"] is False
    assert "winner_distribution" not in manifest["holdout"]


def test_future_holdout_rejects_time_overlap() -> None:
    training = _events("train", 1_743_000_000_000)
    holdout = _events("not-future", 1_742_000_000_000)
    with pytest.raises(ValueError, match="strictly after"):
        build_future_holdout_manifest(training, holdout, expected_matches=1)


def test_future_holdout_rejects_match_or_region_overlap() -> None:
    training = _events("same", 1_742_000_000_000)
    with pytest.raises(ValueError, match="match IDs"):
        build_future_holdout_manifest(
            training,
            _events("same", 1_743_000_000_000),
            expected_matches=1,
        )
    with pytest.raises(ValueError, match="same non-empty region"):
        build_future_holdout_manifest(
            training,
            _events("future", 1_743_000_000_000, region="europe"),
            expected_matches=1,
        )


def test_future_external_region_requires_explicit_policy() -> None:
    manifest = build_future_holdout_manifest(
        _events("train", 1_742_000_000_000),
        _events("future", 1_743_000_000_000, region="europe"),
        expected_matches=1,
        allow_region_shift=True,
    )
    assert manifest["validation_design"] == "future_external_region"
    assert manifest["separation"]["same_regions"] is False
    assert manifest["separation"]["region_shift_explicitly_allowed"] is True
    assert "cannot be attributed separately" in manifest["evaluation_policy"][
        "interpretation_after_release"
    ]


def test_future_holdout_rejects_prior_evaluated_match_overlap() -> None:
    with pytest.raises(ValueError, match="previously evaluated cohort"):
        build_future_holdout_manifest(
            _events("train", 1_742_000_000_000),
            _events("future", 1_743_000_000_000, region="europe"),
            expected_matches=1,
            allow_region_shift=True,
            prior_evaluated_events={
                "explored": _events("future", 1_741_000_000_000, region="europe")
            },
        )


def test_frozen_manifest_is_bound_to_prior_cohort_checksum(tmp_path) -> None:
    training_path = tmp_path / "training.parquet"
    holdout_path = tmp_path / "holdout.parquet"
    prior_path = tmp_path / "prior.parquet"
    manifest_path = tmp_path / "manifest.json"
    _events("train", 1_742_000_000_000).to_parquet(training_path, index=False)
    _events("future", 1_743_000_000_000, region="europe").to_parquet(
        holdout_path, index=False
    )
    _events("prior", 1_741_000_000_000, region="europe").to_parquet(
        prior_path, index=False
    )

    first = freeze_future_holdout(
        training_path,
        holdout_path,
        manifest_path,
        expected_matches=1,
        allow_region_shift=True,
        prior_cohort_inputs=[prior_path],
    )
    second = freeze_future_holdout(
        training_path,
        holdout_path,
        manifest_path,
        expected_matches=1,
        allow_region_shift=True,
        prior_cohort_inputs=[prior_path],
    )
    assert first == second

    _events("changed-prior", 1_741_000_000_000, region="europe").to_parquet(
        prior_path, index=False
    )
    with pytest.raises(FileExistsError, match="different inputs"):
        freeze_future_holdout(
            training_path,
            holdout_path,
            manifest_path,
            expected_matches=1,
            allow_region_shift=True,
            prior_cohort_inputs=[prior_path],
        )
