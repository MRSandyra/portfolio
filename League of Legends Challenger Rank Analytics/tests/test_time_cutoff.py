import pandas as pd

from lol_analytics.features import (
    build_match_features,
    eligible_at_horizon,
    snapshot_at_or_before,
)


def _base_event(timestamp: int, event_type: str, **extra: object) -> dict[str, object]:
    return {"matchId": "cutoff-match", "match_order": 0, "gameStartTimestamp": 1_742_000_000_000, "timestamp": timestamp, "type": event_type, "winningTeam": 100, "gameVersion": "15.7.1", "region": "test", **extra}


def test_snapshot_uses_last_row_at_or_before_cutoff() -> None:
    frame = pd.DataFrame([
        _base_event(599_999, "STATE_SNAPSHOT", totalGold_1=100),
        _base_event(600_000, "STATE_SNAPSHOT", totalGold_1=200),
        _base_event(600_001, "STATE_SNAPSHOT", totalGold_1=999),
    ])
    assert snapshot_at_or_before(frame, 600_000)["totalGold_1"] == 200


def test_future_events_do_not_enter_ten_minute_features() -> None:
    snapshot = _base_event(600_000, "STATE_SNAPSHOT")
    for slot in range(1, 11):
        snapshot[f"totalGold_{slot}"] = 1_000
        snapshot[f"xp_{slot}"] = 900
        snapshot[f"minionsKilled_{slot}"] = 50
        snapshot[f"jungleMinionsKilled_{slot}"] = 10
        snapshot[f"level_{slot}"] = 7
    frame = pd.DataFrame([
        _base_event(590_000, "CHAMPION_KILL", killerTeamId=100, victimId=6),
        snapshot,
        _base_event(610_000, "CHAMPION_KILL", killerTeamId=200, victimId=1),
        _base_event(1_800_000, "GAME_END"),
    ])
    features = build_match_features(frame).iloc[0]
    assert features["kill_diff_10"] == 1
    assert features["first_blood_side_10"] == 1


def test_match_ending_before_horizon_is_excluded() -> None:
    features = pd.DataFrame(
        {
            "matchId": ["short", "long"],
            "duration_min": [18.0, 25.0],
            "snapshot_timestamp_20_ms": [1_079_000, 1_199_000],
        }
    )
    cohort = eligible_at_horizon(features, 20)
    assert cohort["matchId"].tolist() == ["long"]

