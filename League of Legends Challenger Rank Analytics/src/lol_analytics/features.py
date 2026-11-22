"""Cutoff-safe match feature engineering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import TEN_MINUTES_MS, TWENTY_MINUTES_MS, VALID_TEAMS


def _numeric_column(events: pd.DataFrame, name: str) -> pd.Series:
    if name not in events:
        return pd.Series(np.nan, index=events.index, dtype=float)
    return pd.to_numeric(events[name], errors="coerce")


def infer_killer_team(events: pd.DataFrame) -> pd.Series:
    """Resolve event owner without using information after the event."""
    killer_team = _numeric_column(events, "killerTeamId")
    result = killer_team.where(killer_team.isin(VALID_TEAMS))

    victim = _numeric_column(events, "victimId")
    result = result.mask(result.isna() & victim.between(1, 5), 200)
    result = result.mask(result.isna() & victim.between(6, 10), 100)

    # BUILDING_KILL.teamId identifies the destroyed structure's team.
    structure_team = _numeric_column(events, "teamId")
    building = events.get("type", pd.Series(index=events.index, dtype=object)).eq("BUILDING_KILL")
    fallback = result.isna() & building & structure_team.isin(VALID_TEAMS)
    result = result.mask(fallback, 300 - structure_team)
    return result.astype("Float64")


def snapshot_at_or_before(match_events: pd.DataFrame, cutoff_ms: int) -> pd.Series:
    """Return the last statistics row at or before a prediction cutoff."""
    gold_column = "totalGold_1"
    if gold_column not in match_events:
        return pd.Series(dtype=float)
    candidates = match_events.loc[
        match_events["timestamp"].le(cutoff_ms) & match_events[gold_column].notna()
    ]
    if candidates.empty:
        return pd.Series(dtype=float)
    return candidates.sort_values("timestamp", kind="stable").iloc[-1]


def _number(value: Any) -> float:
    return 0.0 if pd.isna(value) else float(value)


def _team_sum(snapshot: pd.Series, team: int, prefix: str) -> float:
    if snapshot.empty:
        return 0.0
    slots = range(1, 6) if team == 100 else range(6, 11)
    return sum(_number(snapshot.get(f"{prefix}_{slot}", 0)) for slot in slots)


def _team_cs(snapshot: pd.Series, team: int) -> float:
    return _team_sum(snapshot, team, "minionsKilled") + _team_sum(
        snapshot, team, "jungleMinionsKilled"
    )


def _event_diff(events: pd.DataFrame, event_type: str, monster_type: str | None = None) -> int:
    selected = events.loc[events["type"].eq(event_type)]
    if monster_type is not None:
        monsters = selected.get(
            "monsterType", pd.Series(index=selected.index, dtype=object)
        )
        selected = selected.loc[monsters.eq(monster_type)]
    return int(selected["event_team"].eq(100).sum() - selected["event_team"].eq(200).sum())


def _first_side(events: pd.DataFrame, event_type: str, monster_type: str | None = None) -> int:
    selected = events.loc[events["type"].eq(event_type)]
    if monster_type is not None:
        monsters = selected.get(
            "monsterType", pd.Series(index=selected.index, dtype=object)
        )
        selected = selected.loc[monsters.eq(monster_type)]
    selected = selected.loc[selected["event_team"].isin(VALID_TEAMS)].sort_values(
        "timestamp", kind="stable"
    )
    if selected.empty:
        return 0
    return 1 if int(selected.iloc[0]["event_team"]) == 100 else -1


def _features_for_cutoff(match_events: pd.DataFrame, cutoff_ms: int, suffix: str) -> dict[str, float | int]:
    snapshot = snapshot_at_or_before(match_events, cutoff_ms)
    observed = match_events.loc[match_events["timestamp"].le(cutoff_ms)]
    return {
        f"snapshot_timestamp_{suffix}_ms": (
            int(snapshot["timestamp"]) if not snapshot.empty else np.nan
        ),
        f"gold_diff_{suffix}": _team_sum(snapshot, 100, "totalGold")
        - _team_sum(snapshot, 200, "totalGold"),
        f"xp_diff_{suffix}": _team_sum(snapshot, 100, "xp") - _team_sum(snapshot, 200, "xp"),
        f"cs_diff_{suffix}": _team_cs(snapshot, 100) - _team_cs(snapshot, 200),
        f"jungle_cs_diff_{suffix}": _team_sum(snapshot, 100, "jungleMinionsKilled")
        - _team_sum(snapshot, 200, "jungleMinionsKilled"),
        f"level_diff_{suffix}": _team_sum(snapshot, 100, "level")
        - _team_sum(snapshot, 200, "level"),
        f"kill_diff_{suffix}": _event_diff(observed, "CHAMPION_KILL"),
        f"tower_diff_{suffix}": _event_diff(observed, "BUILDING_KILL"),
        f"dragon_diff_{suffix}": _event_diff(observed, "ELITE_MONSTER_KILL", "DRAGON"),
        f"herald_diff_{suffix}": _event_diff(observed, "ELITE_MONSTER_KILL", "RIFTHERALD"),
        f"baron_diff_{suffix}": _event_diff(observed, "ELITE_MONSTER_KILL", "BARON_NASHOR"),
        f"first_blood_side_{suffix}": _first_side(observed, "CHAMPION_KILL"),
        f"first_dragon_side_{suffix}": _first_side(observed, "ELITE_MONSTER_KILL", "DRAGON"),
        f"first_tower_side_{suffix}": _first_side(observed, "BUILDING_KILL"),
        f"first_herald_side_{suffix}": _first_side(
            observed, "ELITE_MONSTER_KILL", "RIFTHERALD"
        ),
    }


def _first_value(events: pd.DataFrame, column: str, default: Any = None) -> Any:
    if column not in events:
        return default
    values = events[column].dropna()
    return values.iloc[0] if not values.empty else default


def build_match_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build one audited feature row per complete match."""
    work = events.copy()
    work["event_team"] = infer_killer_team(work)
    rows: list[dict[str, Any]] = []

    for match_id, match in work.groupby("matchId", sort=False):
        match = match.sort_values("timestamp", kind="stable")
        winner = int(_first_value(match, "winningTeam", -1))
        if winner not in VALID_TEAMS or not match["type"].eq("GAME_END").any():
            continue
        game_version = str(_first_value(match, "gameVersion", "unknown"))
        row: dict[str, Any] = {
            "matchId": match_id,
            "match_order": int(_first_value(match, "match_order", len(rows))),
            "game_start_timestamp_ms": int(_first_value(match, "gameStartTimestamp")),
            "gameVersion": game_version,
            "patch": ".".join(game_version.split(".")[:2]),
            "region": _first_value(match, "region", "unknown"),
            "duration_min": float(match["timestamp"].max()) / 60_000,
            "winningTeam": winner,
            "win": int(winner == 100),
        }
        row.update(_features_for_cutoff(match, TEN_MINUTES_MS, "10"))
        row.update(_features_for_cutoff(match, TWENTY_MINUTES_MS, "20"))
        rows.append(row)

    if not rows:
        raise ValueError("No complete, valid matches were available for feature engineering")
    return pd.DataFrame(rows).sort_values(
        ["game_start_timestamp_ms", "matchId"], kind="stable"
    ).reset_index(drop=True)


def eligible_at_horizon(match_features: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    """Keep matches still in progress with an observed state at the requested horizon."""
    if horizon_minutes not in {10, 20}:
        raise ValueError("horizon_minutes must be 10 or 20")
    snapshot_column = f"snapshot_timestamp_{horizon_minutes}_ms"
    if snapshot_column not in match_features:
        raise ValueError(f"Missing horizon provenance column: {snapshot_column}")
    cutoff_ms = horizon_minutes * 60_000
    mask = match_features["duration_min"].mul(60_000).ge(cutoff_ms) & match_features[
        snapshot_column
    ].notna()
    return match_features.loc[mask].copy()

