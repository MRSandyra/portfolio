"""Project constants and the explicit feature-availability contract."""

from __future__ import annotations

from dataclasses import dataclass

TEN_MINUTES_MS = 600_000
TWENTY_MINUTES_MS = 1_200_000
VALID_TEAMS = (100, 200)
RANDOM_STATE = 42


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    available_at_ms: int
    description: str


def _horizon_specs(minutes: int) -> list[FeatureSpec]:
    cutoff = minutes * 60_000
    suffix = str(minutes)
    names = {
        f"gold_diff_{suffix}": "Team 100 minus Team 200 total gold",
        f"xp_diff_{suffix}": "Team 100 minus Team 200 experience",
        f"cs_diff_{suffix}": "Team 100 minus Team 200 lane plus jungle CS",
        f"jungle_cs_diff_{suffix}": "Team 100 minus Team 200 jungle CS",
        f"level_diff_{suffix}": "Team 100 minus Team 200 summed levels",
        f"kill_diff_{suffix}": "Team 100 minus Team 200 champion kills",
        f"tower_diff_{suffix}": "Team 100 minus Team 200 tower kills",
        f"dragon_diff_{suffix}": "Team 100 minus Team 200 dragon kills",
        f"herald_diff_{suffix}": "Team 100 minus Team 200 Rift Herald kills",
        f"baron_diff_{suffix}": "Team 100 minus Team 200 Baron kills",
        f"first_blood_side_{suffix}": "1 Team 100, -1 Team 200, 0 not observed",
        f"first_dragon_side_{suffix}": "1 Team 100, -1 Team 200, 0 not observed",
        f"first_tower_side_{suffix}": "1 Team 100, -1 Team 200, 0 not observed",
        f"first_herald_side_{suffix}": "1 Team 100, -1 Team 200, 0 not observed",
    }
    return [FeatureSpec(name, cutoff, description) for name, description in names.items()]


FEATURE_SPECS = tuple(_horizon_specs(10) + _horizon_specs(20))
FEATURE_REGISTRY = {spec.name: spec for spec in FEATURE_SPECS}

# These fields are useful for audit/reporting but are never model inputs.
NON_MODEL_COLUMNS = frozenset(
    {
        "matchId",
        "match_order",
        "gameStartTimestamp",
        "game_start_timestamp_ms",
        "gameVersion",
        "patch",
        "region",
        "duration_min",
        "snapshot_timestamp_10_ms",
        "snapshot_timestamp_20_ms",
        "winningTeam",
        "win",
        "team100_champs",
        "team200_champs",
    }
)


def features_available_at(cutoff_ms: int) -> list[str]:
    """Return registered features known no later than ``cutoff_ms``."""
    return [spec.name for spec in FEATURE_SPECS if spec.available_at_ms <= cutoff_ms]

