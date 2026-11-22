"""Generate the small deterministic event fixture committed under data/sample/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_START_TIMESTAMP_MS = 1_742_000_000_000
MATCH_SPACING_MS = 3_600_000


def _game_start(order: int) -> int:
    return BASE_START_TIMESTAMP_MS + order * MATCH_SPACING_MS


def _snapshot(match: int, order: int, timestamp: int, winner: int) -> dict[str, object]:
    minutes = timestamp // 60_000
    direction = 1 if winner == 100 else -1
    row: dict[str, object] = {
        "matchId": f"sample-{match:03d}",
        "match_order": order,
        "gameStartTimestamp": _game_start(order),
        "timestamp": timestamp,
        "type": "STATE_SNAPSHOT",
        "winningTeam": winner,
        "gameVersion": "15.7.1",
        "region": "fixture",
    }
    for slot in range(1, 11):
        team_sign = 1 if slot <= 5 else -1
        advantage = direction * team_sign
        row[f"totalGold_{slot}"] = 1_000 + minutes * 220 + advantage * (30 + order)
        row[f"xp_{slot}"] = 800 + minutes * 180 + advantage * (25 + order)
        row[f"minionsKilled_{slot}"] = minutes * 6 + advantage * 2
        row[f"jungleMinionsKilled_{slot}"] = minutes + (slot % 2) + advantage
        row[f"championId_{slot}"] = 10 + slot + (order % 3) * 10
        row[f"level_{slot}"] = min(18, 1 + minutes // 2 + (1 if advantage > 0 else 0))
    return row


def generate(matches: int = 12) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order in range(matches):
        winner = 100 if order % 2 == 0 else 200
        loser = 300 - winner
        match_id = f"sample-{order:03d}"

        def event(timestamp: int, event_type: str, **extra: object) -> dict[str, object]:
            return {
                "matchId": match_id,
                "match_order": order,
                "gameStartTimestamp": _game_start(order),
                "timestamp": timestamp,
                "type": event_type,
                "winningTeam": winner,
                "gameVersion": "15.7.1",
                "region": "fixture",
                **extra,
            }

        match_rows = [
            event(
                180_000,
                "CHAMPION_KILL",
                killerTeamId=winner,
                victimId=6 if winner == 100 else 1,
            ),
            event(
                360_000,
                "ELITE_MONSTER_KILL",
                killerTeamId=winner,
                monsterType="DRAGON",
                monsterSubType="FIRE_DRAGON",
            ),
            event(
                480_000,
                "ELITE_MONSTER_KILL",
                killerTeamId=loser,
                monsterType="RIFTHERALD",
            ),
            _snapshot(order, order, 600_000, winner),
            event(
                720_000,
                "CHAMPION_KILL",
                killerTeamId=loser,
                victimId=1 if winner == 100 else 6,
            ),
            event(
                900_000,
                "BUILDING_KILL",
                teamId=loser,
                buildingType="TOWER_BUILDING",
            ),
            _snapshot(order, order, 1_200_000, winner),
            event(1_800_000, "GAME_END"),
        ]
        for event_id, row in enumerate(match_rows):
            row["eventId"] = event_id
            row["platformId"] = "FIXTURE"
        rows.extend(match_rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/sample/events.jsonl"))
    parser.add_argument("--matches", type=int, default=12)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in generate(args.matches):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

