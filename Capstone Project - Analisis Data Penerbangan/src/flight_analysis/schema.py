"""Column contracts and canonicalization for the source CSV."""

from __future__ import annotations

import re

import pandas as pd


COLUMN_ALIASES = {
    "airline_name": "airline",
    "travel_time": "travel_time",
    "number_of_stoppage": "stops",
    "ticket_prize_doller": "fare_usd",
    "depreture_airport": "departure_airport",
    "depreture_time": "departure_time",
    "1st_stoppage_waiting_hour": "first_stop_wait",
    "1st_stoppage": "first_stop_airport",
    "destination_airport": "destination_airport",
    "arrival_time": "arrival_time",
    "arrival_date": "arrival_date",
    "airline": "airline",
    "stops": "stops",
    "fare_usd": "fare_usd",
    "departure_airport": "departure_airport",
    "departure_time": "departure_time",
    "first_stop_wait": "first_stop_wait",
    "first_stop_airport": "first_stop_airport",
}

REQUIRED_COLUMNS = (
    "airline",
    "travel_time",
    "stops",
    "fare_usd",
    "departure_airport",
    "departure_time",
    "destination_airport",
    "arrival_time",
    "arrival_date",
)

OPTIONAL_COLUMNS = ("first_stop_wait", "first_stop_airport")


def normalize_header(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename known source headers and discard spreadsheet overflow columns."""
    rename_map: dict[object, str] = {}
    seen: set[str] = set()
    for column in frame.columns:
        canonical = COLUMN_ALIASES.get(normalize_header(column))
        if canonical is None:
            continue
        if canonical in seen:
            raise ValueError(f"Multiple source columns map to '{canonical}'.")
        rename_map[column] = canonical
        seen.add(canonical)
    return frame.loc[:, list(rename_map)].rename(columns=rename_map).copy()


def missing_required_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(set(REQUIRED_COLUMNS) - set(frame.columns))

