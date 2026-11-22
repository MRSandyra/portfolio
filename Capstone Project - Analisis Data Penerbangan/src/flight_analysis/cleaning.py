"""Deterministic cleaning and feature engineering for flight listings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flight_analysis.features import (
    coerce_fares,
    parse_arrival_date,
    parse_clock_minutes,
    parse_duration_minutes,
    parse_stop_count,
)
from flight_analysis.schema import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    canonicalize_columns,
    missing_required_columns,
)


@dataclass(frozen=True)
class CleanResult:
    cleaned: pd.DataFrame
    rejected: pd.DataFrame


def _clean_text(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.eq(""))


def _invalid_airline(series: pd.Series) -> pd.Series:
    lowered = series.astype("string").str.lower()
    return series.isna() | lowered.str.contains(
        r"%|retail\s+price", regex=True, na=False
    )


def clean_flights(raw: pd.DataFrame) -> CleanResult:
    """Clean source rows and reject records that cannot support core analysis."""
    frame = canonicalize_columns(raw)
    missing = missing_required_columns(frame)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for column in OPTIONAL_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame.loc[:, [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]].copy()
    frame.insert(0, "source_row", np.arange(2, len(frame) + 2))

    for column in [item for item in frame.columns if item != "source_row"]:
        frame[column] = _clean_text(frame[column])

    duplicate_mask = frame.drop(columns="source_row").duplicated(keep="first")
    fare = coerce_fares(frame["fare_usd"])
    travel_minutes = frame["travel_time"].map(parse_duration_minutes)
    stop_count = frame["stops"].map(parse_stop_count)
    reasons = pd.Series("", index=frame.index, dtype="string")

    def add_reason(mask: pd.Series, reason: str) -> None:
        nonlocal reasons
        already_has_reason = reasons.ne("")
        reasons = reasons.mask(mask & ~already_has_reason, reason)
        reasons = reasons.mask(mask & already_has_reason, reasons + ";" + reason)

    add_reason(duplicate_mask, "exact_duplicate")
    add_reason(_invalid_airline(frame["airline"]), "invalid_airline")
    add_reason(fare.isna(), "invalid_fare")
    add_reason(travel_minutes.isna() | travel_minutes.le(0), "invalid_travel_time")
    add_reason(stop_count.isna(), "invalid_stop_count")

    rejected_mask = reasons.ne("")
    rejected = frame.loc[rejected_mask].copy()
    rejected.insert(1, "reject_reason", reasons.loc[rejected_mask])

    clean = frame.loc[~rejected_mask].copy()
    clean["fare_usd"] = fare.loc[~rejected_mask].astype("float64")
    clean["travel_minutes"] = travel_minutes.loc[~rejected_mask].astype("int64")
    clean["stop_count"] = stop_count.loc[~rejected_mask].astype("int64")
    clean["departure_minutes"] = clean["departure_time"].map(parse_clock_minutes).astype("Int64")
    clean["arrival_minutes"] = clean["arrival_time"].map(parse_clock_minutes).astype("Int64")

    parsed_dates = clean["arrival_date"].map(parse_arrival_date)
    clean["arrival_month"] = parsed_dates.map(lambda item: item[0]).astype("Int64")
    clean["arrival_day"] = parsed_dates.map(lambda item: item[1]).astype("Int64")

    clean["departure_airport"] = clean["departure_airport"].str.upper()
    clean["destination_airport"] = clean["destination_airport"].str.upper()
    complete_route = clean["departure_airport"].notna() & clean["destination_airport"].notna()
    clean["route"] = pd.Series(pd.NA, index=clean.index, dtype="string")
    clean.loc[complete_route, "route"] = (
        clean.loc[complete_route, "departure_airport"]
        + "-"
        + clean.loc[complete_route, "destination_airport"]
    )
    clean["fare_per_travel_hour_usd"] = (
        clean["fare_usd"] / clean["travel_minutes"] * 60
    ).round(2)
    clean["is_nonstop"] = clean["stop_count"].eq(0)

    return CleanResult(
        cleaned=clean.reset_index(drop=True),
        rejected=rejected.reset_index(drop=True),
    )
