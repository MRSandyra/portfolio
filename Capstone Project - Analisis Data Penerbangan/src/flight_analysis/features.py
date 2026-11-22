"""Pure parsers and feature engineering functions."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


_DURATION_RE = re.compile(
    r"^\s*(?:(?P<hours>\d+)\s*h)?\s*(?:(?P<minutes>\d+)\s*m)?\s*$",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(
    r"^\s*(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<period>[ap](?:m)?)?\s*$",
    re.IGNORECASE,
)


def parse_duration_minutes(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    match = _DURATION_RE.fullmatch(str(value))
    if not match or not (match.group("hours") or match.group("minutes")):
        return np.nan
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    if minutes >= 60:
        return np.nan
    return float(hours * 60 + minutes)


def parse_stop_count(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"nonstop", "non-stop", "direct", "0 stop", "0 stops"}:
        return 0.0
    match = re.search(r"\d+", text)
    return float(int(match.group())) if match else np.nan


def parse_clock_minutes(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    match = _CLOCK_RE.fullmatch(str(value))
    if not match:
        return np.nan
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    period = (match.group("period") or "").lower()
    if minute > 59:
        return np.nan
    if period:
        if not 1 <= hour <= 12:
            return np.nan
        hour = hour % 12 + (12 if period.startswith("p") else 0)
    elif hour > 23:
        return np.nan
    return float(hour * 60 + minute)


def parse_arrival_date(value: Any) -> tuple[float, float]:
    """Return month/day for a yearless label; year 2000 validates Feb 29."""
    if pd.isna(value) or not str(value).strip():
        return np.nan, np.nan
    parsed = pd.to_datetime(f"{str(value).strip()} 2000", errors="coerce")
    if pd.isna(parsed):
        return np.nan, np.nan
    return float(parsed.month), float(parsed.day)


def coerce_fares(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(r"[$,\s]", "", regex=True)
    fares = pd.to_numeric(cleaned, errors="coerce")
    return fares.where(fares > 0)
