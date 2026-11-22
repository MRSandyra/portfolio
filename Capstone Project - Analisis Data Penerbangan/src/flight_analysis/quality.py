"""Auditable data-quality checks for raw and transformed flight offers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from flight_analysis.features import (
    coerce_fares,
    parse_arrival_date,
    parse_duration_minutes,
    parse_stop_count,
)
from flight_analysis.schema import canonicalize_columns, missing_required_columns


@dataclass(frozen=True)
class QualityCheck:
    name: str
    status: str
    severity: str
    count: int
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityReport:
    stage: str
    row_count: int
    checks: tuple[QualityCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "row_count": self.row_count,
            "checks": [asdict(check) for check in self.checks],
        }


def _missing_check(frame: pd.DataFrame) -> QualityCheck:
    counts = {
        column: int(frame[column].isna().sum())
        for column in frame.columns
        if int(frame[column].isna().sum()) > 0
    }
    count = sum(counts.values())
    return QualityCheck(
        name="missing_fields",
        status="warn" if count else "pass",
        severity="warning",
        count=count,
        message="Missing values are reported, not imputed from neighboring rows.",
        details={"by_column": counts},
    )


def audit_dataframe(frame: pd.DataFrame, stage: str = "raw") -> QualityReport:
    """Run schema, duplicate, missing, fare, and date-logic checks."""
    canonical = canonicalize_columns(frame)
    missing_schema = missing_required_columns(canonical)
    checks: list[QualityCheck] = [
        QualityCheck(
            name="schema",
            status="fail" if missing_schema else "pass",
            severity="error",
            count=len(missing_schema),
            message="Required source columns must be present after canonicalization.",
            details={"missing_columns": missing_schema},
        )
    ]
    if missing_schema:
        return QualityReport(stage=stage, row_count=len(frame), checks=tuple(checks))

    duplicate_count = int(canonical.duplicated().sum())
    checks.append(
        QualityCheck(
            name="duplicates",
            status="warn" if duplicate_count else "pass",
            severity="warning",
            count=duplicate_count,
            message="Exact duplicate offers are removed before analysis.",
        )
    )
    checks.append(_missing_check(canonical))

    fares = coerce_fares(canonical["fare_usd"])
    invalid_fare = int((canonical["fare_usd"].notna() & fares.isna()).sum())
    checks.append(
        QualityCheck(
            name="invalid_fares",
            status="fail" if invalid_fare else "pass",
            severity="error",
            count=invalid_fare,
            message="Present fares must be numeric and greater than zero.",
        )
    )

    duration_values = canonical["travel_time"].map(parse_duration_minutes)
    stop_values = canonical["stops"].map(parse_stop_count)
    invalid_duration = int(
        (canonical["travel_time"].notna() & duration_values.isna()).sum()
    )
    invalid_stops = int((canonical["stops"].notna() & stop_values.isna()).sum())
    checks.append(
        QualityCheck(
            name="travel_logic",
            status="fail" if invalid_duration + invalid_stops else "pass",
            severity="error",
            count=invalid_duration + invalid_stops,
            message="Travel duration and stop count must parse to valid non-negative values.",
            details={
                "invalid_travel_time": invalid_duration,
                "invalid_stop_count": invalid_stops,
            },
        )
    )

    parsed_dates = canonical["arrival_date"].map(parse_arrival_date)
    valid_date = parsed_dates.map(lambda item: not pd.isna(item[0]))
    invalid_date = int((canonical["arrival_date"].notna() & ~valid_date).sum())
    missing_date = int(canonical["arrival_date"].isna().sum())
    checks.append(
        QualityCheck(
            name="arrival_date_format",
            status="fail" if invalid_date else ("warn" if missing_date else "pass"),
            severity="error" if invalid_date else "warning",
            count=invalid_date + missing_date,
            message="Arrival labels must be valid month/day values; missing dates remain missing.",
            details={"invalid": invalid_date, "missing": missing_date},
        )
    )
    checks.append(
        QualityCheck(
            name="chronological_date_logic",
            status="not_testable",
            severity="warning",
            count=len(canonical),
            message=(
                "Chronological arrival-after-departure logic cannot be tested because the "
                "source has no departure date or year/time-zone context."
            ),
        )
    )
    return QualityReport(stage=stage, row_count=len(frame), checks=tuple(checks))

