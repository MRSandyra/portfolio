"""End-to-end orchestration for reproducible analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from flight_analysis.analysis import build_analysis_tables, render_executive_summary
from flight_analysis.cleaning import clean_flights
from flight_analysis.quality import audit_dataframe
from flight_analysis.visualization import create_charts


GENERATED_FILES = (
    "cleaned_flight_offers.csv",
    "rejected_rows.csv",
    "quality_report.json",
    "kpis.csv",
    "airline_summary.csv",
    "route_summary.csv",
    "stop_summary.csv",
    "EXECUTIVE_SUMMARY.md",
    "charts/airline_offer_availability.png",
    "charts/fare_by_stop_count.png",
    "charts/top_route_fares.png",
)


def _clean_managed_outputs(output_dir: Path) -> None:
    for relative_path in GENERATED_FILES:
        target = output_dir / relative_path
        if target.is_file():
            target.unlink()


def run_pipeline(input_path: Path, output_dir: Path, clean_output: bool = False) -> dict[str, object]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        _clean_managed_outputs(output_dir)

    raw = pd.read_csv(input_path, dtype="string", low_memory=False)
    raw_quality = audit_dataframe(raw, stage="raw")
    schema_check = next(check for check in raw_quality.checks if check.name == "schema")
    if schema_check.status == "fail":
        missing = schema_check.details["missing_columns"]
        raise ValueError(f"Schema validation failed; missing: {', '.join(missing)}")

    result = clean_flights(raw)
    if result.cleaned.empty:
        raise ValueError("No valid rows remain after cleaning.")
    clean_quality = audit_dataframe(result.cleaned, stage="cleaned")
    tables = build_analysis_tables(result.cleaned)

    result.cleaned.to_csv(output_dir / "cleaned_flight_offers.csv", index=False)
    result.rejected.to_csv(output_dir / "rejected_rows.csv", index=False)
    tables.kpis.to_csv(output_dir / "kpis.csv", index=False)
    tables.airlines.to_csv(output_dir / "airline_summary.csv", index=False)
    tables.routes.to_csv(output_dir / "route_summary.csv", index=False)
    tables.stops.to_csv(output_dir / "stop_summary.csv", index=False)

    quality_payload = {"raw": raw_quality.to_dict(), "cleaned": clean_quality.to_dict()}
    (output_dir / "quality_report.json").write_text(
        json.dumps(quality_payload, indent=2), encoding="utf-8"
    )
    (output_dir / "EXECUTIVE_SUMMARY.md").write_text(
        render_executive_summary(result.cleaned, len(result.rejected), tables),
        encoding="utf-8",
    )
    charts = create_charts(tables.airlines, tables.routes, tables.stops, output_dir / "charts")
    return {
        "input_rows": len(raw),
        "valid_rows": len(result.cleaned),
        "rejected_rows": len(result.rejected),
        "output_dir": str(output_dir),
        "charts": [str(path) for path in charts],
    }

