from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

from .logging_utils import log_event
from .models import to_dict


LOGGER = logging.getLogger(__name__)


class ReportValidationError(ValueError):
    pass


class JsonReportWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def write(self, report: dict[str, Any]) -> Path:
        validate_report(report)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.output_dir / f"stock_analysis_{timestamp}.json"
        temporary = path.with_suffix(".json.tmp")
        encoded = json.dumps(to_dict(report), indent=2, ensure_ascii=False, allow_nan=False)
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        from .artifacts import write_checksum

        write_checksum(path)
        log_event(LOGGER, logging.INFO, "report_written", path=str(path), bytes=len(encoded.encode("utf-8")))
        return path


def validate_report(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise ReportValidationError("report must be an object")
    required_types: dict[str, type] = {
        "schema_version": int,
        "report_type": str,
        "report_id": str,
        "generated_at": str,
        "run": dict,
        "disclaimer": str,
        "methodology": dict,
        "sources": list,
        "failures": list,
        "articles": list,
        "stocks": dict,
        "ticker_universe": dict,
    }
    for field, expected_type in required_types.items():
        if field not in report:
            raise ReportValidationError(f"report missing required field: {field}")
        if not isinstance(report[field], expected_type):
            raise ReportValidationError(f"report field {field} must be {expected_type.__name__}")
    try:
        generated_at = datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReportValidationError("generated_at must be an ISO-8601 timestamp") from exc
    if generated_at.tzinfo is None:
        raise ReportValidationError("generated_at must include a timezone")
    if not report["disclaimer"].strip():
        raise ReportValidationError("disclaimer cannot be empty")
    if report["schema_version"] != 1 or report["report_type"] != "analysis":
        raise ReportValidationError("unsupported analysis report schema")
    if not report["report_id"].strip() or report["run"].get("run_id") != report["report_id"]:
        raise ReportValidationError("report_id must match run.run_id")
    software = report["run"].get("software")
    if not isinstance(software, dict) or not all(
        isinstance(software.get(field), str) and software[field].strip()
        for field in ("name", "version", "python")
    ):
        raise ReportValidationError("run software identity is incomplete")
    if report["methodology"].get("status") != "exploratory_uncalibrated":
        raise ReportValidationError("analysis methodology must remain exploratory_uncalibrated")
