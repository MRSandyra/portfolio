import json

import pytest

from indostock_analyzer.artifacts import validate_artifact, verify_checksum
from indostock_analyzer.output import JsonReportWriter, ReportValidationError


def valid_report():
    return {
        "schema_version": 1,
        "report_type": "analysis",
        "report_id": "run-1",
        "generated_at": "2026-11-22T12:00:00+00:00",
        "run": {
            "run_id": "run-1",
            "software": {"name": "indostock-analyzer", "version": "fixture", "python": "3.11"},
        },
        "disclaimer": "Not financial advice.",
        "methodology": {"status": "exploratory_uncalibrated"},
        "sources": [],
        "failures": [],
        "articles": [],
        "stocks": {},
        "ticker_universe": {},
    }


def test_report_writer_writes_valid_json_atomically(tmp_path):
    path = JsonReportWriter(tmp_path).write(valid_report())

    assert json.loads(path.read_text(encoding="utf-8")) == valid_report()
    assert validate_artifact(json.loads(path.read_text(encoding="utf-8"))) == "analysis"
    assert verify_checksum(path) is True
    assert list(tmp_path.glob("*.tmp")) == []


def test_report_writer_rejects_incomplete_report(tmp_path):
    report = valid_report()
    del report["disclaimer"]

    with pytest.raises(ReportValidationError, match="disclaimer"):
        JsonReportWriter(tmp_path).write(report)
