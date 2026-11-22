import pandas as pd

from flight_analysis.quality import audit_dataframe


def test_quality_report_detects_required_failure_modes() -> None:
    row = {
        "Airline name ": "Emirates",
        "Travel  Time": "21h 15m",
        "Number of  Stoppage ": "1 Stop",
        "Ticket prize(Doller)": "$",
        "Depreture  Airport": "DAC",
        "Depreture Time": "08:55p",
        "Destination Airport": None,
        "Arrival Time": "09:30a",
        "Arrival Date": "Apr 31",
    }
    frame = pd.DataFrame([row, row])
    checks = {check.name: check for check in audit_dataframe(frame).checks}
    assert checks["schema"].status == "pass"
    assert checks["duplicates"].count == 1
    assert checks["missing_fields"].count == 2
    assert checks["invalid_fares"].count == 2
    assert checks["arrival_date_format"].details["invalid"] == 2
    assert checks["chronological_date_logic"].status == "not_testable"

