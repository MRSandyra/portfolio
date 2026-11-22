from pathlib import Path

from flight_analysis.pipeline import run_pipeline


def test_pipeline_smoke_creates_auditable_outputs(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "flight_sample.csv"
    output = tmp_path / "artifacts"
    result = run_pipeline(fixture, output, clean_output=True)
    assert result["input_rows"] == 5
    assert result["valid_rows"] == 5
    expected = {
        "cleaned_flight_offers.csv", "rejected_rows.csv", "quality_report.json",
        "airline_summary.csv", "route_summary.csv", "stop_summary.csv",
        "EXECUTIVE_SUMMARY.md",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    assert (output / "charts" / "airline_offer_availability.png").is_file()

