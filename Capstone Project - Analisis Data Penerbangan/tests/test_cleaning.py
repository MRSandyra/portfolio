import pandas as pd

from flight_analysis.cleaning import clean_flights


def _raw_rows() -> pd.DataFrame:
    rows = [
        ["Emirates", "21h 15m", "1 Stop", "1283", "dac", "08:55p", "JFK", "09:30a", "Apr 8"],
        ["Emirates", "21h 15m", "1 Stop", "1283", "dac", "08:55p", "JFK", "09:30a", "Apr 8"],
        ["Turkish", "18h 5m", "1 Stop", "bad", "DAC", "06:35a", "JFK", "05:50p", "Apr 8"],
        ["Qatar", "16h 0m", "Nonstop", "1431", "DAC", "05:30a", "EWR", "10:30p", "Apr 8"],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Airline name ", "Travel  Time", "Number of  Stoppage ",
            "Ticket prize(Doller)", "Depreture  Airport", "Depreture Time",
            "Destination Airport", "Arrival Time", "Arrival Date",
        ],
    )


def test_cleaning_deduplicates_rejects_bad_fare_and_engineers_features() -> None:
    result = clean_flights(_raw_rows())
    assert len(result.cleaned) == 2
    assert len(result.rejected) == 2
    assert set(result.rejected["reject_reason"]) == {"exact_duplicate", "invalid_fare"}
    assert result.cleaned.loc[0, "route"] == "DAC-JFK"
    assert result.cleaned.loc[1, "stop_count"] == 0
    assert result.cleaned.loc[0, "travel_minutes"] == 1275


def test_cleaning_does_not_backfill_missing_values_from_other_rows() -> None:
    raw = _raw_rows().iloc[[0]].copy()
    raw.loc[raw.index[0], "Destination Airport"] = pd.NA
    result = clean_flights(raw)
    assert len(result.cleaned) == 1
    assert pd.isna(result.cleaned.loc[0, "route"])

