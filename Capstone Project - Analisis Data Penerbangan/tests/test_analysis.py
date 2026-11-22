import pandas as pd

from flight_analysis.analysis import build_analysis_tables


def test_airline_offer_count_is_not_repeated_or_squared() -> None:
    frame = pd.DataFrame(
        {
            "airline": ["A", "A", "B"],
            "fare_usd": [100.0, 120.0, 90.0],
            "route": ["X-Y", "X-Y", "X-Z"],
            "departure_airport": ["X", "X", "X"],
            "destination_airport": ["Y", "Y", "Z"],
            "travel_minutes": [60, 70, 80],
            "stop_count": [0, 0, 1],
            "arrival_month": [4, 4, 4],
        }
    )
    tables = build_analysis_tables(frame)
    counts = dict(zip(tables.airlines["airline"], tables.airlines["offer_count"]))
    assert counts == {"A": 2, "B": 1}

