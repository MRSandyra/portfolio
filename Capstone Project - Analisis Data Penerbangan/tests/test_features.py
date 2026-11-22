import math

from flight_analysis.features import (
    parse_arrival_date,
    parse_clock_minutes,
    parse_duration_minutes,
    parse_stop_count,
)


def test_duration_parser_handles_hours_and_minutes() -> None:
    assert parse_duration_minutes("21h 15m") == 1275
    assert parse_duration_minutes("55m") == 55
    assert math.isnan(parse_duration_minutes("1h 75m"))


def test_stop_parser_handles_nonstop_and_plural_labels() -> None:
    assert parse_stop_count("Nonstop") == 0
    assert parse_stop_count("2 Stops") == 2
    assert math.isnan(parse_stop_count("unknown"))


def test_clock_parser_supports_source_am_pm_format() -> None:
    assert parse_clock_minutes("01:40a") == 100
    assert parse_clock_minutes("08:55p") == 1255
    assert math.isnan(parse_clock_minutes("25:10"))


def test_yearless_date_parser_validates_calendar_day() -> None:
    assert parse_arrival_date("Apr 8") == (4, 8)
    assert parse_arrival_date("Feb 29") == (2, 29)
    month, day = parse_arrival_date("Apr 31")
    assert math.isnan(month) and math.isnan(day)

