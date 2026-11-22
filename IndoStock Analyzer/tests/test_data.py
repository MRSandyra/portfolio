from datetime import datetime, timezone

import pandas as pd
import pytest

from indostock_analyzer.cache import JsonTtlCache
from indostock_analyzer.config import RuntimeConfig
from indostock_analyzer.data import (
    DataValidationError,
    YahooFinanceDataProvider,
    assess_fundamental_quality,
    calculate_technical,
)
from indostock_analyzer.scoring import score_technical


def test_empty_price_history_is_rejected_as_missing_data():
    with pytest.raises(ValueError, match="empty price history"):
        calculate_technical(pd.DataFrame(columns=["Close", "Volume"]))


def test_short_history_does_not_invent_long_window_indicators():
    history = pd.DataFrame({"Close": [100.0, 102.0], "Volume": [10, 12]})

    result = calculate_technical(history)

    assert "ma_trend" not in result
    assert "rsi" not in result
    assert "macd_signal" not in result
    assert result["current_price"] == 102.0
    assert result["observation_count"] == 2


@pytest.mark.parametrize(
    "history",
    [
        pd.DataFrame({"Close": [100.0, -1.0], "Volume": [10, 12]}),
        pd.DataFrame({"Close": [100.0, 101.0], "Volume": [10, -1]}),
        pd.DataFrame({"Close": [100.0, float("inf")], "Volume": [10, 12]}),
    ],
)
def test_invalid_price_history_is_rejected(history):
    with pytest.raises(DataValidationError):
        calculate_technical(history)


def test_long_history_produces_only_bounded_indicators():
    history = pd.DataFrame(
        {
            "Close": [100.0 + index for index in range(60)],
            "Volume": [1000 + index for index in range(60)],
        },
        index=pd.date_range("2026-01-01", periods=60, freq="B"),
    )

    result = calculate_technical(
        history,
        quote_currency="IDR",
        reference_time=datetime(2026, 3, 31, tzinfo=timezone.utc),
    )

    assert result["observation_count"] == 60
    assert result["ma_trend"] == "BULLISH"
    assert result["macd_signal"] in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert 0 <= result["rsi"] <= 100
    assert 0 <= result["price_position"] <= 1
    assert result["units"]["current_price"] == "IDR"
    assert result["data_quality"]["stale"] is False


def test_stale_prices_are_excluded_from_technical_scoring():
    history = pd.DataFrame(
        {"Close": [100.0 + index for index in range(30)], "Volume": [1000] * 30},
        index=pd.date_range("2026-01-01", periods=30, freq="B"),
    )
    result = calculate_technical(
        history,
        max_price_age_days=7,
        quote_currency="IDR",
        reference_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )

    assert result["data_quality"]["stale"] is True
    assert "price_data_is_stale" in result["data_quality"]["warnings"]
    assert score_technical(result).score is None


def test_quality_report_flags_gaps_extreme_returns_and_actions():
    history = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 180.0],
            "Volume": [1000, None, 2000],
            "Stock Splits": [0, 0, 2],
            "Dividends": [0, 10, 0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-12"]),
    )
    result = calculate_technical(
        history,
        extreme_return_threshold_percent=40,
        quote_currency="IDR",
        reference_time=datetime(2026, 1, 13, tzinfo=timezone.utc),
    )
    quality = result["data_quality"]

    assert quality["calendar_gap_count"] == 1
    assert quality["extreme_return_count"] == 1
    assert quality["stock_split_count"] == 1
    assert quality["dividend_event_count"] == 1
    assert "volume_observations_missing" in quality["warnings"]


def test_fundamental_quality_tracks_units_freshness_and_extremes():
    data = {
        "pe_ratio": 1500.0,
        "quote_currency": "USD",
        "statement_period_end": "2025-01-01",
    }
    quality = assess_fundamental_quality(
        data,
        max_age_days=200,
        reference_date=datetime(2026, 1, 1).date(),
    )

    assert quality["stale"] is True
    assert quality["usable_for_scoring"] is False
    assert quality["extreme_fields"] == ["pe_ratio"]
    assert "quote_currency_is_not_idr" in quality["warnings"]


def test_unverified_fundamental_freshness_is_not_usable_for_scoring():
    quality = assess_fundamental_quality(
        {"pe_ratio": 12.0, "quote_currency": "IDR"},
        max_age_days=200,
        reference_date=datetime(2026, 1, 1).date(),
    )

    assert quality["freshness_verified"] is False
    assert quality["usable_for_scoring"] is False


class FakeTicker:
    def __init__(self):
        self.info_calls = 0

    def get_info(self):
        self.info_calls += 1
        return {"longName": "Bank Example", "trailingPE": 12.0, "marketCap": 1000}


class EmptyTicker(FakeTicker):
    def get_info(self):
        self.info_calls += 1
        return {}


class FakeProvider(YahooFinanceDataProvider):
    def __init__(self, config, cache, ticker):
        super().__init__(config, cache=cache)
        self.fake_ticker = ticker

    def _ticker(self, symbol):
        return self.fake_ticker


class FailingWriteCache:
    def get(self, namespace, key):
        return None

    def set(self, namespace, key, value):
        raise OSError("cache disk unavailable")


def test_provider_uses_short_lived_cache_without_second_source_call(tmp_path):
    config = RuntimeConfig(retries=0, min_interval_seconds=0).validate()
    cache = JsonTtlCache(tmp_path, ttl_seconds=60)
    ticker = FakeTicker()
    provider = FakeProvider(config, cache, ticker)

    first = provider.get_fundamental("BBRI")
    second = provider.get_fundamental("BBRI")

    assert first.ok is True
    assert first.from_cache is False
    assert second.ok is True
    assert second.from_cache is True
    assert ticker.info_calls == 1


def test_invalid_symbol_fails_before_source_call(tmp_path):
    config = RuntimeConfig(retries=3, min_interval_seconds=0).validate()
    ticker = FakeTicker()
    provider = FakeProvider(config, JsonTtlCache(tmp_path, 60), ticker)

    result = provider.get_fundamental("INVALID")

    assert result.ok is False
    assert result.error_code == "invalid_request"
    assert ticker.info_calls == 0


def test_cache_write_failure_does_not_discard_successful_source_data():
    config = RuntimeConfig(retries=0, min_interval_seconds=0).validate()
    ticker = FakeTicker()
    provider = FakeProvider(config, FailingWriteCache(), ticker)

    result = provider.get_fundamental("BBRI")

    assert result.ok is True
    assert result.value["pe_ratio"] == 12.0
    assert ticker.info_calls == 1


def test_unavailable_ticker_data_returns_structured_failure(tmp_path):
    config = RuntimeConfig(retries=3, min_interval_seconds=0).validate()
    ticker = EmptyTicker()
    provider = FakeProvider(config, JsonTtlCache(tmp_path, 60), ticker)

    result = provider.get_fundamental("BBRI")

    assert result.ok is False
    assert result.error_code == "invalid_source_data"
    assert ticker.info_calls == 1
