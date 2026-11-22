from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .cache import JsonTtlCache
from .config import RuntimeConfig
from .logging_utils import log_event
from .models import FetchResult, SourceAttribution, source_from_dict, to_dict
from .retry import RateLimiter, RetryPolicy, retry_call


LOGGER = logging.getLogger(__name__)
IDX_SYMBOL = re.compile(r"^[A-Z]{4}$")
FUNDAMENTAL_SCORE_FIELDS = frozenset(
    {"pe_ratio", "pb_ratio", "roe", "debt_to_equity", "earnings_growth"}
)


class DataValidationError(ValueError):
    pass


class TimeoutSession(requests.Session):
    def __init__(self, timeout: float):
        super().__init__()
        self.timeout = timeout

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        return super().request(method, url, **kwargs)


def build_http_session(config: RuntimeConfig) -> TimeoutSession:
    session = TimeoutSession(config.request_timeout)
    session.headers.update({"User-Agent": config.user_agent})
    transport_retry = Retry(
        total=config.retries,
        connect=config.retries,
        read=config.retries,
        status=config.retries,
        backoff_factor=config.backoff_seconds,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=transport_retry))
    return session


class YahooFinanceDataProvider:
    """Retrieve attributed market data with bounded retries and pacing."""

    def __init__(
        self,
        config: RuntimeConfig,
        session: requests.Session | None = None,
        cache: JsonTtlCache | None = None,
    ):
        self.config = config
        self.session = session or build_http_session(config)
        self.cache = cache
        self.rate_limiter = RateLimiter(config.min_interval_seconds)
        self.retry_policy = RetryPolicy(config.retries, config.backoff_seconds)

    def _ticker(self, symbol: str) -> Any:
        import yfinance as yf

        return yf.Ticker(f"{symbol}.JK", session=self.session)

    @staticmethod
    def _source(symbol: str) -> SourceAttribution:
        return SourceAttribution(
            name="Yahoo Finance",
            url=f"https://finance.yahoo.com/quote/{symbol}.JK",
        )

    def get_fundamental(self, symbol: str) -> FetchResult[dict[str, object]]:
        source = self._source(symbol)
        try:
            normalized_symbol = validate_symbol(symbol)
        except DataValidationError as exc:
            return FetchResult(None, source, _error_message(exc), "invalid_request")

        cache_key = f"{normalized_symbol}:max-age={self.config.max_fundamental_age_days}"
        cached = self._read_cache("fundamental", cache_key, validate_fundamental_data)
        if cached is not None:
            return cached

        def retrieve() -> dict[str, object]:
            self.rate_limiter.wait()
            info = self._ticker(normalized_symbol).get_info()
            if not info:
                raise DataValidationError("empty fundamental response")
            debt_to_equity = _finite_number(info.get("debtToEquity"))
            # Yahoo reports this field as a percentage (e.g. 150 means a 1.5 ratio).
            if debt_to_equity is not None:
                debt_to_equity /= 100.0
            values: dict[str, object] = {
                "name": info.get("longName"),
                "sector": info.get("sector"),
                "pe_ratio": _finite_number(info.get("trailingPE")),
                "pb_ratio": _finite_number(info.get("priceToBook")),
                "roe": _finite_number(info.get("returnOnEquity")),
                "debt_to_equity": debt_to_equity,
                "earnings_growth": _finite_number(info.get("earningsGrowth")),
                "dividend_yield": _finite_number(info.get("dividendYield")),
                "market_cap": _finite_number(info.get("marketCap")),
                "quote_currency": _clean_string(info.get("currency")),
                "financial_currency": _clean_string(info.get("financialCurrency")),
            }
            statement_period = _source_date(
                info.get("mostRecentQuarter") or info.get("lastFiscalYearEnd")
            )
            if statement_period is not None:
                values["statement_period_end"] = statement_period.isoformat()
            result = {key: value for key, value in values.items() if value is not None}
            result["units"] = {
                "pe_ratio": "ratio",
                "pb_ratio": "ratio",
                "roe": "decimal_fraction",
                "debt_to_equity": "ratio",
                "earnings_growth": "decimal_fraction",
                "dividend_yield": "decimal_fraction",
                "market_cap": result.get("quote_currency") or "source_quote_currency_unknown",
            }
            result["data_quality"] = assess_fundamental_quality(
                result,
                max_age_days=self.config.max_fundamental_age_days,
            )
            validate_fundamental_data(result)
            return result

        try:
            value = retry_call(
                retrieve,
                self.retry_policy,
                should_retry=lambda exc: not isinstance(exc, DataValidationError),
                operation_name="yahoo_fundamental",
            )
            result = FetchResult(value, source)
            self._write_cache("fundamental", cache_key, result)
            return result
        except Exception as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "source_fetch_failed",
                source=source.name,
                data_kind="fundamental",
                symbol=normalized_symbol,
                error_type=type(exc).__name__,
            )
            return FetchResult(None, source, _error_message(exc), _error_code(exc))

    def get_technical(self, symbol: str, period: str = "6mo") -> FetchResult[dict[str, object]]:
        source = self._source(symbol)
        try:
            normalized_symbol = validate_symbol(symbol)
        except DataValidationError as exc:
            return FetchResult(None, source, _error_message(exc), "invalid_request")

        cache_key = (
            f"{normalized_symbol}:{period}:1d:max-age={self.config.max_price_age_days}:"
            f"extreme={self.config.extreme_return_threshold_percent:g}"
        )
        cached = self._read_cache("technical", cache_key, validate_technical_data)
        if cached is not None:
            return cached

        def retrieve() -> dict[str, object]:
            self.rate_limiter.wait()
            ticker = self._ticker(normalized_symbol)
            history = ticker.history(
                period=period,
                interval="1d",
                timeout=self.config.request_timeout,
                raise_errors=True,
                auto_adjust=True,
                actions=True,
            )
            metadata = getattr(ticker, "history_metadata", {}) or {}
            quote_currency = metadata.get("currency") if isinstance(metadata, dict) else None
            return calculate_technical(
                history,
                max_price_age_days=self.config.max_price_age_days,
                extreme_return_threshold_percent=self.config.extreme_return_threshold_percent,
                quote_currency=_clean_string(quote_currency),
            )

        try:
            value = retry_call(
                retrieve,
                self.retry_policy,
                should_retry=lambda exc: not isinstance(exc, DataValidationError),
                operation_name="yahoo_technical",
            )
            result = FetchResult(value, source)
            self._write_cache("technical", cache_key, result)
            return result
        except Exception as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "source_fetch_failed",
                source=source.name,
                data_kind="technical",
                symbol=normalized_symbol,
                error_type=type(exc).__name__,
            )
            return FetchResult(None, source, _error_message(exc), _error_code(exc))

    def _read_cache(
        self,
        namespace: str,
        key: str,
        validator: Any,
    ) -> FetchResult[dict[str, object]] | None:
        if self.cache is None:
            return None
        entry = self.cache.get(f"yahoo-{namespace}", key)
        if entry is None or not isinstance(entry.value, dict):
            return None
        try:
            value = entry.value["value"]
            if not isinstance(value, dict):
                raise DataValidationError("cached market data must be an object")
            validator(value)
            source = source_from_dict(entry.value["source"])
        except (KeyError, TypeError, ValueError, DataValidationError) as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "cache_payload_invalid",
                namespace=namespace,
                error_type=type(exc).__name__,
            )
            return None
        log_event(
            LOGGER,
            logging.INFO,
            "cache_hit",
            namespace=namespace,
            age_seconds=round(entry.age_seconds, 3),
        )
        return FetchResult(value, source, from_cache=True)

    def _write_cache(self, namespace: str, key: str, result: FetchResult[dict[str, object]]) -> None:
        if self.cache is None or not result.ok:
            return
        try:
            self.cache.set(
                f"yahoo-{namespace}",
                key,
                {"value": to_dict(result.value), "source": to_dict(result.source)},
            )
        except (OSError, TypeError, ValueError) as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "cache_write_failed",
                namespace=namespace,
                error_type=type(exc).__name__,
            )


def calculate_technical(
    history: pd.DataFrame,
    max_price_age_days: int = 7,
    extreme_return_threshold_percent: float = 40.0,
    quote_currency: str | None = None,
    reference_time: datetime | None = None,
) -> dict[str, object]:
    validate_price_history(history)

    close = history["Close"].dropna()
    current_price = float(close.iloc[-1])

    ma_trend: str | None = None
    if len(close) >= 50:
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma_trend = "BULLISH" if ma20 > ma50 else "BEARISH" if ma20 < ma50 else "NEUTRAL"

    current_rsi: float | None = None
    if len(close) >= 15:
        delta = close.diff()
        average_gain = delta.clip(lower=0).rolling(14).mean()
        average_loss = (-delta.clip(upper=0)).rolling(14).mean()
        latest_loss = float(average_loss.iloc[-1])
        latest_gain = float(average_gain.iloc[-1])
        if latest_loss == 0:
            current_rsi = 100.0 if latest_gain > 0 else 50.0
        else:
            relative_strength = latest_gain / latest_loss
            current_rsi = float(100 - (100 / (1 + relative_strength)))

    macd_signal: str | None = None
    if len(close) >= 26:
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        histogram = (exp12 - exp26) - (exp12 - exp26).ewm(span=9, adjust=False).mean()
        latest_histogram = float(histogram.iloc[-1])
        macd_signal = "BULLISH" if latest_histogram > 0 else "BEARISH" if latest_histogram < 0 else "NEUTRAL"

    observed = close.tail(20)
    support = float(observed.min())
    resistance = float(observed.max())
    price_position = (current_price - support) / (resistance - support) if resistance != support else 0.5

    volumes = history["Volume"].dropna()
    volume_surge: bool | None = None
    if not volumes.empty:
        volume_surge = bool(float(volumes.iloc[-1]) > float(volumes.mean()) * 1.5)

    result: dict[str, object] = {
        "current_price": current_price,
        "observation_count": int(len(close)),
        "latest_observation": str(close.index[-1]),
        "price_change_1d": (
            float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
            if len(close) > 1 and float(close.iloc[-2]) != 0
            else 0.0
        ),
        "support": support,
        "resistance": resistance,
        "price_position": float(price_position),
        "quote_currency": quote_currency or "UNKNOWN",
        "price_adjustment": "yfinance_auto_adjusted",
        "units": {
            "current_price": quote_currency or "source_quote_currency_unknown",
            "price_change_1d": "percent",
            "support": quote_currency or "source_quote_currency_unknown",
            "resistance": quote_currency or "source_quote_currency_unknown",
            "rsi": "index_0_100",
            "volume": "shares",
        },
    }
    if ma_trend is not None:
        result["ma_trend"] = ma_trend
    if macd_signal is not None:
        result["macd_signal"] = macd_signal
    if current_rsi is not None:
        result["rsi"] = current_rsi
    if volume_surge is not None:
        result["volume_surge"] = volume_surge
    result["data_quality"] = assess_price_history_quality(
        history,
        max_price_age_days=max_price_age_days,
        extreme_return_threshold_percent=extreme_return_threshold_percent,
        quote_currency=quote_currency,
        reference_time=reference_time,
    )
    validate_technical_data(result)
    return result


def validate_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not IDX_SYMBOL.fullmatch(normalized):
        raise DataValidationError("IDX symbol must contain exactly four ASCII letters")
    return normalized


def validate_fundamental_data(data: dict[str, object]) -> None:
    if not isinstance(data, dict) or not data:
        raise DataValidationError("fundamental response is empty")
    available = FUNDAMENTAL_SCORE_FIELDS.intersection(data)
    if not available:
        raise DataValidationError("fundamental response has no scorable metrics")
    for field in available:
        if _finite_number(data[field]) is None:
            raise DataValidationError(f"fundamental field {field} is not finite")
    market_cap = data.get("market_cap")
    if market_cap is not None and (_finite_number(market_cap) or 0) <= 0:
        raise DataValidationError("market_cap must be positive when present")
    quality = data.get("data_quality")
    if not isinstance(quality, dict) or not isinstance(quality.get("warnings"), list):
        raise DataValidationError("fundamental data_quality metadata is missing")


def validate_price_history(history: pd.DataFrame) -> None:
    required_columns = {"Close", "Volume"}
    if history.empty:
        raise DataValidationError("empty price history")
    if not required_columns.issubset(history.columns):
        missing = ", ".join(sorted(required_columns - set(history.columns)))
        raise DataValidationError(f"price history missing columns: {missing}")
    if history.index.has_duplicates:
        raise DataValidationError("price history contains duplicate observations")
    if not history.index.is_monotonic_increasing:
        raise DataValidationError("price history must be chronological")

    close = history["Close"].dropna()
    if close.empty:
        raise DataValidationError("price history has no closing prices")
    for value in close:
        number = _finite_number(value)
        if number is None or number <= 0:
            raise DataValidationError("closing prices must be finite and positive")
    for value in history["Volume"].dropna():
        number = _finite_number(value)
        if number is None or number < 0:
            raise DataValidationError("volumes must be finite and non-negative")


def validate_technical_data(data: dict[str, object]) -> None:
    required = {
        "current_price",
        "observation_count",
        "latest_observation",
        "support",
        "resistance",
        "price_position",
    }
    missing = required - set(data)
    if missing:
        raise DataValidationError(f"technical data missing fields: {', '.join(sorted(missing))}")
    current_price = _finite_number(data["current_price"])
    support = _finite_number(data["support"])
    resistance = _finite_number(data["resistance"])
    position = _finite_number(data["price_position"])
    if current_price is None or current_price <= 0:
        raise DataValidationError("current_price must be finite and positive")
    if support is None or resistance is None or support <= 0 or resistance < support:
        raise DataValidationError("support/resistance range is invalid")
    if position is None or not 0 <= position <= 1:
        raise DataValidationError("price_position must be within 0 and 1")
    observations = _finite_number(data["observation_count"])
    if observations is None or observations < 1:
        raise DataValidationError("observation_count must be positive")
    if not isinstance(data["latest_observation"], str) or not data["latest_observation"].strip():
        raise DataValidationError("latest_observation must be present")
    if not isinstance(data.get("units"), dict):
        raise DataValidationError("technical units metadata is missing")
    quality = data.get("data_quality")
    if not isinstance(quality, dict) or not isinstance(quality.get("warnings"), list):
        raise DataValidationError("technical data_quality metadata is missing")


def assess_fundamental_quality(
    data: dict[str, object],
    max_age_days: int,
    reference_date: date | None = None,
) -> dict[str, object]:
    reference = reference_date or datetime.now(timezone.utc).date()
    warnings: list[str] = []
    available = sorted(FUNDAMENTAL_SCORE_FIELDS.intersection(data))
    missing = sorted(FUNDAMENTAL_SCORE_FIELDS - set(data))
    if missing:
        warnings.append("fundamental_metrics_missing")

    statement_age_days: int | None = None
    stale = False
    freshness_verified = False
    raw_period = data.get("statement_period_end")
    if isinstance(raw_period, str):
        try:
            statement_date = date.fromisoformat(raw_period[:10])
            statement_age_days = (reference - statement_date).days
            if statement_age_days < 0:
                warnings.append("fundamental_period_is_in_the_future")
            elif statement_age_days > max_age_days:
                stale = True
                warnings.append("fundamental_data_is_stale")
            else:
                freshness_verified = True
        except ValueError:
            warnings.append("fundamental_period_is_invalid")
    else:
        warnings.append("fundamental_period_is_unavailable")

    quote_currency = data.get("quote_currency")
    if not isinstance(quote_currency, str) or not quote_currency:
        warnings.append("quote_currency_is_unavailable")
    elif quote_currency.upper() != "IDR":
        warnings.append("quote_currency_is_not_idr")

    extreme_limits = {
        "pe_ratio": 1000.0,
        "pb_ratio": 100.0,
        "roe": 10.0,
        "debt_to_equity": 100.0,
        "earnings_growth": 100.0,
    }
    extreme_fields = [
        field
        for field, limit in extreme_limits.items()
        if field in data and (_finite_number(data[field]) is not None)
        and abs(float(data[field])) > limit
    ]
    if extreme_fields:
        warnings.append("fundamental_extreme_values_detected")

    return {
        "status": "warning" if warnings else "ok",
        "stale": stale,
        "freshness_verified": freshness_verified,
        "usable_for_scoring": freshness_verified and not stale,
        "statement_age_days": statement_age_days,
        "available_metrics": available,
        "missing_metrics": missing,
        "extreme_fields": extreme_fields,
        "warnings": warnings,
    }


def assess_price_history_quality(
    history: pd.DataFrame,
    max_price_age_days: int,
    extreme_return_threshold_percent: float,
    quote_currency: str | None,
    reference_time: datetime | None = None,
) -> dict[str, object]:
    warnings: list[str] = []
    reference = reference_time or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)

    close = history["Close"]
    volume = history["Volume"]
    missing_close_count = int(close.isna().sum())
    missing_volume_count = int(volume.isna().sum())
    if missing_close_count:
        warnings.append("closing_price_observations_missing")
    if missing_volume_count:
        warnings.append("volume_observations_missing")

    stale = False
    freshness_verified = False
    age_days: float | None = None
    calendar_gap_count = 0
    max_calendar_gap_days = 0
    if not isinstance(history.index, pd.DatetimeIndex):
        warnings.append("observation_timestamps_unavailable")
    else:
        datetime_index = pd.to_datetime(history.index, errors="coerce", utc=True)
        if datetime_index.isna().any():
            warnings.append("observation_timestamps_unavailable")
            datetime_index = None
        if datetime_index is None:
            latest = None
        else:
            latest = datetime_index[-1].to_pydatetime()
        if latest is None:
            pass
        else:
            age_days = round((reference.astimezone(timezone.utc) - latest).total_seconds() / 86400, 3)
            if age_days < -1:
                warnings.append("latest_price_is_in_the_future")
            elif age_days > max_price_age_days:
                stale = True
                warnings.append("price_data_is_stale")
            else:
                freshness_verified = True
            gaps = pd.Series(datetime_index).diff().dropna().dt.total_seconds() / 86400
            large_gaps = gaps[gaps > 4]
            calendar_gap_count = int(len(large_gaps))
            max_calendar_gap_days = int(large_gaps.max()) if not large_gaps.empty else 0
            if calendar_gap_count:
                warnings.append("calendar_gaps_detected")

    valid_close = close.dropna().astype(float)
    returns = valid_close.pct_change(fill_method=None).abs() * 100
    extreme_return_count = int((returns > extreme_return_threshold_percent).sum())
    if extreme_return_count:
        warnings.append("extreme_daily_returns_detected")

    split_count = _nonzero_action_count(history, "Stock Splits")
    dividend_count = _nonzero_action_count(history, "Dividends")
    if split_count or dividend_count:
        warnings.append("corporate_actions_present")
    if not quote_currency:
        warnings.append("quote_currency_is_unavailable")
    elif quote_currency.upper() != "IDR":
        warnings.append("quote_currency_is_not_idr")

    return {
        "status": "warning" if warnings else "ok",
        "stale": stale,
        "freshness_verified": freshness_verified,
        "usable_for_scoring": freshness_verified and not stale,
        "age_days": age_days,
        "missing_close_count": missing_close_count,
        "missing_volume_count": missing_volume_count,
        "calendar_gap_count": calendar_gap_count,
        "max_calendar_gap_days": max_calendar_gap_days,
        "extreme_return_threshold_percent": float(extreme_return_threshold_percent),
        "extreme_return_count": extreme_return_count,
        "stock_split_count": split_count,
        "dividend_event_count": dividend_count,
        "warnings": warnings,
    }


def _nonzero_action_count(history: pd.DataFrame, column: str) -> int:
    if column not in history.columns:
        return 0
    values = pd.to_numeric(history[column], errors="coerce").fillna(0)
    return int((values != 0).sum())


def _source_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc).date()
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def _clean_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().upper()
    return cleaned or None


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _error_code(exc: Exception) -> str:
    if isinstance(exc, DataValidationError):
        return "invalid_source_data"
    if isinstance(exc, (TimeoutError, requests.Timeout)) or "timeout" in type(exc).__name__.lower():
        return "source_timeout"
    if isinstance(exc, requests.RequestException):
        return "source_http_error"
    return "source_error"
