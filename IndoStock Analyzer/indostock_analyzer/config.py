from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import sys


DEFAULT_USER_AGENT = "IndoStockAnalyzer/2.0 (educational-research; respectful-crawler)"


def default_ticker_universe_path() -> Path:
    source_checkout = Path(__file__).resolve().parent.parent / "config" / "idx_tickers.json"
    if source_checkout.is_file():
        return source_checkout
    return Path(sys.prefix) / "share" / "indostock-analyzer" / "idx_tickers.json"


@dataclass(frozen=True)
class RuntimeConfig:
    max_news: int = 30
    request_timeout: float = 20.0
    retries: int = 2
    backoff_seconds: float = 1.0
    min_interval_seconds: float = 2.0
    user_agent: str = DEFAULT_USER_AGENT
    chrome_driver_path: Path | None = None
    headless: bool = True
    news_language: str = "id"
    output_dir: Path = Path("output")
    cache_enabled: bool = True
    cache_dir: Path = Path(".cache/indostock")
    cache_ttl_seconds: float = 300.0
    log_level: str = "INFO"
    ticker_universe_path: Path = field(default_factory=default_ticker_universe_path)
    ticker_universe_max_age_days: int = 45
    max_price_age_days: int = 7
    max_fundamental_age_days: int = 200
    extreme_return_threshold_percent: float = 40.0
    sentiment_provider: str | None = None
    sentiment_validation_report: Path | None = None
    sentiment_validation_sha256: str | None = None
    sentiment_review_bundle: Path | None = None
    sentiment_review_bundle_sha256: str | None = None
    sentiment_reviewer_trust_store: Path | None = None
    sentiment_reviewer_trust_store_sha256: str | None = None
    sentiment_review_revocations: Path | None = None
    sentiment_review_revocations_sha256: str | None = None
    sentiment_review_checkpoint: Path | None = None

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        driver = os.getenv("INDOSTOCK_CHROMEDRIVER")
        return cls(
            max_news=int(os.getenv("INDOSTOCK_MAX_NEWS", "30")),
            request_timeout=float(os.getenv("INDOSTOCK_TIMEOUT", "20")),
            retries=int(os.getenv("INDOSTOCK_RETRIES", "2")),
            backoff_seconds=float(os.getenv("INDOSTOCK_BACKOFF", "1")),
            min_interval_seconds=float(os.getenv("INDOSTOCK_MIN_INTERVAL", "2")),
            user_agent=os.getenv("INDOSTOCK_USER_AGENT", DEFAULT_USER_AGENT),
            chrome_driver_path=Path(driver).expanduser() if driver else None,
            headless=os.getenv("INDOSTOCK_HEADLESS", "1") != "0",
            news_language=os.getenv("INDOSTOCK_NEWS_LANGUAGE", "id"),
            output_dir=Path(os.getenv("INDOSTOCK_OUTPUT_DIR", "output")),
            cache_enabled=os.getenv("INDOSTOCK_CACHE_ENABLED", "1") != "0",
            cache_dir=Path(os.getenv("INDOSTOCK_CACHE_DIR", ".cache/indostock")),
            cache_ttl_seconds=float(os.getenv("INDOSTOCK_CACHE_TTL", "300")),
            log_level=os.getenv("INDOSTOCK_LOG_LEVEL", "INFO").upper(),
            ticker_universe_path=Path(
                os.getenv("INDOSTOCK_TICKER_UNIVERSE", str(default_ticker_universe_path()))
            ),
            ticker_universe_max_age_days=int(
                os.getenv("INDOSTOCK_UNIVERSE_MAX_AGE_DAYS", "45")
            ),
            max_price_age_days=int(os.getenv("INDOSTOCK_MAX_PRICE_AGE_DAYS", "7")),
            max_fundamental_age_days=int(
                os.getenv("INDOSTOCK_MAX_FUNDAMENTAL_AGE_DAYS", "200")
            ),
            extreme_return_threshold_percent=float(
                os.getenv("INDOSTOCK_EXTREME_RETURN_PERCENT", "40")
            ),
            sentiment_provider=os.getenv("INDOSTOCK_SENTIMENT_PROVIDER") or None,
            sentiment_validation_report=(
                Path(value).expanduser()
                if (value := os.getenv("INDOSTOCK_SENTIMENT_VALIDATION_REPORT"))
                else None
            ),
            sentiment_validation_sha256=(
                os.getenv("INDOSTOCK_SENTIMENT_VALIDATION_SHA256") or None
            ),
            sentiment_review_bundle=(
                Path(value).expanduser()
                if (value := os.getenv("INDOSTOCK_SENTIMENT_REVIEW_BUNDLE"))
                else None
            ),
            sentiment_review_bundle_sha256=(
                os.getenv("INDOSTOCK_SENTIMENT_REVIEW_BUNDLE_SHA256") or None
            ),
            sentiment_reviewer_trust_store=(
                Path(value).expanduser()
                if (value := os.getenv("INDOSTOCK_SENTIMENT_REVIEWER_TRUST_STORE"))
                else None
            ),
            sentiment_reviewer_trust_store_sha256=(
                os.getenv("INDOSTOCK_SENTIMENT_REVIEWER_TRUST_STORE_SHA256") or None
            ),
            sentiment_review_revocations=(
                Path(value).expanduser()
                if (value := os.getenv("INDOSTOCK_SENTIMENT_REVIEW_REVOCATIONS"))
                else None
            ),
            sentiment_review_revocations_sha256=(
                os.getenv("INDOSTOCK_SENTIMENT_REVIEW_REVOCATIONS_SHA256") or None
            ),
            sentiment_review_checkpoint=(
                Path(value).expanduser()
                if (value := os.getenv("INDOSTOCK_SENTIMENT_REVIEW_CHECKPOINT"))
                else None
            ),
        )

    def validate(self) -> "RuntimeConfig":
        errors: list[str] = []
        if self.max_news < 1:
            errors.append("max_news must be at least 1")
        if self.request_timeout <= 0:
            errors.append("request_timeout must be positive")
        if self.retries < 0:
            errors.append("retries cannot be negative")
        if self.backoff_seconds < 0:
            errors.append("backoff_seconds cannot be negative")
        if self.min_interval_seconds < 0:
            errors.append("min_interval_seconds cannot be negative")
        if not self.user_agent.strip():
            errors.append("user_agent cannot be empty")
        if self.news_language not in {"id", "en"}:
            errors.append("news_language must be 'id' or 'en'")
        if self.cache_enabled and self.cache_ttl_seconds <= 0:
            errors.append("cache_ttl_seconds must be positive when cache is enabled")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            errors.append("log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        if self.ticker_universe_max_age_days < 1:
            errors.append("ticker_universe_max_age_days must be positive")
        if self.max_price_age_days < 1:
            errors.append("max_price_age_days must be positive")
        if self.max_fundamental_age_days < 1:
            errors.append("max_fundamental_age_days must be positive")
        if self.extreme_return_threshold_percent <= 0:
            errors.append("extreme_return_threshold_percent must be positive")
        sentiment_configuration = (
            bool(self.sentiment_provider),
            bool(self.sentiment_validation_report),
            bool(self.sentiment_validation_sha256),
            bool(self.sentiment_review_bundle),
            bool(self.sentiment_review_bundle_sha256),
            bool(self.sentiment_reviewer_trust_store),
            bool(self.sentiment_reviewer_trust_store_sha256),
            bool(self.sentiment_review_revocations),
            bool(self.sentiment_review_revocations_sha256),
            bool(self.sentiment_review_checkpoint),
        )
        if len(set(sentiment_configuration)) != 1:
            errors.append(
                "sentiment_provider, sentiment_validation_report, and "
                "sentiment_validation_sha256, sentiment_review_bundle, "
                "sentiment_review_bundle_sha256, sentiment_reviewer_trust_store, "
                "sentiment_reviewer_trust_store_sha256, sentiment_review_revocations, and "
                "sentiment_review_revocations_sha256, and sentiment_review_checkpoint "
                "must be configured together"
            )
        if self.sentiment_validation_sha256 and not re.fullmatch(
            r"[0-9a-f]{64}", self.sentiment_validation_sha256
        ):
            errors.append(
                "sentiment_validation_sha256 must be 64 lowercase hexadecimal characters"
            )
        if self.sentiment_reviewer_trust_store_sha256 and not re.fullmatch(
            r"[0-9a-f]{64}", self.sentiment_reviewer_trust_store_sha256
        ):
            errors.append(
                "sentiment_reviewer_trust_store_sha256 must be 64 lowercase hexadecimal characters"
            )
        for name, value in (
            ("sentiment_review_bundle_sha256", self.sentiment_review_bundle_sha256),
            ("sentiment_review_revocations_sha256", self.sentiment_review_revocations_sha256),
        ):
            if value and not re.fullmatch(r"[0-9a-f]{64}", value):
                errors.append(f"{name} must be 64 lowercase hexadecimal characters")
        if errors:
            raise ValueError("; ".join(errors))
        return self
