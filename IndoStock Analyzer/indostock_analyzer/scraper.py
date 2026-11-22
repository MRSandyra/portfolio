from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

from .cache import JsonTtlCache
from .config import RuntimeConfig
from .logging_utils import log_event
from .models import FetchResult, NewsItem, SourceAttribution, source_from_dict, to_dict
from .retry import RateLimitError, RateLimiter, RetryPolicy, retry_call


INDOPREMIER_NEWS_URL = "https://www.indopremier.com/ipotnews/newsPages.php?level4=topnews"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NetworkRateLimitSignal:
    found: bool
    retry_after: float | None = None
    url: str | None = None


class IndoPremierNewsScraper:
    def __init__(
        self,
        config: RuntimeConfig,
        driver_factory: Callable[[], Any] | None = None,
        cache: JsonTtlCache | None = None,
    ):
        self.config = config
        self.driver_factory = driver_factory or self._create_driver
        self.cache = cache
        self.rate_limiter = RateLimiter(config.min_interval_seconds)
        self.retry_policy = RetryPolicy(config.retries, config.backoff_seconds)

    def _create_driver(self) -> Any:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        options = Options()
        if self.config.headless:
            options.add_argument("--headless=new")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(f"--user-agent={self.config.user_agent}")
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        driver_path = self.config.chrome_driver_path
        if driver_path is None:
            # Selenium Manager resolves a compatible driver; no repo-local path required.
            driver = webdriver.Chrome(options=options)
        else:
            resolved = Path(driver_path).expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"configured ChromeDriver does not exist: {resolved}")
            driver = webdriver.Chrome(service=Service(str(resolved)), options=options)
        driver.set_page_load_timeout(self.config.request_timeout)
        return driver

    def scrape(self) -> FetchResult[list[NewsItem]]:
        source = SourceAttribution("IndoPremier / IPOT News", INDOPREMIER_NEWS_URL)
        cached = self._read_cache()
        if cached is not None:
            return cached

        def retrieve() -> list[NewsItem]:
            self.rate_limiter.wait()
            driver = self.driver_factory()
            try:
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import WebDriverWait

                driver.get(INDOPREMIER_NEWS_URL)
                self._raise_for_network_rate_limit(driver)
                page_text = str(getattr(driver, "page_source", "")).lower()
                if page_reports_rate_limit(page_text):
                    raise RateLimitError("source reported rate limiting")

                items = WebDriverWait(driver, self.config.request_timeout).until(
                    lambda current: current.find_elements(By.CSS_SELECTOR, "dl.listNews")
                )
                self._raise_for_network_rate_limit(driver)
                news = parse_news_elements(items, self.config.max_news)
                if not news:
                    raise ValueError("no usable news items found")
                return news
            finally:
                try:
                    driver.quit()
                except Exception as exc:
                    log_event(
                        LOGGER,
                        logging.WARNING,
                        "browser_cleanup_failed",
                        error_type=type(exc).__name__,
                    )

        try:
            value = retry_call(
                retrieve,
                self.retry_policy,
                should_retry=lambda exc: not isinstance(exc, FileNotFoundError),
                operation_name="indopremier_news",
            )
            result = FetchResult(value, source)
            self._write_cache(result)
            return result
        except Exception as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "source_fetch_failed",
                source=source.name,
                data_kind="news",
                error_type=type(exc).__name__,
            )
            return FetchResult(
                None,
                source,
                f"{type(exc).__name__}: {exc}"[:500],
                scraper_error_code(exc),
            )

    def _raise_for_network_rate_limit(self, driver: Any) -> None:
        try:
            entries = driver.get_log("performance")
        except Exception:
            return
        signal = detect_network_rate_limit(entries)
        if signal.found:
            raise RateLimitError(
                f"source returned HTTP 429 for {signal.url or 'a browser request'}",
                retry_after=signal.retry_after,
            )

    def _read_cache(self) -> FetchResult[list[NewsItem]] | None:
        if self.cache is None:
            return None
        entry = self.cache.get("news", "indopremier-topnews")
        if entry is None or not isinstance(entry.value, dict):
            return None
        try:
            source = source_from_dict(entry.value["source"])
            raw_articles = entry.value["articles"]
            if not isinstance(raw_articles, list) or not raw_articles:
                raise ValueError("cached news list is empty")
            articles = [
                NewsItem(title=item["title"], source=source_from_dict(item["source"]))
                for item in raw_articles
                if isinstance(item, dict) and isinstance(item.get("title"), str)
            ]
            if not articles:
                raise ValueError("cached news list has no valid articles")
        except (KeyError, TypeError, ValueError) as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "cache_payload_invalid",
                namespace="news",
                error_type=type(exc).__name__,
            )
            return None
        log_event(
            LOGGER,
            logging.INFO,
            "cache_hit",
            namespace="news",
            age_seconds=round(entry.age_seconds, 3),
        )
        return FetchResult(articles[: self.config.max_news], source, from_cache=True)

    def _write_cache(self, result: FetchResult[list[NewsItem]]) -> None:
        if self.cache is None or not result.ok:
            return
        try:
            self.cache.set(
                "news",
                "indopremier-topnews",
                {
                    "source": to_dict(result.source),
                    "articles": [to_dict(article) for article in result.value or []],
                },
            )
        except (OSError, TypeError, ValueError) as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "cache_write_failed",
                namespace="news",
                error_type=type(exc).__name__,
            )


def parse_news_elements(elements: list[Any], max_news: int) -> list[NewsItem]:
    news: list[NewsItem] = []
    seen: set[str] = set()
    for item in elements[:max_news]:
        try:
            link = item.find_element("tag name", "a")
            title = " ".join(str(link.text).split())
            href = str(link.get_attribute("href") or INDOPREMIER_NEWS_URL).strip()
        except Exception:
            continue
        if len(title) <= 15 or title in seen:
            continue
        if not href.startswith(("https://", "http://")):
            href = INDOPREMIER_NEWS_URL
        seen.add(title)
        news.append(NewsItem(title, SourceAttribution("IndoPremier / IPOT News", href)))
    return news


def page_reports_rate_limit(page_text: str) -> bool:
    normalized = page_text.lower()
    return any(
        marker in normalized
        for marker in ("too many requests", "error 429", "429 too many", "rate limit exceeded")
    )


def detect_network_rate_limit(entries: list[object]) -> NetworkRateLimitSignal:
    for entry in entries:
        try:
            if not isinstance(entry, dict):
                continue
            outer = json.loads(str(entry.get("message", "")))
            message = outer.get("message", {})
            if message.get("method") != "Network.responseReceived":
                continue
            response = message.get("params", {}).get("response", {})
            if int(response.get("status", 0)) != 429:
                continue
            headers = response.get("headers", {})
            retry_after_value = next(
                (value for key, value in headers.items() if str(key).lower() == "retry-after"),
                None,
            )
            retry_after = _positive_float(retry_after_value)
            return NetworkRateLimitSignal(True, retry_after, str(response.get("url") or "") or None)
        except (TypeError, ValueError, AttributeError, json.JSONDecodeError):
            continue
    return NetworkRateLimitSignal(False)


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def scraper_error_code(exc: Exception) -> str:
    if isinstance(exc, RateLimitError):
        return "source_rate_limited"
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "source_timeout"
    if isinstance(exc, FileNotFoundError):
        return "browser_configuration_error"
    if isinstance(exc, ValueError):
        return "invalid_source_data"
    return "source_error"
