from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Protocol

from . import DISCLAIMER
from .logging_utils import log_event
from .models import FetchResult, NewsItem, SentimentResult, SourceAttribution, to_dict, utc_now_iso
from .provenance import minimal_run_metadata
from .scoring import combine_scores, score_fundamental, score_sentiments, score_technical
from .sentiment import INDONESIAN_TEXTBLOB_AUDIT, SentimentAnalyzer, TextBlobSentimentAnalyzer
from .tickers import TickerExtractor


LOGGER = logging.getLogger(__name__)


class NewsSource(Protocol):
    def scrape(self) -> FetchResult[list[NewsItem]]: ...


class MarketDataSource(Protocol):
    def get_fundamental(self, symbol: str) -> FetchResult[dict[str, object]]: ...

    def get_technical(self, symbol: str) -> FetchResult[dict[str, object]]: ...


class Analyzer:
    def __init__(
        self,
        news_source: NewsSource,
        market_data: MarketDataSource,
        sentiment: SentimentAnalyzer | None = None,
        ticker_extractor: TickerExtractor | None = None,
        news_language: str = "id",
        run_metadata: dict[str, object] | None = None,
    ):
        self.news_source = news_source
        self.market_data = market_data
        self.sentiment = sentiment or TextBlobSentimentAnalyzer()
        self.ticker_extractor = ticker_extractor or TickerExtractor()
        self.news_language = news_language
        self.run_metadata = run_metadata or minimal_run_metadata()

    def run(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "schema_version": 1,
            "report_type": "analysis",
            "report_id": self.run_metadata["run_id"],
            "generated_at": utc_now_iso(),
            "run": self.run_metadata,
            "disclaimer": DISCLAIMER,
            "methodology": {
                "status": "exploratory_uncalibrated",
                "backtested": False,
                "component_weights": {"sentiment": 0.2, "fundamental": 0.5, "technical": 0.3},
                "missing_data": "Unavailable components are excluded and coverage is reported.",
                "sentiment_audit": INDONESIAN_TEXTBLOB_AUDIT,
                "sentiment_model": self.sentiment.metadata(),
            },
            "sources": [],
            "failures": [],
            "articles": [],
            "stocks": {},
            "ticker_universe": self.ticker_extractor.metadata(),
        }

        try:
            news_result = self.news_source.scrape()
        except Exception as exc:
            report["failures"].append(
                {
                    "stage": "news",
                    "error": _error_message(exc),
                    "error_code": "pipeline_source_error",
                }
            )
            log_event(
                LOGGER,
                logging.ERROR,
                "pipeline_component_unavailable",
                stage="news",
                error_code="pipeline_source_error",
            )
            return report

        report["sources"].append({**to_dict(news_result.source), "from_cache": news_result.from_cache})
        if not news_result.ok:
            report["failures"].append(
                {
                    "stage": "news",
                    "source": to_dict(news_result.source),
                    "error": news_result.error,
                    "error_code": news_result.error_code,
                }
            )
            return report

        articles_by_ticker: dict[str, list[tuple[NewsItem, SentimentResult]]] = defaultdict(list)
        for article in news_result.value or []:
            tickers = self.ticker_extractor.extract(article.title)
            assessments = {
                ticker: self.sentiment.analyze(article.title, self.news_language, target=ticker)
                for ticker in tickers
            }
            report["articles"].append(
                {
                    "title": article.title,
                    "tickers": tickers,
                    "source": to_dict(article.source),
                    "sentiment_by_ticker": {
                        ticker: to_dict(assessment)
                        for ticker, assessment in assessments.items()
                    },
                }
            )
            for ticker in tickers:
                articles_by_ticker[ticker].append((article, assessments[ticker]))

        for ticker, entries in articles_by_ticker.items():
            sentiment_score = score_sentiments([entry[1] for entry in entries])
            fundamental_result = self._fetch("fundamental", ticker, report)
            technical_result = self._fetch("technical", ticker, report)
            fundamental_score = score_fundamental(fundamental_result.value)
            technical_score = score_technical(technical_result.value)
            indicator = combine_scores(
                {
                    "sentiment": sentiment_score.score,
                    "fundamental": fundamental_score.score,
                    "technical": technical_score.score,
                }
            )
            report["stocks"][ticker] = {
                "article_count": len(entries),
                "sentiment": to_dict(sentiment_score),
                "fundamental": {
                    "rating": to_dict(fundamental_score),
                    "data": to_dict(fundamental_result.value),
                    "source": to_dict(fundamental_result.source),
                    "error": fundamental_result.error,
                    "error_code": fundamental_result.error_code,
                    "from_cache": fundamental_result.from_cache,
                },
                "technical": {
                    "rating": to_dict(technical_score),
                    "data": to_dict(technical_result.value),
                    "source": to_dict(technical_result.source),
                    "error": technical_result.error,
                    "error_code": technical_result.error_code,
                    "from_cache": technical_result.from_cache,
                },
                "exploratory_indicator": to_dict(indicator),
            }
        return report

    def _fetch(self, kind: str, ticker: str, report: dict[str, Any]) -> FetchResult[dict[str, object]]:
        fallback_source = SourceAttribution("Yahoo Finance", f"https://finance.yahoo.com/quote/{ticker}.JK")
        try:
            if kind == "fundamental":
                result = self.market_data.get_fundamental(ticker)
            else:
                result = self.market_data.get_technical(ticker)
        except Exception as exc:
            result = FetchResult(None, fallback_source, _error_message(exc), "pipeline_source_error")
        if not result.ok:
            report["failures"].append(
                {
                    "stage": kind,
                    "ticker": ticker,
                    "source": to_dict(result.source),
                    "error": result.error,
                    "error_code": result.error_code,
                }
            )
            log_event(
                LOGGER,
                logging.ERROR,
                "pipeline_component_unavailable",
                stage=kind,
                ticker=ticker,
                error_code=result.error_code,
            )
        return result


def _error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]
