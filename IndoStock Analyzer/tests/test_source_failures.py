from indostock_analyzer.analyzer import Analyzer
from indostock_analyzer.models import FetchResult, NewsItem, SourceAttribution


NEWS_SOURCE = SourceAttribution("Example News", "https://example.test/news")
MARKET_SOURCE = SourceAttribution("Example Market", "https://example.test/market")


class FailedNewsSource:
    def scrape(self):
        return FetchResult(None, NEWS_SOURCE, "timeout")


class NewsSourceWithTicker:
    def scrape(self):
        return FetchResult([NewsItem("BBRI mencatat perubahan laba kuartalan", NEWS_SOURCE)], NEWS_SOURCE)


class FailedMarketSource:
    def get_fundamental(self, symbol):
        return FetchResult(None, MARKET_SOURCE, "HTTP 429")

    def get_technical(self, symbol):
        raise TimeoutError("upstream timed out")


def test_news_source_failure_is_reported_without_crashing():
    report = Analyzer(FailedNewsSource(), FailedMarketSource()).run()

    assert report["stocks"] == {}
    assert report["failures"][0]["stage"] == "news"
    assert report["failures"][0]["error"] == "timeout"


def test_market_source_failures_leave_indicator_unavailable():
    report = Analyzer(NewsSourceWithTicker(), FailedMarketSource()).run()
    indicator = report["stocks"]["BBRI"]["exploratory_indicator"]

    assert indicator["score"] is None
    assert indicator["coverage"] == 0.0
    assert {failure["stage"] for failure in report["failures"]} == {"fundamental", "technical"}
