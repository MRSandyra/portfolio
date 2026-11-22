import json

from indostock_analyzer.scraper import (
    INDOPREMIER_NEWS_URL,
    detect_network_rate_limit,
    page_reports_rate_limit,
    parse_news_elements,
)


class FakeLink:
    def __init__(self, text, href):
        self.text = text
        self.href = href

    def get_attribute(self, name):
        return self.href if name == "href" else None


class FakeItem:
    def __init__(self, link=None, fails=False):
        self.link = link
        self.fails = fails

    def find_element(self, by, value):
        assert (by, value) == ("tag name", "a")
        if self.fails:
            raise LookupError("missing link")
        return self.link


def test_news_parser_normalizes_deduplicates_and_skips_invalid_items():
    elements = [
        FakeItem(FakeLink("  BBRI   mencatat pertumbuhan laba  ", "https://example.test/1")),
        FakeItem(FakeLink("BBRI mencatat pertumbuhan laba", "https://example.test/2")),
        FakeItem(FakeLink("terlalu pendek", "javascript:alert(1)")),
        FakeItem(fails=True),
        FakeItem(FakeLink("TLKM mengumumkan perubahan manajemen", "javascript:alert(1)")),
    ]

    news = parse_news_elements(elements, max_news=10)

    assert [item.title for item in news] == [
        "BBRI mencatat pertumbuhan laba",
        "TLKM mengumumkan perubahan manajemen",
    ]
    assert news[1].source.url == INDOPREMIER_NEWS_URL


def test_network_log_detects_429_and_retry_after():
    message = {
        "message": {
            "method": "Network.responseReceived",
            "params": {
                "response": {
                    "status": 429,
                    "url": "https://www.indopremier.com/news",
                    "headers": {"Retry-After": "7"},
                }
            },
        }
    }
    signal = detect_network_rate_limit([{"message": json.dumps(message)}])

    assert signal.found is True
    assert signal.retry_after == 7.0
    assert signal.url == "https://www.indopremier.com/news"


def test_malformed_browser_logs_are_ignored():
    assert detect_network_rate_limit([{"message": "bad-json"}, None]).found is False
    assert page_reports_rate_limit("429 Too Many Requests") is True
