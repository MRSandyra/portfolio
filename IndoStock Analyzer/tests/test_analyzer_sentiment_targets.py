from indostock_analyzer.analyzer import Analyzer
from indostock_analyzer.models import FetchResult, NewsItem, SentimentResult, SourceAttribution


SOURCE = SourceAttribution("Fixture", "https://example.test")


class TwoTickerNews:
    def scrape(self):
        return FetchResult([NewsItem("BBRI dan TLKM mengumumkan kerja sama", SOURCE)], SOURCE)


class NoMarketData:
    def get_fundamental(self, symbol):
        return FetchResult(None, SOURCE, "not in fixture")

    def get_technical(self, symbol):
        return FetchResult(None, SOURCE, "not in fixture")


class TargetAwareSentiment:
    def __init__(self):
        self.targets = []

    def analyze(self, text, language, target=None):
        self.targets.append(target)
        label = "POSITIVE" if target == "BBRI" else "NEGATIVE"
        polarity = 0.8 if target == "BBRI" else -0.8
        return SentimentResult(polarity, None, label, "fixture", language, True, "fixture", target)

    def metadata(self):
        return {"method": "fixture", "validated": True}


def test_analyzer_assesses_multi_ticker_headline_per_target():
    sentiment = TargetAwareSentiment()

    report = Analyzer(TwoTickerNews(), NoMarketData(), sentiment=sentiment).run()

    assert sentiment.targets == ["BBRI", "TLKM"]
    article_sentiment = report["articles"][0]["sentiment_by_ticker"]
    assert article_sentiment["BBRI"]["label"] == "POSITIVE"
    assert article_sentiment["TLKM"]["label"] == "NEGATIVE"
