from indostock_analyzer.tickers import TickerExtractor


def test_extracts_supported_ticker_forms_in_first_seen_order():
    extractor = TickerExtractor({"BBRI", "TLKM", "BBCA"})

    assert extractor.extract("BBRI naik; $TLKM mengikuti BBRI dan BBCA.JK stabil") == [
        "BBRI",
        "TLKM",
        "BBCA",
    ]


def test_does_not_treat_arbitrary_four_letter_words_as_tickers():
    extractor = TickerExtractor({"BBRI"})

    assert extractor.extract("DARI ASIA untuk BBRI") == ["BBRI"]
