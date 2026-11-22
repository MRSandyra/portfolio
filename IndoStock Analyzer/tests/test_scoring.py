import pytest

from indostock_analyzer.scoring import clamp_score, combine_scores, score_fundamental, score_technical


def test_missing_data_remains_unavailable_instead_of_neutral():
    assert score_fundamental(None).score is None
    assert score_fundamental({"name": "Example"}).score is None
    assert score_technical(None).score is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-100, 1.0), (1, 1.0), (3.25, 3.25), (5, 5.0), (100, 5.0)],
)
def test_score_boundaries(value, expected):
    assert clamp_score(value) == expected


def test_component_scores_are_bounded_under_extreme_inputs():
    high = score_fundamental(
        {"pe_ratio": 1, "pb_ratio": 0.1, "roe": 1, "debt_to_equity": 0, "earnings_growth": 2}
    )
    low = score_fundamental(
        {"pe_ratio": 100, "pb_ratio": 10, "roe": -1, "debt_to_equity": 10, "earnings_growth": -1}
    )

    assert high.score == 5.0
    assert low.score == 1.0


def test_composite_reweights_available_data_and_reports_coverage():
    result = combine_scores({"sentiment": None, "fundamental": 4.0, "technical": None})

    assert result.score == 4.0
    assert result.coverage == 0.5
    assert result.label == "positive exploratory range"


def test_stale_fundamental_is_not_scored():
    result = score_fundamental(
        {"pe_ratio": 10, "data_quality": {"stale": True, "warnings": ["stale"]}}
    )

    assert result.score is None
    assert result.reasons == ("Fundamental data is stale",)


def test_unverified_technical_freshness_is_not_scored():
    result = score_technical(
        {
            "price_position": 0.5,
            "data_quality": {
                "stale": False,
                "usable_for_scoring": False,
                "warnings": ["timestamp unavailable"],
            },
        }
    )

    assert result.score is None
    assert result.reasons == ("Technical freshness is unverified",)
