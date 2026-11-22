import pandas as pd
import pytest

from global_inequality.data import prepare_percentiles


def _complete_distribution() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country_code": ["TST"] * 100,
            "year": [2020] * 100,
            "reporting_level": ["national"] * 100,
            "welfare_type": ["income"] * 100,
            "percentile": list(range(1, 101)),
            "avg_welfare": list(range(1, 101)),
            "pop_share": [0.01] * 100,
            "welfare_share": [value / 5050 for value in range(1, 101)],
            "quantile": list(range(1, 101)),
            "pop": [1000] * 100,
        }
    )


def test_complete_distribution_is_selected_and_sorted() -> None:
    data = pd.concat(
        [
            _complete_distribution().sample(frac=1, random_state=4),
            _complete_distribution().assign(reporting_level="urban"),
        ],
        ignore_index=True,
    )
    prepared = prepare_percentiles(data)
    assert len(prepared) == 100
    assert prepared["percentile"].tolist() == list(range(1, 101))
    assert set(prepared["reporting_level"]) == {"national"}


def test_missing_value_fails_instead_of_dropping_a_bin() -> None:
    data = _complete_distribution()
    data.loc[4, "welfare_share"] = None
    with pytest.raises(ValueError, match="missing or non-numeric"):
        prepare_percentiles(data)


def test_missing_unused_quantile_is_allowed() -> None:
    data = _complete_distribution()
    data.loc[99, "quantile"] = None
    assert len(prepare_percentiles(data)) == 100


def test_incomplete_distribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="incomplete percentile distribution"):
        prepare_percentiles(_complete_distribution().iloc[:-1])


def test_duplicate_percentile_is_rejected() -> None:
    data = _complete_distribution()
    data.loc[99, "percentile"] = 99
    with pytest.raises(ValueError, match="duplicate percentile"):
        prepare_percentiles(data)


def test_negative_welfare_share_is_rejected() -> None:
    data = _complete_distribution()
    data.loc[0, "welfare_share"] = -0.1
    with pytest.raises(ValueError, match="negative welfare_share"):
        prepare_percentiles(data)


def test_unordered_average_welfare_is_rejected() -> None:
    data = _complete_distribution()
    data.loc[50, "avg_welfare"] = 1
    with pytest.raises(ValueError, match="avg_welfare is not ordered"):
        prepare_percentiles(data)


def test_incomplete_share_total_is_rejected() -> None:
    data = _complete_distribution()
    data.loc[0, "pop_share"] = 0.02
    with pytest.raises(ValueError, match="pop_share does not sum to one"):
        prepare_percentiles(data)

