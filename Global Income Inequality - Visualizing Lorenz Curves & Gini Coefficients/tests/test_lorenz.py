import numpy as np
import pandas as pd
import pytest

from global_inequality.lorenz import gini_coefficient, gini_from_group, lorenz_curve


def test_lorenz_endpoints_and_monotonicity() -> None:
    population, welfare = lorenz_curve([0.2, 0.3, 0.5], [0.05, 0.25, 0.70])
    assert population[0] == welfare[0] == 0
    assert population[-1] == welfare[-1] == 1
    assert np.all(np.diff(population) > 0)
    assert np.all(np.diff(welfare) >= 0)
    assert np.all(welfare <= population)


def test_perfect_equality_has_zero_gini() -> None:
    assert gini_coefficient([0.1, 0.2, 0.7], [0.1, 0.2, 0.7]) == pytest.approx(0)


def test_known_equal_population_distribution() -> None:
    # For observations [1, 2, 3, 4], the exact finite-sample Gini is 0.25.
    assert gini_coefficient([1, 1, 1, 1], [1, 2, 3, 4]) == pytest.approx(0.25)


def test_grouped_maximum_concentration() -> None:
    # With four equal bins and all welfare in the last bin, grouped Gini is 0.75.
    assert gini_coefficient([1, 1, 1, 1], [0, 0, 0, 1]) == pytest.approx(0.75)


def test_scale_invariance() -> None:
    expected = gini_coefficient([0.25] * 4, [0.05, 0.15, 0.30, 0.50])
    actual = gini_coefficient([25] * 4, [5, 15, 30, 50])
    assert actual == pytest.approx(expected)


def test_replication_invariance_for_adjacent_sub_bins() -> None:
    population = np.array([0.2, 0.3, 0.5])
    welfare = np.array([0.05, 0.25, 0.70])
    assert gini_coefficient(
        np.repeat(population / 2, 2), np.repeat(welfare / 2, 2)
    ) == pytest.approx(gini_coefficient(population, welfare))


def test_gini_matches_one_minus_twice_lorenz_area() -> None:
    population, welfare = lorenz_curve([0.2, 0.3, 0.5], [0.05, 0.25, 0.70])
    assert gini_coefficient([0.2, 0.3, 0.5], [0.05, 0.25, 0.70]) == pytest.approx(
        1 - 2 * np.trapezoid(welfare, population)
    )


@pytest.mark.parametrize(
    ("population", "welfare"),
    [
        ([0.5], [1.0]),
        ([0.5, 0.5], [1.0]),
        ([0.5, -0.5], [0.5, 0.5]),
        ([0.5, 0.5], [-0.1, 1.1]),
        ([0.5, np.nan], [0.5, 0.5]),
    ],
)
def test_invalid_shares_are_rejected(population: list[float], welfare: list[float]) -> None:
    with pytest.raises(ValueError):
        gini_coefficient(population, welfare)


def test_dataframe_helper_orders_percentiles_without_mutating_input() -> None:
    group = pd.DataFrame(
        {
            "percentile": [3, 1, 2],
            "pop_share": [1, 1, 1],
            "welfare_share": [3, 1, 2],
        }
    )
    original = group.copy(deep=True)
    assert gini_from_group(group) == pytest.approx(2 / 9)
    pd.testing.assert_frame_equal(group, original)

