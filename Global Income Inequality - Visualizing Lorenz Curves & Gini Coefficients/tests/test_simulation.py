import pandas as pd
import pytest

from global_inequality.simulation import simulate_bottom_subsidy, simulate_progressive_tax


def _distribution() -> pd.DataFrame:
    percentile = list(range(1, 101))
    population = [100.0] * 100
    welfare = pd.Series([float(value) for value in percentile])
    totals = welfare * pd.Series(population)
    return pd.DataFrame(
        {
            "percentile": percentile,
            "pop_share": [0.01] * 100,
            "avg_welfare": welfare,
            "pop": population,
            "welfare_share": totals / totals.sum(),
        }
    )


def test_progressive_tax_is_arithmetically_budget_neutral() -> None:
    original = _distribution()
    result = simulate_progressive_tax(original)
    original_total = (original["avg_welfare"] * original["pop"]).sum()
    simulated_total = (
        result.distribution["avg_welfare"] * result.distribution["pop"]
    ).sum()
    assert simulated_total == pytest.approx(original_total)
    assert result.distribution["welfare_share"].sum() == pytest.approx(1)
    assert result.gini < 1


def test_external_subsidy_increases_total_welfare() -> None:
    original = _distribution()
    result = simulate_bottom_subsidy(original, amount_per_person_per_day=0.5)
    original_total = (original["avg_welfare"] * original["pop"]).sum()
    simulated_total = (
        result.distribution["avg_welfare"] * result.distribution["pop"]
    ).sum()
    assert simulated_total > original_total
    assert "external funding" in result.description

