import pandas as pd
import pytest

from lol_analytics.config import TEN_MINUTES_MS
from lol_analytics.leakage import audit_feature_set
from lol_analytics.modeling import evaluate_model, fit_model, train_and_evaluate


def test_future_and_outcome_columns_fail_leakage_audit() -> None:
    future = audit_feature_set(["gold_diff_10", "gold_diff_20"], TEN_MINUTES_MS)
    assert future.future_features == ("gold_diff_20",)
    assert not future.passed
    outcome = audit_feature_set(["gold_diff_10", "win"], TEN_MINUTES_MS)
    assert outcome.forbidden_columns == ("win",)
    assert not outcome.passed


def test_scaler_is_fitted_on_train_only() -> None:
    train = pd.DataFrame({"matchId": ["a", "b", "c", "d"], "match_order": range(4), "win": [0, 1, 0, 1], "gold_diff_10": [0.0, 2.0, 4.0, 6.0]})
    test = pd.DataFrame({"matchId": ["e", "f"], "match_order": [4, 5], "win": [0, 1], "gold_diff_10": [1_000.0, 2_000.0]})
    result = train_and_evaluate(train, test, ["gold_diff_10"])
    assert result.model.named_steps["scaler"].mean_[0] == pytest.approx(3.0)


def test_evaluating_multiple_cohorts_does_not_refit_preprocessing() -> None:
    train = pd.DataFrame(
        {"win": [0, 1, 0, 1], "gold_diff_10": [0.0, 2.0, 4.0, 6.0]}
    )
    first = pd.DataFrame(
        {"win": [0, 1], "gold_diff_10": [100.0, 200.0]}
    )
    second = pd.DataFrame(
        {"win": [0, 1], "gold_diff_10": [1_000.0, 2_000.0]}
    )
    model = fit_model(train, ["gold_diff_10"])
    training_mean = model.named_steps["scaler"].mean_[0]

    evaluate_model(model, first, ["gold_diff_10"])
    evaluate_model(model, second, ["gold_diff_10"])

    assert model.named_steps["scaler"].mean_[0] == pytest.approx(training_mean)

