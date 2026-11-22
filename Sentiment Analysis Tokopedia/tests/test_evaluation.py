import pytest

from tokopedia_sentiment.evaluation import evaluate_probabilities, expected_calibration_error


def test_evaluation_contains_per_class_confusion_and_calibration_metrics() -> None:
    report = evaluate_probabilities([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert set(report["per_class"]) == {"negative", "positive"}
    assert report["confusion_matrix"] == [[1, 1], [1, 1]]
    assert set(report["probability_quality"]) == {
        "brier_score",
        "log_loss",
        "expected_calibration_error_10_bins",
    }
    assert report["accuracy"] == pytest.approx(0.5)


def test_perfect_probabilities_have_small_calibration_error() -> None:
    error = expected_calibration_error([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
    assert error == pytest.approx(0.0)
