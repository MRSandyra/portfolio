import numpy as np
import pytest

from lol_analytics.metrics import classification_metrics


def test_metrics_match_known_binary_example() -> None:
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == pytest.approx(0.75)


def test_metrics_reject_single_class_target() -> None:
    with pytest.raises(ValueError, match="both target classes"):
        classification_metrics(np.array([1, 1]), np.array([0.7, 0.8]))

