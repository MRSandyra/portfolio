import pytest

from lol_analytics.data import read_events
from lol_analytics.external_validation import evaluate_external_features
from lol_analytics.features import build_match_features


def _disjoint_sample_cohorts():
    features = build_match_features(read_events("data/sample/events.jsonl"))
    training = features.iloc[:6].copy()
    external = features.iloc[6:].copy()
    training["region"] = "americas"
    external["region"] = "europe"
    return training, external


def test_external_validation_fits_training_and_writes_external_predictions(tmp_path) -> None:
    training, external = _disjoint_sample_cohorts()
    summary = evaluate_external_features(
        training,
        external,
        tmp_path,
        bootstrap_iterations=100,
    )

    assert summary["match_id_overlap"] == 0
    assert summary["region_overlap"] == []
    assert set(summary["models"]) == {"early_10m", "mid_20m"}
    assert summary["models"]["early_10m"]["training_matches"] == 6
    assert summary["models"]["early_10m"]["external_matches"] == 6
    assert "gold_diff_20" not in summary["models"]["early_10m"]["features"]
    assert "gold_diff_20" in summary["models"]["mid_20m"]["features"]
    assert summary["paired_comparison"]["external_matches"] == 6
    for name in [
        "external_validation.json",
        "external_validation.md",
        "external_model_10m.joblib",
        "external_model_20m.joblib",
        "external_predictions_10m.csv",
        "external_predictions_20m.csv",
        "external_paired_predictions.csv",
    ]:
        assert (tmp_path / name).is_file()


def test_external_validation_rejects_match_overlap(tmp_path) -> None:
    training, external = _disjoint_sample_cohorts()
    external.loc[external.index[0], "matchId"] = training.iloc[0]["matchId"]

    with pytest.raises(ValueError, match="disjoint matchId"):
        evaluate_external_features(
            training,
            external,
            tmp_path,
            bootstrap_iterations=100,
        )


def test_external_region_validation_rejects_region_overlap(tmp_path) -> None:
    training, external = _disjoint_sample_cohorts()
    external["region"] = "americas"

    with pytest.raises(ValueError, match="disjoint region"):
        evaluate_external_features(
            training,
            external,
            tmp_path,
            bootstrap_iterations=100,
        )
