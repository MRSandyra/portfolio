import pytest

from lol_analytics.data import read_events
from lol_analytics.features import build_match_features
from lol_analytics.shift_validation import evaluate_shift_features


def _shift_cohorts():
    features = build_match_features(read_events("data/sample/events.jsonl"))
    training = features.iloc[:4].copy()
    same = features.iloc[4:8].copy()
    cross = features.iloc[8:].copy()
    training["region"] = "americas"
    same["region"] = "americas"
    cross["region"] = "europe"
    shared_times = training["game_start_timestamp_ms"].to_numpy()
    same["game_start_timestamp_ms"] = shared_times
    cross["game_start_timestamp_ms"] = shared_times
    return training, same, cross


def test_shift_validation_fits_once_per_horizon_and_writes_outputs(
    tmp_path, monkeypatch
) -> None:
    training, same, cross = _shift_cohorts()
    import lol_analytics.shift_validation as module

    original_fit = module.fit_model
    fit_calls = []

    def tracked_fit(*args, **kwargs):
        fit_calls.append(1)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(module, "fit_model", tracked_fit)
    summary = evaluate_shift_features(
        training,
        same,
        cross,
        tmp_path,
        bootstrap_iterations=100,
    )

    assert len(fit_calls) == 2
    assert summary["match_id_overlap"] == 0
    assert summary["controlled_support"]["shared_patches"]
    assert set(summary["models"]) == {"early_10m", "mid_20m"}
    assert all(model["fit_count"] == 1 for model in summary["models"].values())
    for name in [
        "shift_validation.json",
        "shift_validation.md",
        "shift_model_10m.joblib",
        "shift_model_20m.joblib",
        "same_region_predictions_10m.csv",
        "same_region_predictions_20m.csv",
        "cross_region_predictions_10m.csv",
        "cross_region_predictions_20m.csv",
    ]:
        assert (tmp_path / name).is_file()


def test_shift_validation_rejects_wrong_region_roles(tmp_path) -> None:
    training, same, cross = _shift_cohorts()
    same["region"] = "europe"
    with pytest.raises(ValueError, match="same_region cohort"):
        evaluate_shift_features(
            training, same, cross, tmp_path, bootstrap_iterations=100
        )


def test_shift_validation_rejects_overlapping_match_ids(tmp_path) -> None:
    training, same, cross = _shift_cohorts()
    same.loc[same.index[0], "matchId"] = training.iloc[0]["matchId"]
    with pytest.raises(ValueError, match="disjoint matchId"):
        evaluate_shift_features(
            training, same, cross, tmp_path, bootstrap_iterations=100
        )


def test_shift_validation_rejects_nonoverlapping_patch_support(tmp_path) -> None:
    training, same, cross = _shift_cohorts()
    cross["patch"] = "99.9"
    with pytest.raises(ValueError, match="shared patch support"):
        evaluate_shift_features(
            training, same, cross, tmp_path, bootstrap_iterations=100
        )
