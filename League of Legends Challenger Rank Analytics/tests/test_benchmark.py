from lol_analytics.benchmark import run_benchmark


def test_benchmark_writes_temporal_and_paired_results(tmp_path) -> None:
    summary = run_benchmark(
        "data/sample/events.jsonl",
        tmp_path,
        bootstrap_iterations=100,
        cv_splits=3,
    )
    assert set(summary["models"]) == {"early_10m", "mid_20m"}
    assert summary["models"]["early_10m"]["temporal_cv"]["n_splits"] == 3
    assert summary["models"]["early_10m"]["temporal_cv"]["pooled_test_matches"] == 9
    assert summary["models"]["mid_20m"]["holdout_roc_auc_interval"]["iterations"] == 100
    assert summary["paired_comparison"]["cohort"] == "matches_eligible_at_20m"
    assert summary["paired_temporal_comparison"]["pooled_test_matches"] == 9
    for name in [
        "benchmark.json",
        "benchmark.md",
        "match_features.parquet",
        "model_10m.joblib",
        "model_20m.joblib",
        "test_predictions_10m.csv",
        "test_predictions_20m.csv",
        "temporal_predictions_10m.csv",
        "temporal_predictions_20m.csv",
        "paired_temporal_predictions.csv",
    ]:
        assert (tmp_path / name).is_file()
