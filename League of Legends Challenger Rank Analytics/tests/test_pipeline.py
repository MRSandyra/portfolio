from lol_analytics.pipeline import run_case_study


def test_end_to_end_sample_reproduction(tmp_path) -> None:
    summary = run_case_study("data/sample/events.jsonl", tmp_path)
    assert summary["matches"] == 12
    assert set(summary["models"]) == {"early_10m", "mid_20m"}
    assert summary["models"]["early_10m"]["leakage_audit"]["passed"] is True
    assert summary["models"]["mid_20m"]["leakage_audit"]["passed"] is True
    assert summary["models"]["early_10m"]["eligible_matches"] == 12
    assert summary["models"]["mid_20m"]["eligible_matches"] == 12
    assert summary["models"]["early_10m"]["split"]["timezone"] == "UTC"
    assert "gold_diff_20" not in summary["models"]["early_10m"]["features"]
    assert "gold_diff_20" in summary["models"]["mid_20m"]["features"]
    for name in ["match_features.parquet", "test_predictions_10m.csv", "test_predictions_20m.csv", "model_10m.joblib", "model_20m.joblib", "run.json", "report.md"]:
        assert (tmp_path / name).is_file()

