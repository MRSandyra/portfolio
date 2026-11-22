import json

import joblib
import pytest

from lol_analytics.config import features_available_at
from lol_analytics.final_model import fit_final_models, select_modal_regularization_c


def _evidence(*, future_accessed: bool = False) -> dict[str, object]:
    models = {}
    for horizon, key, counts in (
        (10, "early_10m", {"0.01": 4, "0.1": 2}),
        (20, "mid_20m", {"0.1": 4, "1.0": 2}),
    ):
        models[key] = {
            "eligible_matches": 12,
            "features": features_available_at(horizon * 60_000),
            "leakage_audit": {"passed": True},
            "repeated_nested_validation": {"selected_c_counts": counts},
        }
    return {
        "input": "data/sample/events.jsonl",
        "matches": 12,
        "validation_design": (
            "repeated_nested_temporal_with_block_bootstrap_refitting"
        ),
        "frozen_future_holdout_accessed": future_accessed,
        "configuration": {"repeats": 3, "outer_splits": 2, "inner_splits": 2},
        "models": models,
    }


def test_modal_policy_uses_smaller_c_for_ties() -> None:
    assert select_modal_regularization_c({"1.0": 3, "0.1": 3, "0.01": 1}) == 0.1
    with pytest.raises(ValueError, match="positive"):
        select_modal_regularization_c({"0": 1})


def test_final_models_fit_without_performance_or_future_holdout(tmp_path) -> None:
    evidence_path = tmp_path / "robustness.json"
    evidence_path.write_text(json.dumps(_evidence()), encoding="utf-8")
    output = tmp_path / "models"
    manifest_path = tmp_path / "final-models.json"

    manifest = fit_final_models(
        "data/sample/events.jsonl",
        evidence_path,
        output,
        manifest_path,
    )
    assert manifest["status"] == "final_models_fitted_unreleased"
    assert manifest["models"]["early_10m"]["selected_regularization_c"] == 0.01
    assert manifest["models"]["mid_20m"]["selected_regularization_c"] == 0.1
    assert manifest["evaluation_policy"]["in_sample_performance_computed"] is False
    assert manifest["evaluation_policy"]["frozen_future_holdout_accessed"] is False
    assert "roc_auc" not in json.dumps(manifest)
    for horizon in (10, 20):
        artifact = output / f"final_model_{horizon}m.joblib"
        assert artifact.is_file()
        bundle = joblib.load(artifact)
        assert bundle["horizon_minutes"] == horizon

    assert fit_final_models(
        "data/sample/events.jsonl",
        evidence_path,
        output,
        manifest_path,
    ) == manifest


def test_final_model_rejects_future_contaminated_selection_evidence(tmp_path) -> None:
    evidence_path = tmp_path / "contaminated.json"
    evidence_path.write_text(
        json.dumps(_evidence(future_accessed=True)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must not access"):
        fit_final_models(
            "data/sample/events.jsonl",
            evidence_path,
            tmp_path / "models",
            tmp_path / "manifest.json",
        )
