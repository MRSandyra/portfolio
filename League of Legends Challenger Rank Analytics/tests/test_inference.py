import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from lol_analytics.config import TEN_MINUTES_MS, features_available_at
from lol_analytics.inference import (
    assemble_inference_bundle,
    build_inference_bundles,
    load_inference_bundle,
    predict_inference_bundle,
)
from lol_analytics.modeling import fit_model
from lol_analytics.monitoring import build_model_input_contracts


POLICY_PATH = Path("data/manifests/cohort-policy-v1.json")


def _early_frame(rows: int = 8) -> pd.DataFrame:
    features = features_available_at(TEN_MINUTES_MS)
    frame = pd.DataFrame(
        {
            feature: np.linspace(-2, 2, rows)
            for feature in features
        }
    )
    for feature in features:
        if feature.startswith("first_"):
            frame[feature] = np.resize(np.array([-1, 0, 1]), rows)
    frame.insert(0, "win", np.resize(np.array([0, 1]), rows))
    frame.insert(0, "matchId", [f"match-{index}" for index in range(rows)])
    return frame


def _identity_bundle() -> dict[str, object]:
    frame = _early_frame()
    features = features_available_at(TEN_MINUTES_MS)
    model = fit_model(frame, features, regularization_c=0.01)
    return assemble_inference_bundle(
        {
            "model": model,
            "features": features,
            "horizon_minutes": 10,
            "regularization_c": 0.01,
            "fit_cohort": "test-development",
            "training_sha256": "training",
            "selection_evidence_sha256": "selection",
        },
        build_model_input_contracts()["models"]["early_10m"],
        {
            "kind": "identity_keep_base_probability",
            "calibration_applied": False,
            "decision": "rejected_keep_base_probability",
        },
    )


def test_identity_bundle_serves_exact_base_probability_without_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _identity_bundle()

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("Inference must not fit a model")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    predictions = predict_inference_bundle(bundle, _early_frame())

    np.testing.assert_array_equal(
        predictions["raw_probability"], predictions["served_probability"]
    )
    assert predictions["calibration_applied"].eq(False).all()  # noqa: E712
    assert predictions["horizon_minutes"].eq(10).all()
    assert predictions["probability_transform"].eq(
        "identity_keep_base_probability"
    ).all()


def test_inference_rejects_cross_horizon_input() -> None:
    frame = _early_frame().assign(gold_diff_20=0.0)
    with pytest.raises(ValueError, match="another horizon"):
        predict_inference_bundle(_identity_bundle(), frame)


def test_bundle_assembly_rejects_inconsistent_transform() -> None:
    bundle = _identity_bundle()
    with pytest.raises(ValueError, match="must not include a calibrator"):
        assemble_inference_bundle(
            bundle,
            bundle["runtime_contract"],
            bundle["probability_transform"],
            calibrator=object(),
        )


def test_loading_bundle_verifies_checksum(tmp_path: Path) -> None:
    artifact = tmp_path / "bundle.joblib"
    joblib.dump(_identity_bundle(), artifact)
    with pytest.raises(ValueError, match="checksum differs"):
        load_inference_bundle(artifact, expected_sha256="0" * 64)


def test_phase_three_rejects_diagnostic_development_before_other_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="does not allow fit_base_model"):
        build_inference_bundles(
            "data/raw/events_asia_train_105.parquet",
            POLICY_PATH,
            tmp_path / "unused-final.json",
            tmp_path / "unused-calibration.json",
            tmp_path / "unused-contract.json",
            tmp_path / "output",
            tmp_path / "manifest.json",
        )


def test_committed_phase_three_manifest_preserves_base_probability() -> None:
    manifest = json.loads(
        Path("data/manifests/inference-bundles-v1.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "inference_bundles_base_probability_unreleased"
    assert manifest["evaluation_guards"] == {
        "calibrator_refit": False,
        "external_diagnostic_cohort_read": False,
        "future_holdout_accessed": False,
        "future_holdout_scored": False,
        "model_refit": False,
    }
    for key in ("early_10m", "mid_20m"):
        record = manifest["bundles"][key]
        assert record["probability_transform"]["kind"] == (
            "identity_keep_base_probability"
        )
        assert record["probability_transform"]["calibration_applied"] is False
        assert not Path(record["source_model_artifact"]["path"]).is_absolute()
