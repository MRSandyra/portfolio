import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from lol_analytics.config import TEN_MINUTES_MS, features_available_at
from lol_analytics.modeling import fit_model
from lol_analytics.quality import sha256_file
from lol_analytics.rank_era_validation import (
    UNAVAILABLE_FEATURES,
    adapt_high_diamond_10m,
    validate_rank_era,
)


def _source_frame() -> pd.DataFrame:
    rows = []
    for index, win in enumerate((0, 1, 0, 1, 0, 1)):
        direction = 1 if win else -1
        blue_kills = 5 + index + int(win)
        red_kills = 5 + index + int(not win)
        blue_dragons = int(win)
        red_dragons = int(not win)
        blue_heralds = int(index % 3 == 0)
        red_heralds = int(index % 3 == 1)
        blue_gold = 15_000 + direction * (300 + index * 10)
        red_gold = 15_000 - direction * (300 + index * 10)
        blue_xp = 17_000 + direction * (100 + index)
        red_xp = 17_000 - direction * (100 + index)
        rows.append(
            {
                "gameId": 4_500_000_000 + index,
                "blueWins": win,
                "blueFirstBlood": win,
                "redFirstBlood": 1 - win,
                "blueKills": blue_kills,
                "redKills": red_kills,
                "blueDeaths": red_kills,
                "redDeaths": blue_kills,
                "blueEliteMonsters": blue_dragons + blue_heralds,
                "redEliteMonsters": red_dragons + red_heralds,
                "blueDragons": blue_dragons,
                "redDragons": red_dragons,
                "blueHeralds": blue_heralds,
                "redHeralds": red_heralds,
                "blueTowersDestroyed": int(win),
                "redTowersDestroyed": int(not win),
                "blueTotalGold": blue_gold,
                "redTotalGold": red_gold,
                "blueAvgLevel": 6.8 + direction * 0.1,
                "redAvgLevel": 6.8 - direction * 0.1,
                "blueTotalExperience": blue_xp,
                "redTotalExperience": red_xp,
                "blueTotalMinionsKilled": 200 + direction * 5,
                "redTotalMinionsKilled": 200 - direction * 5,
                "blueTotalJungleMinionsKilled": 45 + direction * 2,
                "redTotalJungleMinionsKilled": 45 - direction * 2,
                "blueGoldDiff": blue_gold - red_gold,
                "redGoldDiff": red_gold - blue_gold,
                "blueExperienceDiff": blue_xp - red_xp,
                "redExperienceDiff": red_xp - blue_xp,
            }
        )
    return pd.DataFrame(rows)


def test_adapter_maps_team_differences_and_marks_unavailable_features() -> None:
    adapted = adapt_high_diamond_10m(_source_frame())
    first = adapted.iloc[0]

    assert first["gold_diff_10"] == -600
    assert first["xp_diff_10"] == -200
    assert first["cs_diff_10"] == -14
    assert first["jungle_cs_diff_10"] == -4
    assert first["level_diff_10"] == pytest.approx(-1.0)
    assert first["first_blood_side_10"] == -1
    assert first["baron_diff_10"] == 0
    assert adapted[list(UNAVAILABLE_FEATURES)].isna().all().all()
    assert adapted.columns.tolist() == [
        "matchId",
        "win",
        *features_available_at(TEN_MINUTES_MS),
    ]


def test_adapter_rejects_missing_or_inconsistent_schema() -> None:
    source = _source_frame().drop(columns="blueGoldDiff")
    with pytest.raises(ValueError, match="missing required columns"):
        adapt_high_diamond_10m(source)

    source = _source_frame()
    source.loc[0, "blueGoldDiff"] += 1
    with pytest.raises(ValueError, match="inconsistent with team totals"):
        adapt_high_diamond_10m(source)


def _write_validation_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = _source_frame()
    csv_path = tmp_path / "external.csv"
    source.to_csv(csv_path, index=False)
    source_manifest_path = tmp_path / "source.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "ref": "test/high-diamond",
                    "rank_scope": "Diamond I to Master solo queue",
                    "snapshot_horizon_minutes": 10,
                    "last_updated_utc": "2020-04-13T13:53:02.740000+00:00",
                },
                "file": {"sha256": sha256_file(csv_path), "rows": len(source)},
            }
        ),
        encoding="utf-8",
    )

    features = features_available_at(TEN_MINUTES_MS)
    training = adapt_high_diamond_10m(source)
    training[list(UNAVAILABLE_FEATURES)] = 0
    model = fit_model(training, features, regularization_c=0.01)
    artifact_path = tmp_path / "final_model_10m.joblib"
    joblib.dump(
        {
            "model": model,
            "features": features,
            "horizon_minutes": 10,
            "regularization_c": 0.01,
            "training_sha256": "training-hash",
        },
        artifact_path,
    )
    final_manifest_path = tmp_path / "final.json"
    final_manifest_path.write_text(
        json.dumps(
            {
                "status": "final_models_fitted_unreleased",
                "training": {"file": {"sha256": "training-hash"}},
                "evaluation_policy": {
                    "frozen_future_holdout_accessed": False,
                    "frozen_future_holdout_scored": False,
                },
                "models": {
                    "early_10m": {
                        "horizon_minutes": 10,
                        "features": features,
                        "artifact": {
                            "path": artifact_path.as_posix(),
                            "sha256": sha256_file(artifact_path),
                        },
                    },
                    "mid_20m": {"artifact": {"path": "must-not-be-loaded.joblib"}},
                },
            }
        ),
        encoding="utf-8",
    )
    return csv_path, source_manifest_path, final_manifest_path


def test_validation_scores_frozen_early_model_without_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, source_manifest, final_manifest = _write_validation_inputs(tmp_path)

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("External validation must not fit a model")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    result = validate_rank_era(
        csv_path,
        source_manifest,
        final_manifest,
        tmp_path / "output",
        bootstrap_iterations=100,
    )

    assert result["validation_design"] == "external_rank_and_historical_era_10m_only"
    assert result["early_10m"]["leakage_audit"]["passed"] is True
    assert result["evaluation_guards"]["model_refit"] is False
    assert result["evaluation_guards"]["mid_20m_model_loaded"] is False
    assert result["evaluation_guards"]["frozen_future_holdout_accessed"] is False
    assert 0 <= result["early_10m"]["metrics"]["roc_auc"] <= 1
    assert (tmp_path / "output" / "external_predictions_10m.csv").is_file()


def test_validation_rejects_changed_model_artifact(tmp_path: Path) -> None:
    csv_path, source_manifest, final_manifest = _write_validation_inputs(tmp_path)
    manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
    artifact = Path(manifest["models"]["early_10m"]["artifact"]["path"])
    artifact.write_bytes(artifact.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="checksum does not match"):
        validate_rank_era(
            csv_path,
            source_manifest,
            final_manifest,
            tmp_path / "output",
            bootstrap_iterations=100,
        )
