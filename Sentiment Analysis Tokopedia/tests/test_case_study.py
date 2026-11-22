from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tokopedia_sentiment.active_learning import create_active_learning_batch
from tokopedia_sentiment.annotation import annotation_key_path, create_annotation_batch
from tokopedia_sentiment.annotation_review import (
    analyze_annotation_agreement,
    build_adjudicated_gold_set,
)
from tokopedia_sentiment.case_study_data import (
    group_split,
    prepare_case_study_reviews,
    temporal_split,
)
from tokopedia_sentiment.case_study_training import (
    CaseStudyTrainingConfig,
    train_case_study,
)
from tokopedia_sentiment.decision import (
    predict_with_thresholds,
    select_minority_thresholds,
)
from tokopedia_sentiment.inference import predict_reviews
from tokopedia_sentiment.modeling import load_artifact
from tokopedia_sentiment.monitoring import compute_drift_report
from tokopedia_sentiment.transformer_experiment import stratified_transformer_training_sample
from tokopedia_sentiment.triage import (
    apply_triage_policy,
    evaluate_triage,
    select_triage_policy,
)
from tokopedia_sentiment.two_stage import fit_two_stage_model


def _three_class_frame(rows: int = 180) -> pd.DataFrame:
    labels = ["negative", "neutral", "positive"]
    ratings = [1, 3, 5]
    cues = ["buruk kecewa", "biasa standar", "bagus puas"]
    start = date(2024, 1, 1)
    records = []
    for index in range(rows):
        class_id = index % 3
        records.append(
            {
                "review_id": index,
                "review_text": f"{cues[class_id]} produk kode {index}",
                "rating": ratings[class_id],
                "sentiment_label": labels[class_id],
                "review_date": (start + timedelta(days=index)).isoformat(),
                "product_id": index % 10,
                "product_category": f"category-{index % 2}",
            }
        )
    return pd.DataFrame(records)


def test_temporal_and_group_splits_prevent_expected_leakage() -> None:
    prepared, _ = prepare_case_study_reviews(_three_class_frame())
    temporal = temporal_split(prepared)
    grouped = group_split(prepared)

    assert temporal.train["date"].max() < temporal.validation["date"].min()
    assert temporal.validation["date"].max() < temporal.test["date"].min()
    assert set(grouped.train["group"]).isdisjoint(grouped.validation["group"])
    assert set(grouped.train["group"]).isdisjoint(grouped.test["group"])
    assert set(grouped.validation["group"]).isdisjoint(grouped.test["group"])


def test_threshold_selection_targets_minority_recall() -> None:
    actual = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    probability = np.array(
        [
            [0.45, 0.05, 0.50],
            [0.40, 0.10, 0.50],
            [0.35, 0.10, 0.55],
            [0.05, 0.45, 0.50],
            [0.10, 0.40, 0.50],
            [0.10, 0.35, 0.55],
            [0.05, 0.05, 0.90],
            [0.10, 0.10, 0.80],
            [0.10, 0.15, 0.75],
            [0.15, 0.10, 0.75],
            [0.05, 0.20, 0.75],
            [0.20, 0.05, 0.75],
        ]
    )
    selection = select_minority_thresholds(
        actual, probability, ["negative", "neutral", "positive"], target_recall=0.7
    )
    predicted = predict_with_thresholds(
        probability,
        ["negative", "neutral", "positive"],
        selection.thresholds,
        selection.majority_class,
    )

    assert selection.target_met is True
    assert (predicted[:3] == 0).mean() >= 0.7
    assert (predicted[3:6] == 1).mean() >= 0.7


def test_inference_discloses_when_threshold_differs_from_argmax() -> None:
    class DummyModel:
        def predict_proba(self, reviews):
            return np.array([[0.01, 0.05, 0.94] for _ in reviews])

        def predict(self, reviews):
            return np.array([2 for _ in reviews])

    prediction = predict_reviews(
        DummyModel(),
        ["produk biasa saja"],
        probabilities_calibrated=True,
        class_names=["negative", "neutral", "positive"],
        thresholds={"negative": 0.03, "neutral": 0.02},
        majority_class="positive",
    )[0]

    assert prediction.sentiment == "neutral"
    assert prediction.argmax_sentiment == "positive"
    assert prediction.confidence == 0.05


def test_triage_policy_uses_precision_constraints_and_abstention() -> None:
    actual = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    probability = np.array(
        [
            [0.80, 0.05, 0.15],
            [0.70, 0.10, 0.20],
            [0.05, 0.75, 0.20],
            [0.10, 0.65, 0.25],
            [0.05, 0.05, 0.90],
            [0.10, 0.05, 0.85],
            [0.20, 0.10, 0.70],
            [0.20, 0.20, 0.60],
        ]
    )
    policy = select_triage_policy(
        actual,
        probability,
        ["negative", "neutral", "positive"],
        minimum_precision={"negative": 0.75, "neutral": 0.75},
        max_queue_fraction=0.30,
        positive_minimum_precision=0.99,
    )
    decisions = apply_triage_policy(probability, ["negative", "neutral", "positive"], policy)
    metrics = evaluate_triage(actual, decisions, ["negative", "neutral", "positive"])

    assert metrics["queues"]["negative"]["precision"] >= 0.75
    assert metrics["queues"]["neutral"]["precision"] >= 0.75
    assert set(decisions).issubset(
        {"negative_candidate", "neutral_candidate", "positive", "abstain"}
    )


def test_two_stage_model_produces_three_class_probabilities() -> None:
    prepared, _ = prepare_case_study_reviews(_three_class_frame(90))
    model = fit_two_stage_model(
        prepared["text"].tolist(), prepared["label"].tolist(), parameter=0.5
    )
    probability = model.predict_proba(["produk buruk", "produk biasa", "produk bagus"])

    assert probability.shape == (3, 3)
    assert probability.sum(axis=1) == pytest.approx(np.ones(3))


def test_drift_report_detects_category_and_vocabulary_change() -> None:
    prepared, _ = prepare_case_study_reviews(_three_class_frame())
    reference = prepared.iloc[:90].copy()
    current = prepared.iloc[90:].copy()
    current["category"] = "new-category"
    current["text"] = current["text"] + " kosakatabaru"

    report = compute_drift_report(reference, current)

    assert report["categorical"]["category"]["jensen_shannon_divergence"] > 0.1
    assert report["alerts"]["category_shift"] is True
    assert report["text"]["current_oov_token_rate"] > 0


def test_active_learning_writes_blind_batch_and_separate_key(tmp_path) -> None:
    prepared, _ = prepare_case_study_reviews(_three_class_frame())

    class DummyModel:
        def predict_proba(self, text):
            index = np.arange(len(text), dtype=float)
            negative = 0.2 + 0.1 * (index % 3 == 0)
            neutral = 0.2 + 0.1 * (index % 3 == 1)
            positive = 1.0 - negative - neutral
            return np.column_stack([negative, neutral, positive])

    metadata = {
        "class_names": ["negative", "neutral", "positive"],
        "decision_policy": {
            "thresholds": {"negative": 0.2, "neutral": 0.2},
            "majority_class": "positive",
        },
    }
    output, key_path = create_active_learning_batch(
        prepared,
        DummyModel(),
        metadata,
        tmp_path / "active.csv",
        batch_size=30,
        exclude_review_ids={0, 1, 2},
    )
    blind = pd.read_csv(output, keep_default_na=False)
    key = pd.read_csv(key_path)

    assert len(blind) == len(key) == 30
    assert "probability_negative" not in blind
    assert "probability_negative" in key
    assert not set(blind["review_id"]) & {0, 1, 2}


def test_transformer_sampling_keeps_minority_and_caps_only_positive() -> None:
    prepared, _ = prepare_case_study_reviews(_three_class_frame())
    sampled = stratified_transformer_training_sample(
        prepared, positive_to_minority_ratio=0.5, seed=42
    )

    assert (sampled["label"] == 0).sum() == (prepared["label"] == 0).sum()
    assert (sampled["label"] == 1).sum() == (prepared["label"] == 1).sum()
    assert (sampled["label"] == 2).sum() == 60


def test_annotation_agreement_and_gold_set_workflow(tmp_path) -> None:
    source = create_annotation_batch(
        _three_class_frame(), tmp_path / "source.csv", samples_per_rating=1
    )
    first = pd.read_csv(source, keep_default_na=False)
    second = first.copy()
    labels = [
        "negative" if "buruk" in text else "neutral" if "biasa" in text else "positive"
        for text in first["review_text"]
    ]
    first["human_label"] = labels
    second["human_label"] = labels
    second.loc[0, "human_label"] = "neutral" if labels[0] != "neutral" else "negative"
    first_path = tmp_path / "a.csv"
    second_path = tmp_path / "b.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)

    disagreement_path = tmp_path / "disagreements.csv"
    report = analyze_annotation_agreement(
        first_path,
        second_path,
        report_path=tmp_path / "agreement.json",
        disagreement_path=disagreement_path,
    )
    disagreements = pd.read_csv(disagreement_path, keep_default_na=False)
    disagreements["adjudicated_label"] = labels[0]
    disagreements.to_csv(disagreement_path, index=False)
    gold_path = build_adjudicated_gold_set(
        first_path, second_path, disagreement_path, tmp_path / "gold.csv"
    )
    gold = pd.read_csv(gold_path)

    assert report["agreements"] == 2
    assert report["disagreements"] == 1
    assert len(gold) == 3
    assert set(gold["label_source"]) == {"annotator_agreement", "adjudicated"}


def test_annotation_batch_is_stratified_and_has_blank_human_labels(tmp_path) -> None:
    frame = _three_class_frame()
    output = create_annotation_batch(frame, tmp_path / "batch.csv", samples_per_rating=5)
    batch = pd.read_csv(output, keep_default_na=False)
    key = pd.read_csv(annotation_key_path(output), keep_default_na=False)

    assert key["rating"].value_counts().to_dict() == {1: 5, 3: 5, 5: 5}
    assert batch["human_label"].eq("").all()
    assert "rating" not in batch
    assert "proxy_label" not in batch
    assert "product_id" not in batch


def test_case_study_training_writes_advanced_report_and_artifact(tmp_path) -> None:
    data_path = tmp_path / "case.csv"
    _three_class_frame().to_csv(data_path, index=False)
    artifact_path = tmp_path / "model.joblib"
    report_directory = tmp_path / "reports"

    report = train_case_study(
        CaseStudyTrainingConfig(
            data_path=data_path,
            artifact_path=artifact_path,
            report_directory=report_directory,
            source_uri="synthetic://pytest",
            bootstrap_iterations=20,
            cv_folds=3,
            cv_repeats=2,
            candidates=(("count_multinomial_nb", 0.1),),
            run_group_holdout=False,
            compare_two_stage=False,
            run_rolling_backtest=False,
        )
    )

    assert report["primary_split"]["strategy"] == "temporal"
    assert report["decision_policy"]["target_recall"] == 0.7
    assert len(report["repeated_product_group_cv"]["folds"]) == 6
    assert report["bootstrap_confidence_intervals"]["method"].startswith("stratified bootstrap")
    bundle = load_artifact(artifact_path)
    assert bundle["metadata"]["class_names"] == ["negative", "neutral", "positive"]
    assert (report_directory / "precision_recall_curves.png").is_file()
