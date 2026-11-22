import pytest

from tokopedia_sentiment.demo_data import DEMO_LABELS, DEMO_TEXTS
from tokopedia_sentiment.inference import predict_reviews
from tokopedia_sentiment.modeling import build_pipeline, load_artifact, save_artifact


@pytest.fixture
def demo_model():
    return build_pipeline().fit(DEMO_TEXTS, DEMO_LABELS)


def test_inference_returns_labels_and_probabilities(demo_model) -> None:
    predictions = predict_reviews(
        demo_model,
        ["rasanya enak kualitas bagus", "barang rusak dan kualitas buruk"],
    )
    assert [prediction.sentiment for prediction in predictions] == ["positive", "negative"]
    for prediction in predictions:
        assert prediction.probability_negative + prediction.probability_positive == pytest.approx(1)
        assert 0.5 <= prediction.confidence <= 1
        assert prediction.probabilities_calibrated is False


@pytest.mark.parametrize("reviews", [[], [""], ["!!! 😡"]])
def test_inference_rejects_empty_input(demo_model, reviews: list[str]) -> None:
    with pytest.raises(ValueError):
        predict_reviews(demo_model, reviews)


def test_artifact_round_trip(tmp_path, demo_model) -> None:
    path = save_artifact(
        demo_model,
        {"model": {"probabilities_calibrated": False}},
        tmp_path / "m.joblib",
    )
    bundle = load_artifact(path)
    prediction = predict_reviews(bundle["model"], ["produk bagus"])[0]
    assert prediction.sentiment == "positive"
