import pandas as pd

from tokopedia_sentiment.data import stratified_train_validation_test_split


def test_split_is_reproducible_stratified_and_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "text": [f"review {index}" for index in range(100)],
            "label": [0, 1] * 50,
            "rating": [1, 5] * 50,
        }
    )
    first = stratified_train_validation_test_split(frame, seed=42)
    second = stratified_train_validation_test_split(frame, seed=42)
    assert len(first.train) == 70
    assert len(first.validation) == 15
    assert len(first.test) == 15
    assert first.train["text"].tolist() == second.train["text"].tolist()
    assert set(first.train["text"]).isdisjoint(first.validation["text"])
    assert set(first.train["text"]).isdisjoint(first.test["text"])
    assert set(first.validation["text"]).isdisjoint(first.test["text"])
    for split in (first.train, first.validation, first.test):
        assert set(split["label"]) == {0, 1}
