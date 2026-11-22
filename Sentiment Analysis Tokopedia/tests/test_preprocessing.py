import pandas as pd
import pytest

from tokopedia_sentiment.data import prepare_reviews, rating_to_label
from tokopedia_sentiment.preprocessing import normalize_indonesian_text


def test_normalizes_bahasa_indonesia_and_preserves_negation() -> None:
    result = normalize_indonesian_text("  Produk BAGUS!\nTidak mengecewakan.  ")
    assert result == "produk bagus tidak mengecewakan"


def test_normalizes_slang_urls_mentions_hashtags_and_noise() -> None:
    result = normalize_indonesian_text(
        "Gk enak bgt!!! 😡 cek https://contoh.id/x @tokopedia #Kecewa"
    )
    assert result == "tidak enak banget cek url user kecewa"


@pytest.mark.parametrize("value", ["", " \t\n ", None, 123])
def test_empty_or_non_text_input_becomes_empty(value: object) -> None:
    assert normalize_indonesian_text(value) == ""


def test_prepare_reviews_removes_empty_duplicate_and_conflicting_text() -> None:
    frame = pd.DataFrame(
        {
            "Review": ["Bagus!", "bagus", "", "sama", "SAMA", "unik"],
            "Rating": [5, 5, 1, 1, 5, 4],
        }
    )
    prepared, summary = prepare_reviews(frame)
    assert prepared["text"].tolist() == ["bagus", "unik"]
    assert summary.empty_text_rows_removed == 1
    assert summary.conflicting_text_rows_removed == 2
    assert summary.duplicate_rows_removed == 1


@pytest.mark.parametrize(("rating", "expected"), [(1, 0), (2, 0), (3, 0), (4, 1), (5, 1)])
def test_label_definition(rating: int, expected: int) -> None:
    assert rating_to_label(rating) == expected


@pytest.mark.parametrize("rating", [0, 6, 3.5])
def test_rejects_invalid_rating(rating: float) -> None:
    with pytest.raises(ValueError, match="integer from 1 to 5"):
        rating_to_label(rating)
