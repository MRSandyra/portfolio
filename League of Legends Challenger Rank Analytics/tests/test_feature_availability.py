from lol_analytics.config import TEN_MINUTES_MS, TWENTY_MINUTES_MS, features_available_at


def test_ten_minute_contract_excludes_twenty_minute_features() -> None:
    early = features_available_at(TEN_MINUTES_MS)
    mid = features_available_at(TWENTY_MINUTES_MS)
    assert early
    assert all(not feature.endswith("_20") for feature in early)
    assert set(early) < set(mid)

