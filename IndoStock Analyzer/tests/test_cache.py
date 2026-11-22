import json

from indostock_analyzer.cache import JsonTtlCache


def test_cache_returns_fresh_entry_and_expires_old_entry(tmp_path):
    current_time = [1000.0]
    cache = JsonTtlCache(tmp_path, ttl_seconds=10, clock=lambda: current_time[0])
    cache.set("market", "BBRI", {"price": 100})

    current_time[0] = 1005.0
    fresh = cache.get("market", "BBRI")
    assert fresh is not None
    assert fresh.value == {"price": 100}
    assert fresh.age_seconds == 5.0

    current_time[0] = 1011.0
    assert cache.get("market", "BBRI") is None


def test_corrupt_cache_entry_fails_closed(tmp_path):
    cache = JsonTtlCache(tmp_path, ttl_seconds=10, clock=lambda: 1000.0)
    cache.set("market", "BBRI", {"price": 100})
    cache_file = next(tmp_path.rglob("*.json"))
    cache_file.write_text("not-json", encoding="utf-8")

    assert cache.get("market", "BBRI") is None
    assert not cache_file.exists()


def test_cache_file_contains_no_raw_key_in_filename(tmp_path):
    cache = JsonTtlCache(tmp_path, ttl_seconds=10, clock=lambda: 1000.0)
    cache.set("market", "BBRI:6mo:1d", {"ok": True})

    cache_file = next(tmp_path.rglob("*.json"))
    assert "BBRI" not in cache_file.name
    assert json.loads(cache_file.read_text(encoding="utf-8"))["value"] == {"ok": True}
