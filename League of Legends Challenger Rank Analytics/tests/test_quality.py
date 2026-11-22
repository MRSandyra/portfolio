from lol_analytics.data import read_events
from lol_analytics.quality import audit_events


def test_sample_fixture_passes_quality_gates() -> None:
    events = read_events("data/sample/events.jsonl")
    audit = audit_events(events, expected_matches=12)
    assert audit["passed"] is True
    assert audit["matches"] == 12
    assert audit["duplicate_event_ids"] == 0
    assert audit["horizon_coverage"]["10m"]["eligible_matches"] == 12
    assert audit["horizon_coverage"]["20m"]["eligible_matches"] == 12
