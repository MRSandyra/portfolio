import hashlib
import json

from indostock_analyzer.config import RuntimeConfig
from indostock_analyzer.provenance import build_run_metadata


def test_run_metadata_hashes_inputs_and_excludes_paths_and_user_agent(tmp_path):
    universe = tmp_path / "universe.json"
    universe.write_text('{"schema_version": 1}\n', encoding="utf-8")
    config = RuntimeConfig(
        ticker_universe_path=universe,
        output_dir=tmp_path / "private-output",
        chrome_driver_path=tmp_path / "private-driver",
        user_agent="Research Bot (contact: private@example.test)",
    )

    metadata = build_run_metadata(config, {"ticker_universe": universe})

    encoded = json.dumps(metadata)
    assert str(tmp_path) not in encoded
    assert "private@example.test" not in encoded
    assert metadata["configuration"]["driver_resolution"] == "explicit"
    assert metadata["input_artifacts"][0]["sha256"] == hashlib.sha256(
        universe.read_bytes()
    ).hexdigest()
    assert len(metadata["configuration_sha256"]) == 64
