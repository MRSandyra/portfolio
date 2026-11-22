from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from indostock_analyzer.cli import build_sentiment_analyzer, config_from_args
from indostock_analyzer.config import RuntimeConfig
from indostock_analyzer.sentiment import SentimentConfigurationError


def test_default_driver_path_is_portable():
    assert RuntimeConfig().chrome_driver_path is None


@pytest.mark.parametrize(
    "config",
    [
        RuntimeConfig(request_timeout=0),
        RuntimeConfig(retries=-1),
        RuntimeConfig(min_interval_seconds=-1),
        RuntimeConfig(cache_ttl_seconds=0),
        RuntimeConfig(user_agent=" "),
    ],
)
def test_invalid_runtime_config_is_rejected(config):
    with pytest.raises(ValueError):
        config.validate()


def test_cli_overrides_cache_and_logging(monkeypatch, tmp_path):
    monkeypatch.delenv("INDOSTOCK_CHROMEDRIVER", raising=False)
    args = Namespace(
        max_news=5,
        timeout=3.0,
        retries=1,
        min_interval=0.0,
        user_agent="TestResearchBot/1.0",
        driver_path=None,
        news_language="id",
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        cache_ttl=60.0,
        no_cache=True,
        log_level="WARNING",
        ticker_universe=None,
        universe_max_age_days=None,
        max_price_age_days=None,
        max_fundamental_age_days=None,
        extreme_return_percent=None,
        sentiment_provider=None,
        sentiment_validation_report=None,
        sentiment_validation_sha256=None,
        sentiment_review_bundle=None,
        sentiment_review_bundle_sha256=None,
        sentiment_reviewer_trust_store=None,
        sentiment_reviewer_trust_store_sha256=None,
        sentiment_review_revocations=None,
        sentiment_review_revocations_sha256=None,
        sentiment_review_checkpoint=None,
        show_browser=False,
    )

    config = config_from_args(args)

    assert config.max_news == 5
    assert config.cache_enabled is False
    assert config.cache_dir == Path(tmp_path / "cache")
    assert config.log_level == "WARNING"


def test_phase_two_quality_thresholds_are_validated():
    with pytest.raises(ValueError, match="max_price_age_days"):
        RuntimeConfig(max_price_age_days=0).validate()
    with pytest.raises(ValueError, match="extreme_return_threshold_percent"):
        RuntimeConfig(extreme_return_threshold_percent=0).validate()


def test_sentiment_provider_and_report_must_be_configured_together(tmp_path):
    with pytest.raises(ValueError, match="configured together"):
        RuntimeConfig(sentiment_provider="candidate").validate()
    with pytest.raises(ValueError, match="configured together"):
        RuntimeConfig(sentiment_validation_report=tmp_path / "report.json").validate()
    with pytest.raises(ValueError, match="configured together"):
        RuntimeConfig(
            sentiment_provider="candidate",
            sentiment_validation_report=tmp_path / "report.json",
        ).validate()
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        RuntimeConfig(
            sentiment_provider="candidate",
            sentiment_validation_report=tmp_path / "report.json",
            sentiment_validation_sha256="NOT-A-DIGEST",
            sentiment_review_bundle=tmp_path / "bundle.json",
            sentiment_review_bundle_sha256="b" * 64,
            sentiment_reviewer_trust_store=tmp_path / "trust.json",
            sentiment_reviewer_trust_store_sha256="a" * 64,
            sentiment_review_revocations=tmp_path / "revocations.json",
            sentiment_review_revocations_sha256="c" * 64,
            sentiment_review_checkpoint=tmp_path / "revocations.checkpoint.json",
        ).validate()


def test_complete_sentiment_runtime_binding_is_accepted(tmp_path):
    config = RuntimeConfig(
        sentiment_provider="candidate",
        sentiment_validation_report=tmp_path / "report.json",
        sentiment_validation_sha256="a" * 64,
        sentiment_review_bundle=tmp_path / "bundle.json",
        sentiment_review_bundle_sha256="b" * 64,
        sentiment_reviewer_trust_store=tmp_path / "trust.json",
        sentiment_reviewer_trust_store_sha256="b" * 64,
        sentiment_review_revocations=tmp_path / "revocations.json",
        sentiment_review_revocations_sha256="c" * 64,
        sentiment_review_checkpoint=tmp_path / "revocations.checkpoint.json",
    )

    assert config.validate() is config


def test_sentiment_report_digest_mismatch_fails_before_provider_load(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    config = RuntimeConfig(
        sentiment_provider="candidate",
        sentiment_validation_report=report,
        sentiment_validation_sha256="0" * 64,
        sentiment_review_bundle=tmp_path / "bundle.json",
        sentiment_review_bundle_sha256="b" * 64,
        sentiment_reviewer_trust_store=tmp_path / "trust.json",
        sentiment_reviewer_trust_store_sha256="a" * 64,
        sentiment_review_revocations=tmp_path / "revocations.json",
        sentiment_review_revocations_sha256="c" * 64,
        sentiment_review_checkpoint=tmp_path / "revocations.checkpoint.json",
    ).validate()

    with pytest.raises(SentimentConfigurationError, match="SHA-256 mismatch"):
        build_sentiment_analyzer(config)


def test_digest_matching_but_internally_inconsistent_report_fails_before_plugin_load(
    monkeypatch, tmp_path
):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "schema_version": 1,
        "report_type": "sentiment_validation",
        "report_id": "forged",
        "dataset": {},
        "model": {},
        "metrics": {},
        "activation_gate": {"active": False, "metrics_eligible": True, "reasons": []},
        "evidence": {
            "level": "independently_reviewed",
            "input_sha256": {
                "annotations": "a" * 64,
                "predictions": "b" * 64,
                "dataset_manifest": "c" * 64,
                "model_manifest": "d" * 64,
            },
        },
    }), encoding="utf-8")
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    provider_loaded = False

    def fake_loader(name):
        nonlocal provider_loaded
        provider_loaded = True

    monkeypatch.setattr("indostock_analyzer.cli.load_sentiment_provider", fake_loader)
    config = RuntimeConfig(
        sentiment_provider="candidate",
        sentiment_validation_report=report,
        sentiment_validation_sha256=digest,
        sentiment_review_bundle=tmp_path / "bundle.json",
        sentiment_review_bundle_sha256="b" * 64,
        sentiment_reviewer_trust_store=tmp_path / "trust.json",
        sentiment_reviewer_trust_store_sha256="a" * 64,
        sentiment_review_revocations=tmp_path / "revocations.json",
        sentiment_review_revocations_sha256="c" * 64,
        sentiment_review_checkpoint=tmp_path / "revocations.checkpoint.json",
    ).validate()

    with pytest.raises(SentimentConfigurationError, match="consistency checks"):
        build_sentiment_analyzer(config)
    assert provider_loaded is False
