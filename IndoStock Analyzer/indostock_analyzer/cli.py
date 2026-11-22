from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from . import DISCLAIMER
from .analyzer import Analyzer
from .artifacts import ArtifactValidationError, validate_artifact
from .cache import JsonTtlCache
from .config import RuntimeConfig
from .data import YahooFinanceDataProvider
from .logging_utils import configure_logging
from .output import JsonReportWriter
from .provenance import build_run_metadata
from .review import (
    ReviewVerificationError,
    verify_sentiment_review_bundle,
)
from .scraper import IndoPremierNewsScraper
from .sentiment import (
    SentimentConfigurationError,
    TextBlobSentimentAnalyzer,
    ValidatedIndonesianSentimentAnalyzer,
    load_sentiment_provider,
    load_validation_report,
)
from .tickers import TickerExtractor
from .universe import UniverseValidationError, load_ticker_universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indostock-analyzer",
        description="Generate uncalibrated exploratory indicators for IDX tickers mentioned in news.",
    )
    parser.add_argument("--max-news", type=int, help="Maximum number of headlines to inspect")
    parser.add_argument("--timeout", type=float, help="Network/browser timeout in seconds")
    parser.add_argument("--retries", type=int, help="Retries after the first attempt")
    parser.add_argument("--min-interval", type=float, help="Minimum seconds between source requests")
    parser.add_argument("--user-agent", help="Transparent crawler user agent")
    parser.add_argument("--driver-path", type=Path, help="Optional ChromeDriver path; Selenium Manager is the default")
    parser.add_argument("--news-language", choices=("id", "en"), help="Headline language")
    parser.add_argument("--output-dir", type=Path, help="Directory for JSON reports")
    parser.add_argument("--cache-dir", type=Path, help="Directory for short-lived source cache")
    parser.add_argument("--cache-ttl", type=float, help="Cache lifetime in seconds")
    parser.add_argument("--no-cache", action="store_true", help="Disable the source cache")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Structured JSON log level",
    )
    parser.add_argument("--ticker-universe", type=Path, help="Validated ticker universe JSON file")
    parser.add_argument(
        "--universe-max-age-days",
        type=int,
        help="Age that marks the ticker snapshot stale",
    )
    parser.add_argument(
        "--max-price-age-days",
        type=int,
        help="Age that makes technical data unavailable for scoring",
    )
    parser.add_argument(
        "--max-fundamental-age-days",
        type=int,
        help="Age that makes dated fundamental data unavailable for scoring",
    )
    parser.add_argument(
        "--extreme-return-percent",
        type=float,
        help="Absolute daily return threshold for a quality warning",
    )
    parser.add_argument("--show-browser", action="store_true", help="Run Chrome with a visible window")
    parser.add_argument(
        "--sentiment-provider",
        help="Installed indostock.sentiment entry-point name (requires a validation report)",
    )
    parser.add_argument(
        "--sentiment-validation-report",
        type=Path,
        help="Independent evaluation report used by the Indonesian sentiment activation gate",
    )
    parser.add_argument(
        "--sentiment-validation-sha256",
        help="Operator-approved SHA-256 of the exact sentiment validation report",
    )
    parser.add_argument(
        "--sentiment-review-bundle",
        type=Path,
        help="Pinned bundle of signed independent-review receipts",
    )
    parser.add_argument(
        "--sentiment-review-bundle-sha256",
        help="Operator-approved SHA-256 of the review bundle",
    )
    parser.add_argument(
        "--sentiment-reviewer-trust-store",
        type=Path,
        help="Operator-managed reviewer public-key trust store",
    )
    parser.add_argument(
        "--sentiment-reviewer-trust-store-sha256",
        help="Operator-approved SHA-256 of the reviewer trust store",
    )
    parser.add_argument(
        "--sentiment-review-revocations",
        type=Path,
        help="Operator-managed review revocation registry",
    )
    parser.add_argument(
        "--sentiment-review-revocations-sha256",
        help="Operator-approved SHA-256 of the revocation registry",
    )
    parser.add_argument(
        "--sentiment-review-checkpoint",
        type=Path,
        help="Persistent monotonic checkpoint for signed revocation registries",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    config = RuntimeConfig.from_env()
    updates = {
        "max_news": args.max_news,
        "request_timeout": args.timeout,
        "retries": args.retries,
        "min_interval_seconds": args.min_interval,
        "user_agent": args.user_agent,
        "chrome_driver_path": args.driver_path,
        "news_language": args.news_language,
        "output_dir": args.output_dir,
        "cache_dir": args.cache_dir,
        "cache_ttl_seconds": args.cache_ttl,
        "log_level": args.log_level,
        "ticker_universe_path": args.ticker_universe,
        "ticker_universe_max_age_days": args.universe_max_age_days,
        "max_price_age_days": args.max_price_age_days,
        "max_fundamental_age_days": args.max_fundamental_age_days,
        "extreme_return_threshold_percent": args.extreme_return_percent,
        "sentiment_provider": args.sentiment_provider,
        "sentiment_validation_report": args.sentiment_validation_report,
        "sentiment_validation_sha256": args.sentiment_validation_sha256,
        "sentiment_review_bundle": args.sentiment_review_bundle,
        "sentiment_review_bundle_sha256": args.sentiment_review_bundle_sha256,
        "sentiment_reviewer_trust_store": args.sentiment_reviewer_trust_store,
        "sentiment_reviewer_trust_store_sha256": args.sentiment_reviewer_trust_store_sha256,
        "sentiment_review_revocations": args.sentiment_review_revocations,
        "sentiment_review_revocations_sha256": args.sentiment_review_revocations_sha256,
        "sentiment_review_checkpoint": args.sentiment_review_checkpoint,
    }
    config = replace(config, **{key: value for key, value in updates.items() if value is not None})
    if args.show_browser:
        config = replace(config, headless=False)
    if args.no_cache:
        config = replace(config, cache_enabled=False)
    return config.validate()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        config = config_from_args(parser.parse_args(argv))
    except ValueError as exc:
        parser.error(str(exc))

    configure_logging(config.log_level)
    print(DISCLAIMER)
    try:
        universe = load_ticker_universe(config.ticker_universe_path)
    except UniverseValidationError as exc:
        print(f"Invalid ticker universe: {exc}", file=sys.stderr)
        return 2
    ticker_extractor = TickerExtractor.from_universe(
        universe,
        max_age_days=config.ticker_universe_max_age_days,
    )
    for warning in ticker_extractor.metadata().get("warnings", []):
        print(f"Ticker universe warning: {warning}", file=sys.stderr)
    cache = (
        JsonTtlCache(config.cache_dir, config.cache_ttl_seconds)
        if config.cache_enabled
        else None
    )
    try:
        sentiment = build_sentiment_analyzer(config)
    except SentimentConfigurationError as exc:
        print(f"Invalid sentiment configuration: {exc}", file=sys.stderr)
        return 2
    artifacts = {"ticker_universe": config.ticker_universe_path}
    if config.sentiment_validation_report:
        artifacts["sentiment_validation_report"] = config.sentiment_validation_report
    if config.sentiment_review_bundle:
        artifacts["sentiment_review_bundle"] = config.sentiment_review_bundle
    if config.sentiment_reviewer_trust_store:
        artifacts["sentiment_reviewer_trust_store"] = config.sentiment_reviewer_trust_store
    if config.sentiment_review_revocations:
        artifacts["sentiment_review_revocations"] = config.sentiment_review_revocations
    if config.sentiment_review_checkpoint and config.sentiment_review_checkpoint.exists():
        artifacts["sentiment_review_checkpoint"] = config.sentiment_review_checkpoint
    try:
        run_metadata = build_run_metadata(config, artifacts)
    except OSError as exc:
        print(f"Cannot fingerprint input artifacts: {exc}", file=sys.stderr)
        return 2
    analyzer = Analyzer(
        IndoPremierNewsScraper(config, cache=cache),
        YahooFinanceDataProvider(config, cache=cache),
        sentiment=sentiment,
        ticker_extractor=ticker_extractor,
        news_language=config.news_language,
        run_metadata=run_metadata,
    )
    report = analyzer.run()
    path = JsonReportWriter(config.output_dir).write(report)

    if report["failures"]:
        print(f"Completed with {len(report['failures'])} source/data failure(s).", file=sys.stderr)
    if not report["stocks"]:
        print("No ticker indicators were produced.", file=sys.stderr)
    else:
        for ticker, data in report["stocks"].items():
            indicator = data["exploratory_indicator"]
            score = "n/a" if indicator["score"] is None else f"{indicator['score']:.2f}/5"
            print(f"{ticker}: {indicator['label']} ({score}, coverage {indicator['coverage']:.0%})")
    print(f"Report: {path}")
    return 0 if report["stocks"] else 2


def build_sentiment_analyzer(config: RuntimeConfig):
    if (
        not config.sentiment_provider
        or not config.sentiment_validation_report
        or not config.sentiment_validation_sha256
        or not config.sentiment_review_bundle
        or not config.sentiment_review_bundle_sha256
        or not config.sentiment_reviewer_trust_store
        or not config.sentiment_reviewer_trust_store_sha256
        or not config.sentiment_review_revocations
        or not config.sentiment_review_revocations_sha256
        or not config.sentiment_review_checkpoint
    ):
        return TextBlobSentimentAnalyzer()
    report = load_validation_report(
        config.sentiment_validation_report,
        config.sentiment_validation_sha256,
    )
    try:
        validate_artifact(report, "sentiment_validation")
    except ArtifactValidationError as exc:
        raise SentimentConfigurationError(
            f"sentiment validation report failed consistency checks: {exc}"
        ) from exc
    try:
        approval = verify_sentiment_review_bundle(
            bundle_path=config.sentiment_review_bundle,
            expected_bundle_sha256=config.sentiment_review_bundle_sha256,
            trust_store_path=config.sentiment_reviewer_trust_store,
            expected_trust_store_sha256=config.sentiment_reviewer_trust_store_sha256,
            revocations_path=config.sentiment_review_revocations,
            expected_revocations_sha256=config.sentiment_review_revocations_sha256,
            revocation_checkpoint_path=config.sentiment_review_checkpoint,
            report=report,
            report_path=config.sentiment_validation_report,
        )
    except ReviewVerificationError as exc:
        raise SentimentConfigurationError(
            f"independent-review quorum verification failed: {exc}"
        ) from exc
    predictor = load_sentiment_provider(config.sentiment_provider)
    return ValidatedIndonesianSentimentAnalyzer(
        predictor=predictor,
        validation_report=report,
        provider_name=config.sentiment_provider,
        report_integrity_approved=True,
        independent_review_approved=True,
        review_approval={
            "quorum": {
                "minimum_distinct_reviewers": approval.minimum_distinct_reviewers,
                "minimum_distinct_organizations": approval.minimum_distinct_organizations,
                "verified_receipts": len(approval.approvals),
            },
            "review_ids": [item.review_id for item in approval.approvals],
            "reviewer_ids": [item.reviewer_id for item in approval.approvals],
            "organizations": sorted({item.organization for item in approval.approvals}),
            "valid_until": approval.valid_until,
            "revocation_registry": {
                "registry_id": approval.revocation_registry_id,
                "sequence": approval.revocation_sequence,
                "sha256": approval.revocation_sha256,
                "authority_ids": [
                    item.authority_id for item in approval.revocation_authorities
                ],
                "organizations": sorted(
                    {item.organization for item in approval.revocation_authorities}
                ),
            },
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
