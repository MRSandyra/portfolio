from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .artifacts import (
    ArtifactValidationError,
    load_artifact,
    release_check,
    validate_artifact,
    verify_checksum,
)
from .evidence import (
    EvidenceVerificationError,
    verify_backtest_evidence,
    verify_sentiment_evidence,
)
from .review import (
    ReviewVerificationError,
    build_sentiment_review_bundle,
    canonical_review_payload,
    canonical_revocation_payload,
    load_review_json,
    prepare_sentiment_review_request,
    verify_sentiment_review_receipt,
    verify_sentiment_review_bundle,
    write_bytes_atomic,
    write_json_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indostock-audit", description="Offline artifact and release checks.")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-report")
    validate.add_argument("path", type=Path)
    validate.add_argument("--type", choices=("analysis", "sentiment_validation", "backtest"))
    verify = commands.add_parser("verify-checksum")
    verify.add_argument("path", type=Path)
    release = commands.add_parser("release-check")
    release.add_argument("root", type=Path, nargs="?", default=Path("."))
    sentiment = commands.add_parser("verify-sentiment-evidence")
    sentiment.add_argument("report", type=Path)
    sentiment.add_argument("--annotations", type=Path, required=True)
    sentiment.add_argument("--predictions", type=Path, required=True)
    sentiment.add_argument("--dataset-manifest", type=Path, required=True)
    sentiment.add_argument("--model-manifest", type=Path, required=True)
    backtest = commands.add_parser("verify-backtest-evidence")
    backtest.add_argument("report", type=Path)
    backtest.add_argument("--observations", type=Path, required=True)
    backtest.add_argument("--manifest", type=Path, required=True)
    prepare_review = commands.add_parser("prepare-sentiment-review-request")
    prepare_review.add_argument("report", type=Path)
    prepare_review.add_argument("--output", type=Path, required=True)
    canonicalize = commands.add_parser("canonicalize-review-receipt")
    canonicalize.add_argument("receipt", type=Path)
    canonicalize.add_argument("--output", type=Path, required=True)
    verify_review = commands.add_parser("verify-sentiment-review-receipt")
    verify_review.add_argument("receipt", type=Path)
    verify_review.add_argument("--report", type=Path, required=True)
    verify_review.add_argument("--trust-store", type=Path, required=True)
    verify_review.add_argument("--trust-store-sha256", required=True)
    canonicalize_revocations = commands.add_parser("canonicalize-review-revocations")
    canonicalize_revocations.add_argument("revocations", type=Path)
    canonicalize_revocations.add_argument("--output", type=Path, required=True)
    build_bundle = commands.add_parser("build-sentiment-review-bundle")
    build_bundle.add_argument("--report", type=Path, required=True)
    build_bundle.add_argument("--receipt", type=Path, action="append", required=True)
    build_bundle.add_argument("--output", type=Path, required=True)
    verify_bundle = commands.add_parser("verify-sentiment-review-bundle")
    verify_bundle.add_argument("bundle", type=Path)
    verify_bundle.add_argument("--bundle-sha256", required=True)
    verify_bundle.add_argument("--report", type=Path, required=True)
    verify_bundle.add_argument("--trust-store", type=Path, required=True)
    verify_bundle.add_argument("--trust-store-sha256", required=True)
    verify_bundle.add_argument("--revocations", type=Path, required=True)
    verify_bundle.add_argument("--revocations-sha256", required=True)
    verify_bundle.add_argument("--checkpoint", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-report":
            artifact_type = validate_artifact(load_artifact(args.path), args.type)
            print(
                f"Format and internal consistency checks passed for {artifact_type}: {args.path}"
            )
            print("Evidence origin and research conclusions were not independently verified.")
        elif args.command == "verify-checksum":
            verify_checksum(args.path)
            print(f"Checksum valid: {args.path}")
        elif args.command == "release-check":
            findings = release_check(args.root)
            if findings:
                for finding in findings:
                    print(finding, file=sys.stderr)
                return 2
            print(f"Release check passed: {args.root.resolve()}")
        elif args.command == "verify-sentiment-evidence":
            result = verify_sentiment_evidence(
                load_artifact(args.report),
                annotations_path=args.annotations,
                predictions_path=args.predictions,
                dataset_manifest_path=args.dataset_manifest,
                model_manifest_path=args.model_manifest,
            )
            _print_recomputation_result(result)
        elif args.command == "verify-backtest-evidence":
            result = verify_backtest_evidence(
                load_artifact(args.report),
                observations_path=args.observations,
                manifest_path=args.manifest,
            )
            _print_recomputation_result(result)
        elif args.command == "prepare-sentiment-review-request":
            request = prepare_sentiment_review_request(
                load_artifact(args.report), args.report
            )
            write_json_atomic(args.output, request)
            print(f"Independent-review request: {args.output}")
            print("No review approval or signature was created.")
        elif args.command == "canonicalize-review-receipt":
            write_bytes_atomic(
                args.output,
                canonical_review_payload(load_review_json(args.receipt)),
            )
            print(f"Canonical unsigned receipt payload: {args.output}")
        elif args.command == "verify-sentiment-review-receipt":
            approval = verify_sentiment_review_receipt(
                receipt_path=args.receipt,
                trust_store_path=args.trust_store,
                expected_trust_store_sha256=args.trust_store_sha256,
                report=load_artifact(args.report),
                report_path=args.report,
            )
            print(
                f"Signed independent-review receipt verified: {approval.review_id} "
                f"({approval.reviewer_id}, expires {approval.expires_at})"
            )
            print("This verifies the receipt signature and bindings, not investment validity.")
        elif args.command == "canonicalize-review-revocations":
            write_bytes_atomic(
                args.output,
                canonical_revocation_payload(load_review_json(args.revocations)),
            )
            print(f"Canonical unsigned revocation-registry payload: {args.output}")
            print("No registry signature or revocation was created.")
        elif args.command == "build-sentiment-review-bundle":
            bundle = build_sentiment_review_bundle(
                load_artifact(args.report), args.report, args.receipt
            )
            write_json_atomic(args.output, bundle)
            print(f"Review quorum bundle: {args.output}")
            print("The bundle aggregates signed receipts; it does not create approvals.")
        else:
            approval = verify_sentiment_review_bundle(
                bundle_path=args.bundle,
                expected_bundle_sha256=args.bundle_sha256,
                trust_store_path=args.trust_store,
                expected_trust_store_sha256=args.trust_store_sha256,
                revocations_path=args.revocations,
                expected_revocations_sha256=args.revocations_sha256,
                revocation_checkpoint_path=args.checkpoint,
                report=load_artifact(args.report),
                report_path=args.report,
            )
            print(
                f"Review quorum verified for {approval.report_id}: "
                f"{len(approval.approvals)} receipts, "
                f"{len({item.organization for item in approval.approvals})} organizations, "
                f"{len(approval.revocation_authorities)} revocation authorities, "
                f"revocation sequence {approval.revocation_sequence}, "
                f"valid until {approval.valid_until}"
            )
            print("This verifies review governance, not investment validity.")
        return 0
    except (ArtifactValidationError, EvidenceVerificationError, ReviewVerificationError) as exc:
        print(f"Artifact validation error: {exc}", file=sys.stderr)
        return 2


def _print_recomputation_result(result) -> None:
    print(
        f"Recomputation matched {result.report_type} report {result.report_id}: "
        f"{len(result.verified_inputs)} bound inputs, "
        f"{len(result.recomputed_sections)} sections"
    )
    print("This confirms reproducibility with these inputs, not independent validation.")


if __name__ == "__main__":
    raise SystemExit(main())
