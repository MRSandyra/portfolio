from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import base64
import binascii
import hashlib
import hmac
import json
import os
from pathlib import Path
import time
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from . import __version__
from .artifacts import ArtifactValidationError, validate_artifact


RECEIPT_TYPE = "sentiment_independent_review"
REVIEW_DECISION = "approved_for_exploratory_sentiment_runtime"
REQUIRED_REVIEW_SCOPE = (
    "annotation_quality",
    "calibration",
    "dataset_provenance",
    "failure_slices",
    "leakage_controls",
    "model_identity",
    "reproducibility",
)
SIGNATURE_DOMAIN = b"IndoStockAnalyzer/sentiment-independent-review/v1\x00"
REVOCATION_SIGNATURE_DOMAIN = b"IndoStockAnalyzer/sentiment-review-revocations/v3\x00"
REVOCATION_REGISTRY_TYPE = "sentiment_review_revocations"
REVOCATION_CHECKPOINT_TYPE = "sentiment_review_revocation_checkpoint"


class ReviewVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class ReviewApproval:
    review_id: str
    reviewer_id: str
    organization: str
    key_id: str
    issued_at: str
    expires_at: str


@dataclass(frozen=True)
class ReviewQuorumApproval:
    report_id: str
    report_sha256: str
    approvals: tuple[ReviewApproval, ...]
    minimum_distinct_reviewers: int
    minimum_distinct_organizations: int
    valid_until: str
    revocation_registry_id: str
    revocation_sequence: int
    revocation_sha256: str
    revocation_authorities: tuple[RevocationAuthorityApproval, ...]


@dataclass(frozen=True)
class RevocationAuthorityApproval:
    authority_id: str
    organization: str
    key_id: str


@dataclass(frozen=True)
class RevocationRegistryApproval:
    registry_id: str
    sequence: int
    sha256: str
    previous_registry_sha256: str | None
    authorities: tuple[RevocationAuthorityApproval, ...]
    generated_at: str
    next_update_at: str


def load_review_json(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReviewVerificationError(f"cannot read review artifact: {path}") from exc
    if expected_sha256 is not None:
        _verify_digest_format(expected_sha256, "reviewer trust-store SHA-256")
        actual = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(actual, expected_sha256):
            raise ReviewVerificationError("reviewer trust-store SHA-256 mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ReviewVerificationError(f"review artifact is not valid UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReviewVerificationError(f"review artifact is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReviewVerificationError("review artifact must contain a JSON object")
    return value


def canonical_review_payload(receipt: Mapping[str, Any]) -> bytes:
    unsigned = {key: value for key, value in receipt.items() if key != "signature"}
    try:
        serialized = json.dumps(
            unsigned,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ReviewVerificationError("review receipt cannot be canonicalized") from exc
    return SIGNATURE_DOMAIN + serialized.encode("utf-8")


def canonical_revocation_payload(registry: Mapping[str, Any]) -> bytes:
    unsigned = {key: value for key, value in registry.items() if key != "signatures"}
    try:
        serialized = json.dumps(
            unsigned,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ReviewVerificationError("revocation registry cannot be canonicalized") from exc
    return REVOCATION_SIGNATURE_DOMAIN + serialized.encode("utf-8")


def prepare_sentiment_review_request(report: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    try:
        validate_artifact(report, "sentiment_validation")
    except ArtifactValidationError as exc:
        raise ReviewVerificationError(f"report consistency check failed: {exc}") from exc
    report_hash = sha256_file(report_path)
    evidence = report.get("evidence")
    input_hashes = evidence.get("input_sha256") if isinstance(evidence, Mapping) else None
    if not isinstance(input_hashes, Mapping):
        raise ReviewVerificationError("sentiment report lacks bound input hashes")
    return {
        "schema_version": 1,
        "request_type": "sentiment_independent_review_request",
        "request_id": f"sentiment-review-{report_hash[:20]}",
        "generator": {"name": "indostock-audit", "version": __version__},
        "report": {
            "report_id": report.get("report_id"),
            "sha256": report_hash,
        },
        "input_sha256": dict(input_hashes),
        "evaluation_policy": report.get("evaluation_policy"),
        "required_receipt_type": RECEIPT_TYPE,
        "required_decision": REVIEW_DECISION,
        "required_scope": list(REQUIRED_REVIEW_SCOPE),
        "statement": (
            "The reviewer must independently obtain the frozen inputs, reproduce or inspect "
            "the evidence, disclose conflicts, and sign a separate receipt."
        ),
    }


def verify_sentiment_review_receipt(
    *,
    receipt_path: Path,
    trust_store_path: Path,
    expected_trust_store_sha256: str,
    report: Mapping[str, Any],
    report_path: Path,
    now: datetime | None = None,
) -> ReviewApproval:
    receipt = load_review_json(receipt_path)
    trust_store = load_review_json(trust_store_path, expected_trust_store_sha256)
    return _verify_receipt_mapping(
        receipt=receipt,
        trust_store=trust_store,
        report=report,
        report_path=report_path,
        now=now,
    )


def _verify_receipt_mapping(
    *,
    receipt: Mapping[str, Any],
    trust_store: Mapping[str, Any],
    report: Mapping[str, Any],
    report_path: Path,
    now: datetime | None = None,
) -> ReviewApproval:
    try:
        validate_artifact(report, "sentiment_validation")
    except ArtifactValidationError as exc:
        raise ReviewVerificationError(f"report consistency check failed: {exc}") from exc
    if receipt.get("schema_version") != 1 or receipt.get("receipt_type") != RECEIPT_TYPE:
        raise ReviewVerificationError("unsupported review receipt schema or type")
    if trust_store.get("schema_version") != 1 or trust_store.get("trust_store_type") != "sentiment_reviewers":
        raise ReviewVerificationError("unsupported reviewer trust-store schema or type")
    review_id = _required_string(receipt, "review_id")
    if receipt.get("decision") != REVIEW_DECISION:
        raise ReviewVerificationError("review receipt decision does not approve exploratory runtime")
    report_binding = receipt.get("report")
    if not isinstance(report_binding, Mapping):
        raise ReviewVerificationError("review receipt report binding is missing")
    report_hash = sha256_file(report_path)
    if report_binding.get("report_id") != report.get("report_id"):
        raise ReviewVerificationError("review receipt report_id mismatch")
    if report_binding.get("sha256") != report_hash:
        raise ReviewVerificationError("review receipt report SHA-256 mismatch")
    report_evidence = report.get("evidence")
    report_hashes = (
        report_evidence.get("input_sha256") if isinstance(report_evidence, Mapping) else None
    )
    receipt_hashes = receipt.get("input_sha256")
    if not isinstance(report_hashes, Mapping) or receipt_hashes != report_hashes:
        raise ReviewVerificationError("review receipt input hashes do not match the report")
    scope = receipt.get("scope")
    if not isinstance(scope, list) or any(not isinstance(item, str) for item in scope):
        raise ReviewVerificationError("review receipt scope must be a string list")
    missing_scope = sorted(set(REQUIRED_REVIEW_SCOPE) - set(scope))
    if missing_scope:
        raise ReviewVerificationError("review receipt is missing required scope: " + ", ".join(missing_scope))
    methodology = receipt.get("methodology")
    if not isinstance(methodology, Mapping):
        raise ReviewVerificationError("review receipt methodology is missing")
    if methodology.get("independent_recomputation_completed") is not True:
        raise ReviewVerificationError("independent recomputation was not attested")
    if methodology.get("manual_failure_slice_review_completed") is not True:
        raise ReviewVerificationError("manual failure-slice review was not attested")
    reviewer = receipt.get("reviewer")
    if not isinstance(reviewer, Mapping):
        raise ReviewVerificationError("review receipt reviewer metadata is missing")
    reviewer_id = _required_string(reviewer, "id")
    organization = _required_string(reviewer, "organization")
    if reviewer.get("conflicts_disclosed") is not True:
        raise ReviewVerificationError("reviewer conflicts were not disclosed")
    signature = receipt.get("signature")
    if not isinstance(signature, Mapping) or signature.get("algorithm") != "Ed25519":
        raise ReviewVerificationError("review receipt requires an Ed25519 signature")
    key_id = _required_string(signature, "key_id")
    if reviewer.get("key_id") != key_id:
        raise ReviewVerificationError("reviewer and signature key_id differ")
    trusted_reviewer = _find_trusted_reviewer(trust_store, key_id)
    if trusted_reviewer.get("status") != "active":
        raise ReviewVerificationError("reviewer key is not active")
    if trusted_reviewer.get("reviewer_id") != reviewer_id:
        raise ReviewVerificationError("reviewer identity does not match trust store")
    if trusted_reviewer.get("organization") != organization:
        raise ReviewVerificationError("reviewer organization does not match trust store")
    allowed_types = trusted_reviewer.get("allowed_receipt_types")
    if not isinstance(allowed_types, list) or RECEIPT_TYPE not in allowed_types:
        raise ReviewVerificationError("reviewer key is not authorized for this receipt type")

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ReviewVerificationError("verification time must include a timezone")
    current = current.astimezone(timezone.utc)
    issued_at = _timestamp(receipt.get("issued_at"), "issued_at")
    expires_at = _timestamp(receipt.get("expires_at"), "expires_at")
    if issued_at > current + timedelta(minutes=5):
        raise ReviewVerificationError("review receipt was issued in the future")
    if expires_at <= current:
        raise ReviewVerificationError("review receipt has expired")
    if expires_at <= issued_at or expires_at - issued_at > timedelta(days=366):
        raise ReviewVerificationError("review receipt validity period is invalid")
    key_valid_from = _timestamp(trusted_reviewer.get("valid_from"), "trust key valid_from")
    key_valid_until = _timestamp(trusted_reviewer.get("valid_until"), "trust key valid_until")
    if not key_valid_from <= issued_at < key_valid_until or expires_at > key_valid_until:
        raise ReviewVerificationError("review receipt lies outside reviewer key validity")

    public_key_bytes = _decode_base64(
        trusted_reviewer.get("public_key_base64"), "reviewer public key"
    )
    signature_bytes = _decode_base64(signature.get("value_base64"), "review signature")
    if len(public_key_bytes) != 32 or len(signature_bytes) != 64:
        raise ReviewVerificationError("Ed25519 key or signature has an invalid length")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        public_key.verify(signature_bytes, canonical_review_payload(receipt))
    except (InvalidSignature, ValueError) as exc:
        raise ReviewVerificationError("review receipt signature is invalid") from exc
    return ReviewApproval(
        review_id=review_id,
        reviewer_id=reviewer_id,
        organization=organization,
        key_id=key_id,
        issued_at=issued_at.isoformat(),
        expires_at=expires_at.isoformat(),
    )


def build_sentiment_review_bundle(
    report: Mapping[str, Any],
    report_path: Path,
    receipt_paths: list[Path],
) -> dict[str, Any]:
    try:
        validate_artifact(report, "sentiment_validation")
    except ArtifactValidationError as exc:
        raise ReviewVerificationError(f"report consistency check failed: {exc}") from exc
    if len(receipt_paths) < 2:
        raise ReviewVerificationError("review bundle requires at least two receipts")
    receipts = [load_review_json(path) for path in receipt_paths]
    return {
        "schema_version": 1,
        "bundle_type": "sentiment_review_quorum",
        "report": {
            "report_id": report.get("report_id"),
            "sha256": sha256_file(report_path),
        },
        "receipts": receipts,
    }


def verify_sentiment_review_bundle(
    *,
    bundle_path: Path,
    expected_bundle_sha256: str,
    trust_store_path: Path,
    expected_trust_store_sha256: str,
    revocations_path: Path,
    expected_revocations_sha256: str,
    revocation_checkpoint_path: Path,
    report: Mapping[str, Any],
    report_path: Path,
    now: datetime | None = None,
) -> ReviewQuorumApproval:
    bundle = _load_pinned_json(bundle_path, expected_bundle_sha256, "review bundle SHA-256")
    trust_store = _load_pinned_json(
        trust_store_path,
        expected_trust_store_sha256,
        "reviewer trust-store SHA-256",
    )
    revocations = _load_pinned_json(
        revocations_path,
        expected_revocations_sha256,
        "review revocation-registry SHA-256",
    )
    try:
        validate_artifact(report, "sentiment_validation")
    except ArtifactValidationError as exc:
        raise ReviewVerificationError(f"report consistency check failed: {exc}") from exc
    if bundle.get("schema_version") != 1 or bundle.get("bundle_type") != "sentiment_review_quorum":
        raise ReviewVerificationError("unsupported review bundle schema or type")
    if trust_store.get("schema_version") != 1 or trust_store.get("trust_store_type") != "sentiment_reviewers":
        raise ReviewVerificationError("unsupported reviewer trust-store schema or type")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ReviewVerificationError("verification time must include a timezone")
    current = current.astimezone(timezone.utc)
    revocation_approval = _verify_revocation_registry(
        revocations,
        trust_store=trust_store,
        expected_sha256=expected_revocations_sha256,
        current=current,
    )
    enforce_revocation_checkpoint(
        revocation_checkpoint_path,
        revocation_approval,
        current=current,
    )
    report_hash = sha256_file(report_path)
    binding = bundle.get("report")
    if not isinstance(binding, Mapping):
        raise ReviewVerificationError("review bundle report binding is missing")
    if binding.get("report_id") != report.get("report_id") or binding.get("sha256") != report_hash:
        raise ReviewVerificationError("review bundle report binding mismatch")
    revoked_reports = _string_set(revocations, "revoked_report_sha256")
    if report_hash in revoked_reports:
        raise ReviewVerificationError("reviewed report has been revoked")
    governance = trust_store.get("governance")
    if not isinstance(governance, Mapping):
        raise ReviewVerificationError("reviewer trust store needs governance policy")
    minimum_reviewers = _positive_integer(
        governance.get("minimum_distinct_reviewers"), "minimum_distinct_reviewers"
    )
    minimum_organizations = _positive_integer(
        governance.get("minimum_distinct_organizations"), "minimum_distinct_organizations"
    )
    if minimum_reviewers < 2 or minimum_organizations < 2:
        raise ReviewVerificationError("review quorum policy must require at least two reviewers and organizations")
    if minimum_organizations > minimum_reviewers:
        raise ReviewVerificationError("organization quorum cannot exceed reviewer quorum")
    receipts = bundle.get("receipts")
    if not isinstance(receipts, list) or len(receipts) < minimum_reviewers:
        raise ReviewVerificationError("review bundle does not contain enough receipts")
    approvals: list[ReviewApproval] = []
    review_ids: set[str] = set()
    reviewer_ids: set[str] = set()
    organizations: set[str] = set()
    key_ids: set[str] = set()
    revoked_reviews = _string_set(revocations, "revoked_review_ids")
    revoked_keys = _string_set(revocations, "revoked_key_ids")
    for index, receipt in enumerate(receipts, 1):
        if not isinstance(receipt, Mapping):
            raise ReviewVerificationError(f"review bundle receipt {index} must be an object")
        approval = _verify_receipt_mapping(
            receipt=receipt,
            trust_store=trust_store,
            report=report,
            report_path=report_path,
            now=current,
        )
        if approval.review_id in review_ids or approval.reviewer_id in reviewer_ids or approval.key_id in key_ids:
            raise ReviewVerificationError("review bundle repeats a review, reviewer, or signing key")
        if approval.review_id in revoked_reviews:
            raise ReviewVerificationError(f"review receipt has been revoked: {approval.review_id}")
        if approval.key_id in revoked_keys:
            raise ReviewVerificationError(f"reviewer key has been revoked: {approval.key_id}")
        review_ids.add(approval.review_id)
        reviewer_ids.add(approval.reviewer_id)
        organizations.add(approval.organization)
        key_ids.add(approval.key_id)
        approvals.append(approval)
    if len(reviewer_ids) < minimum_reviewers or len(organizations) < minimum_organizations:
        raise ReviewVerificationError("review bundle does not satisfy distinct reviewer/organization quorum")
    valid_until = min(_timestamp(item.expires_at, "approval expires_at") for item in approvals)
    return ReviewQuorumApproval(
        report_id=str(report.get("report_id")),
        report_sha256=report_hash,
        approvals=tuple(approvals),
        minimum_distinct_reviewers=minimum_reviewers,
        minimum_distinct_organizations=minimum_organizations,
        valid_until=valid_until.isoformat(),
        revocation_registry_id=revocation_approval.registry_id,
        revocation_sequence=revocation_approval.sequence,
        revocation_sha256=revocation_approval.sha256,
        revocation_authorities=revocation_approval.authorities,
    )


def _verify_revocation_registry(
    value: Mapping[str, Any],
    *,
    trust_store: Mapping[str, Any],
    expected_sha256: str,
    current: datetime,
) -> RevocationRegistryApproval:
    if value.get("schema_version") != 3 or value.get("registry_type") != REVOCATION_REGISTRY_TYPE:
        raise ReviewVerificationError("unsupported review revocation-registry schema or type")
    if "signature" in value:
        raise ReviewVerificationError("legacy single revocation signature is not accepted")
    registry_id = _required_string(value, "registry_id")
    sequence = _positive_integer(value.get("sequence"), "revocation sequence")
    previous_digest = value.get("previous_registry_sha256")
    if sequence == 1:
        if previous_digest is not None:
            raise ReviewVerificationError("initial revocation registry cannot have a previous digest")
    else:
        if not isinstance(previous_digest, str):
            raise ReviewVerificationError("revocation registry needs a previous digest")
        _verify_digest_format(previous_digest, "previous revocation-registry SHA-256")
    generated_at = _timestamp(value.get("generated_at"), "revocation generated_at")
    next_update_at = _timestamp(value.get("next_update_at"), "revocation next_update_at")
    if generated_at > current + timedelta(minutes=5):
        raise ReviewVerificationError("revocation registry was generated in the future")
    if next_update_at <= current:
        raise ReviewVerificationError("review revocation registry is stale")
    if next_update_at <= generated_at or next_update_at - generated_at > timedelta(days=31):
        raise ReviewVerificationError("revocation registry refresh period is invalid")
    for field in ("revoked_review_ids", "revoked_key_ids", "revoked_report_sha256"):
        _string_set(value, field)
    governance = trust_store.get("governance")
    if not isinstance(governance, Mapping):
        raise ReviewVerificationError("reviewer trust store needs governance policy")
    minimum_authorities = _positive_integer(
        governance.get("minimum_revocation_authorities"),
        "minimum_revocation_authorities",
    )
    minimum_organizations = _positive_integer(
        governance.get("minimum_revocation_organizations"),
        "minimum_revocation_organizations",
    )
    if minimum_authorities < 2 or minimum_organizations < 2:
        raise ReviewVerificationError(
            "revocation quorum policy must require at least two authorities and organizations"
        )
    if minimum_organizations > minimum_authorities:
        raise ReviewVerificationError(
            "revocation organization quorum cannot exceed authority quorum"
        )
    signatures = value.get("signatures")
    if not isinstance(signatures, list) or len(signatures) < minimum_authorities:
        raise ReviewVerificationError("revocation registry does not contain enough signatures")
    payload = canonical_revocation_payload(value)
    approvals: list[RevocationAuthorityApproval] = []
    authority_ids: set[str] = set()
    organizations: set[str] = set()
    key_ids: set[str] = set()
    for index, signature in enumerate(signatures, 1):
        if not isinstance(signature, Mapping) or signature.get("algorithm") != "Ed25519":
            raise ReviewVerificationError(
                f"revocation signature {index} must use Ed25519"
            )
        key_id = _required_string(signature, "key_id")
        authority = _find_revocation_authority(trust_store, key_id)
        if authority.get("status") != "active":
            raise ReviewVerificationError("revocation-authority key is not active")
        authority_id = _required_string(authority, "authority_id")
        organization = _required_string(authority, "organization")
        if signature.get("authority_id") != authority_id:
            raise ReviewVerificationError(
                "revocation signature authority does not match trust store"
            )
        if authority_id in authority_ids or key_id in key_ids:
            raise ReviewVerificationError(
                "revocation registry repeats an authority or signing key"
            )
        allowed_types = authority.get("allowed_registry_types")
        if not isinstance(allowed_types, list) or REVOCATION_REGISTRY_TYPE not in allowed_types:
            raise ReviewVerificationError(
                "revocation-authority key is not authorized for this registry"
            )
        key_valid_from = _timestamp(
            authority.get("valid_from"), "revocation key valid_from"
        )
        key_valid_until = _timestamp(
            authority.get("valid_until"), "revocation key valid_until"
        )
        if not key_valid_from <= generated_at < key_valid_until or next_update_at > key_valid_until:
            raise ReviewVerificationError(
                "revocation registry lies outside authority-key validity"
            )
        public_key_bytes = _decode_base64(
            authority.get("public_key_base64"), "revocation-authority public key"
        )
        signature_bytes = _decode_base64(
            signature.get("value_base64"), "revocation signature"
        )
        if len(public_key_bytes) != 32 or len(signature_bytes) != 64:
            raise ReviewVerificationError(
                "Ed25519 revocation key or signature has an invalid length"
            )
        try:
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            public_key.verify(signature_bytes, payload)
        except (InvalidSignature, ValueError) as exc:
            raise ReviewVerificationError(
                f"revocation registry signature {index} is invalid"
            ) from exc
        authority_ids.add(authority_id)
        organizations.add(organization)
        key_ids.add(key_id)
        approvals.append(
            RevocationAuthorityApproval(
                authority_id=authority_id,
                organization=organization,
                key_id=key_id,
            )
        )
    if len(authority_ids) < minimum_authorities or len(organizations) < minimum_organizations:
        raise ReviewVerificationError(
            "revocation registry does not satisfy distinct authority/organization quorum"
        )
    return RevocationRegistryApproval(
        registry_id=registry_id,
        sequence=sequence,
        sha256=expected_sha256,
        previous_registry_sha256=previous_digest,
        authorities=tuple(approvals),
        generated_at=generated_at.isoformat(),
        next_update_at=next_update_at.isoformat(),
    )


def enforce_revocation_checkpoint(
    path: Path,
    approval: RevocationRegistryApproval,
    *,
    current: datetime,
) -> None:
    if current.tzinfo is None:
        raise ReviewVerificationError("verification time must include a timezone")
    current = current.astimezone(timezone.utc)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_checkpoint_lock(path):
            checkpoint = _load_checkpoint(path) if path.exists() else None
            if checkpoint is not None:
                checkpoint_registry = _required_string(checkpoint, "registry_id")
                checkpoint_sequence = _positive_integer(
                    checkpoint.get("sequence"), "checkpoint sequence"
                )
                checkpoint_digest = _required_string(checkpoint, "registry_sha256")
                _verify_digest_format(checkpoint_digest, "checkpoint registry SHA-256")
                observed_at = _timestamp(checkpoint.get("observed_at"), "checkpoint observed_at")
                if checkpoint_registry != approval.registry_id:
                    raise ReviewVerificationError("revocation registry_id differs from checkpoint")
                if current + timedelta(minutes=5) < observed_at:
                    raise ReviewVerificationError("verification clock moved behind revocation checkpoint")
                if approval.sequence < checkpoint_sequence:
                    raise ReviewVerificationError("revocation registry rollback detected")
                if approval.sequence == checkpoint_sequence:
                    if not hmac.compare_digest(approval.sha256, checkpoint_digest):
                        raise ReviewVerificationError("revocation registry equivocation detected")
                else:
                    if approval.sequence != checkpoint_sequence + 1:
                        raise ReviewVerificationError("revocation registry sequence gap detected")
                    if approval.previous_registry_sha256 != checkpoint_digest:
                        raise ReviewVerificationError(
                            "revocation registry hash chain does not match checkpoint"
                        )
            write_json_atomic(
                path,
                {
                    "schema_version": 1,
                    "checkpoint_type": REVOCATION_CHECKPOINT_TYPE,
                    "registry_id": approval.registry_id,
                    "sequence": approval.sequence,
                    "registry_sha256": approval.sha256,
                    "observed_at": current.isoformat(),
                },
            )
    except OSError as exc:
        raise ReviewVerificationError(
            f"cannot update revocation checkpoint: {path}"
        ) from exc


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except OSError as exc:
        raise ReviewVerificationError(f"cannot read revocation checkpoint: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewVerificationError("revocation checkpoint is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReviewVerificationError("revocation checkpoint must contain a JSON object")
    if value.get("schema_version") != 1 or value.get("checkpoint_type") != REVOCATION_CHECKPOINT_TYPE:
        raise ReviewVerificationError("unsupported revocation checkpoint schema or type")
    return value


@contextmanager
def _exclusive_checkpoint_lock(path: Path):
    lock_path = path.with_name(f".{path.name}.lock")
    deadline = time.monotonic() + 5.0
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise ReviewVerificationError(
                    f"revocation checkpoint is locked: {lock_path}"
                ) from exc
            time.sleep(0.05)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _string_set(value: Mapping[str, Any], field: str) -> set[str]:
    items = value.get(field)
    if not isinstance(items, list) or any(not isinstance(item, str) or not item for item in items):
        raise ReviewVerificationError(f"revocation field {field} must be a string list")
    if len(items) != len(set(items)):
        raise ReviewVerificationError(f"revocation field {field} contains duplicates")
    if field == "revoked_report_sha256":
        for item in items:
            _verify_digest_format(item, "revoked report SHA-256")
    return set(items)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReviewVerificationError(f"{name} must be a positive integer")
    return value


def _load_pinned_json(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    _verify_digest_format(expected_sha256, name)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReviewVerificationError(f"cannot read review artifact: {path}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected_sha256):
        raise ReviewVerificationError(f"{name} mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewVerificationError(f"review artifact is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReviewVerificationError("review artifact must contain a JSON object")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReviewVerificationError(f"cannot hash review input: {path}") from exc
    return digest.hexdigest()


def _find_trusted_reviewer(trust_store: Mapping[str, Any], key_id: str) -> Mapping[str, Any]:
    reviewers = trust_store.get("reviewers")
    if not isinstance(reviewers, list):
        raise ReviewVerificationError("reviewer trust store needs a reviewers list")
    matches = [
        item for item in reviewers
        if isinstance(item, Mapping) and item.get("key_id") == key_id
    ]
    if len(matches) != 1:
        raise ReviewVerificationError("reviewer key_id is missing or duplicated in trust store")
    return matches[0]


def _find_revocation_authority(
    trust_store: Mapping[str, Any], key_id: str
) -> Mapping[str, Any]:
    authorities = trust_store.get("revocation_authorities")
    if not isinstance(authorities, list):
        raise ReviewVerificationError(
            "reviewer trust store needs a revocation_authorities list"
        )
    matches = [
        item
        for item in authorities
        if isinstance(item, Mapping) and item.get("key_id") == key_id
    ]
    if len(matches) != 1:
        raise ReviewVerificationError(
            "revocation-authority key_id is missing or duplicated in trust store"
        )
    return matches[0]


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ReviewVerificationError(f"review field {key} must be a non-empty string")
    return item.strip()


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ReviewVerificationError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewVerificationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ReviewVerificationError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _decode_base64(value: object, name: str) -> bytes:
    if not isinstance(value, str):
        raise ReviewVerificationError(f"{name} must be base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ReviewVerificationError(f"{name} is not valid base64") from exc


def _verify_digest_format(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ReviewVerificationError(f"{name} must be 64 lowercase hexadecimal characters")
