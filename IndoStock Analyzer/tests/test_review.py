import base64
from datetime import datetime, timezone
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from indostock_analyzer.artifact_cli import main as audit_cli_main
from indostock_analyzer.cli import build_sentiment_analyzer
from indostock_analyzer.config import RuntimeConfig
from indostock_analyzer.review import (
    REQUIRED_REVIEW_SCOPE,
    REVIEW_DECISION,
    ReviewVerificationError,
    build_sentiment_review_bundle,
    canonical_review_payload,
    canonical_revocation_payload,
    load_review_json,
    prepare_sentiment_review_request,
    verify_sentiment_review_receipt,
    verify_sentiment_review_bundle,
)


NOW = datetime(2026, 11, 22, 8, 0, tzinfo=timezone.utc)


def sentiment_report():
    return {
        "schema_version": 1,
        "report_type": "sentiment_validation",
        "report_id": "candidate-phase8",
        "dataset": {
            "name": "frozen-holdout",
            "version": "1",
            "license": "CC-BY-4.0",
            "source_url": "https://example.test/dataset",
            "content_sha256": "a" * 64,
            "domain": "indonesian_financial_headlines",
            "split_strategy": "temporal_holdout",
            "target_aware": True,
        },
        "model": {
            "id": "example/model",
            "revision": "immutable-revision",
            "license": "Apache-2.0",
            "source_url": "https://example.test/model",
        },
        "metrics": {
            "evaluated_samples": 300,
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "expected_calibration_error": 0.1,
            "confusion_matrix": {
                "NEGATIVE": {"NEGATIVE": 100, "NEUTRAL": 0, "POSITIVE": 0},
                "NEUTRAL": {"NEGATIVE": 0, "NEUTRAL": 100, "POSITIVE": 0},
                "POSITIVE": {"NEGATIVE": 0, "NEUTRAL": 0, "POSITIVE": 100},
            },
            "per_class": {
                label: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 100}
                for label in ("NEGATIVE", "NEUTRAL", "POSITIVE")
            },
        },
        "activation_gate": {
            "active": False,
            "metrics_eligible": True,
            "reasons": [
                "external_independent_review_receipt_required",
                "report_integrity_not_operator_approved",
            ],
        },
        "evaluation_policy": {
            "minimum_test_samples": 300,
            "minimum_macro_f1": 0.75,
            "minimum_class_recall": 0.65,
            "maximum_ece": 0.15,
            "required_domain": "indonesian_financial_headlines",
            "required_split_strategy": "temporal_holdout",
            "require_target_aware": True,
            "require_external_review_receipt": True,
        },
        "evidence": {
            "level": "generated_from_bound_inputs",
            "independently_reviewed": False,
            "input_sha256": {
                "annotations": "a" * 64,
                "predictions": "b" * 64,
                "dataset_manifest": "c" * 64,
                "model_manifest": "d" * 64,
            },
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def signed_fixture(tmp_path):
    report_path = tmp_path / "report.json"
    trust_path = tmp_path / "trust-store.json"
    receipt_path = tmp_path / "receipt.json"
    report = sentiment_report()
    write_json(report_path, report)
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust_store = {
        "schema_version": 1,
        "trust_store_type": "sentiment_reviewers",
        "reviewers": [{
            "key_id": "reviewer-key-2026",
            "reviewer_id": "reviewer-001",
            "organization": "Independent Research Lab",
            "public_key_base64": base64.b64encode(public_key).decode("ascii"),
            "status": "active",
            "allowed_receipt_types": ["sentiment_independent_review"],
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_until": "2027-12-31T00:00:00Z",
        }],
    }
    write_json(trust_path, trust_store)
    receipt = {
        "schema_version": 1,
        "receipt_type": "sentiment_independent_review",
        "review_id": "review-phase8-001",
        "decision": REVIEW_DECISION,
        "issued_at": "2026-11-22T07:00:00Z",
        "expires_at": "2027-05-22T07:00:00Z",
        "report": {
            "report_id": report["report_id"],
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
        "input_sha256": report["evidence"]["input_sha256"],
        "scope": list(REQUIRED_REVIEW_SCOPE),
        "methodology": {
            "independent_recomputation_completed": True,
            "manual_failure_slice_review_completed": True,
        },
        "reviewer": {
            "id": "reviewer-001",
            "organization": "Independent Research Lab",
            "key_id": "reviewer-key-2026",
            "conflicts_disclosed": True,
        },
    }
    receipt["signature"] = {
        "algorithm": "Ed25519",
        "key_id": "reviewer-key-2026",
        "value_base64": base64.b64encode(
            private_key.sign(canonical_review_payload(receipt))
        ).decode("ascii"),
    }
    write_json(receipt_path, receipt)
    trust_hash = hashlib.sha256(trust_path.read_bytes()).hexdigest()
    return report_path, trust_path, receipt_path, trust_hash, private_key


def quorum_fixture(
    tmp_path,
    *,
    same_organization=False,
    same_revocation_organization=False,
    revoked_review_ids=None,
    stale=False,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    report_path, trust_path, receipt_one_path, _, _ = signed_fixture(tmp_path)
    trust_store = load_review_json(trust_path)
    trust_store["governance"] = {
        "minimum_distinct_reviewers": 2,
        "minimum_distinct_organizations": 2,
        "minimum_revocation_authorities": 2,
        "minimum_revocation_organizations": 2,
    }
    second_key = Ed25519PrivateKey.generate()
    second_public = second_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    second_organization = (
        "Independent Research Lab" if same_organization else "External Model Audit Group"
    )
    trust_store["reviewers"].append({
        "key_id": "reviewer-key-2026-b",
        "reviewer_id": "reviewer-002",
        "organization": second_organization,
        "public_key_base64": base64.b64encode(second_public).decode("ascii"),
        "status": "active",
        "allowed_receipt_types": ["sentiment_independent_review"],
        "valid_from": "2026-01-01T00:00:00Z",
        "valid_until": "2027-12-31T00:00:00Z",
    })
    revocation_signers = []
    revocation_authorities = []
    for index, (authority_id, organization) in enumerate((
        ("review-governance-a", "Governance Foundation A"),
        (
            "review-governance-b",
            "Governance Foundation A"
            if same_revocation_organization
            else "Governance Foundation B",
        ),
    ), 1):
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        key_id = f"revocation-key-2026-{index}"
        revocation_signers.append((private_key, authority_id, key_id))
        revocation_authorities.append({
            "key_id": key_id,
            "authority_id": authority_id,
            "organization": organization,
            "public_key_base64": base64.b64encode(public_key).decode("ascii"),
            "status": "active",
            "allowed_registry_types": ["sentiment_review_revocations"],
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_until": "2027-12-31T00:00:00Z",
        })
    trust_store["revocation_authorities"] = revocation_authorities
    write_json(trust_path, trust_store)
    second_receipt = load_review_json(receipt_one_path)
    second_receipt["review_id"] = "review-phase9-002"
    second_receipt["reviewer"] = {
        "id": "reviewer-002",
        "organization": second_organization,
        "key_id": "reviewer-key-2026-b",
        "conflicts_disclosed": True,
    }
    second_receipt["signature"] = {
        "algorithm": "Ed25519",
        "key_id": "reviewer-key-2026-b",
        "value_base64": base64.b64encode(
            second_key.sign(canonical_review_payload(second_receipt))
        ).decode("ascii"),
    }
    receipt_two_path = tmp_path / "receipt-two.json"
    write_json(receipt_two_path, second_receipt)
    report = load_review_json(report_path)
    bundle = build_sentiment_review_bundle(
        report, report_path, [receipt_one_path, receipt_two_path]
    )
    bundle_path = tmp_path / "review-bundle.json"
    write_json(bundle_path, bundle)
    revocations = {
        "schema_version": 3,
        "registry_type": "sentiment_review_revocations",
        "registry_id": "indostock-sentiment-reviews",
        "sequence": 1,
        "previous_registry_sha256": None,
        "generated_at": "2026-11-01T00:00:00Z",
        "next_update_at": (
            "2026-11-20T00:00:00Z" if stale else "2026-12-01T00:00:00Z"
        ),
        "revoked_review_ids": revoked_review_ids or [],
        "revoked_key_ids": [],
        "revoked_report_sha256": [],
    }
    payload = canonical_revocation_payload(revocations)
    revocations["signatures"] = [
        {
            "algorithm": "Ed25519",
            "authority_id": authority_id,
            "key_id": key_id,
            "value_base64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
        }
        for private_key, authority_id, key_id in revocation_signers
    ]
    revocations_path = tmp_path / "revocations.json"
    write_json(revocations_path, revocations)
    return {
        "report": report_path,
        "trust": trust_path,
        "bundle": bundle_path,
        "revocations": revocations_path,
        "checkpoint": tmp_path / "revocations.checkpoint.json",
        "receipt_one": receipt_one_path,
        "receipt_two": receipt_two_path,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "trust_sha256": hashlib.sha256(trust_path.read_bytes()).hexdigest(),
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "revocations_sha256": hashlib.sha256(revocations_path.read_bytes()).hexdigest(),
        "revocation_signers": revocation_signers,
    }


def write_revocation_update(
    fixture,
    path,
    *,
    sequence,
    previous_sha256,
    revoked_review_ids=None,
):
    registry = load_review_json(fixture["revocations"])
    registry.pop("signatures", None)
    registry["sequence"] = sequence
    registry["previous_registry_sha256"] = previous_sha256
    registry["revoked_review_ids"] = revoked_review_ids or []
    payload = canonical_revocation_payload(registry)
    registry["signatures"] = [
        {
            "algorithm": "Ed25519",
            "authority_id": authority_id,
            "key_id": key_id,
            "value_base64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
        }
        for private_key, authority_id, key_id in fixture["revocation_signers"]
    ]
    write_json(path, registry)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_signed_external_review_receipt_verifies_and_cli_reports_boundary(tmp_path):
    report_path, trust_path, receipt_path, trust_hash, _ = signed_fixture(tmp_path)

    approval = verify_sentiment_review_receipt(
        receipt_path=receipt_path,
        trust_store_path=trust_path,
        expected_trust_store_sha256=trust_hash,
        report=load_review_json(report_path),
        report_path=report_path,
        now=NOW,
    )

    assert approval.review_id == "review-phase8-001"
    assert approval.reviewer_id == "reviewer-001"
    assert audit_cli_main([
        "verify-sentiment-review-receipt",
        str(receipt_path),
        "--report", str(report_path),
        "--trust-store", str(trust_path),
        "--trust-store-sha256", trust_hash,
    ]) == 0


def test_runtime_activates_only_after_signed_receipt_verification(monkeypatch, tmp_path):
    fixture = quorum_fixture(tmp_path)

    class Predictor:
        def predict(self, text, target=None):
            return {"NEGATIVE": 0.1, "NEUTRAL": 0.2, "POSITIVE": 0.7}

    monkeypatch.setattr(
        "indostock_analyzer.cli.load_sentiment_provider",
        lambda provider_name: Predictor(),
    )
    config = RuntimeConfig(
        sentiment_provider="reviewed-provider",
        sentiment_validation_report=fixture["report"],
        sentiment_validation_sha256=fixture["report_sha256"],
        sentiment_review_bundle=fixture["bundle"],
        sentiment_review_bundle_sha256=fixture["bundle_sha256"],
        sentiment_reviewer_trust_store=fixture["trust"],
        sentiment_reviewer_trust_store_sha256=fixture["trust_sha256"],
        sentiment_review_revocations=fixture["revocations"],
        sentiment_review_revocations_sha256=fixture["revocations_sha256"],
        sentiment_review_checkpoint=fixture["checkpoint"],
    ).validate()

    analyzer = build_sentiment_analyzer(config)
    result = analyzer.analyze("BBRI membukukan kenaikan laba", "id", target="BBRI")

    assert analyzer.decision.active is True
    assert analyzer.metadata()["review_approval"]["quorum"]["verified_receipts"] == 2
    assert result.supported is True
    assert result.evidence_level == "externally_signed_review"


def test_distinct_reviewer_and_organization_quorum_verifies(tmp_path):
    fixture = quorum_fixture(tmp_path)

    approval = verify_sentiment_review_bundle(
        bundle_path=fixture["bundle"],
        expected_bundle_sha256=fixture["bundle_sha256"],
        trust_store_path=fixture["trust"],
        expected_trust_store_sha256=fixture["trust_sha256"],
        revocations_path=fixture["revocations"],
        expected_revocations_sha256=fixture["revocations_sha256"],
        revocation_checkpoint_path=fixture["checkpoint"],
        report=load_review_json(fixture["report"]),
        report_path=fixture["report"],
        now=NOW,
    )

    assert len(approval.approvals) == 2
    assert {item.organization for item in approval.approvals} == {
        "Independent Research Lab", "External Model Audit Group"
    }
    assert approval.minimum_distinct_organizations == 2
    assert approval.revocation_sequence == 1
    assert {item.authority_id for item in approval.revocation_authorities} == {
        "review-governance-a",
        "review-governance-b",
    }
    assert {item.organization for item in approval.revocation_authorities} == {
        "Governance Foundation A",
        "Governance Foundation B",
    }
    cli_bundle = tmp_path / "cli-bundle.json"
    assert audit_cli_main([
        "build-sentiment-review-bundle",
        "--report", str(fixture["report"]),
        "--receipt", str(fixture["receipt_one"]),
        "--receipt", str(fixture["receipt_two"]),
        "--output", str(cli_bundle),
    ]) == 0
    cli_bundle_hash = hashlib.sha256(cli_bundle.read_bytes()).hexdigest()
    assert audit_cli_main([
        "verify-sentiment-review-bundle",
        str(cli_bundle),
        "--bundle-sha256", cli_bundle_hash,
        "--report", str(fixture["report"]),
        "--trust-store", str(fixture["trust"]),
        "--trust-store-sha256", fixture["trust_sha256"],
        "--revocations", str(fixture["revocations"]),
        "--revocations-sha256", fixture["revocations_sha256"],
        "--checkpoint", str(tmp_path / "cli-revocations.checkpoint.json"),
    ]) == 0
    revocation_payload = tmp_path / "revocations.payload"
    assert audit_cli_main([
        "canonicalize-review-revocations",
        str(fixture["revocations"]),
        "--output", str(revocation_payload),
    ]) == 0
    assert revocation_payload.read_bytes() == canonical_revocation_payload(
        load_review_json(fixture["revocations"])
    )
    assert revocation_payload.read_bytes().startswith(
        b"IndoStockAnalyzer/sentiment-review-revocations/v3\x00"
    )


def test_quorum_rejects_same_organization_revocation_and_stale_registry(tmp_path):
    same_org = quorum_fixture(tmp_path / "same-org", same_organization=True)
    with pytest.raises(ReviewVerificationError, match="organization quorum"):
        verify_sentiment_review_bundle(
            bundle_path=same_org["bundle"],
            expected_bundle_sha256=same_org["bundle_sha256"],
            trust_store_path=same_org["trust"],
            expected_trust_store_sha256=same_org["trust_sha256"],
            revocations_path=same_org["revocations"],
            expected_revocations_sha256=same_org["revocations_sha256"],
            revocation_checkpoint_path=same_org["checkpoint"],
            report=load_review_json(same_org["report"]),
            report_path=same_org["report"],
            now=NOW,
        )

    revoked = quorum_fixture(
        tmp_path / "revoked", revoked_review_ids=["review-phase9-002"]
    )
    with pytest.raises(ReviewVerificationError, match="has been revoked"):
        verify_sentiment_review_bundle(
            bundle_path=revoked["bundle"],
            expected_bundle_sha256=revoked["bundle_sha256"],
            trust_store_path=revoked["trust"],
            expected_trust_store_sha256=revoked["trust_sha256"],
            revocations_path=revoked["revocations"],
            expected_revocations_sha256=revoked["revocations_sha256"],
            revocation_checkpoint_path=revoked["checkpoint"],
            report=load_review_json(revoked["report"]),
            report_path=revoked["report"],
            now=NOW,
        )

    stale = quorum_fixture(tmp_path / "stale", stale=True)
    with pytest.raises(ReviewVerificationError, match="registry is stale"):
        verify_sentiment_review_bundle(
            bundle_path=stale["bundle"],
            expected_bundle_sha256=stale["bundle_sha256"],
            trust_store_path=stale["trust"],
            expected_trust_store_sha256=stale["trust_sha256"],
            revocations_path=stale["revocations"],
            expected_revocations_sha256=stale["revocations_sha256"],
            revocation_checkpoint_path=stale["checkpoint"],
            report=load_review_json(stale["report"]),
            report_path=stale["report"],
            now=NOW,
        )


def test_signed_revocation_checkpoint_rejects_rollback_fork_gap_and_clock_rollback(tmp_path):
    fixture = quorum_fixture(tmp_path)
    report = load_review_json(fixture["report"])

    verify_sentiment_review_bundle(
        bundle_path=fixture["bundle"],
        expected_bundle_sha256=fixture["bundle_sha256"],
        trust_store_path=fixture["trust"],
        expected_trust_store_sha256=fixture["trust_sha256"],
        revocations_path=fixture["revocations"],
        expected_revocations_sha256=fixture["revocations_sha256"],
        revocation_checkpoint_path=fixture["checkpoint"],
        report=report,
        report_path=fixture["report"],
        now=NOW,
    )
    sequence_two_path = tmp_path / "revocations-sequence-two.json"
    sequence_two_hash = write_revocation_update(
        fixture,
        sequence_two_path,
        sequence=2,
        previous_sha256=fixture["revocations_sha256"],
    )
    sequence_two_approval = verify_sentiment_review_bundle(
        bundle_path=fixture["bundle"],
        expected_bundle_sha256=fixture["bundle_sha256"],
        trust_store_path=fixture["trust"],
        expected_trust_store_sha256=fixture["trust_sha256"],
        revocations_path=sequence_two_path,
        expected_revocations_sha256=sequence_two_hash,
        revocation_checkpoint_path=fixture["checkpoint"],
        report=report,
        report_path=fixture["report"],
        now=datetime(2026, 11, 22, 8, 1, tzinfo=timezone.utc),
    )
    assert sequence_two_approval.revocation_sequence == 2
    checkpoint = load_review_json(fixture["checkpoint"])
    assert checkpoint["sequence"] == 2
    assert checkpoint["registry_sha256"] == sequence_two_hash

    with pytest.raises(ReviewVerificationError, match="rollback detected"):
        verify_sentiment_review_bundle(
            bundle_path=fixture["bundle"],
            expected_bundle_sha256=fixture["bundle_sha256"],
            trust_store_path=fixture["trust"],
            expected_trust_store_sha256=fixture["trust_sha256"],
            revocations_path=fixture["revocations"],
            expected_revocations_sha256=fixture["revocations_sha256"],
            revocation_checkpoint_path=fixture["checkpoint"],
            report=report,
            report_path=fixture["report"],
            now=datetime(2026, 11, 22, 8, 2, tzinfo=timezone.utc),
        )

    fork_path = tmp_path / "revocations-sequence-two-fork.json"
    fork_hash = write_revocation_update(
        fixture,
        fork_path,
        sequence=2,
        previous_sha256=fixture["revocations_sha256"],
        revoked_review_ids=["unrelated-review"],
    )
    with pytest.raises(ReviewVerificationError, match="equivocation detected"):
        verify_sentiment_review_bundle(
            bundle_path=fixture["bundle"],
            expected_bundle_sha256=fixture["bundle_sha256"],
            trust_store_path=fixture["trust"],
            expected_trust_store_sha256=fixture["trust_sha256"],
            revocations_path=fork_path,
            expected_revocations_sha256=fork_hash,
            revocation_checkpoint_path=fixture["checkpoint"],
            report=report,
            report_path=fixture["report"],
            now=datetime(2026, 11, 22, 8, 2, tzinfo=timezone.utc),
        )

    gap_path = tmp_path / "revocations-sequence-four.json"
    gap_hash = write_revocation_update(
        fixture,
        gap_path,
        sequence=4,
        previous_sha256=sequence_two_hash,
    )
    with pytest.raises(ReviewVerificationError, match="sequence gap detected"):
        verify_sentiment_review_bundle(
            bundle_path=fixture["bundle"],
            expected_bundle_sha256=fixture["bundle_sha256"],
            trust_store_path=fixture["trust"],
            expected_trust_store_sha256=fixture["trust_sha256"],
            revocations_path=gap_path,
            expected_revocations_sha256=gap_hash,
            revocation_checkpoint_path=fixture["checkpoint"],
            report=report,
            report_path=fixture["report"],
            now=datetime(2026, 11, 22, 8, 2, tzinfo=timezone.utc),
        )

    broken_chain_path = tmp_path / "revocations-sequence-three-broken.json"
    broken_chain_hash = write_revocation_update(
        fixture,
        broken_chain_path,
        sequence=3,
        previous_sha256="0" * 64,
    )
    with pytest.raises(ReviewVerificationError, match="hash chain does not match"):
        verify_sentiment_review_bundle(
            bundle_path=fixture["bundle"],
            expected_bundle_sha256=fixture["bundle_sha256"],
            trust_store_path=fixture["trust"],
            expected_trust_store_sha256=fixture["trust_sha256"],
            revocations_path=broken_chain_path,
            expected_revocations_sha256=broken_chain_hash,
            revocation_checkpoint_path=fixture["checkpoint"],
            report=report,
            report_path=fixture["report"],
            now=datetime(2026, 11, 22, 8, 2, tzinfo=timezone.utc),
        )

    with pytest.raises(ReviewVerificationError, match="clock moved behind"):
        verify_sentiment_review_bundle(
            bundle_path=fixture["bundle"],
            expected_bundle_sha256=fixture["bundle_sha256"],
            trust_store_path=fixture["trust"],
            expected_trust_store_sha256=fixture["trust_sha256"],
            revocations_path=sequence_two_path,
            expected_revocations_sha256=sequence_two_hash,
            revocation_checkpoint_path=fixture["checkpoint"],
            report=report,
            report_path=fixture["report"],
            now=datetime(2026, 11, 22, 7, 50, tzinfo=timezone.utc),
        )


def test_revocation_registry_rejects_invalid_authority_signature(tmp_path):
    fixture = quorum_fixture(tmp_path)
    registry = load_review_json(fixture["revocations"])
    registry["signatures"][0]["value_base64"] = base64.b64encode(b"x" * 64).decode("ascii")
    tampered_path = tmp_path / "tampered-revocations.json"
    write_json(tampered_path, registry)

    with pytest.raises(ReviewVerificationError, match="signature 1 is invalid"):
        verify_sentiment_review_bundle(
            bundle_path=fixture["bundle"],
            expected_bundle_sha256=fixture["bundle_sha256"],
            trust_store_path=fixture["trust"],
            expected_trust_store_sha256=fixture["trust_sha256"],
            revocations_path=tampered_path,
            expected_revocations_sha256=hashlib.sha256(tampered_path.read_bytes()).hexdigest(),
            revocation_checkpoint_path=fixture["checkpoint"],
            report=load_review_json(fixture["report"]),
            report_path=fixture["report"],
            now=NOW,
        )


def test_revocation_quorum_rejects_same_organization_and_duplicate_authority(tmp_path):
    same_org = quorum_fixture(
        tmp_path / "same-org", same_revocation_organization=True
    )
    with pytest.raises(ReviewVerificationError, match="authority/organization quorum"):
        verify_sentiment_review_bundle(
            bundle_path=same_org["bundle"],
            expected_bundle_sha256=same_org["bundle_sha256"],
            trust_store_path=same_org["trust"],
            expected_trust_store_sha256=same_org["trust_sha256"],
            revocations_path=same_org["revocations"],
            expected_revocations_sha256=same_org["revocations_sha256"],
            revocation_checkpoint_path=same_org["checkpoint"],
            report=load_review_json(same_org["report"]),
            report_path=same_org["report"],
            now=NOW,
        )

    insufficient = quorum_fixture(tmp_path / "insufficient")
    registry = load_review_json(insufficient["revocations"])
    registry["signatures"] = registry["signatures"][:1]
    insufficient_path = tmp_path / "insufficient" / "one-signature.json"
    write_json(insufficient_path, registry)
    with pytest.raises(ReviewVerificationError, match="not contain enough signatures"):
        verify_sentiment_review_bundle(
            bundle_path=insufficient["bundle"],
            expected_bundle_sha256=insufficient["bundle_sha256"],
            trust_store_path=insufficient["trust"],
            expected_trust_store_sha256=insufficient["trust_sha256"],
            revocations_path=insufficient_path,
            expected_revocations_sha256=hashlib.sha256(
                insufficient_path.read_bytes()
            ).hexdigest(),
            revocation_checkpoint_path=insufficient["checkpoint"],
            report=load_review_json(insufficient["report"]),
            report_path=insufficient["report"],
            now=NOW,
        )

    duplicate = quorum_fixture(tmp_path / "duplicate")
    registry = load_review_json(duplicate["revocations"])
    registry["signatures"][1] = dict(registry["signatures"][0])
    duplicate_path = tmp_path / "duplicate" / "duplicate-authority.json"
    write_json(duplicate_path, registry)
    with pytest.raises(ReviewVerificationError, match="repeats an authority"):
        verify_sentiment_review_bundle(
            bundle_path=duplicate["bundle"],
            expected_bundle_sha256=duplicate["bundle_sha256"],
            trust_store_path=duplicate["trust"],
            expected_trust_store_sha256=duplicate["trust_sha256"],
            revocations_path=duplicate_path,
            expected_revocations_sha256=hashlib.sha256(duplicate_path.read_bytes()).hexdigest(),
            revocation_checkpoint_path=duplicate["checkpoint"],
            report=load_review_json(duplicate["report"]),
            report_path=duplicate["report"],
            now=NOW,
        )

def test_review_request_and_canonical_payload_do_not_create_approval(tmp_path):
    report_path, _, receipt_path, _, _ = signed_fixture(tmp_path)
    report = load_review_json(report_path)

    request = prepare_sentiment_review_request(report, report_path)

    assert request["required_scope"] == list(REQUIRED_REVIEW_SCOPE)
    assert "signature" not in request
    assert "approved" not in request
    request_path = tmp_path / "request.json"
    payload_path = tmp_path / "receipt.payload"
    assert audit_cli_main([
        "prepare-sentiment-review-request", str(report_path), "--output", str(request_path)
    ]) == 0
    assert audit_cli_main([
        "canonicalize-review-receipt", str(receipt_path), "--output", str(payload_path)
    ]) == 0
    assert payload_path.read_bytes() == canonical_review_payload(load_review_json(receipt_path))


def test_review_receipt_rejects_tampering_expiry_and_unpinned_trust_store(tmp_path):
    report_path, trust_path, receipt_path, trust_hash, private_key = signed_fixture(tmp_path)
    report = load_review_json(report_path)
    receipt = load_review_json(receipt_path)
    receipt["signature"]["value_base64"] = base64.b64encode(b"x" * 64).decode("ascii")
    tampered_path = tmp_path / "tampered-receipt.json"
    write_json(tampered_path, receipt)
    with pytest.raises(ReviewVerificationError, match="signature is invalid"):
        verify_sentiment_review_receipt(
            receipt_path=tampered_path,
            trust_store_path=trust_path,
            expected_trust_store_sha256=trust_hash,
            report=report,
            report_path=report_path,
            now=NOW,
        )

    expired = load_review_json(receipt_path)
    expired["expires_at"] = "2026-11-22T07:30:00Z"
    expired["signature"]["value_base64"] = base64.b64encode(
        private_key.sign(canonical_review_payload(expired))
    ).decode("ascii")
    expired_path = tmp_path / "expired-receipt.json"
    write_json(expired_path, expired)
    with pytest.raises(ReviewVerificationError, match="expired"):
        verify_sentiment_review_receipt(
            receipt_path=expired_path,
            trust_store_path=trust_path,
            expected_trust_store_sha256=trust_hash,
            report=report,
            report_path=report_path,
            now=NOW,
        )

    with pytest.raises(ReviewVerificationError, match="trust-store SHA-256 mismatch"):
        verify_sentiment_review_receipt(
            receipt_path=receipt_path,
            trust_store_path=trust_path,
            expected_trust_store_sha256="0" * 64,
            report=report,
            report_path=report_path,
            now=NOW,
        )
