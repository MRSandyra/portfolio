# Signed external-review and revocation protocol

Phase 8 separates report generation, external review, and runtime operation. It does not bundle a reviewer, private key, model, dataset, or approval receipt. Outputs remain exploratory and are not financial advice.

## Roles

- **Producer:** freezes sentiment inputs, generates the candidate report, and completes Phase 7 recomputation.
- **Reviewer:** obtains frozen evidence independently, performs independent recomputation plus manual failure-slice review, discloses conflicts, and signs a detached receipt with an externally managed Ed25519 private key.
- **Operator:** chooses trusted reviewer public keys, pins the exact trust-store hash and report hash, and configures runtime activation.

The same person can technically hold multiple roles; software cannot prove organizational separation. Governance must prevent that when independence matters.

## Review request

The producer creates an unsigned request:

```bash
indostock-audit prepare-sentiment-review-request evaluations/candidate-report.json \
  --output review-requests/candidate.json
```

This contains no approval or signature. It lists the exact report digest, bound input digests, evaluation policy, required decision, and review scope.

## Receipt signing

Start from `config/sentiment_review_receipt.example.json`. The reviewer completes every field, then produces canonical signing bytes:

```bash
indostock-audit canonicalize-review-receipt review-receipts/unsigned.json \
  --output review-receipts/unsigned.payload
```

The payload begins with `IndoStockAnalyzer/sentiment-independent-review/v1`, a NUL byte, and canonical UTF-8 JSON with sorted keys, compact separators, no NaN values, and the top-level `signature` field removed. Sign those exact bytes using an external Ed25519 tool or key-management service, then place the base64 signature in `signature.value_base64`.

This repository deliberately provides no private-key generation or signing command. Never commit private keys, signing credentials, generated receipts, datasets, predictions, or review payloads.

## Trust store and verification

Copy `config/sentiment_reviewer_trust_store.example.json` to operator-managed local configuration and replace all placeholders. Public keys are not secrets, but trust-store contents are authorization policy and must be reviewed. Obtain its SHA-256 through an approved channel and configure that digest separately.

```bash
indostock-audit verify-sentiment-review-receipt review-receipts/candidate.json \
  --report evaluations/candidate-report.json \
  --trust-store config/sentiment_reviewer_trust_store.json \
  --trust-store-sha256 64-lowercase-hex-digest
```

Verification checks exact report/input bindings, required scope, methodology attestations, identity against the trust store, conflict disclosure, key authorization, key validity, receipt issue/expiry dates, and the Ed25519 signature. Receipt lifetime cannot exceed 366 days.

## Quorum bundle and revocation

Phase 9 runtime activation does not accept a single receipt. Combine separately signed receipts:

```bash
indostock-audit build-sentiment-review-bundle \
  --report evaluations/candidate-report.json \
  --receipt review-receipts/reviewer-a.json \
  --receipt review-receipts/reviewer-b.json \
  --output review-receipts/candidate-bundle.json
```

The pinned trust store contains `minimum_distinct_reviewers` and `minimum_distinct_organizations`, both at least two. Repeated review IDs, reviewer IDs, or keys do not count twice. Each organization must also match the identity authorized by the trust store.

Maintain a local registry based on `config/sentiment_review_revocations.example.json`. It can revoke a review ID, signing key, or exact report digest. Phase 10 introduced a stable registry ID, monotonic sequence, predecessor digest, and local checkpoint. Phase 11 requires schema v3 and a quorum of Ed25519 signatures from distinct revocation authorities and organizations listed in the pinned trust store. The legacy single-signature schema v2 is no longer accepted. The registry fails closed after `next_update_at`; its refresh interval cannot exceed 31 days.

Generate the canonical unsigned bytes and sign them with an external key-management process:

```bash
indostock-audit canonicalize-review-revocations \
  config/sentiment_review_revocations.unsigned.json \
  --output config/sentiment_review_revocations.payload
```

The canonical payload uses the `IndoStockAnalyzer/sentiment-review-revocations/v3` domain and removes the top-level `signatures` list. Each authority signs the same bytes independently, and the operator adds every signature to that list. The application never accepts or stores authority private keys. For sequence one, `previous_registry_sha256` is null. Every later sequence must reference the exact digest of its predecessor.

The trust-store governance policy must set `minimum_revocation_authorities` and `minimum_revocation_organizations` to at least two. Duplicate authority IDs or keys do not count twice. All supplied signatures must be valid; an invalid extra signature fails the registry instead of being ignored.

```bash
indostock-audit verify-sentiment-review-bundle review-receipts/candidate-bundle.json \
  --bundle-sha256 64-lowercase-hex-bundle-digest \
  --report evaluations/candidate-report.json \
  --trust-store config/sentiment_reviewer_trust_store.json \
  --trust-store-sha256 64-lowercase-hex-trust-digest \
  --revocations config/sentiment_review_revocations.json \
  --revocations-sha256 64-lowercase-hex-revocation-digest \
  --checkpoint state/sentiment_review_revocations.checkpoint.json
```

On first use, the separately pinned registry becomes the local trust-on-first-use checkpoint. Later verification accepts only the same sequence and digest or the immediately following sequence whose previous digest matches. Rollback, same-sequence forks, gaps, broken chains, and a clock moving more than five minutes behind the last observation fail closed. Apply every signed update in order and preserve the checkpoint across deployments.

## Runtime configuration

All ten values are required together:

```text
INDOSTOCK_SENTIMENT_PROVIDER
INDOSTOCK_SENTIMENT_VALIDATION_REPORT
INDOSTOCK_SENTIMENT_VALIDATION_SHA256
INDOSTOCK_SENTIMENT_REVIEW_BUNDLE
INDOSTOCK_SENTIMENT_REVIEW_BUNDLE_SHA256
INDOSTOCK_SENTIMENT_REVIEWER_TRUST_STORE
INDOSTOCK_SENTIMENT_REVIEWER_TRUST_STORE_SHA256
INDOSTOCK_SENTIMENT_REVIEW_REVOCATIONS
INDOSTOCK_SENTIMENT_REVIEW_REVOCATIONS_SHA256
INDOSTOCK_SENTIMENT_REVIEW_CHECKPOINT
```

The runtime verifies report, quorum bundle, trust policy, signed revocation chain, and monotonic checkpoint before loading the plugin. Missing, expired, stale, revoked, tampered, unauthorized, duplicate, incorrectly bound, rolled-back, forked, or skipped evidence leaves Indonesian sentiment disabled.

## Cryptographic boundary

The implementation uses PyCA `cryptography`'s Ed25519 verification API. A valid signature demonstrates that someone possessing the private key signed the canonical payload. Reviewer and revocation-authority quorums only count declared identities and organizations; they cannot establish true organizational independence or rule out collusion. Key compromise and emergency revocation require timely operator distribution of a new pinned registry or trust store. The local checkpoint is not a public transparency log and cannot resist an actor able to replace both runtime configuration and checkpoint storage.

References: [PyCA Ed25519 documentation](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/) and [cryptography package record](https://pypi.org/project/cryptography/50.0.2/).
