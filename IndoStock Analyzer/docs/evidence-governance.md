# Evidence governance and activation boundary

This project separates four questions that must not be collapsed into one claim:

1. **Format and internal consistency:** required fields exist and reported arithmetic agrees with the confusion matrix or backtest periods.
2. **Reproducibility:** the bound input files match their hashes and rerunning the recorded evaluation policy reproduces the material report sections.
3. **Metric eligibility:** the sentiment report meets the configured sample, provenance, temporal-split, class-recall, macro-F1, and calibration thresholds.
4. **Runtime evidence approval:** the exact report is pinned by SHA-256 and covered by a valid multi-reviewer bundle whose receipts are signed by operator-trusted keys and are not listed in the current pinned revocation registry.

A checksum answers only whether bytes match an expected digest. It does not prove that annotations are correct, observations are genuinely point-in-time, the reviewer is independent, or a model will generalize. `indostock-audit validate-report` therefore reports internal consistency rather than calling the research evidence verified.

Phase 7 adds exact-input recomputation. It closes the gap where an internally plausible ECE, slice, calibration, or association value could be altered without contradicting summary fields. It still uses the same implementation, so it is a reproducibility check rather than independent validation.

## Sentiment evidence lifecycle

`indostock-sentiment evaluate` produces a candidate with:

- `evidence.level: generated_from_bound_inputs`;
- SHA-256 bindings for annotations, predictions, dataset manifest, and model manifest;
- `activation_gate.active: false`, including when `metrics_eligible` is true.

An external process must review the frozen inputs, annotation protocol and agreement, model identity, leakage controls, aggregate metrics, calibration, and failure slices. Phase 8 keeps the report immutable and records approval in detached Ed25519-signed receipts. Phase 9 requires a quorum of distinct reviewers and organizations plus a fresh pinned revocation registry. Phase 10 authenticates that registry and enforces a local monotonic hash-chain checkpoint. Phase 11 requires a distinct-organization quorum for the revocation authorities themselves. A self-asserted review field inside the report is not accepted.

The operator must obtain the report digest, signed-receipt bundle, reviewer trust policy, and signed revocation registry through approved channels and configure all ten values together:

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

The runtime verifies the report digest, pinned trust store, multi-reviewer quorum bundle, current signed revocation registry, monotonic checkpoint, exact artifact bindings, and every signature before loading the plugin. A review claim inside the report, insufficient organizational quorum, revoked approval, stale registry, untrusted signature, rollback, fork, sequence gap, or expired receipt remains inactive. See [`independent-review-protocol.md`](independent-review-protocol.md).

## Backtest evidence boundary

Backtest reports bind the observations and manifest hashes and retain `research_only_not_investment_validated`. The validator recomputes material portfolio summaries and policy flags from the reported periods, but it cannot reconstruct the original observations from hashes or establish market-data correctness. Independent replication from licensed, frozen point-in-time inputs remains necessary before making any stronger research claim.

## Remaining trust assumptions

- The operator receives the expected digest through a trustworthy channel.
- The signature proves possession of a trusted private key, not reviewer competence, independence, or truthfulness.
- Reviewer authorization, private-key protection, rotation, compromise response, and revocation remain governance responsibilities.
- Reviewer and revocation-authority quorums rely on declared organizational identities and do not prove absence of common ownership or collusion.
- Revocation snapshots are offline and can lag newly issued revocations until the next operator refresh.
- The local checkpoint is not a public transparency log. A privileged actor who can replace both configuration and checkpoint storage can re-bootstrap a still-valid older registry.
- Initial checkpoint creation is trust-on-first-use around the separately pinned digest and does not independently discover the latest global sequence.
- Receipt expiry and registry freshness require an accurate system clock. The checkpoint detects backward movement after first use but does not provide externally trusted time.
- Input hashes bind files but do not validate their truth, legality, completeness, or representativeness.
- Model execution and prediction generation are outside this repository and need their own reproducible environment and supply-chain controls.
- No evidence level changes the project disclaimer: outputs are exploratory research and not financial advice.
