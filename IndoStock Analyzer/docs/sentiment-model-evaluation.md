# Indonesian sentiment evaluation and activation

## Audit conclusion

TextBlob's default Pattern sentiment analyzer is English-oriented and is not evidence of validity on Indonesian financial headlines. IndoNLU provides Indonesian sentiment resources, but its SmSA task contains online comments/reviews rather than issuer-specific financial headlines. ID-SMSA is closer to finance but contains stock-market tweets, which still differ in style, context, and label target from news headlines.

Public financial model cards also report materially different results and use different or augmented datasets. Scores from those cards are not directly comparable and are not accepted as local validation. Relevant references:

- [TextBlob API: PatternAnalyzer](https://textblob.readthedocs.io/en/latest/api_reference.html)
- [IndoNLU paper](https://aclanthology.org/2020.aacl-main.85/)
- [NusaCrowd SmSA dataset definition](https://github.com/IndoNLP/nusa-crowd/blob/master/nusacrowd/nusa_datasets/smsa/smsa.py)
- [ID-SMSA dataset article](https://www.sciencedirect.com/science/article/pii/S2352340925003038)
- [ID-SMSA dataset record](https://data.mendeley.com/datasets/tn4vzs8tdw)
- [Example Indonesian FinBERT model card](https://huggingface.co/michaelmanurung/finbert-indonesia)

The consequence is deliberate: Indonesian sentiment remains `UNASSESSED` unless an independently reviewed local report passes metric checks and the operator pins the exact report digest.

## Required comparison

Evaluate candidate methods on the exact same frozen temporal holdout:

1. Majority-class baseline.
2. A transparent lexical/rule baseline with Indonesian negation handling.
3. At least one Indonesian pretrained model fine-tuned for the task.
4. Any finance-specific candidate proposed for activation.

Keep model artifacts outside this repository. Pin an immutable model revision/hash and license in the model manifest. Do not use test-set examples for prompt selection, training, threshold choice, or error-driven rule changes.

## Default metric and evidence gates

Metric eligibility requires all of the following:

- evaluation domain exactly `indonesian_financial_headlines`;
- chronological `temporal_holdout` split and target-aware examples;
- dataset license, source provenance, and frozen-file SHA-256 are recorded;
- immutable model ID and revision are recorded;
- at least 300 evaluable held-out records;
- macro F1 at least 0.75;
- recall at least 0.65 for each of `NEGATIVE`, `NEUTRAL`, and `POSITIVE`;
- expected calibration error no greater than 0.15.

Metric eligibility alone does not activate a provider. Runtime activation also requires an external Ed25519-signed review receipt, a pinned reviewer trust store, and an operator-approved SHA-256 matching the exact report bytes. A self-asserted `evidence.level` is not accepted. See [`evidence-governance.md`](evidence-governance.md) and [`independent-review-protocol.md`](independent-review-protocol.md).

These are engineering guardrails, not proof of investment usefulness. Before changing them, document the cost of each error class and preserve the original report.

Always inspect slices for negation, multiple entities, finance jargon, irony, numeric context, forecasts, rumors, and corporate actions. A passing aggregate score can conceal an unsafe failure mode; slice results are reported but currently require human review rather than an automatic threshold.

## Reproducible evaluation

Create permitted local annotation and prediction JSONL files plus manifests based on the examples in `config/`. Then run:

```bash
indostock-sentiment evaluate \
  --annotations annotations/holdout.jsonl \
  --predictions evaluations/candidate-predictions.jsonl \
  --dataset-manifest config/sentiment_dataset_manifest.json \
  --model-manifest config/sentiment_model_manifest.json \
  --report-id candidate-YYYYMMDD \
  --output evaluations/candidate-report.json
```

The report includes provenance, bound input hashes, agreement, distribution, accuracy, macro F1, class metrics, confusion matrix, calibration error, slices, and gate reasons. It remains an inactive evidence candidate. After external review, configure the report, report digest, signed receipt, reviewer trust store, and trust-store digest together. The provider must be installed separately through Python entry-point group `indostock.sentiment` and expose `predict(text, target)` returning probabilities for `NEGATIVE`, `NEUTRAL`, and `POSITIVE`.

Compare two or more reports only when they identify the same frozen holdout:

```bash
indostock-sentiment compare-reports evaluations/baseline.json evaluations/candidate.json
```
