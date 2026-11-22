# Human handoff

All automatable engineering and offline evaluation work is complete. The following steps require human judgment, legal authority, or an external GitHub account.

## 1. Independent annotation

Files prepared locally and ignored by Git:

- `annotations/review_batch.csv`: 500 rating-stratified pilot reviews.
- `annotations/active_learning_batch.csv`: 500 uncertain/disagreeing reviews with zero pilot overlap.
- matching `_key.csv` files: retain privately; never show to annotators.

Create two copies of each blind batch, one per Indonesian-speaking annotator. Annotators fill only `human_label`, `uncertain`, `annotator_id`, and optional `notes`, following [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md).

Run agreement analysis after both copies are returned:

```bash
uv run tokopedia-sentiment annotation-agreement \
  --annotator-a annotations/annotator_a.csv \
  --annotator-b annotations/annotator_b.csv \
  --report reports/annotation/agreement.json \
  --disagreements annotations/disagreements.csv
```

Have a third reviewer complete `adjudicated_label` for every disagreement/uncertain row. Then build the validated set:

```bash
uv run tokopedia-sentiment build-gold-set \
  --annotator-a annotations/annotator_a.csv \
  --annotator-b annotations/annotator_b.csv \
  --adjudication annotations/disagreements.csv \
  --output annotations/gold_set.csv
```

Freeze a final gold-test partition before any further model tuning. Report annotator count, raw agreement, Cohen's kappa, adjudication policy, and support per class.

## 2. Dataset permission

The public Kaggle page must be checked manually while signed in. Save the displayed license name, URL, access date, and any redistribution/commercial restrictions. Obtain written permission from the owner if portfolio publication exceeds “Educational & Research Purpose.” Do not publish the CSV, annotation batches, or review excerpts until rights are clear.

## 3. GitHub connection

This workspace has no `.git` directory or remote. Create/connect the intended GitHub repository, confirm that `git status` does not list CSV, `.env`, model, report, cache, or annotation files, then push the source. Verify the existing CI workflow runs successfully and enable branch protection requiring the lint/test job.

## 4. Optional full transformer run

The locked Windows PyTorch wheel was CPU-only. The completed one-epoch smoke run is documented and underperformed the classical model. For a full comparison, install a CUDA-enabled PyTorch build compatible with the RTX 5060 Ti in a separate environment, then run the default two-epoch transformer command. Do not replace the selected model unless the same frozen temporal/product/gold evaluations improve.
