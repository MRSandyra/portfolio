"""Command-line interface for training, inference, evaluation, and a tiny demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tokopedia_sentiment.active_learning import create_active_learning_batch
from tokopedia_sentiment.annotation import annotation_key_path, create_annotation_batch
from tokopedia_sentiment.annotation_review import (
    analyze_annotation_agreement,
    build_adjudicated_gold_set,
)
from tokopedia_sentiment.case_study_data import prepare_case_study_reviews
from tokopedia_sentiment.case_study_training import (
    DATASET_SOURCE,
    CaseStudyTrainingConfig,
    train_case_study,
)
from tokopedia_sentiment.constants import DEFAULT_DATASET_URL
from tokopedia_sentiment.data import load_reviews, prepare_reviews
from tokopedia_sentiment.demo_data import DEMO_LABELS, DEMO_TEXTS
from tokopedia_sentiment.evaluation import evaluate_probabilities
from tokopedia_sentiment.inference import predict_reviews
from tokopedia_sentiment.modeling import build_pipeline, load_artifact
from tokopedia_sentiment.monitoring import compute_drift_report, write_drift_report
from tokopedia_sentiment.preprocessing import normalize_indonesian_text
from tokopedia_sentiment.training import TrainingConfig, train
from tokopedia_sentiment.transformer_experiment import (
    TransformerConfig,
    run_transformer_experiment,
)
from tokopedia_sentiment.triage import apply_triage_policy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokopedia-sentiment",
        description="Reproducible Indonesian sentiment baseline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train from a local Kaggle CSV")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    train_parser.add_argument("--reports", type=Path, default=Path("reports"))
    train_parser.add_argument("--source-uri", default=DEFAULT_DATASET_URL)
    train_parser.add_argument("--text-column", default="Review")
    train_parser.add_argument("--rating-column", default="Rating")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--calibrate", action="store_true")

    predict_parser = commands.add_parser("predict", help="predict with a saved local artifact")
    predict_parser.add_argument("--artifact", type=Path, required=True)
    predict_parser.add_argument("review", nargs="+")

    evaluate_parser = commands.add_parser(
        "evaluate", help="evaluate a model on a separate labeled CSV"
    )
    evaluate_parser.add_argument("--artifact", type=Path, required=True)
    evaluate_parser.add_argument("--data", type=Path, required=True)
    evaluate_parser.add_argument("--text-column", default="Review")
    evaluate_parser.add_argument("--rating-column", default="Rating")

    benchmark_parser = commands.add_parser(
        "benchmark", help="run the three-class 2025 portfolio benchmark"
    )
    benchmark_parser.add_argument("--data", type=Path, required=True)
    benchmark_parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/case-study-model.joblib")
    )
    benchmark_parser.add_argument("--reports", type=Path, default=Path("reports/case-study"))
    benchmark_parser.add_argument("--source-uri", default=DATASET_SOURCE)
    benchmark_parser.add_argument(
        "--split-strategy", choices=["temporal", "group", "random"], default="temporal"
    )
    benchmark_parser.add_argument("--target-minority-recall", type=float, default=0.70)
    benchmark_parser.add_argument("--bootstrap-iterations", type=int, default=500)
    benchmark_parser.add_argument("--cv-folds", type=int, default=3)
    benchmark_parser.add_argument("--cv-repeats", type=int, default=2)
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.add_argument("--no-group-holdout", action="store_true")
    benchmark_parser.add_argument("--no-two-stage", action="store_true")
    benchmark_parser.add_argument("--no-rolling-backtest", action="store_true")

    annotation_parser = commands.add_parser(
        "sample-annotation", help="create an ignored human-labeling batch"
    )
    annotation_parser.add_argument("--data", type=Path, required=True)
    annotation_parser.add_argument(
        "--output", type=Path, default=Path("annotations/review_batch.csv")
    )
    annotation_parser.add_argument("--samples-per-rating", type=int, default=100)
    annotation_parser.add_argument("--seed", type=int, default=42)

    active_parser = commands.add_parser(
        "sample-active-learning", help="sample uncertain/disagreeing reviews for annotation"
    )
    active_parser.add_argument("--data", type=Path, required=True)
    active_parser.add_argument("--artifact", type=Path, required=True)
    active_parser.add_argument(
        "--output", type=Path, default=Path("annotations/active_learning_batch.csv")
    )
    active_parser.add_argument("--batch-size", type=int, default=500)
    active_parser.add_argument("--exclude-batch", type=Path)

    triage_parser = commands.add_parser(
        "triage", help="route reviews to precision-constrained queues or abstain"
    )
    triage_parser.add_argument("--artifact", type=Path, required=True)
    triage_parser.add_argument("review", nargs="+")

    drift_parser = commands.add_parser("drift-report", help="compare historical time windows")
    drift_parser.add_argument("--data", type=Path, required=True)
    drift_parser.add_argument("--reference-end", required=True)
    drift_parser.add_argument("--current-start", required=True)
    drift_parser.add_argument("--output", type=Path, default=Path("reports/drift/metrics.json"))

    transformer_parser = commands.add_parser(
        "transformer-benchmark", help="fine-tune optional IndoBERT-Lite baseline"
    )
    transformer_parser.add_argument("--data", type=Path, required=True)
    transformer_parser.add_argument("--output", type=Path, default=Path("artifacts/indobert-lite"))
    transformer_parser.add_argument(
        "--report", type=Path, default=Path("reports/transformer/metrics.json")
    )
    transformer_parser.add_argument("--epochs", type=int, default=2)
    transformer_parser.add_argument("--batch-size", type=int, default=32)
    transformer_parser.add_argument("--max-length", type=int, default=128)
    transformer_parser.add_argument("--positive-ratio", type=float, default=5.0)
    transformer_parser.add_argument("--seed", type=int, default=42)

    agreement_parser = commands.add_parser(
        "annotation-agreement", help="measure two annotators and create disagreement queue"
    )
    agreement_parser.add_argument("--annotator-a", type=Path, required=True)
    agreement_parser.add_argument("--annotator-b", type=Path, required=True)
    agreement_parser.add_argument(
        "--report", type=Path, default=Path("reports/annotation/agreement.json")
    )
    agreement_parser.add_argument(
        "--disagreements", type=Path, default=Path("annotations/disagreements.csv")
    )

    gold_parser = commands.add_parser(
        "build-gold-set", help="merge agreements and completed adjudication"
    )
    gold_parser.add_argument("--annotator-a", type=Path, required=True)
    gold_parser.add_argument("--annotator-b", type=Path, required=True)
    gold_parser.add_argument("--adjudication", type=Path, required=True)
    gold_parser.add_argument("--output", type=Path, default=Path("annotations/gold_set.csv"))

    demo_parser = commands.add_parser("demo", help="run without the full dataset")
    demo_parser.add_argument(
        "review",
        nargs="*",
        default=["produk enak dan bagus", "barang rusak dan sangat mengecewakan"],
    )
    return parser


def _command_train(arguments: argparse.Namespace) -> int:
    metadata = train(
        TrainingConfig(
            data_path=arguments.data,
            artifact_path=arguments.artifact,
            report_directory=arguments.reports,
            source_uri=arguments.source_uri,
            text_column=arguments.text_column,
            rating_column=arguments.rating_column,
            seed=arguments.seed,
            calibrate=arguments.calibrate,
        )
    )
    print(json.dumps(metadata, indent=2))
    return 0


def _command_predict(arguments: argparse.Namespace) -> int:
    bundle = load_artifact(arguments.artifact)
    calibrated = bool(bundle["metadata"].get("model", {}).get("probabilities_calibrated"))
    metadata = bundle["metadata"]
    policy = metadata.get("decision_policy", {})
    predictions = predict_reviews(
        bundle["model"],
        arguments.review,
        probabilities_calibrated=calibrated,
        class_names=metadata.get("class_names"),
        thresholds=policy.get("thresholds"),
        majority_class=policy.get("majority_class"),
    )
    print(json.dumps([prediction.to_dict() for prediction in predictions], indent=2))
    return 0


def _command_evaluate(arguments: argparse.Namespace) -> int:
    bundle = load_artifact(arguments.artifact)
    frame, summary = prepare_reviews(
        load_reviews(arguments.data),
        text_column=arguments.text_column,
        rating_column=arguments.rating_column,
    )
    probability = bundle["model"].predict_proba(frame["text"].tolist())[:, 1]
    result = {
        "preparation": summary.to_dict(),
        "metrics": evaluate_probabilities(frame["label"].to_numpy(), probability),
    }
    print(json.dumps(result, indent=2))
    return 0


def _command_demo(arguments: argparse.Namespace) -> int:
    model = build_pipeline().fit(DEMO_TEXTS, DEMO_LABELS)
    predictions = predict_reviews(model, arguments.review, probabilities_calibrated=False)
    payload = {
        "warning": "Toy synthetic data: predictions are a behavior demo, not quality evidence.",
        "predictions": [prediction.to_dict() for prediction in predictions],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _command_benchmark(arguments: argparse.Namespace) -> int:
    report = train_case_study(
        CaseStudyTrainingConfig(
            data_path=arguments.data,
            artifact_path=arguments.artifact,
            report_directory=arguments.reports,
            source_uri=arguments.source_uri,
            split_strategy=arguments.split_strategy,
            target_minority_recall=arguments.target_minority_recall,
            bootstrap_iterations=arguments.bootstrap_iterations,
            cv_folds=arguments.cv_folds,
            cv_repeats=arguments.cv_repeats,
            seed=arguments.seed,
            run_group_holdout=not arguments.no_group_holdout,
            compare_two_stage=not arguments.no_two_stage,
            run_rolling_backtest=not arguments.no_rolling_backtest,
        )
    )
    summary = {
        "artifact": str(arguments.artifact),
        "reports": str(arguments.reports),
        "dataset": report["dataset"],
        "primary_split": report["primary_split"],
        "model_selection": report["model_selection"],
        "decision_policy": report["decision_policy"],
        "triage_policy": report["triage_policy"],
        "test_metrics": report["test_metrics"],
        "triage_metrics": report["triage_metrics"],
        "bootstrap_confidence_intervals": report["bootstrap_confidence_intervals"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _command_sample_annotation(arguments: argparse.Namespace) -> int:
    destination = create_annotation_batch(
        pd.read_csv(arguments.data),
        arguments.output,
        samples_per_rating=arguments.samples_per_rating,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "output": str(destination),
                "adjudication_key": str(annotation_key_path(destination)),
                "warning": (
                    "Both files are sensitive and ignored; do not show the key to annotators."
                ),
            },
            indent=2,
        )
    )
    return 0


def _command_sample_active_learning(arguments: argparse.Namespace) -> int:
    bundle = load_artifact(arguments.artifact)
    prepared, _ = prepare_case_study_reviews(pd.read_csv(arguments.data))
    excluded: set[object] = set()
    if arguments.exclude_batch:
        excluded = set(pd.read_csv(arguments.exclude_batch)["review_id"])
    output, key = create_active_learning_batch(
        prepared,
        bundle["model"],
        bundle["metadata"],
        arguments.output,
        batch_size=arguments.batch_size,
        exclude_review_ids=excluded,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "adjudication_key": str(key),
                "excluded_review_ids": len(excluded),
                "warning": "Keep the key hidden from annotators.",
            },
            indent=2,
        )
    )
    return 0


def _command_triage(arguments: argparse.Namespace) -> int:
    bundle = load_artifact(arguments.artifact)
    metadata = bundle["metadata"]
    policy = metadata.get("triage_policy")
    if not policy:
        raise ValueError("artifact has no triage policy; rerun the enhanced benchmark")
    probability = bundle["model"].predict_proba(arguments.review)
    decisions = apply_triage_policy(probability, metadata["class_names"], policy)
    payload = []
    for review, row, decision in zip(arguments.review, probability, decisions, strict=True):
        payload.append(
            {
                "text": review,
                "normalized_text": normalize_indonesian_text(review),
                "decision": str(decision),
                "argmax_sentiment": metadata["class_names"][int(row.argmax())],
                "probabilities": {
                    name: float(row[index]) for index, name in enumerate(metadata["class_names"])
                },
            }
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _command_drift_report(arguments: argparse.Namespace) -> int:
    prepared, _ = prepare_case_study_reviews(pd.read_csv(arguments.data))
    reference_end = pd.Timestamp(arguments.reference_end)
    current_start = pd.Timestamp(arguments.current_start)
    reference = prepared.loc[prepared["date"] <= reference_end]
    current = prepared.loc[prepared["date"] >= current_start]
    if reference.empty or current.empty:
        raise ValueError("reference and current windows must both contain rows")
    report = compute_drift_report(reference, current)
    destination = write_drift_report(report, arguments.output)
    print(json.dumps({"output": str(destination), "report": report}, indent=2))
    return 0


def _command_transformer_benchmark(arguments: argparse.Namespace) -> int:
    report = run_transformer_experiment(
        TransformerConfig(
            data_path=arguments.data,
            output_directory=arguments.output,
            report_path=arguments.report,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            max_length=arguments.max_length,
            positive_to_minority_ratio=arguments.positive_ratio,
            seed=arguments.seed,
        )
    )
    print(
        json.dumps(
            {
                "report": str(arguments.report),
                "model_output": str(arguments.output),
                "device": report["device"],
                "history": report["history"],
                "test_metrics": report["test_metrics"],
                "triage_metrics": report["triage_metrics"],
            },
            indent=2,
        )
    )
    return 0


def _command_annotation_agreement(arguments: argparse.Namespace) -> int:
    report = analyze_annotation_agreement(
        arguments.annotator_a,
        arguments.annotator_b,
        report_path=arguments.report,
        disagreement_path=arguments.disagreements,
    )
    print(json.dumps(report, indent=2))
    return 0


def _command_build_gold_set(arguments: argparse.Namespace) -> int:
    output = build_adjudicated_gold_set(
        arguments.annotator_a,
        arguments.annotator_b,
        arguments.adjudication,
        arguments.output,
    )
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    commands = {
        "train": _command_train,
        "predict": _command_predict,
        "evaluate": _command_evaluate,
        "demo": _command_demo,
        "benchmark": _command_benchmark,
        "sample-annotation": _command_sample_annotation,
        "sample-active-learning": _command_sample_active_learning,
        "triage": _command_triage,
        "drift-report": _command_drift_report,
        "transformer-benchmark": _command_transformer_benchmark,
        "annotation-agreement": _command_annotation_agreement,
        "build-gold-set": _command_build_gold_set,
    }
    return commands[arguments.command](arguments)


if __name__ == "__main__":
    raise SystemExit(main())
