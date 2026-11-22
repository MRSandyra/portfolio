from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .sentiment_evaluation import (
    EvaluationDataError,
    build_validation_report,
    compare_validation_reports,
    load_json,
    load_jsonl,
    sha256_file,
    validate_annotations,
    validate_predictions,
    write_json_atomic,
)
from .artifacts import write_checksum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indostock-sentiment",
        description="Validate annotations and evaluate Indonesian financial-headline sentiment.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-annotations")
    validate.add_argument("annotations", type=Path)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--annotations", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--dataset-manifest", type=Path, required=True)
    evaluate.add_argument("--model-manifest", type=Path, required=True)
    evaluate.add_argument("--report-id", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare-reports")
    compare.add_argument("reports", type=Path, nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "compare-reports":
            comparison = compare_validation_reports(load_json(path) for path in args.reports)
            print(json.dumps(comparison, ensure_ascii=False, indent=2))
            return 0
        annotations = validate_annotations(load_jsonl(args.annotations))
        if args.command == "validate-annotations":
            summary = annotations["summary"]
            print(
                f"Valid annotations: {summary['total_records']} records, "
                f"{summary['evaluable_records']} evaluable"
            )
            print(f"Agreement: {summary['agreement']}")
            return 0

        predictions = validate_predictions(load_jsonl(args.predictions))
        dataset = load_json(args.dataset_manifest)
        actual_hash = sha256_file(args.annotations)
        if dataset.get("content_sha256") != actual_hash:
            raise EvaluationDataError(
                "dataset manifest content_sha256 does not match the annotation file"
            )
        model_manifest = load_json(args.model_manifest)
        report = build_validation_report(
            annotation_result=annotations,
            predictions=predictions,
            dataset=dataset,
            model=model_manifest,
            report_id=args.report_id,
            evidence={
                "level": "generated_from_bound_inputs",
                "independently_reviewed": False,
                "generator": {"name": "indostock-sentiment", "version": __version__},
                "input_sha256": {
                    "annotations": actual_hash,
                    "predictions": sha256_file(args.predictions),
                    "dataset_manifest": sha256_file(args.dataset_manifest),
                    "model_manifest": sha256_file(args.model_manifest),
                },
            },
        )
        write_json_atomic(args.output, report)
        write_checksum(args.output)
        gate = report["activation_gate"]
        print(f"Report: {args.output}")
        print(f"Runtime activation: {'ACTIVE' if gate['active'] else 'INACTIVE'}")
        print(f"Metric eligibility: {'PASS' if gate['metrics_eligible'] else 'FAIL'}")
        if gate["reasons"]:
            print("Reasons: " + ", ".join(gate["reasons"]))
        return 0
    except EvaluationDataError as exc:
        print(f"Evaluation data error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
