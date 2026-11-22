"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_benchmark
from .calibration import run_calibration_phase
from .cohort_policy import (
    audit_cohort_policy,
    authorize_cohort_file_operation,
    freeze_cohort_policy,
)
from .data import acquire_huggingface
from .external_validation import run_external_validation
from .final_results import audit_final_results, freeze_final_results
from .final_model import fit_final_models
from .future_holdout import freeze_future_holdout
from .holdout_release import release_future_holdout
from .inference import build_inference_bundles
from .nested_validation import DEFAULT_C_GRID, run_nested_validation
from .monitoring import run_monitoring_audit, write_model_input_contracts
from .pipeline import run_case_study, run_reproduction
from .quality import audit_event_file, write_manifest
from .rank_era_validation import acquire_high_diamond_10m, validate_rank_era
from .release_readiness import build_release_readiness
from .robustness_validation import run_robustness_validation
from .shift_validation import run_shift_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lol-analytics")
    subcommands = parser.add_subparsers(dest="command", required=True)

    reproduce = subcommands.add_parser("reproduce", help="run features, audit, split, train, report")
    reproduce.add_argument("--input", type=Path, default=Path("data/sample/events.jsonl"))
    reproduce.add_argument("--output", type=Path, default=Path("artifacts/sample"))
    reproduce.add_argument("--horizon", choices=("10", "20", "both"), default="both")
    reproduce.add_argument("--test-fraction", type=float, default=0.25)

    acquire = subcommands.add_parser("acquire", help="stream complete matches from Hugging Face")
    acquire.add_argument("--output", type=Path, default=Path("data/raw/events.parquet"))
    acquire.add_argument("--matches", type=int, default=300)
    acquire.add_argument("--dataset", default="gptilt/lol-ultimate-events-challenger-10m")
    acquire.add_argument("--split", default="train_region_americas")
    acquire.add_argument("--manifest", type=Path)
    acquire.add_argument("--batch-size", type=int, default=10_000)
    acquire.add_argument("--skip-matches", type=int, default=0)

    freeze = subcommands.add_parser(
        "freeze-holdout",
        help="validate and register a strictly future unscored holdout",
    )
    freeze.add_argument("--training-input", type=Path, required=True)
    freeze.add_argument("--holdout-input", type=Path, required=True)
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--expected-matches", type=int, required=True)
    freeze.add_argument("--allow-region-shift", action="store_true")
    freeze.add_argument(
        "--source-dataset", default="gptilt/lol-ultimate-events-challenger-10m"
    )
    freeze.add_argument("--source-config", default="events")
    freeze.add_argument("--source-split", required=True)
    freeze.add_argument("--source-skip-matches", type=int, default=0)
    freeze.add_argument("--source-batch-size", type=int, default=10_000)
    freeze.add_argument(
        "--prior-cohort-input",
        type=Path,
        action="append",
        default=[],
        help="previously evaluated cohort that must be match-disjoint; repeat as needed",
    )

    audit = subcommands.add_parser("audit-data", help="audit a canonical raw event file")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--expected-matches", type=int)

    benchmark = subcommands.add_parser(
        "benchmark", help="run temporal benchmark with bootstrap uncertainty"
    )
    benchmark.add_argument("--input", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--test-fraction", type=float, default=0.25)
    benchmark.add_argument("--bootstrap-iterations", type=int, default=2_000)
    benchmark.add_argument("--cv-splits", type=int, default=3)

    external = subcommands.add_parser(
        "validate-external",
        help="fit on one event file and evaluate a disjoint external event file",
    )
    external.add_argument("--training-input", type=Path, required=True)
    external.add_argument("--external-input", type=Path, required=True)
    external.add_argument("--output", type=Path, required=True)
    external.add_argument(
        "--cohort-policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    external.add_argument("--bootstrap-iterations", type=int, default=2_000)

    shifts = subcommands.add_parser(
        "validate-shifts",
        help="fit once and compare same-region with cross-region test cohorts",
    )
    shifts.add_argument("--training-input", type=Path, required=True)
    shifts.add_argument("--same-region-input", type=Path, required=True)
    shifts.add_argument("--cross-region-input", type=Path, required=True)
    shifts.add_argument("--output", type=Path, required=True)
    shifts.add_argument(
        "--cohort-policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    shifts.add_argument("--bootstrap-iterations", type=int, default=2_000)

    nested = subcommands.add_parser(
        "validate-nested",
        help="nested expanding-window hyperparameter selection and evaluation",
    )
    nested.add_argument("--input", type=Path, required=True)
    nested.add_argument("--output", type=Path, required=True)
    nested.add_argument(
        "--cohort-policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    nested.add_argument("--outer-splits", type=int, default=3)
    nested.add_argument("--inner-splits", type=int, default=2)
    nested.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    nested.add_argument("--bootstrap-iterations", type=int, default=2_000)

    robustness = subcommands.add_parser(
        "validate-robustness",
        help="repeated nested temporal validation with block-bootstrap refitting",
    )
    robustness.add_argument("--input", type=Path, required=True)
    robustness.add_argument("--output", type=Path, required=True)
    robustness.add_argument(
        "--cohort-policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    robustness.add_argument("--outer-splits", type=int, default=3)
    robustness.add_argument("--inner-splits", type=int, default=2)
    robustness.add_argument("--repeats", type=int, default=5)
    robustness.add_argument("--max-shift-fraction", type=float, default=0.4)
    robustness.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    robustness.add_argument("--bootstrap-iterations", type=int, default=2_000)
    robustness.add_argument("--refit-bootstrap-iterations", type=int, default=100)
    robustness.add_argument("--refit-block-size", type=int, default=50)

    final_models = subcommands.add_parser(
        "fit-final-models",
        help="freeze the selection policy and fit one unreleased model per horizon",
    )
    final_models.add_argument("--training-input", type=Path, required=True)
    final_models.add_argument("--selection-evidence", type=Path, required=True)
    final_models.add_argument("--output", type=Path, required=True)
    final_models.add_argument("--manifest", type=Path, required=True)
    final_models.add_argument(
        "--cohort-policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )

    acquire_rank = subcommands.add_parser(
        "acquire-high-diamond",
        help="download and verify the pinned public Diamond I-Master 10-minute cohort",
    )
    acquire_rank.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/high-diamond-10m/high_diamond_ranked_10min.csv"),
    )
    acquire_rank.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/high-diamond-10m.json"),
    )

    rank_era = subcommands.add_parser(
        "validate-rank-era",
        help="score only the frozen early 10-minute model across rank and historical era",
    )
    rank_era.add_argument("--input", type=Path, required=True)
    rank_era.add_argument("--source-manifest", type=Path, required=True)
    rank_era.add_argument("--final-model-manifest", type=Path, required=True)
    rank_era.add_argument("--output", type=Path, required=True)
    rank_era.add_argument(
        "--cohort-policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    rank_era.add_argument("--bootstrap-iterations", type=int, default=2_000)

    contracts = subcommands.add_parser(
        "write-data-contracts",
        help="write the versioned runtime model-input contracts",
    )
    contracts.add_argument(
        "--output",
        type=Path,
        default=Path("data/contracts/model-input-v1.json"),
    )

    monitoring = subcommands.add_parser(
        "audit-monitoring",
        help="audit calibration, drift, and runtime contracts without holdout release",
    )
    monitoring.add_argument("--training-input", type=Path, required=True)
    monitoring.add_argument("--external-input", type=Path, required=True)
    monitoring.add_argument("--external-source-manifest", type=Path, required=True)
    monitoring.add_argument("--final-model-manifest", type=Path, required=True)
    monitoring.add_argument("--robustness-summary", type=Path, required=True)
    monitoring.add_argument(
        "--contract",
        type=Path,
        default=Path("data/contracts/model-input-v1.json"),
    )
    monitoring.add_argument("--output", type=Path, required=True)
    monitoring.add_argument(
        "--cohort-policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    monitoring.add_argument("--bins", type=int, default=10)
    monitoring.add_argument("--bootstrap-iterations", type=int, default=2_000)

    cohort_policy = subcommands.add_parser(
        "freeze-cohort-policy",
        help="freeze checksum-bound roles for every existing cohort",
    )
    cohort_policy.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    cohort_policy.add_argument(
        "--manifest-dir", type=Path, default=Path("data/manifests")
    )

    cohort_audit = subcommands.add_parser(
        "audit-cohort-policy",
        help="verify frozen cohort roles and optional raw-file checksums",
    )
    cohort_audit.add_argument(
        "--policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    cohort_audit.add_argument("--verify-raw-files", action="store_true")

    cohort_authorize = subcommands.add_parser(
        "authorize-cohort-operation",
        help="fail unless a registered cohort role permits an operation",
    )
    cohort_authorize.add_argument(
        "--policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    cohort_authorize.add_argument("--input", type=Path, required=True)
    cohort_authorize.add_argument("--operation", required=True)

    calibration = subcommands.add_parser(
        "calibrate-temporal",
        help="temporally cross-fit and conditionally package development-only calibrators",
    )
    calibration.add_argument("--development-input", type=Path, required=True)
    calibration.add_argument(
        "--cohort-policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    calibration.add_argument("--robustness-summary", type=Path, required=True)
    calibration.add_argument("--final-model-manifest", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--manifest", type=Path, required=True)
    calibration.add_argument("--bins", type=int, default=10)
    calibration.add_argument("--bootstrap-iterations", type=int, default=2_000)

    inference = subcommands.add_parser(
        "build-inference-bundles",
        help="package checksum-bound final inference bundles without refitting",
    )
    inference.add_argument("--development-input", type=Path, required=True)
    inference.add_argument(
        "--cohort-policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    inference.add_argument("--final-model-manifest", type=Path, required=True)
    inference.add_argument("--calibration-manifest", type=Path, required=True)
    inference.add_argument(
        "--contract",
        type=Path,
        default=Path("data/contracts/model-input-v1.json"),
    )
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--manifest", type=Path, required=True)

    readiness = subcommands.add_parser(
        "build-release-readiness",
        help="run development rolling checks and gate a one-time holdout release",
    )
    readiness.add_argument("--development-input", type=Path, required=True)
    readiness.add_argument("--future-holdout-input", type=Path, required=True)
    readiness.add_argument(
        "--cohort-policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    readiness.add_argument("--future-holdout-manifest", type=Path, required=True)
    readiness.add_argument("--final-model-manifest", type=Path, required=True)
    readiness.add_argument("--calibration-manifest", type=Path, required=True)
    readiness.add_argument("--inference-manifest", type=Path, required=True)
    readiness.add_argument(
        "--contract",
        type=Path,
        default=Path("data/contracts/model-input-v1.json"),
    )
    readiness.add_argument("--robustness-summary", type=Path, required=True)
    readiness.add_argument(
        "--ci", type=Path, default=Path(".github/workflows/ci.yml")
    )
    readiness.add_argument("--output", type=Path, required=True)
    readiness.add_argument("--manifest", type=Path, required=True)
    readiness.add_argument("--windows", type=int, default=5)
    readiness.add_argument("--calibration-bins", type=int, default=10)

    release = subcommands.add_parser(
        "release-future-holdout",
        help="evaluate the frozen future holdout exactly once",
    )
    release.add_argument("--future-holdout-input", type=Path, required=True)
    release.add_argument(
        "--cohort-policy",
        type=Path,
        default=Path("data/manifests/cohort-policy-v1.json"),
    )
    release.add_argument(
        "--readiness-manifest",
        type=Path,
        default=Path("data/manifests/release-readiness-v1.json"),
    )
    release.add_argument(
        "--future-holdout-manifest",
        type=Path,
        default=Path("data/manifests/europe-future-500.json"),
    )
    release.add_argument(
        "--inference-manifest",
        type=Path,
        default=Path("data/manifests/inference-bundles-v1.json"),
    )
    release.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/future-holdout-release-v1"),
    )
    release.add_argument("--confirm-one-time-release", action="store_true")
    release.add_argument("--bootstrap-iterations", type=int, default=2_000)
    release.add_argument("--calibration-bins", type=int, default=10)

    final_results = subcommands.add_parser(
        "freeze-final-results",
        help="freeze the consumed holdout metrics and artifact checksums",
    )
    final_results.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("data/manifests/future-holdout-release-v1.json"),
    )
    final_results.add_argument(
        "--release-started-manifest",
        type=Path,
        default=Path("data/manifests/future-holdout-release-started-v1.json"),
    )
    final_results.add_argument(
        "--inference-manifest",
        type=Path,
        default=Path("data/manifests/inference-bundles-v1.json"),
    )
    final_results.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path("data/manifests/calibration-v1.json"),
    )
    final_results.add_argument(
        "--final-model-manifest",
        type=Path,
        default=Path("data/manifests/final-models-2000.json"),
    )
    final_results.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/final-results-v1.json"),
    )

    final_audit = subcommands.add_parser(
        "audit-final-results",
        help="audit final manifest links without rescoring the holdout",
    )
    final_audit.add_argument(
        "--final-results",
        type=Path,
        default=Path("data/manifests/final-results-v1.json"),
    )
    final_audit.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("data/manifests/future-holdout-release-v1.json"),
    )
    final_audit.add_argument(
        "--release-started-manifest",
        type=Path,
        default=Path("data/manifests/future-holdout-release-started-v1.json"),
    )
    final_audit.add_argument(
        "--inference-manifest",
        type=Path,
        default=Path("data/manifests/inference-bundles-v1.json"),
    )
    final_audit.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path("data/manifests/calibration-v1.json"),
    )
    final_audit.add_argument(
        "--final-model-manifest",
        type=Path,
        default=Path("data/manifests/final-models-2000.json"),
    )
    final_audit.add_argument("--verify-artifacts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "acquire":
        path = acquire_huggingface(
            args.output,
            matches=args.matches,
            dataset=args.dataset,
            split=args.split,
            batch_size=args.batch_size,
            skip_matches=args.skip_matches,
        )
        result: dict[str, object] = {"data": path.as_posix()}
        if args.manifest:
            manifest = audit_event_file(path, expected_matches=args.matches)
            manifest["source"] = {
                "dataset": args.dataset,
                "config": "events",
                "split": args.split,
                "requested_matches": args.matches,
                "skipped_complete_matches": args.skip_matches,
            }
            result["manifest"] = write_manifest(args.manifest, manifest).as_posix()
            result["audit_passed"] = manifest["passed"]
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-holdout":
        summary = freeze_future_holdout(
            args.training_input,
            args.holdout_input,
            args.manifest,
            expected_matches=args.expected_matches,
            allow_region_shift=args.allow_region_shift,
            source={
                "dataset": args.source_dataset,
                "config": args.source_config,
                "split": args.source_split,
                "skipped_complete_matches": args.source_skip_matches,
                "batch_size": args.source_batch_size,
                "requested_matches": args.expected_matches,
            },
            prior_cohort_inputs=args.prior_cohort_input,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "audit-data":
        manifest = audit_event_file(args.input, expected_matches=args.expected_matches)
        write_manifest(args.output, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["passed"] else 1
    if args.command == "benchmark":
        summary = run_benchmark(
            args.input,
            args.output,
            test_fraction=args.test_fraction,
            bootstrap_iterations=args.bootstrap_iterations,
            cv_splits=args.cv_splits,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-external":
        summary = run_external_validation(
            args.training_input,
            args.external_input,
            args.output,
            cohort_policy_path=args.cohort_policy,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-shifts":
        summary = run_shift_validation(
            args.training_input,
            args.same_region_input,
            args.cross_region_input,
            args.output,
            cohort_policy_path=args.cohort_policy,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-nested":
        summary = run_nested_validation(
            args.input,
            args.output,
            cohort_policy_path=args.cohort_policy,
            c_grid=args.c_grid,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-robustness":
        summary = run_robustness_validation(
            args.input,
            args.output,
            cohort_policy_path=args.cohort_policy,
            c_grid=args.c_grid,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            repeats=args.repeats,
            max_shift_fraction=args.max_shift_fraction,
            bootstrap_iterations=args.bootstrap_iterations,
            refit_bootstrap_iterations=args.refit_bootstrap_iterations,
            refit_block_size=args.refit_block_size,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "fit-final-models":
        summary = fit_final_models(
            args.training_input,
            args.selection_evidence,
            args.output,
            args.manifest,
            cohort_policy_path=args.cohort_policy,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "acquire-high-diamond":
        summary = acquire_high_diamond_10m(args.output, args.manifest)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-rank-era":
        summary = validate_rank_era(
            args.input,
            args.source_manifest,
            args.final_model_manifest,
            args.output,
            cohort_policy_path=args.cohort_policy,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "write-data-contracts":
        summary = write_model_input_contracts(args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "audit-monitoring":
        summary = run_monitoring_audit(
            args.training_input,
            args.external_input,
            args.external_source_manifest,
            args.final_model_manifest,
            args.robustness_summary,
            args.contract,
            args.output,
            cohort_policy_path=args.cohort_policy,
            bins=args.bins,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-cohort-policy":
        summary = freeze_cohort_policy(
            args.output,
            manifest_dir=args.manifest_dir,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "audit-cohort-policy":
        summary = audit_cohort_policy(
            args.policy,
            verify_raw_files=args.verify_raw_files,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "authorize-cohort-operation":
        cohort = authorize_cohort_file_operation(
            args.policy,
            args.input,
            args.operation,
        )
        print(json.dumps({"authorized": True, "cohort": cohort}, indent=2))
        return 0
    if args.command == "calibrate-temporal":
        summary = run_calibration_phase(
            args.development_input,
            args.cohort_policy,
            args.robustness_summary,
            args.final_model_manifest,
            args.output,
            args.manifest,
            bins=args.bins,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "build-inference-bundles":
        summary = build_inference_bundles(
            args.development_input,
            args.cohort_policy,
            args.final_model_manifest,
            args.calibration_manifest,
            args.contract,
            args.output,
            args.manifest,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "build-release-readiness":
        summary = build_release_readiness(
            args.development_input,
            args.future_holdout_input,
            args.cohort_policy,
            args.future_holdout_manifest,
            args.final_model_manifest,
            args.calibration_manifest,
            args.inference_manifest,
            args.contract,
            args.robustness_summary,
            args.ci,
            args.output,
            args.manifest,
            windows=args.windows,
            calibration_bins=args.calibration_bins,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "release-future-holdout":
        summary = release_future_holdout(
            args.future_holdout_input,
            args.cohort_policy,
            args.readiness_manifest,
            args.future_holdout_manifest,
            args.inference_manifest,
            args.output,
            confirm_one_time_release=args.confirm_one_time_release,
            bootstrap_iterations=args.bootstrap_iterations,
            calibration_bins=args.calibration_bins,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-final-results":
        summary = freeze_final_results(
            args.release_manifest,
            args.release_started_manifest,
            args.inference_manifest,
            args.calibration_manifest,
            args.final_model_manifest,
            args.output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "audit-final-results":
        summary = audit_final_results(
            args.final_results,
            args.release_manifest,
            args.release_started_manifest,
            args.inference_manifest,
            args.calibration_manifest,
            args.final_model_manifest,
            verify_artifacts=args.verify_artifacts,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.horizon == "both":
        summary = run_case_study(args.input, args.output, test_fraction=args.test_fraction)
    else:
        summary = run_reproduction(
            args.input,
            args.output,
            horizon_minutes=int(args.horizon),
            test_fraction=args.test_fraction,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

