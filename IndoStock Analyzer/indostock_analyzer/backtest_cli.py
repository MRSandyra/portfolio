from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import __version__
from .backtest import (
    BacktestDataError,
    build_backtest_report,
    load_json,
    load_jsonl,
    sha256_file,
    validate_manifest,
    validate_observations,
    write_json_atomic,
)
from .artifacts import write_checksum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indostock-backtest",
        description="Run a point-in-time historical research evaluation; not financial advice.",
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = validate_manifest(load_json(args.manifest), sha256_file(args.observations))
        observations = validate_observations(load_jsonl(args.observations), manifest)
        report = build_backtest_report(
            observations,
            manifest,
            args.report_id,
            evidence={
                "level": "generated_from_bound_inputs",
                "independently_verified": False,
                "generator": {"name": "indostock-backtest", "version": __version__},
                "input_sha256": {
                    "observations": sha256_file(args.observations),
                    "manifest": sha256_file(args.manifest),
                },
            },
        )
        write_json_atomic(args.output, report)
        write_checksum(args.output)
    except BacktestDataError as exc:
        print(f"Backtest data error: {exc}", file=sys.stderr)
        return 2
    print(f"Report: {args.output}")
    print("Status: research_only_not_investment_validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
