"""Command-line interface for the flight-offer analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flight_analysis.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze flight offer listings reproducibly.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run quality checks, cleaning, analysis, and charts.")
    run.add_argument("--input", type=Path, required=True, help="Path to source flight.csv")
    run.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    run.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove only known generated files before rebuilding them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        try:
            result = run_pipeline(args.input, args.output_dir, args.clean_output)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 2
        print(json.dumps(result, indent=2))
        return 0
    return 2
