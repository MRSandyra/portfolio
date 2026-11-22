"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import create_sample, download_source
from .report import build_report

DEFAULT_RAW = Path("data/raw/pip_percentiles_2021_20260922.csv")
DEFAULT_SAMPLE = Path("data/sample/pip_percentiles_sample.csv")


def download_main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the pinned PIP source CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(download_source(args.output, overwrite=args.overwrite))


def sample_main() -> None:
    parser = argparse.ArgumentParser(description="Derive the deterministic committed sample.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_SAMPLE)
    args = parser.parse_args()
    print(create_sample(args.input, args.output))


def report_main() -> None:
    parser = argparse.ArgumentParser(description="Generate the static inequality report.")
    parser.add_argument("--input", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, default=Path("report"))
    args = parser.parse_args()
    print(build_report(args.input, args.output))

