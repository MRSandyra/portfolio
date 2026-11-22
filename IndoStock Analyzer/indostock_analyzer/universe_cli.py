from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .universe import (
    UniverseValidationError,
    load_ticker_universe,
    read_universe_csv,
    write_ticker_universe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indostock-universe")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate an existing universe JSON")
    validate.add_argument("path", type=Path)
    validate.add_argument("--max-age-days", type=int, default=45)

    import_csv = commands.add_parser("import-csv", help="Build a universe JSON from an official CSV export")
    import_csv.add_argument("input", type=Path)
    import_csv.add_argument("output", type=Path)
    import_csv.add_argument("--as-of", type=date.fromisoformat, required=True)
    import_csv.add_argument("--source-name", required=True)
    import_csv.add_argument("--source-url", required=True)
    import_csv.add_argument(
        "--coverage",
        choices=("complete", "curated_subset"),
        default="complete",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            universe = load_ticker_universe(args.path)
            metadata = universe.metadata(args.max_age_days)
            print(
                f"Valid universe: {metadata['active_symbol_count']} active symbols, "
                f"coverage={metadata['coverage']}, age={metadata['age_days']} days"
            )
            for warning in metadata["warnings"]:
                print(f"Warning: {warning}")
            return 0

        entries = read_universe_csv(args.input)
        write_ticker_universe(
            args.output,
            entries,
            as_of=args.as_of,
            source_name=args.source_name,
            source_url=args.source_url,
            coverage=args.coverage,
        )
        load_ticker_universe(args.output)
        print(f"Wrote {len(entries)} ticker entries to {args.output}")
        return 0
    except UniverseValidationError as exc:
        print(f"Universe error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
