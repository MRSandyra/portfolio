"""Convenience wrapper for precision-constrained triage."""

from tokopedia_sentiment.cli import main

raise SystemExit(main(["triage", *__import__("sys").argv[1:]]))
