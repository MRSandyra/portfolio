"""Convenience wrapper for drift reporting."""

from tokopedia_sentiment.cli import main

raise SystemExit(main(["drift-report", *__import__("sys").argv[1:]]))
