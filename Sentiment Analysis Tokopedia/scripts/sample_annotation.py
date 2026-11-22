"""Convenience wrapper for creating a human-annotation batch."""

from tokopedia_sentiment.cli import main

raise SystemExit(main(["sample-annotation", *__import__("sys").argv[1:]]))
