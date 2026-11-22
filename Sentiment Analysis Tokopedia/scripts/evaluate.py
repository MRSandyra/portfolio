"""Convenience wrapper for evaluating a saved model on a labeled CSV."""

from tokopedia_sentiment.cli import main

raise SystemExit(main(["evaluate", *__import__("sys").argv[1:]]))
