"""Convenience wrapper for the optional IndoBERT-Lite experiment."""

from tokopedia_sentiment.cli import main

raise SystemExit(main(["transformer-benchmark", *__import__("sys").argv[1:]]))
