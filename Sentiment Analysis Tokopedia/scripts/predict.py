"""Convenience wrapper for saved-model inference."""

from tokopedia_sentiment.cli import main

raise SystemExit(main(["predict", *__import__("sys").argv[1:]]))
