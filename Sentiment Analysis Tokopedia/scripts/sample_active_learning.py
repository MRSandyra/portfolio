"""Convenience wrapper for active-learning annotation sampling."""

from tokopedia_sentiment.cli import main

raise SystemExit(main(["sample-active-learning", *__import__("sys").argv[1:]]))
