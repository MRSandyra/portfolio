"""Convenience wrapper for annotation agreement analysis."""

from tokopedia_sentiment.cli import main

raise SystemExit(main(["annotation-agreement", *__import__("sys").argv[1:]]))
