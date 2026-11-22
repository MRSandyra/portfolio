"""Convenience wrapper for the three-class portfolio benchmark."""

from tokopedia_sentiment.cli import main

raise SystemExit(main(["benchmark", *__import__("sys").argv[1:]]))
