"""Convenience wrapper for adjudicated gold-set assembly."""

from tokopedia_sentiment.cli import main

raise SystemExit(main(["build-gold-set", *__import__("sys").argv[1:]]))
