"""Convenience wrapper: python scripts/train.py --data path/to/reviewTokped.csv"""

from tokopedia_sentiment.cli import main

raise SystemExit(main(["train", *__import__("sys").argv[1:]]))
