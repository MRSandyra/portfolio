"""Deterministic text normalization shared by training and inference."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Mapping

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@[\w_]+", flags=re.UNICODE)
_HASHTAG_RE = re.compile(r"#([\w_]+)", flags=re.UNICODE)
_NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\b[\w]+\b", flags=re.UNICODE)
_ELONGATED_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)

# Deliberately small and review-domain focused. Avoid aggressive stemming because it can
# erase negation or change product names. Callers may pass an extended mapping.
DEFAULT_SLANG_MAP: dict[str, str] = {
    "bgt": "banget",
    "blm": "belum",
    "brg": "barang",
    "dgn": "dengan",
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "sm": "sama",
    "udh": "sudah",
    "udah": "sudah",
    "yg": "yang",
}


def normalize_indonesian_text(
    text: object,
    *,
    slang_map: Mapping[str, str] | None = None,
) -> str:
    """Normalize noisy Indonesian review text without removing negation.

    Null-like/non-string values become an empty string. URLs and mentions become stable
    tokens, while hashtag words are retained. Character elongation is reduced to two
    characters (``enaaak`` -> ``enaak``) rather than guessed into a dictionary word.
    """
    if not isinstance(text, str):
        return ""

    normalized = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    normalized = _URL_RE.sub(" url ", normalized)
    normalized = _MENTION_RE.sub(" user ", normalized)
    normalized = _HASHTAG_RE.sub(r" \1 ", normalized)
    normalized = _NON_WORD_RE.sub(" ", normalized)
    normalized = normalized.replace("_", " ")
    normalized = _ELONGATED_RE.sub(r"\1\1", normalized)

    replacements = DEFAULT_SLANG_MAP if slang_map is None else slang_map
    tokens = _TOKEN_RE.findall(normalized)
    tokens = [replacements.get(token, token) for token in tokens]
    return _WHITESPACE_RE.sub(" ", " ".join(tokens)).strip()
