"""Text normalization that preserves Indic scripts."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
}


def normalize_text(
    text: str,
    *,
    strip_punctuation: bool = True,
    casefold: bool = True,
) -> str:
    """Normalize text for fuzzy comparison without dropping Indic marks."""

    if not text:
        return ""

    normalized = unicodedata.normalize("NFC", text)
    normalized = "".join(" " if char in _ZERO_WIDTH else char for char in normalized)
    if casefold:
        normalized = normalized.casefold()
    if strip_punctuation:
        normalized = "".join(_punctuation_to_space(char) for char in normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def _punctuation_to_space(char: str) -> str:
    category = unicodedata.category(char)
    if category[0] in {"P", "S"}:
        return " "
    return char
