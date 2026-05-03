"""Optional EasyOCR backend that mirrors the tesseract entry point."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .backend_config import TESSERACT_TO_ISO_LANGUAGES
from .dependencies import parse_language_spec
from .exceptions import MissingDependencyError, ProcessingError

_READER_LOCK = threading.Lock()
_READER_CACHE: dict[tuple[str, ...], Any] = {}


def map_languages(spec: str) -> list[str]:
    codes: list[str] = []
    for token in parse_language_spec(spec):
        mapped = TESSERACT_TO_ISO_LANGUAGES.get(token, token)
        if mapped not in codes:
            codes.append(mapped)
    return codes


def get_reader(languages: tuple[str, ...]):
    try:
        import easyocr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "Python package 'easyocr' is not installed. "
            "Install with: python -m pip install -e '.[ocr-easy]'"
        ) from exc

    with _READER_LOCK:
        reader = _READER_CACHE.get(languages)
        if reader is None:
            try:
                reader = easyocr.Reader(list(languages), gpu=False, verbose=False)
            except Exception as exc:  # pragma: no cover - depends on torch runtime
                raise MissingDependencyError(
                    f"EasyOCR could not load languages {languages}: {exc}"
                ) from exc
            _READER_CACHE[languages] = reader
        return reader


def run_easyocr(image_path: Path, *, languages: str) -> str:
    codes = map_languages(languages)
    if not codes:
        raise ProcessingError(f"EasyOCR requires at least one supported language: {languages}")
    reader = get_reader(tuple(codes))
    try:
        result = reader.readtext(str(image_path), detail=0, paragraph=True)
    except Exception as exc:
        raise ProcessingError(f"EasyOCR failed for {image_path}: {exc}") from exc
    return " ".join(piece.strip() for piece in result if piece and piece.strip())
