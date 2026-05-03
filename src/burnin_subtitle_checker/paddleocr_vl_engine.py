"""Optional PaddleOCR-VL backend for stylised subtitle fonts."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .exceptions import MissingDependencyError, ProcessingError

_PIPELINE_LOCK = threading.Lock()
_PIPELINE: Any | None = None


def get_pipeline():
    try:
        from paddleocr import PaddleOCRVL  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "Python package 'paddleocr' with PaddleOCR-VL support is not installed. "
            "Install with: python -m pip install -e '.[ocr-paddle]'"
        ) from exc

    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            try:
                _PIPELINE = PaddleOCRVL()
            except Exception as exc:  # pragma: no cover - depends on Paddle runtime
                raise MissingDependencyError(f"PaddleOCR-VL could not load: {exc}") from exc
        return _PIPELINE


def run_paddleocr_vl(image_path: Path, *, languages: str) -> str:
    _ = languages
    pipeline = get_pipeline()
    try:
        output = pipeline.predict(str(image_path))
    except Exception as exc:
        raise ProcessingError(f"PaddleOCR-VL failed for {image_path}: {exc}") from exc
    return _join_text_parts(_extract_text_parts(output))


def _extract_text_parts(value: Any) -> list[str]:
    parts: list[str] = []
    for item in _walk(value):
        if isinstance(item, str):
            text = item.strip()
            if text and _looks_like_ocr_text(text):
                parts.append(text)
    return parts


def _walk(value: Any) -> Iterable[Any]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        preferred = (
            "text",
            "markdown",
            "rec_texts",
            "prunedResult",
            "layoutParsingResults",
            "res",
        )
        for key in preferred:
            if key in value:
                yield from _walk(value[key])
        return
    if isinstance(value, list | tuple):
        for item in value:
            yield from _walk(item)
        return
    for attr in ("res", "json", "markdown", "text"):
        if hasattr(value, attr):
            yield from _walk(getattr(value, attr))
            return
    if hasattr(value, "to_dict"):
        try:
            yield from _walk(value.to_dict())
        except Exception:
            return


def _looks_like_ocr_text(text: str) -> bool:
    lowered = text.lower()
    noisy_prefixes = ("output image saved", "input_path", "page_index")
    return not any(lowered.startswith(prefix) for prefix in noisy_prefixes)


def _join_text_parts(parts: list[str]) -> str:
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part not in seen:
            deduped.append(part)
            seen.add(part)
    return " ".join(deduped)
