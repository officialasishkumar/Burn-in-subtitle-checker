"""Optional AI4Bharat Indic OCR backend.

The public AI4Bharat OCR assets are still model-family oriented rather than a
stable Python package API, so this adapter supports a transformers-compatible
image-to-text model id and keeps the repository contract narrow.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .backend_config import AI4BHARAT_OCR_MODEL_ID
from .exceptions import MissingDependencyError, ProcessingError

_PIPELINE_LOCK = threading.Lock()
_PIPELINE_CACHE: dict[str, Any] = {}


def resolve_model_id() -> str:
    return os.environ.get("BURNSUB_AI4BHARAT_OCR_MODEL_ID", AI4BHARAT_OCR_MODEL_ID)


def get_pipeline(model_id: str):
    try:
        from transformers import pipeline  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "AI4Bharat OCR requires 'transformers', 'torch', and 'pillow'. "
            "Install with: python -m pip install -e '.[ocr-indic]'"
        ) from exc

    with _PIPELINE_LOCK:
        cached = _PIPELINE_CACHE.get(model_id)
        if cached is not None:
            return cached
        try:
            created = pipeline("image-to-text", model=model_id)
        except Exception as exc:  # pragma: no cover - depends on model runtime/cache
            raise MissingDependencyError(
                "AI4Bharat OCR model could not be loaded. "
                f"Model id: {model_id}. Set BURNSUB_AI4BHARAT_OCR_MODEL_ID if your "
                f"weights use a different id. Original error: {exc}"
            ) from exc
        _PIPELINE_CACHE[model_id] = created
        return created


def run_ai4bharat_ocr(image_path: Path, *, languages: str) -> str:
    _ = languages
    model_id = resolve_model_id()
    recognizer = get_pipeline(model_id)
    try:
        result = recognizer(str(image_path))
    except Exception as exc:
        raise ProcessingError(f"AI4Bharat OCR failed for {image_path}: {exc}") from exc
    return _text_from_result(result)


def _text_from_result(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("generated_text", "text", "prediction", "transcription"):
            value = result.get(key)
            if value:
                return str(value).strip()
        return ""
    if isinstance(result, list | tuple):
        return " ".join(_text_from_result(item) for item in result if item).strip()
    text = getattr(result, "text", None)
    if text:
        return str(text).strip()
    return str(result).strip()
