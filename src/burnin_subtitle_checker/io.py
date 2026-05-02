"""JSON IO helpers for structured pipeline data."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .compare import ReferenceWindow
from .exceptions import ConfigError
from .models import OcrSegment, TranscriptSegment, ocr_from_mapping, transcript_from_mapping
from .srt import load_reference_srt


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _extract_records(payload: Any, preferred_key: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get(preferred_key, payload.get("segments"))
    else:
        records = None
    if not isinstance(records, list):
        raise ConfigError(f"Expected a JSON list or an object with a '{preferred_key}' list")
    if not all(isinstance(item, dict) for item in records):
        raise ConfigError("Each JSON segment must be an object")
    return records


def load_transcript(path: Path) -> list[TranscriptSegment]:
    records = _extract_records(read_json(path), "transcript")
    try:
        segments = [transcript_from_mapping(index, item) for index, item in enumerate(records)]
    except ValueError as exc:
        raise ConfigError(f"Invalid transcript JSON in {path}: {exc}") from exc
    return sorted(segments, key=lambda item: (item.start, item.end, item.index))


def load_ocr(path: Path) -> list[OcrSegment]:
    records = _extract_records(read_json(path), "ocr")
    try:
        segments = [ocr_from_mapping(index, item) for index, item in enumerate(records)]
    except ValueError as exc:
        raise ConfigError(f"Invalid OCR JSON in {path}: {exc}") from exc
    return sorted(segments, key=lambda item: (item.start, item.end, item.index))


def transcript_payload(
    segments: Iterable[TranscriptSegment],
    source: str | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "segments": [segment.to_dict() for segment in segments],
    }


def ocr_payload(segments: Iterable[OcrSegment], source: str | None = None) -> dict[str, Any]:
    return {
        "source": source,
        "segments": [segment.to_dict() for segment in segments],
    }


def load_reference_windows(path: Path) -> list[ReferenceWindow]:
    cues = load_reference_srt(path)
    return [
        ReferenceWindow(start=cue.start, end=cue.end, text=cue.text)
        for cue in cues
    ]
