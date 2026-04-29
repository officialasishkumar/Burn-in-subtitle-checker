"""Structured data models for pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OcrSegment:
    index: int
    start: float
    end: float
    timestamp: float
    text: str
    language: str
    crop_path: str | None = None
    frame_path: str | None = None
    sampled_timestamps: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ComparedSegment:
    index: int
    start: float
    end: float
    timestamp: float
    audio_text: str
    subtitle_text: str
    normalized_audio_text: str
    normalized_subtitle_text: str
    score: float
    status: str
    crop_path: str | None = None
    frame_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number, got {value!r}") from exc


def transcript_from_mapping(index: int, data: dict[str, Any]) -> TranscriptSegment:
    text = data.get("text", data.get("audio_text", ""))
    start = _coerce_float(data.get("start"), "start")
    end = _coerce_float(data.get("end"), "end")
    if end < start:
        raise ValueError(f"end must be greater than or equal to start for segment {index}")
    return TranscriptSegment(
        index=int(data.get("index", index)),
        start=start,
        end=end,
        text=str(text or "").strip(),
    )


def ocr_from_mapping(index: int, data: dict[str, Any]) -> OcrSegment:
    start = _coerce_float(data.get("start"), "start")
    end = _coerce_float(data.get("end"), "end")
    if end < start:
        raise ValueError(f"end must be greater than or equal to start for segment {index}")
    timestamp = _coerce_float(data.get("timestamp", (start + end) / 2), "timestamp")
    text = data.get("text", data.get("subtitle_text", ""))
    errors = data.get("errors") or []
    sampled = data.get("sampled_timestamps") or [timestamp]
    return OcrSegment(
        index=int(data.get("index", index)),
        start=start,
        end=end,
        timestamp=timestamp,
        text=str(text or "").strip(),
        language=str(data.get("language", "")),
        crop_path=_path_to_str(data.get("crop_path")),
        frame_path=_path_to_str(data.get("frame_path")),
        sampled_timestamps=[float(item) for item in sampled],
        errors=[str(item) for item in errors],
    )


def _path_to_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, Path):
        return str(value)
    return str(value)
