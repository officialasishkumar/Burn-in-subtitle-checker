"""ASR adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .exceptions import MissingDependencyError, ProcessingError
from .models import TranscriptSegment


def transcribe_video(
    video_path: Path,
    *,
    backend: str = "whisper",
    model_name: str = "base",
    language: str | None = None,
    device: str | None = None,
) -> list[TranscriptSegment]:
    if backend == "whisper":
        return _transcribe_with_openai_whisper(
            video_path,
            model_name=model_name,
            language=language,
            device=device,
        )
    if backend == "faster-whisper":
        return _transcribe_with_faster_whisper(
            video_path,
            model_name=model_name,
            language=language,
            device=device,
        )
    raise MissingDependencyError(f"Unsupported ASR backend: {backend}")


def _transcribe_with_openai_whisper(
    video_path: Path,
    *,
    model_name: str,
    language: str | None,
    device: str | None,
) -> list[TranscriptSegment]:
    try:
        import whisper  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "Python package 'openai-whisper' is not installed. "
            "Install with: python -m pip install -e '.[asr]'"
        ) from exc

    kwargs: dict[str, Any] = {}
    if device and device != "auto":
        kwargs["device"] = device
    try:
        model = whisper.load_model(model_name, **kwargs)
        transcribe_kwargs: dict[str, Any] = {
            "language": None if language in {None, "auto"} else language,
            "task": "transcribe",
        }
        if device in {None, "auto", "cpu"}:
            transcribe_kwargs["fp16"] = False
        transcription = model.transcribe(str(video_path), **transcribe_kwargs)
    except Exception as exc:
        raise ProcessingError(f"Whisper transcription failed: {exc}") from exc
    return _segments_from_whisper_payload(transcription)


def _transcribe_with_faster_whisper(
    video_path: Path,
    *,
    model_name: str,
    language: str | None,
    device: str | None,
) -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "Python package 'faster-whisper' is not installed. "
            "Install with: python -m pip install -e '.[asr-fast]'"
        ) from exc

    try:
        model = WhisperModel(model_name, device="auto" if device in {None, "auto"} else device)
        segments, _info = model.transcribe(
            str(video_path),
            language=None if language in {None, "auto"} else language,
            task="transcribe",
        )
        return [
            TranscriptSegment(
                index=index,
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text,
            )
            for index, segment in enumerate(segments)
        ]
    except Exception as exc:
        raise ProcessingError(f"faster-whisper transcription failed: {exc}") from exc


def _segments_from_whisper_payload(payload: dict[str, Any]) -> list[TranscriptSegment]:
    raw_segments = payload.get("segments") or []
    segments: list[TranscriptSegment] = []
    for index, segment in enumerate(raw_segments):
        segments.append(
            TranscriptSegment(
                index=index,
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=str(segment.get("text", "")).strip(),
            )
        )
    return segments
