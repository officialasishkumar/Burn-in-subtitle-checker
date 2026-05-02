"""ASR adapters with hallucination guards and VAD support."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .exceptions import MissingDependencyError, ProcessingError
from .models import TranscriptSegment

# Whisper sometimes hallucinates these phrases for silent or noisy stretches.
_DEFAULT_HALLUCINATION_PHRASES = (
    "thanks for watching",
    "thank you for watching",
    "subscribe to my channel",
    "subscribe to the channel",
    "please like and subscribe",
    "अधिक जानकारी के लिए सब्सक्राइब",
    "ご視聴ありがとうございました",
    "♪♪",
)


def transcribe_video(
    video_path: Path,
    *,
    backend: str = "whisper",
    model_name: str = "base",
    language: str | None = None,
    device: str | None = None,
    vad: bool = False,
    no_speech_threshold: float = 0.6,
    drop_hallucinations: bool = True,
    initial_prompt: str | None = None,
) -> list[TranscriptSegment]:
    if backend == "whisper":
        segments = _transcribe_with_openai_whisper(
            video_path,
            model_name=model_name,
            language=language,
            device=device,
            initial_prompt=initial_prompt,
        )
    elif backend == "faster-whisper":
        segments = _transcribe_with_faster_whisper(
            video_path,
            model_name=model_name,
            language=language,
            device=device,
            vad=vad,
            initial_prompt=initial_prompt,
        )
    else:
        raise MissingDependencyError(f"Unsupported ASR backend: {backend}")

    return _post_process_segments(
        segments,
        no_speech_threshold=no_speech_threshold,
        drop_hallucinations=drop_hallucinations,
    )


def _transcribe_with_openai_whisper(
    video_path: Path,
    *,
    model_name: str,
    language: str | None,
    device: str | None,
    initial_prompt: str | None,
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
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
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
    vad: bool,
    initial_prompt: str | None,
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
        kwargs: dict[str, Any] = {
            "language": None if language in {None, "auto"} else language,
            "task": "transcribe",
            "vad_filter": vad,
        }
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        segments, _info = model.transcribe(str(video_path), **kwargs)
        results: list[TranscriptSegment] = []
        for index, segment in enumerate(segments):
            results.append(
                TranscriptSegment(
                    index=index,
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(getattr(segment, "text", "") or "").strip(),
                    confidence=_safe_float(getattr(segment, "avg_logprob", None)),
                    no_speech_prob=_safe_float(getattr(segment, "no_speech_prob", None)),
                )
            )
        return results
    except Exception as exc:
        raise ProcessingError(f"faster-whisper transcription failed: {exc}") from exc


def _segments_from_whisper_payload(payload: dict[str, Any]) -> list[TranscriptSegment]:
    raw_segments = payload.get("segments") or []
    segments: list[TranscriptSegment] = []
    for index, segment in enumerate(raw_segments):
        text = segment.get("text") or ""
        segments.append(
            TranscriptSegment(
                index=index,
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=str(text).strip(),
                confidence=_safe_float(segment.get("avg_logprob")),
                no_speech_prob=_safe_float(segment.get("no_speech_prob")),
            )
        )
    return segments


def _post_process_segments(
    segments: Iterable[TranscriptSegment],
    *,
    no_speech_threshold: float,
    drop_hallucinations: bool,
) -> list[TranscriptSegment]:
    cleaned: list[TranscriptSegment] = []
    for segment in segments:
        if segment.end < segment.start:
            continue
        if segment.no_speech_prob is not None and segment.no_speech_prob >= no_speech_threshold:
            continue
        if drop_hallucinations and _looks_like_hallucination(segment.text):
            continue
        cleaned.append(segment)
    for new_index, segment in enumerate(cleaned):
        segment.index = new_index
    return cleaned


def _looks_like_hallucination(text: str) -> bool:
    if not text:
        return False
    folded = text.casefold().strip()
    return any(phrase in folded for phrase in _DEFAULT_HALLUCINATION_PHRASES)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
