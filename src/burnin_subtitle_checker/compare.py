"""Timed segment comparison and mismatch classification."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import ComparedSegment, OcrSegment, TranscriptSegment
from .normalize import normalize_text

try:  # pragma: no cover - exercised when optional dependency is installed
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional import failure should not break core package
    fuzz = None

try:  # pragma: no cover - exercised when optional dependency is installed
    from jiwer import cer as jiwer_cer
    from jiwer import wer as jiwer_wer
except Exception:  # pragma: no cover - optional import failure should not break core package
    jiwer_cer = None
    jiwer_wer = None


@dataclass(frozen=True, slots=True)
class _IndexedOcrSegment:
    position: int
    segment: OcrSegment


def similarity_score(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    char_score = _character_similarity(left_norm, right_norm)
    token_score = _token_similarity(left_norm, right_norm)
    if token_score is None:
        return char_score
    return round(min(char_score, token_score), 4)


def word_error_rate(left: str, right: str) -> float | None:
    return _jiwer_error_rate(left, right, metric=jiwer_wer)


def character_error_rate(left: str, right: str) -> float | None:
    return _jiwer_error_rate(left, right, metric=jiwer_cer)


def _jiwer_error_rate(left: str, right: str, *, metric) -> float | None:
    if metric is None:
        return None
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm and not right_norm:
        return 0.0
    if not left_norm or not right_norm:
        return 1.0
    try:
        return round(float(metric(left_norm, right_norm)), 4)
    except ValueError:
        return None


def _character_similarity(left_norm: str, right_norm: str) -> float:
    if fuzz is not None:
        return round(float(fuzz.ratio(left_norm, right_norm)) / 100, 4)
    return round(SequenceMatcher(None, left_norm, right_norm).ratio(), 4)


def _token_similarity(left_norm: str, right_norm: str) -> float | None:
    left_tokens = left_norm.split()
    right_tokens = right_norm.split()
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return None
    return round(SequenceMatcher(None, left_tokens, right_tokens).ratio(), 4)


def compare_segments(
    transcript_segments: Iterable[TranscriptSegment],
    ocr_segments: Iterable[OcrSegment],
    *,
    threshold: float = 0.75,
    max_alignment_gap: float = 1.5,
) -> list[ComparedSegment]:
    transcript = list(transcript_segments)
    ocr = list(ocr_segments)
    indexed_ocr = [_IndexedOcrSegment(position, segment) for position, segment in enumerate(ocr)]
    ocr_by_index = _build_ocr_index(indexed_ocr)
    ocr_by_time = sorted(indexed_ocr, key=lambda item: item.segment.timestamp)
    ocr_timestamps = [item.segment.timestamp for item in ocr_by_time]
    used_ocr_positions: set[int] = set()
    compared: list[ComparedSegment] = []

    for audio_segment in transcript:
        matched_ocr = _find_matching_ocr(
            audio_segment,
            ocr_by_index=ocr_by_index,
            ocr_by_time=ocr_by_time,
            ocr_timestamps=ocr_timestamps,
            used_ocr_positions=used_ocr_positions,
            max_alignment_gap=max_alignment_gap,
        )
        if matched_ocr is not None:
            used_ocr_positions.add(matched_ocr.position)
        subtitle_segment = matched_ocr.segment if matched_ocr is not None else None

        subtitle_text = subtitle_segment.text if subtitle_segment else ""
        normalized_audio = normalize_text(audio_segment.text)
        normalized_subtitle = normalize_text(subtitle_text)
        score = similarity_score(audio_segment.text, subtitle_text)
        wer = word_error_rate(audio_segment.text, subtitle_text)
        cer = character_error_rate(audio_segment.text, subtitle_text)
        notes: list[str] = []

        if not normalized_audio and not normalized_subtitle:
            status = "NO_TEXT"
            notes.append("Audio and subtitle text are both empty after normalization.")
        elif not normalized_audio:
            status = "NO_AUDIO"
            notes.append("Audio transcription text is empty after normalization.")
        elif not normalized_subtitle:
            status = "NO_SUBTITLE"
            notes.append("OCR subtitle text is empty after normalization.")
        elif score >= threshold:
            status = "OK"
        else:
            status = "REVIEW"

        if subtitle_segment and subtitle_segment.errors:
            notes.extend(subtitle_segment.errors)
        if subtitle_segment is None:
            notes.append("No OCR segment aligned with this audio segment.")

        compared.append(
            ComparedSegment(
                index=audio_segment.index,
                start=audio_segment.start,
                end=audio_segment.end,
                timestamp=(
                    subtitle_segment.timestamp
                    if subtitle_segment
                    else audio_segment.midpoint
                ),
                audio_text=audio_segment.text,
                subtitle_text=subtitle_text,
                normalized_audio_text=normalized_audio,
                normalized_subtitle_text=normalized_subtitle,
                score=score,
                word_error_rate=wer,
                character_error_rate=cer,
                status=status,
                crop_path=subtitle_segment.crop_path if subtitle_segment else None,
                frame_path=subtitle_segment.frame_path if subtitle_segment else None,
                notes=notes,
            )
        )

    return compared


def count_review_rows(rows: Iterable[ComparedSegment]) -> int:
    return sum(1 for row in rows if row.status != "OK")


def _find_matching_ocr(
    audio_segment: TranscriptSegment,
    *,
    ocr_by_index: dict[int, list[_IndexedOcrSegment]],
    ocr_by_time: list[_IndexedOcrSegment],
    ocr_timestamps: list[float],
    used_ocr_positions: set[int],
    max_alignment_gap: float,
) -> _IndexedOcrSegment | None:
    by_index = _closest_unused_ocr(
        audio_segment,
        ocr_by_index.get(audio_segment.index, []),
        used_ocr_positions=used_ocr_positions,
        max_alignment_gap=max_alignment_gap,
    )
    if by_index is not None:
        return by_index

    midpoint = audio_segment.midpoint
    left = bisect_left(ocr_timestamps, midpoint - max_alignment_gap)
    right = bisect_right(ocr_timestamps, midpoint + max_alignment_gap)
    return _closest_unused_ocr(
        audio_segment,
        ocr_by_time[left:right],
        used_ocr_positions=used_ocr_positions,
        max_alignment_gap=max_alignment_gap,
    )


def _build_ocr_index(
    indexed_ocr: list[_IndexedOcrSegment],
) -> dict[int, list[_IndexedOcrSegment]]:
    by_index: dict[int, list[_IndexedOcrSegment]] = {}
    for item in indexed_ocr:
        by_index.setdefault(item.segment.index, []).append(item)
    return by_index


def _closest_unused_ocr(
    audio_segment: TranscriptSegment,
    candidates: Iterable[_IndexedOcrSegment],
    *,
    used_ocr_positions: set[int],
    max_alignment_gap: float,
) -> _IndexedOcrSegment | None:
    unused_candidates = (
        item
        for item in candidates
        if item.position not in used_ocr_positions
        and _within_alignment_gap(audio_segment, item.segment, max_alignment_gap)
    )
    return min(
        unused_candidates,
        key=lambda item: (
            abs(item.segment.timestamp - audio_segment.midpoint),
            abs(item.segment.index - audio_segment.index),
            item.position,
        ),
        default=None,
    )


def _within_alignment_gap(
    audio_segment: TranscriptSegment,
    ocr_segment: OcrSegment,
    max_alignment_gap: float,
) -> bool:
    return abs(ocr_segment.timestamp - audio_segment.midpoint) <= max_alignment_gap
