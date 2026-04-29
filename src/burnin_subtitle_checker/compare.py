"""Timed segment comparison and mismatch classification."""

from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher

from .models import ComparedSegment, OcrSegment, TranscriptSegment
from .normalize import normalize_text

try:  # pragma: no cover - exercised when optional dependency is installed
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional import failure should not break core package
    fuzz = None


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
    used_ocr_indexes: set[int] = set()
    compared: list[ComparedSegment] = []

    for audio_segment in transcript:
        subtitle_segment = _find_matching_ocr(
            audio_segment,
            ocr,
            used_ocr_indexes=used_ocr_indexes,
            max_alignment_gap=max_alignment_gap,
        )
        if subtitle_segment is not None:
            used_ocr_indexes.add(subtitle_segment.index)

        subtitle_text = subtitle_segment.text if subtitle_segment else ""
        normalized_audio = normalize_text(audio_segment.text)
        normalized_subtitle = normalize_text(subtitle_text)
        score = similarity_score(audio_segment.text, subtitle_text)
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
    ocr_segments: list[OcrSegment],
    *,
    used_ocr_indexes: set[int],
    max_alignment_gap: float,
) -> OcrSegment | None:
    by_index = next(
        (
            item
            for item in ocr_segments
            if item.index == audio_segment.index
            and _within_alignment_gap(audio_segment, item, max_alignment_gap)
        ),
        None,
    )
    if by_index is not None:
        return by_index

    midpoint = audio_segment.midpoint
    candidates = [
        item
        for item in ocr_segments
        if item.index not in used_ocr_indexes
        and abs(item.timestamp - midpoint) <= max_alignment_gap
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item.timestamp - midpoint))


def _within_alignment_gap(
    audio_segment: TranscriptSegment,
    ocr_segment: OcrSegment,
    max_alignment_gap: float,
) -> bool:
    return abs(ocr_segment.timestamp - audio_segment.midpoint) <= max_alignment_gap
