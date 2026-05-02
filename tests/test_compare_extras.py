from burnin_subtitle_checker.compare import (
    ReferenceWindow,
    compare_segments,
    composite_similarity,
)
from burnin_subtitle_checker.models import OcrSegment, TranscriptSegment


def test_composite_similarity_handles_identical_text():
    assert composite_similarity("वो कहाँ गई थी", "वो कहाँ गई थी") == 1.0


def test_composite_similarity_handles_empty_inputs():
    assert composite_similarity("", "") == 1.0
    assert composite_similarity("hello", "") == 0.0


def test_composite_similarity_drops_for_different_text():
    score = composite_similarity("hello world", "totally different text here")
    assert 0.0 < score < 0.6


def test_compare_uses_wer_threshold_to_flag_high_score_rows(monkeypatch):
    monkeypatch.setattr("burnin_subtitle_checker.compare.jiwer_wer", lambda left, right: 0.4)
    monkeypatch.setattr("burnin_subtitle_checker.compare.jiwer_cer", lambda left, right: 0.05)

    rows = compare_segments(
        [TranscriptSegment(index=0, start=0.0, end=1.0, text="hello world friends")],
        [
            OcrSegment(
                index=0,
                start=0.0,
                end=1.0,
                timestamp=0.5,
                text="hello world friend",
                language="eng",
            )
        ],
        threshold=0.5,
        wer_threshold=0.2,
    )

    assert rows[0].status == "REVIEW"
    assert any("Word error rate" in note for note in rows[0].notes)


def test_compare_with_reference_window_marks_subtitle_drift():
    transcript = [TranscriptSegment(index=0, start=10.0, end=12.0, text="hello world")]
    ocr = [
        OcrSegment(
            index=0,
            start=10.0,
            end=12.0,
            timestamp=11.0,
            text="hello world",
            language="eng",
        )
    ]
    reference_windows = [
        ReferenceWindow(start=10.0, end=12.0, text="goodnight everyone"),
    ]

    rows = compare_segments(transcript, ocr, reference_windows=reference_windows)

    assert rows[0].reference_text == "goodnight everyone"
    assert rows[0].status == "REVIEW"
    assert any("reference" in note.lower() for note in rows[0].notes)


def test_compare_picks_single_best_overlap_reference_cue():
    transcript = [TranscriptSegment(index=0, start=10.0, end=12.0, text="hello world")]
    ocr = [
        OcrSegment(
            index=0,
            start=10.0,
            end=12.0,
            timestamp=11.0,
            text="hello world",
            language="eng",
        )
    ]
    reference_windows = [
        ReferenceWindow(start=8.0, end=10.0, text="adjacent earlier cue"),
        ReferenceWindow(start=10.0, end=12.0, text="hello world"),
        ReferenceWindow(start=12.0, end=14.0, text="adjacent later cue"),
    ]

    rows = compare_segments(transcript, ocr, reference_windows=reference_windows)

    assert rows[0].reference_text == "hello world"
    assert rows[0].status == "OK"


def test_compare_records_composite_score_field():
    transcript = [TranscriptSegment(index=0, start=0.0, end=1.0, text="hello world")]
    ocr = [
        OcrSegment(
            index=0,
            start=0.0,
            end=1.0,
            timestamp=0.5,
            text="hello world",
            language="eng",
        )
    ]
    rows = compare_segments(transcript, ocr)
    assert rows[0].composite_score == 1.0
