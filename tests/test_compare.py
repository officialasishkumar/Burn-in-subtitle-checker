from burnin_subtitle_checker.compare import compare_segments
from burnin_subtitle_checker.models import OcrSegment, TranscriptSegment


def test_compare_marks_ok_and_review_rows():
    transcript = [
        TranscriptSegment(index=0, start=1.0, end=3.0, text="वो कहाँ गई थी"),
        TranscriptSegment(index=1, start=4.0, end=6.0, text="वो कहाँ गई थी"),
    ]
    ocr = [
        OcrSegment(index=0, start=1.0, end=3.0, timestamp=2.0, text="वो कहाँ गई थी", language="hin"),
        OcrSegment(index=1, start=4.0, end=6.0, timestamp=5.0, text="राम स्कूल गया", language="hin"),
    ]

    rows = compare_segments(transcript, ocr, threshold=0.75)

    assert rows[0].status == "OK"
    assert rows[0].score == 1.0
    assert rows[1].status == "REVIEW"
    assert rows[1].score < 0.75


def test_compare_flags_empty_ocr_as_no_subtitle():
    transcript = [TranscriptSegment(index=0, start=1.0, end=2.0, text="ठीक है भाई")]
    ocr = [OcrSegment(index=0, start=1.0, end=2.0, timestamp=1.5, text="", language="hin")]

    rows = compare_segments(transcript, ocr)

    assert rows[0].status == "NO_SUBTITLE"
    assert rows[0].score == 0.0


def test_compare_does_not_pair_same_index_when_timestamp_is_far_away():
    transcript = [TranscriptSegment(index=0, start=1.0, end=2.0, text="hello world")]
    ocr = [
        OcrSegment(
            index=0,
            start=100.0,
            end=101.0,
            timestamp=100.5,
            text="hello world",
            language="eng",
        )
    ]

    rows = compare_segments(transcript, ocr, max_alignment_gap=1.0)

    assert rows[0].status == "NO_SUBTITLE"
    assert rows[0].subtitle_text == ""
