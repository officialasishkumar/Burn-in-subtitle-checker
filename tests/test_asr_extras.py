from burnin_subtitle_checker.asr import _post_process_segments
from burnin_subtitle_checker.models import TranscriptSegment


def test_post_process_drops_high_no_speech_segments():
    segments = [
        TranscriptSegment(index=0, start=0.0, end=1.0, text="real speech", no_speech_prob=0.1),
        TranscriptSegment(index=1, start=1.0, end=2.0, text="silence", no_speech_prob=0.95),
    ]
    cleaned = _post_process_segments(segments, no_speech_threshold=0.6, drop_hallucinations=False)
    assert len(cleaned) == 1
    assert cleaned[0].text == "real speech"
    assert cleaned[0].index == 0


def test_post_process_drops_known_hallucination_phrases():
    segments = [
        TranscriptSegment(index=0, start=0.0, end=1.0, text="thanks for watching!"),
        TranscriptSegment(index=1, start=1.0, end=2.0, text="next sentence"),
    ]
    cleaned = _post_process_segments(segments, no_speech_threshold=1.0, drop_hallucinations=True)
    assert [seg.text for seg in cleaned] == ["next sentence"]


def test_post_process_renumbers_indexes_after_filtering():
    segments = [
        TranscriptSegment(index=0, start=0.0, end=1.0, text="silence", no_speech_prob=0.95),
        TranscriptSegment(index=1, start=1.0, end=2.0, text="hi"),
        TranscriptSegment(index=2, start=2.0, end=3.0, text="there"),
    ]
    cleaned = _post_process_segments(segments, no_speech_threshold=0.6, drop_hallucinations=False)
    assert [seg.index for seg in cleaned] == [0, 1]
