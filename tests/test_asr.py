import pytest

from burnin_subtitle_checker.asr import _segments_from_whisper_payload, transcribe_video
from burnin_subtitle_checker.exceptions import MissingDependencyError


def test_segments_from_whisper_payload_normalizes_timed_segments():
    rows = _segments_from_whisper_payload(
        {
            "segments": [
                {"start": 1, "end": 2.5, "text": "  hello world  "},
                {"start": "3.0", "end": "4.25", "text": None},
            ]
        }
    )

    assert rows[0].index == 0
    assert rows[0].start == 1.0
    assert rows[0].end == 2.5
    assert rows[0].text == "hello world"
    assert rows[1].text == ""


def test_transcribe_video_rejects_unknown_backend(tmp_path):
    with pytest.raises(MissingDependencyError):
        transcribe_video(tmp_path / "video.mp4", backend="unknown")
