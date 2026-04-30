from pathlib import Path

import pytest

from burnin_subtitle_checker.exceptions import ConfigError
from burnin_subtitle_checker.models import TranscriptSegment
from burnin_subtitle_checker.ocr import ocr_video_segments


def test_ocr_rejects_invalid_crop_percent_before_processing(tmp_path, monkeypatch):
    monkeypatch.setattr("burnin_subtitle_checker.ocr.require_executable", lambda *args: "")
    monkeypatch.setattr(
        "burnin_subtitle_checker.ocr.require_tesseract_languages",
        lambda *args: None,
    )

    with pytest.raises(ConfigError):
        ocr_video_segments(
            tmp_path / "video.mp4",
            [TranscriptSegment(index=0, start=0.0, end=1.0, text="hello")],
            output_dir=tmp_path,
            crop_bottom_percent=0,
        )


def test_ocr_removes_crop_files_when_artifacts_are_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("burnin_subtitle_checker.ocr.require_executable", lambda *args: "")
    monkeypatch.setattr(
        "burnin_subtitle_checker.ocr.require_tesseract_languages",
        lambda *args: None,
    )

    captured_paths: list[Path] = []

    def fake_capture_frame(video_path, timestamp, output_path, **kwargs):
        output_path.write_text("image", encoding="utf-8")
        captured_paths.append(output_path)
        return output_path

    monkeypatch.setattr("burnin_subtitle_checker.ocr.capture_frame_region", fake_capture_frame)
    monkeypatch.setattr(
        "burnin_subtitle_checker.ocr.run_tesseract",
        lambda *args, **kwargs: "hello",
    )

    rows = ocr_video_segments(
        tmp_path / "video.mp4",
        [TranscriptSegment(index=0, start=0.0, end=1.0, text="hello")],
        output_dir=tmp_path,
        languages="eng",
        save_artifacts=False,
    )

    assert rows[0].text == "hello"
    assert rows[0].crop_path is None
    assert captured_paths
    assert not captured_paths[0].exists()


def test_ocr_chooses_offset_candidate_that_best_matches_audio(tmp_path, monkeypatch):
    monkeypatch.setattr("burnin_subtitle_checker.ocr.require_executable", lambda *args: "")
    monkeypatch.setattr(
        "burnin_subtitle_checker.ocr.require_tesseract_languages",
        lambda *args: None,
    )

    def fake_capture_frame(video_path, timestamp, output_path, **kwargs):
        output_path.write_text("image", encoding="utf-8")
        return output_path

    def fake_tesseract(image_path, **kwargs):
        if "1.000" in image_path.name:
            return "hello world"
        return "totally different subtitle with more characters"

    monkeypatch.setattr("burnin_subtitle_checker.ocr.capture_frame_region", fake_capture_frame)
    monkeypatch.setattr("burnin_subtitle_checker.ocr.run_tesseract", fake_tesseract)

    rows = ocr_video_segments(
        tmp_path / "video.mp4",
        [TranscriptSegment(index=0, start=0.0, end=2.0, text="hello world")],
        output_dir=tmp_path,
        languages="eng",
        frame_offsets=[0.0, 0.25],
    )

    assert rows[0].text == "hello world"
    assert rows[0].timestamp == 1.0
