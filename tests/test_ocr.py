import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from burnin_subtitle_checker.exceptions import ConfigError
from burnin_subtitle_checker.models import TranscriptSegment
from burnin_subtitle_checker.ocr import ocr_video_segments, run_tesseract


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


def test_ocr_can_preprocess_crops_with_opencv(tmp_path, monkeypatch):
    monkeypatch.setattr("burnin_subtitle_checker.ocr.require_executable", lambda *args: "")
    monkeypatch.setattr(
        "burnin_subtitle_checker.ocr.require_tesseract_languages",
        lambda *args: None,
    )
    monkeypatch.setattr("burnin_subtitle_checker.ocr.python_module_available", lambda *args: True)

    writes: list[str] = []
    fake_cv2 = SimpleNamespace(
        IMREAD_GRAYSCALE=0,
        INTER_CUBIC=1,
        ADAPTIVE_THRESH_GAUSSIAN_C=2,
        THRESH_BINARY=3,
        imread=lambda path, flag: object(),
        imwrite=lambda path, image: writes.append(path) or True,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    def fake_capture_frame(video_path, timestamp, output_path, **kwargs):
        output_path.write_text("image", encoding="utf-8")
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
        preprocess="grayscale",
        upscale_factor=1.0,
    )

    assert rows[0].text == "hello"
    assert writes


def test_tesseract_receives_custom_tessdata_dir(tmp_path, monkeypatch):
    captured = {}
    image = tmp_path / "crop.png"
    image.write_text("image", encoding="utf-8")
    tessdata_dir = tmp_path / "tessdata"
    tessdata_dir.mkdir()

    def fake_run_command(args, timeout):
        captured["args"] = args
        captured["timeout"] = timeout
        return SimpleNamespace(returncode=0, stdout=" hello \n", stderr="")

    monkeypatch.setattr("burnin_subtitle_checker.ocr.run_command", fake_run_command)

    assert run_tesseract(image, languages="hin", psm=6, tessdata_dir=tessdata_dir) == "hello"
    assert "--tessdata-dir" in captured["args"]
    assert str(tessdata_dir) in captured["args"]


def test_paddleocr_vl_engine_does_not_require_tesseract(tmp_path, monkeypatch):
    def fail_require_executable(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("PaddleOCR-VL should not require tesseract")

    def fake_capture_frame(video_path, timestamp, output_path, **kwargs):
        output_path.write_text("image", encoding="utf-8")
        return output_path

    monkeypatch.setattr("burnin_subtitle_checker.ocr.require_executable", fail_require_executable)
    monkeypatch.setattr("burnin_subtitle_checker.ocr.capture_frame_region", fake_capture_frame)
    monkeypatch.setattr(
        "burnin_subtitle_checker.paddleocr_vl_engine.run_paddleocr_vl",
        lambda image_path, *, languages: "नमस्ते",
    )

    rows = ocr_video_segments(
        tmp_path / "video.mp4",
        [TranscriptSegment(index=0, start=0.0, end=1.0, text="नमस्ते")],
        output_dir=tmp_path,
        languages="hin",
        engine="paddleocr-vl",
        save_artifacts=False,
    )

    assert rows[0].text == "नमस्ते"
    assert rows[0].engine == "paddleocr-vl"


def test_ai4bharat_ocr_engine_does_not_require_tesseract(tmp_path, monkeypatch):
    def fail_require_executable(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("AI4Bharat OCR should not require tesseract")

    def fake_capture_frame(video_path, timestamp, output_path, **kwargs):
        output_path.write_text("image", encoding="utf-8")
        return output_path

    monkeypatch.setattr("burnin_subtitle_checker.ocr.require_executable", fail_require_executable)
    monkeypatch.setattr("burnin_subtitle_checker.ocr.capture_frame_region", fake_capture_frame)
    monkeypatch.setattr(
        "burnin_subtitle_checker.ai4bharat_ocr_engine.run_ai4bharat_ocr",
        lambda image_path, *, languages: "ನಮಸ್ಕಾರ",
    )

    rows = ocr_video_segments(
        tmp_path / "video.mp4",
        [TranscriptSegment(index=0, start=0.0, end=1.0, text="ನಮಸ್ಕಾರ")],
        output_dir=tmp_path,
        languages="kan",
        engine="ai4bharat",
        save_artifacts=False,
    )

    assert rows[0].text == "ನಮಸ್ಕಾರ"
    assert rows[0].engine == "ai4bharat"
