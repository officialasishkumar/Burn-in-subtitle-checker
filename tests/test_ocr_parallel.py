import threading
from pathlib import Path

import pytest

from burnin_subtitle_checker.exceptions import ConfigError
from burnin_subtitle_checker.models import TranscriptSegment
from burnin_subtitle_checker.ocr import ocr_video_segments


def _stub_capture(monkeypatch, *, paths: list[Path] | None = None):
    def fake_capture(video_path, timestamp, output_path, **_kwargs):
        output_path.write_text("frame", encoding="utf-8")
        if paths is not None:
            paths.append(output_path)
        return output_path

    monkeypatch.setattr("burnin_subtitle_checker.ocr.capture_frame_region", fake_capture)


def _stub_engine_dependencies(monkeypatch):
    monkeypatch.setattr("burnin_subtitle_checker.ocr.require_executable", lambda *args: "")
    monkeypatch.setattr(
        "burnin_subtitle_checker.ocr.require_tesseract_languages",
        lambda *args: None,
    )


def test_ocr_runs_in_parallel(monkeypatch, tmp_path):
    _stub_engine_dependencies(monkeypatch)
    _stub_capture(monkeypatch)

    barrier = threading.Barrier(parties=4)

    def fake_tesseract(image_path, **_kwargs):
        barrier.wait(timeout=2)
        return image_path.stem

    monkeypatch.setattr("burnin_subtitle_checker.ocr.run_tesseract", fake_tesseract)

    transcript = [
        TranscriptSegment(index=i, start=float(i), end=float(i) + 1.0, text="x")
        for i in range(4)
    ]

    rows = ocr_video_segments(
        tmp_path / "video.mp4",
        transcript,
        output_dir=tmp_path,
        languages="eng",
        save_artifacts=False,
        workers=4,
    )

    assert [row.index for row in rows] == [0, 1, 2, 3]
    assert all(row.text for row in rows)


def test_ocr_writes_checkpoint_jsonl(monkeypatch, tmp_path):
    _stub_engine_dependencies(monkeypatch)
    _stub_capture(monkeypatch)
    monkeypatch.setattr(
        "burnin_subtitle_checker.ocr.run_tesseract",
        lambda image_path, **_kwargs: "burned-in",
    )

    transcript = [
        TranscriptSegment(index=i, start=float(i), end=float(i) + 1, text="")
        for i in range(2)
    ]
    checkpoint = tmp_path / "ocr.partial.jsonl"

    ocr_video_segments(
        tmp_path / "video.mp4",
        transcript,
        output_dir=tmp_path,
        languages="eng",
        save_artifacts=False,
        checkpoint_path=checkpoint,
    )

    lines = [line for line in checkpoint.read_text().splitlines() if line.strip()]
    assert len(lines) == 2


def test_ocr_resumes_from_checkpoint_without_running_engine(monkeypatch, tmp_path):
    _stub_engine_dependencies(monkeypatch)
    _stub_capture(monkeypatch)

    checkpoint = tmp_path / "ocr.partial.jsonl"
    checkpoint.write_text(
        '{"index": 0, "start": 0.0, "end": 1.0, "timestamp": 0.5, '
        '"text": "from checkpoint", "language": "eng", "sampled_timestamps": [0.5], '
        '"errors": [], "engine": "tesseract"}\n',
        encoding="utf-8",
    )

    def boom(image_path, **_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("OCR should not run for cached segment")

    monkeypatch.setattr("burnin_subtitle_checker.ocr.run_tesseract", boom)

    transcript = [TranscriptSegment(index=0, start=0.0, end=1.0, text="x")]

    rows = ocr_video_segments(
        tmp_path / "video.mp4",
        transcript,
        output_dir=tmp_path,
        languages="eng",
        save_artifacts=False,
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert rows[0].text == "from checkpoint"


def test_ocr_rejects_invalid_workers(tmp_path):
    with pytest.raises(ConfigError):
        ocr_video_segments(
            tmp_path / "video.mp4",
            [TranscriptSegment(index=0, start=0.0, end=1.0, text="hi")],
            output_dir=tmp_path,
            languages="eng",
            workers=0,
        )


def test_ocr_rejects_unknown_engine(monkeypatch, tmp_path):
    _stub_engine_dependencies(monkeypatch)
    with pytest.raises(ConfigError):
        ocr_video_segments(
            tmp_path / "video.mp4",
            [TranscriptSegment(index=0, start=0.0, end=1.0, text="hi")],
            output_dir=tmp_path,
            languages="eng",
            engine="bogus",
        )
