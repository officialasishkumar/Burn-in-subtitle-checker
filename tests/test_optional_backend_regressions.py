"""Slow optional backend smoke tests against committed fixtures.

These tests are skipped unless the optional runtime packages are importable.
If model weights are not present and the runtime refuses to download them,
the test skips with that backend's error instead of failing unrelated setups.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from burnin_subtitle_checker.asr import transcribe_video
from burnin_subtitle_checker.exceptions import BurnSubError
from burnin_subtitle_checker.models import TranscriptSegment
from burnin_subtitle_checker.ocr import ocr_video_segments

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "fixtures" / "realspeech" / "bundle" / "video.mp4"


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for optional backend fixture tests.",
)


@pytest.mark.skipif(
    not (_has_module("transformers") and _has_module("torch")),
    reason="IndicWhisper runtime is not installed.",
)
def test_indicwhisper_backend_runs_on_video_fixture_if_installed():
    try:
        segments = transcribe_video(
            VIDEO,
            backend="indicwhisper",
            model_name="small",
            language="hi",
            device="cpu",
        )
    except BurnSubError as exc:
        pytest.skip(str(exc))

    assert all(segment.end >= segment.start for segment in segments)


@pytest.mark.skipif(
    not (_has_module("nemo.collections.asr") or _has_module("transformers")),
    reason="IndicConformer runtime is not installed.",
)
def test_indic_conformer_backend_runs_on_video_fixture_if_installed():
    try:
        segments = transcribe_video(
            VIDEO,
            backend="indic-conformer",
            model_name="large",
            language="hi",
            device="cpu",
            conformer_decoder="ctc",
        )
    except BurnSubError as exc:
        pytest.skip(str(exc))

    assert len(segments) == 1
    assert segments[0].end >= segments[0].start


@pytest.mark.skipif(not _has_module("paddleocr"), reason="PaddleOCR-VL is not installed.")
def test_paddleocr_vl_backend_runs_on_video_fixture_if_installed(tmp_path):
    transcript = [TranscriptSegment(index=0, start=0.0, end=1.0, text="hello")]
    try:
        rows = ocr_video_segments(
            VIDEO,
            transcript,
            output_dir=tmp_path,
            languages="eng",
            crop_bottom_percent=25,
            engine="paddleocr-vl",
            save_artifacts=False,
        )
    except BurnSubError as exc:
        pytest.skip(str(exc))

    assert rows[0].engine == "paddleocr-vl"


@pytest.mark.skipif(
    not (_has_module("transformers") and _has_module("torch")),
    reason="AI4Bharat OCR runtime is not installed.",
)
def test_ai4bharat_ocr_backend_runs_on_video_fixture_if_installed(tmp_path):
    transcript = [TranscriptSegment(index=0, start=0.0, end=1.0, text="hello")]
    try:
        rows = ocr_video_segments(
            VIDEO,
            transcript,
            output_dir=tmp_path,
            languages="hin+kan",
            crop_bottom_percent=25,
            engine="ai4bharat",
            save_artifacts=False,
        )
    except BurnSubError as exc:
        pytest.skip(str(exc))

    assert rows[0].engine == "ai4bharat"
