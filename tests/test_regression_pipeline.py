"""End-to-end regression test against a video with deliberate burned-in mismatches.

This test renders ``fixtures/regression/spec.json`` into a synthetic MP4 with
burned-in subtitles using Pillow + ffmpeg, then runs ``burnsub check`` against
it. It asserts that each segment's status matches the ``expected_status`` field
in the spec.

The test is skipped automatically when ffmpeg, tesseract, or Pillow are not
available so it does not break developer machines without those dependencies.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SPEC = ROOT / "fixtures" / "regression" / "spec.json"
BUILD_SCRIPT = ROOT / "scripts" / "build_regression_fixture.py"

REQUIRED_BINARIES = ("ffmpeg", "ffprobe", "tesseract")
REQUIRED_TESSERACT_LANG = "eng"


def _missing_binaries() -> list[str]:
    return [name for name in REQUIRED_BINARIES if shutil.which(name) is None]


def _has_tesseract_eng() -> bool:
    if shutil.which("tesseract") is None:
        return False
    completed = subprocess.run(
        ["tesseract", "--list-langs"],
        capture_output=True,
        text=True,
        check=False,
    )
    return REQUIRED_TESSERACT_LANG in completed.stdout.split()


def _has_pillow() -> bool:
    return importlib.util.find_spec("PIL") is not None


pytestmark = [
    pytest.mark.skipif(
        bool(_missing_binaries()),
        reason=f"Missing native binaries: {_missing_binaries()}",
    ),
    pytest.mark.skipif(not _has_pillow(), reason="Pillow is required to render frames."),
    pytest.mark.skipif(not _has_tesseract_eng(), reason="tesseract eng pack required."),
]


def test_regression_video_pipeline(tmp_path):
    spec = json.loads(FIXTURE_SPEC.read_text(encoding="utf-8"))
    expected_statuses = [segment["expected_status"] for segment in spec["segments"]]

    bundle_dir = tmp_path / "bundle"
    report_dir = tmp_path / "report"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    build = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), str(bundle_dir)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    check = subprocess.run(
        [
            sys.executable,
            "-m",
            "burnin_subtitle_checker.cli",
            "--quiet",
            "check",
            str(bundle_dir / "video.mp4"),
            "--transcript-json",
            str(bundle_dir / "transcript.json"),
            "--reference-srt",
            str(bundle_dir / "reference.srt"),
            "--output-dir",
            str(report_dir),
            "--ocr-languages",
            "eng",
            "--crop-bottom-percent",
            "25",
            "--workers",
            "4",
            "--threshold",
            "0.75",
            "--wer-threshold",
            "0.2",
            "--frame-offsets",
            "0",
            "--formats",
            "json",
            "--save-artifacts",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert check.returncode == 0, check.stderr

    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    statuses = [segment["status"] for segment in report["segments"]]

    assert statuses == expected_statuses, (
        "Per-segment status drift; full report:\n"
        + json.dumps(report["segments"], indent=2, ensure_ascii=False)
    )

    summary = report["summary"]
    assert summary["total"] == len(expected_statuses)
    assert summary["ok"] == expected_statuses.count("OK")
    assert summary["review"] == expected_statuses.count("REVIEW")
    assert summary["no_subtitle"] == expected_statuses.count("NO_SUBTITLE")
