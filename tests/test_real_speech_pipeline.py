"""Real-speech end-to-end regression for the full ASR + OCR + compare pipeline.

This test runs ``burnsub check`` against the committed real-speech bundle in
``fixtures/realspeech/bundle/`` (built from ``fixtures/realspeech/spec.json``
with ``scripts/build_real_speech_fixture.py``). Whisper actually transcribes
the burned-in audio, Tesseract OCRs the burned-in text, and the comparator
must catch the deliberately wrong subtitles.

Set ``BURNSUB_REBUILD_FIXTURE=1`` to regenerate the bundle (requires Pillow
plus a system TTS: ``espeak-ng`` on Linux or ``say`` on macOS).

Set ``BURNSUB_RUN_REAL_SPEECH=1`` to opt into running this test; it is slow
because Whisper inference downloads a model and runs on CPU.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "fixtures" / "realspeech" / "spec.json"
BUILD_SCRIPT = ROOT / "scripts" / "build_real_speech_fixture.py"
BUNDLE_DIR = ROOT / "fixtures" / "realspeech" / "bundle"

REQUIRED_BINARIES = ("ffmpeg", "ffprobe", "tesseract")

EXPECTED_STATUS_BY_KIND = {
    "match": "OK",
    "single_word_swap": "REVIEW",
    "multi_word_swap": "REVIEW",
    "wholly_different": "REVIEW",
    "blank_subtitle": "NO_SUBTITLE",
}


def _missing_binaries() -> list[str]:
    return [name for name in REQUIRED_BINARIES if shutil.which(name) is None]


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_eng_pack() -> bool:
    if shutil.which("tesseract") is None:
        return False
    completed = subprocess.run(
        ["tesseract", "--list-langs"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "eng" in completed.stdout.split()


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("BURNSUB_RUN_REAL_SPEECH") != "1",
        reason="Set BURNSUB_RUN_REAL_SPEECH=1 to opt into real Whisper ASR.",
    ),
    pytest.mark.skipif(
        bool(_missing_binaries()),
        reason=f"Missing native binaries: {_missing_binaries()}",
    ),
    pytest.mark.skipif(not _has_module("whisper"), reason="Needs openai-whisper installed."),
    pytest.mark.skipif(not _has_eng_pack(), reason="Needs Tesseract eng language pack."),
]


def test_real_speech_pipeline_detects_intentional_mismatches(tmp_path):
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    report = tmp_path / "report"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    if os.environ.get("BURNSUB_REBUILD_FIXTURE") == "1":
        if not (shutil.which("espeak-ng") or shutil.which("say")):
            pytest.skip("BURNSUB_REBUILD_FIXTURE=1 needs espeak-ng or 'say'.")
        if not _has_module("PIL"):
            pytest.skip("BURNSUB_REBUILD_FIXTURE=1 needs Pillow.")
        bundle = tmp_path / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        build = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), str(bundle)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert build.returncode == 0, build.stderr
    else:
        bundle = BUNDLE_DIR
        if not (bundle / "video.mp4").exists():
            pytest.skip(
                "Committed real-speech bundle missing; rebuild with "
                "BURNSUB_REBUILD_FIXTURE=1 or run scripts/build_real_speech_fixture.py."
            )

    check = subprocess.run(
        [
            sys.executable,
            "-m",
            "burnin_subtitle_checker.cli",
            "--quiet",
            "check",
            str(bundle / "video.mp4"),
            "--output-dir",
            str(report),
            "--ocr-languages",
            "eng",
            "--crop-bottom-percent",
            "25",
            "--workers",
            "4",
            "--frame-offsets",
            "0",
            "--asr-backend",
            "whisper",
            "--asr-model",
            "tiny",
            "--asr-language",
            "en",
            "--device",
            "cpu",
            "--threshold",
            "0.85",
            "--wer-threshold",
            "0.15",
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

    payload = json.loads((report / "report.json").read_text(encoding="utf-8"))
    statuses = [segment["status"] for segment in payload["segments"]]
    spec_kinds = [segment.get("kind", "match") for segment in spec["segments"]]

    # Whisper's segmentation should produce one row per spoken phrase
    # because the fixture inserts ~0.4s silence between phrases.
    assert (
        abs(len(statuses) - len(spec_kinds)) <= 1
    ), f"Whisper produced {len(statuses)} segments; expected ~{len(spec_kinds)}"

    paired_count = min(len(statuses), len(spec_kinds))
    paired = list(zip(spec_kinds[:paired_count], statuses[:paired_count], strict=True))

    expected = Counter(EXPECTED_STATUS_BY_KIND[kind] for kind, _ in paired)
    observed = Counter(status for _, status in paired)

    # The blank-subtitle and wholly-different scenarios are always
    # detectable: one NO_SUBTITLE row, plus one REVIEW row.
    assert observed["NO_SUBTITLE"] >= 1, _diagnostic("NO_SUBTITLE", paired, payload)
    assert observed["REVIEW"] >= max(1, expected["REVIEW"] - 1), _diagnostic(
        "REVIEW", paired, payload
    )
    # Allow up to one match row to drift to REVIEW because of ASR noise.
    assert observed["OK"] >= expected["OK"] - 1, _diagnostic("OK", paired, payload)

    per_kind_status = {kind: status for kind, status in paired}
    if "wholly_different" in per_kind_status:
        assert per_kind_status["wholly_different"] == "REVIEW", _diagnostic(
            "wholly_different", paired, payload
        )
    if "blank_subtitle" in per_kind_status:
        assert per_kind_status["blank_subtitle"] == "NO_SUBTITLE", _diagnostic(
            "blank_subtitle", paired, payload
        )


def _diagnostic(label: str, paired, payload) -> str:
    return (
        f"Failed {label} expectation. Pairs (kind -> status):\n"
        + "\n".join(f"  {kind} -> {status}" for kind, status in paired)
        + "\n\nSegments:\n"
        + json.dumps(payload["segments"], indent=2, ensure_ascii=False)
    )
