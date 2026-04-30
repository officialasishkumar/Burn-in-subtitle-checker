import subprocess

import pytest

from burnin_subtitle_checker.dependencies import (
    collect_doctor_results,
    parse_language_spec,
    run_command,
)
from burnin_subtitle_checker.exceptions import ConfigError, ProcessingError


def test_parse_language_spec_rejects_empty_values():
    with pytest.raises(ConfigError):
        parse_language_spec(" , + ")


def test_run_command_reports_timeouts(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr("burnin_subtitle_checker.dependencies.subprocess.run", fake_run)

    with pytest.raises(ProcessingError, match="timed out"):
        run_command(["slow-tool"], timeout=1)


def test_doctor_checks_opencv_when_preprocessing_is_requested(monkeypatch):
    monkeypatch.setattr("burnin_subtitle_checker.dependencies.executable_path", lambda name: name)
    monkeypatch.setattr("burnin_subtitle_checker.dependencies.tesseract_languages", lambda: {"eng"})
    monkeypatch.setattr(
        "burnin_subtitle_checker.dependencies.python_module_available",
        lambda module: module == "cv2",
    )

    results = collect_doctor_results(
        ocr_languages="eng",
        ocr_preprocess="threshold",
        asr_backend="none",
    )

    assert any(result.name == "opencv preprocessing" and result.ok for result in results)
