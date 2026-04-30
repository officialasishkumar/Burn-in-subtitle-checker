import subprocess

import pytest

from burnin_subtitle_checker.dependencies import parse_language_spec, run_command
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
