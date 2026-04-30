"""Runtime dependency checks used by the CLI."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ConfigError, MissingDependencyError, ProcessingError


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def executable_path(name: str) -> str | None:
    return shutil.which(name)


def require_executable(name: str, install_hint: str | None = None) -> str:
    path = executable_path(name)
    if path:
        return path
    hint = f" Install it with: {install_hint}" if install_hint else ""
    raise MissingDependencyError(f"Missing required executable '{name}'.{hint}")


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise MissingDependencyError(f"Missing required executable '{args[0]}'.") from exc
    except subprocess.TimeoutExpired as exc:
        command = " ".join(args)
        raise ProcessingError(f"Command timed out after {timeout:g}s: {command}") from exc


def tesseract_languages() -> set[str]:
    require_executable("tesseract", "sudo apt install tesseract-ocr")
    completed = run_command(["tesseract", "--list-langs"])
    if completed.returncode != 0:
        raise MissingDependencyError(
            f"Could not list Tesseract languages: {completed.stderr.strip()}"
        )
    lines = completed.stdout.splitlines()
    return {line.strip() for line in lines[1:] if line.strip()}


def require_tesseract_languages(language_spec: str) -> None:
    requested = parse_language_spec(language_spec)
    installed = tesseract_languages()
    missing = [lang for lang in requested if lang not in installed]
    if missing:
        raise MissingDependencyError(
            "Missing Tesseract language pack(s): "
            + ", ".join(missing)
            + ". Install with: sudo apt install "
            + " ".join(f"tesseract-ocr-{lang}" for lang in missing)
        )


def parse_language_spec(value: str) -> list[str]:
    languages = [item.strip() for item in value.replace(",", "+").split("+") if item.strip()]
    if not languages:
        raise ConfigError("OCR language spec must include at least one language code")
    return languages


def python_module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def collect_doctor_results(
    *,
    ocr_languages: str,
    asr_backend: str,
    ocr_preprocess: str = "none",
    video_path: Path | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(CheckResult("python", True, sys.version.split()[0]))

    for executable, hint in [
        ("ffmpeg", "sudo apt install ffmpeg"),
        ("ffprobe", "sudo apt install ffmpeg"),
        ("tesseract", "sudo apt install tesseract-ocr"),
    ]:
        path = executable_path(executable)
        results.append(CheckResult(executable, bool(path), path or f"missing; {hint}"))

    if executable_path("tesseract"):
        try:
            installed = tesseract_languages()
            missing = [lang for lang in parse_language_spec(ocr_languages) if lang not in installed]
            detail = (
                "installed: " + ", ".join(sorted(installed))
                if not missing
                else "missing: " + ", ".join(missing)
            )
            results.append(CheckResult("tesseract languages", not missing, detail))
        except MissingDependencyError as exc:
            results.append(CheckResult("tesseract languages", False, str(exc)))
    else:
        results.append(CheckResult("tesseract languages", False, "tesseract executable missing"))

    if ocr_preprocess != "none":
        results.append(
            CheckResult(
                "opencv preprocessing",
                python_module_available("cv2"),
                "module: cv2",
            )
        )

    if asr_backend == "none":
        results.append(CheckResult("asr backend", True, "skipped"))
    elif asr_backend == "whisper":
        results.append(
            CheckResult("asr backend", python_module_available("whisper"), "module: whisper")
        )
    elif asr_backend == "faster-whisper":
        results.append(
            CheckResult(
                "asr backend",
                python_module_available("faster_whisper"),
                "module: faster_whisper",
            )
        )
    else:
        results.append(CheckResult("asr backend", False, f"unknown backend: {asr_backend}"))

    if video_path is not None:
        if not video_path.exists():
            results.append(CheckResult("video", False, f"not found: {video_path}"))
        elif executable_path("ffprobe"):
            completed = run_command(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    str(video_path),
                ]
            )
            has_audio = completed.returncode == 0 and "audio" in completed.stdout
            detail = "audio stream found" if has_audio else "no audio stream detected"
            results.append(CheckResult("video audio", has_audio, detail))

    return results
