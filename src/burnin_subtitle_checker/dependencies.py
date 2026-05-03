"""Runtime dependency checks used by the CLI."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .backend_config import (
    ASR_BACKEND_MODULES,
    ASR_INDIC_CONFORMER_BACKEND,
    ASR_INDICWHISPER_BACKEND,
    ASR_WHISPER_BACKEND,
    INDIC_CONFORMER_MODEL_ID,
    INDIC_OCR_TESSERACT_LANGUAGES,
    INDICWHISPER_MODEL_FAMILY,
    OCR_ENGINE_MODULES,
    OCR_TESSERACT_ENGINE,
    candidate_indic_tessdata_dirs,
    indic_tessdata_install_command,
)
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


def tesseract_languages(tessdata_dir: Path | str | None = None) -> set[str]:
    require_executable("tesseract", "sudo apt install tesseract-ocr")
    command = ["tesseract", "--list-langs"]
    if tessdata_dir is not None:
        command.extend(["--tessdata-dir", str(tessdata_dir)])
    completed = run_command(command)
    if completed.returncode != 0:
        raise MissingDependencyError(
            f"Could not list Tesseract languages: {completed.stderr.strip()}"
        )
    lines = completed.stdout.splitlines()
    return {line.strip() for line in lines[1:] if line.strip()}


def require_tesseract_languages(
    language_spec: str,
    *,
    tessdata_dir: Path | str | None = None,
) -> None:
    requested = parse_language_spec(language_spec)
    installed = tesseract_languages(tessdata_dir)
    missing = [lang for lang in requested if lang not in installed]
    if missing:
        location = f" in {tessdata_dir}" if tessdata_dir is not None else ""
        raise MissingDependencyError(
            f"Missing Tesseract language pack(s){location}: "
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
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def cuda_available() -> tuple[bool, str]:
    if not python_module_available("torch"):
        return False, "torch not installed"
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on torch runtime
        return False, f"torch import failed: {exc}"
    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if count else ""
            return True, f"{count} device(s); first: {name}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, "Apple Metal (MPS) available"
        return False, "no CUDA / MPS device detected"
    except Exception as exc:  # pragma: no cover - depends on torch runtime
        return False, f"torch device probe failed: {exc}"


def executable_version(name: str, *, args: list[str] | None = None) -> str | None:
    path = executable_path(name)
    if not path:
        return None
    try:
        completed = run_command([path, *(args or ["-version"])], timeout=10)
    except (MissingDependencyError, ProcessingError):
        return None
    text = (completed.stdout or completed.stderr or "").strip().splitlines()
    return text[0] if text else None


def traineddata_languages_in_dir(path: Path) -> set[str]:
    if not path.exists() or not path.is_dir():
        return set()
    return {item.stem for item in path.glob("*.traineddata") if item.is_file()}


def find_indic_tessdata_dir(language_spec: str) -> Path | None:
    requested = set(parse_language_spec(language_spec))
    for path in candidate_indic_tessdata_dirs():
        installed = traineddata_languages_in_dir(path)
        if requested.issubset(installed) and installed.intersection(INDIC_OCR_TESSERACT_LANGUAGES):
            return path
    return None


def indic_tessdata_detail(language_spec: str) -> tuple[Path | None, str]:
    requested = set(parse_language_spec(language_spec))
    requested_indic = requested.intersection(INDIC_OCR_TESSERACT_LANGUAGES)
    indic_dir = find_indic_tessdata_dir(language_spec)
    if indic_dir is not None:
        return indic_dir, f"Indic-OCR data: {indic_dir}"
    if not requested_indic:
        return None, "not needed for non-Indic OCR languages"

    partial_details: list[str] = []
    for path in candidate_indic_tessdata_dirs():
        installed = traineddata_languages_in_dir(path)
        if installed.intersection(requested_indic):
            missing = sorted(requested - installed)
            partial_details.append(f"{path} missing: {', '.join(missing)}")
    if partial_details:
        return None, "; ".join(partial_details)
    return None, "not detected; install with: " + indic_tessdata_install_command()


def huggingface_cache_detail(model_id: str) -> str:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_dir = cache_root / ("models--" + model_id.replace("/", "--"))
    if model_dir.exists():
        return f"weights cache: {model_dir}"
    return "weights cache not found; first run will download"


def collect_doctor_results(
    *,
    ocr_languages: str,
    asr_backend: str,
    ocr_engine: str = "tesseract",
    ocr_preprocess: str = "none",
    video_path: Path | None = None,
    tessdata_dir: Path | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(CheckResult("python", True, sys.version.split()[0]))

    executable_checks = [
        ("ffmpeg", "sudo apt install ffmpeg", ["-version"]),
        ("ffprobe", "sudo apt install ffmpeg", ["-version"]),
    ]
    if ocr_engine == OCR_TESSERACT_ENGINE:
        executable_checks.append(("tesseract", "sudo apt install tesseract-ocr", ["--version"]))
    for executable, hint, version_args in executable_checks:
        path = executable_path(executable)
        if path:
            version = executable_version(executable, args=version_args) or path
            results.append(CheckResult(executable, True, version))
        else:
            results.append(CheckResult(executable, False, f"missing; {hint}"))

    if ocr_engine == OCR_TESSERACT_ENGINE:
        if executable_path("tesseract"):
            try:
                installed = (
                    tesseract_languages(tessdata_dir) if tessdata_dir else tesseract_languages()
                )
                missing = [
                    lang for lang in parse_language_spec(ocr_languages) if lang not in installed
                ]
                detail = (
                    "installed: " + ", ".join(sorted(installed))
                    if not missing
                    else "missing: " + ", ".join(missing)
                )
                results.append(CheckResult("tesseract languages", not missing, detail))
            except MissingDependencyError as exc:
                results.append(CheckResult("tesseract languages", False, str(exc)))
        else:
            results.append(
                CheckResult("tesseract languages", False, "tesseract executable missing")
            )
        try:
            if tessdata_dir is not None:
                installed = traineddata_languages_in_dir(tessdata_dir)
                detail = (
                    f"custom tessdata: {tessdata_dir}"
                    if installed.intersection(INDIC_OCR_TESSERACT_LANGUAGES)
                    else f"custom tessdata has no Indic-OCR marker languages: {tessdata_dir}"
                )
            else:
                _indic_dir, detail = indic_tessdata_detail(ocr_languages)
            results.append(CheckResult("Indic-OCR tessdata", True, detail))
        except ConfigError as exc:
            results.append(CheckResult("Indic-OCR tessdata", False, str(exc)))
    elif ocr_engine in OCR_ENGINE_MODULES:
        module, extra = OCR_ENGINE_MODULES[ocr_engine]
        ok = python_module_available(module)
        results.append(
            CheckResult(
                f"ocr engine: {ocr_engine}",
                ok,
                f"module: {module}" if ok else f"missing; install '{extra}'",
            )
        )
    else:
        results.append(CheckResult("ocr engine", False, f"unknown engine: {ocr_engine}"))

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
    elif asr_backend == "auto":
        indic_module, indic_extra = ASR_BACKEND_MODULES[ASR_INDICWHISPER_BACKEND]
        whisper_module, whisper_extra = ASR_BACKEND_MODULES[ASR_WHISPER_BACKEND]
        if python_module_available(indic_module):
            results.append(
                CheckResult(
                    "asr backend: auto",
                    True,
                    "will use indicwhisper; " + huggingface_cache_detail(INDICWHISPER_MODEL_FAMILY),
                )
            )
        elif python_module_available(whisper_module):
            results.append(
                CheckResult(
                    "asr backend: auto",
                    True,
                    f"IndicWhisper missing; will fall back to whisper ({whisper_module})",
                )
            )
        else:
            results.append(
                CheckResult(
                    "asr backend: auto",
                    False,
                    f"missing; install '{indic_extra}' or '{whisper_extra}'",
                )
            )
    elif asr_backend in ASR_BACKEND_MODULES:
        module, extra = ASR_BACKEND_MODULES[asr_backend]
        ok = python_module_available(module)
        cache_detail = ""
        if asr_backend == ASR_INDICWHISPER_BACKEND:
            cache_detail = "; " + huggingface_cache_detail(INDICWHISPER_MODEL_FAMILY)
        elif asr_backend == ASR_INDIC_CONFORMER_BACKEND:
            cache_detail = "; " + huggingface_cache_detail(INDIC_CONFORMER_MODEL_ID)
        results.append(
            CheckResult(
                "asr backend",
                ok,
                f"module: {module}{cache_detail}" if ok else f"missing; install '{extra}'",
            )
        )
    else:
        results.append(CheckResult("asr backend", False, f"unknown backend: {asr_backend}"))

    has_gpu, gpu_detail = cuda_available()
    accelerator_detail = gpu_detail if has_gpu else f"cpu only ({gpu_detail})"
    results.append(CheckResult("accelerator", True, accelerator_detail))

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
