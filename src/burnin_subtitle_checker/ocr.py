"""Burned-in subtitle OCR using ffmpeg frame crops and Tesseract CLI."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from .compare import similarity_score
from .dependencies import (
    parse_language_spec,
    python_module_available,
    require_executable,
    require_tesseract_languages,
    run_command,
)
from .exceptions import ConfigError, MissingDependencyError, ProcessingError
from .media import capture_frame_region
from .models import OcrSegment, TranscriptSegment
from .normalize import normalize_text


def parse_frame_offsets(value: str) -> list[float]:
    offsets: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            offsets.append(float(item))
        except ValueError as exc:
            raise ConfigError("--frame-offsets must be a comma-separated list of seconds") from exc
    if not offsets:
        raise ConfigError("--frame-offsets must contain at least one offset")
    return offsets


def ocr_video_segments(
    video_path: Path,
    transcript_segments: list[TranscriptSegment],
    *,
    output_dir: Path,
    languages: str = "hin+kan+eng",
    crop_bottom_percent: float = 15.0,
    crop_box: tuple[int, int, int, int] | None = None,
    frame_offsets: list[float] | None = None,
    psm: int = 6,
    preprocess: str = "none",
    upscale_factor: float = 2.0,
    save_artifacts: bool = True,
) -> list[OcrSegment]:
    _validate_ocr_options(
        crop_bottom_percent=crop_bottom_percent,
        psm=psm,
        preprocess=preprocess,
        upscale_factor=upscale_factor,
    )
    require_executable("tesseract", "sudo apt install tesseract-ocr")
    require_tesseract_languages(languages)
    parse_language_spec(languages)
    require_ocr_preprocess_backend(preprocess)
    offsets = frame_offsets or [0.0]

    ocr_segments: list[OcrSegment] = []
    artifact_context = (
        nullcontext(output_dir / "artifacts" / "crops")
        if save_artifacts
        else TemporaryDirectory(prefix="burnsub-ocr-")
    )

    with artifact_context as artifact_root:
        artifact_dir = Path(artifact_root)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for segment in transcript_segments:
            candidates: list[tuple[str, float, Path, str | None]] = []
            errors: list[str] = []
            for offset in offsets:
                timestamp = max(segment.midpoint + offset, 0)
                crop_path = artifact_dir / f"segment-{segment.index:05d}-{timestamp:.3f}.png"
                try:
                    capture_frame_region(
                        video_path,
                        timestamp,
                        crop_path,
                        crop_bottom_percent=crop_bottom_percent,
                        crop_box=crop_box,
                    )
                    preprocess_crop_for_ocr(
                        crop_path,
                        mode=preprocess,
                        upscale_factor=upscale_factor,
                    )
                    text = run_tesseract(crop_path, languages=languages, psm=psm)
                    candidates.append((text, timestamp, crop_path, None))
                except ProcessingError as exc:
                    errors.append(f"{timestamp:.3f}s: {exc}")

            best = _choose_best_ocr_candidate(candidates, reference_text=segment.text)
            if best is None:
                ocr_segments.append(
                    OcrSegment(
                        index=segment.index,
                        start=segment.start,
                        end=segment.end,
                        timestamp=segment.midpoint,
                        text="",
                        language=languages,
                        sampled_timestamps=[
                            max(segment.midpoint + offset, 0) for offset in offsets
                        ],
                        errors=errors or ["OCR produced no candidates."],
                    )
                )
                continue

            text, timestamp, crop_path, warning = best
            if warning:
                errors.append(warning)
            ocr_segments.append(
                OcrSegment(
                    index=segment.index,
                    start=segment.start,
                    end=segment.end,
                    timestamp=timestamp,
                    text=text,
                    language=languages,
                    crop_path=str(crop_path) if save_artifacts else None,
                    sampled_timestamps=[max(segment.midpoint + offset, 0) for offset in offsets],
                    errors=errors,
                )
            )
    return ocr_segments


def run_tesseract(image_path: Path, *, languages: str, psm: int = 6) -> str:
    completed = run_command(
        [
            "tesseract",
            str(image_path),
            "stdout",
            "-l",
            languages,
            "--psm",
            str(psm),
            "-c",
            "preserve_interword_spaces=1",
        ],
        timeout=120,
    )
    if completed.returncode != 0:
        raise ProcessingError(f"Tesseract OCR failed for {image_path}: {completed.stderr.strip()}")
    return " ".join(line.strip() for line in completed.stdout.splitlines() if line.strip())


def require_ocr_preprocess_backend(mode: str) -> None:
    if mode == "none":
        return
    if python_module_available("cv2"):
        return
    raise MissingDependencyError(
        "OpenCV preprocessing requires Python package 'opencv-python-headless'. "
        "Install with: python -m pip install -e '.[ocr-preprocess]'"
    )


def preprocess_crop_for_ocr(
    image_path: Path,
    *,
    mode: str = "none",
    upscale_factor: float = 2.0,
) -> Path:
    if mode == "none":
        return image_path

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "OpenCV preprocessing requires Python package 'opencv-python-headless'. "
            "Install with: python -m pip install -e '.[ocr-preprocess]'"
        ) from exc

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ProcessingError(f"OpenCV could not read OCR crop: {image_path}")

    if upscale_factor != 1.0:
        height, width = image.shape[:2]
        target_size = (
            max(1, int(round(width * upscale_factor))),
            max(1, int(round(height * upscale_factor))),
        )
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

    if mode == "threshold":
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )

    if not cv2.imwrite(str(image_path), image):
        raise ProcessingError(f"OpenCV could not write preprocessed OCR crop: {image_path}")
    return image_path


def _choose_best_ocr_candidate(
    candidates: list[tuple[str, float, Path, str | None]],
    *,
    reference_text: str = "",
) -> tuple[str, float, Path, str | None] | None:
    if not candidates:
        return None
    reference = normalize_text(reference_text)
    if not reference:
        return max(candidates, key=lambda item: len(normalize_text(item[0])))
    return max(
        candidates,
        key=lambda item: (similarity_score(reference, item[0]), len(normalize_text(item[0]))),
    )


def _validate_ocr_options(
    *,
    crop_bottom_percent: float,
    psm: int,
    preprocess: str,
    upscale_factor: float,
) -> None:
    if not 0 < crop_bottom_percent <= 100:
        raise ConfigError("--crop-bottom-percent must be greater than 0 and at most 100")
    if psm < 0:
        raise ConfigError("--tesseract-psm must be non-negative")
    if preprocess not in {"none", "grayscale", "threshold"}:
        raise ConfigError("--ocr-preprocess must be one of: none, grayscale, threshold")
    if upscale_factor <= 0:
        raise ConfigError("--ocr-upscale-factor must be greater than 0")
