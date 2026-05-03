"""Burned-in subtitle OCR with pluggable engines and parallel execution."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .backend_config import (
    OCR_AI4BHARAT_ENGINE,
    OCR_EASYOCR_ENGINE,
    OCR_ENGINE_CHOICES,
    OCR_PADDLE_VL_ENGINE,
    OCR_TESSERACT_ENGINE,
)
from .compare import similarity_score
from .dependencies import (
    find_indic_tessdata_dir,
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
from .progress import ProgressReporter

OcrCallable = Callable[[Path, str, int], str]


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


@dataclass(frozen=True, slots=True)
class _OcrCandidate:
    segment_index: int
    timestamp: float
    crop_path: Path
    text: str
    error: str | None


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
    engine: str = "tesseract",
    workers: int = 1,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    progress: ProgressReporter | None = None,
    tessdata_dir: Path | None = None,
) -> list[OcrSegment]:
    _validate_ocr_options(
        crop_bottom_percent=crop_bottom_percent,
        psm=psm,
        preprocess=preprocess,
        upscale_factor=upscale_factor,
        workers=workers,
        engine=engine,
    )

    ocr_callable = _resolve_engine(
        engine,
        languages=languages,
        psm=psm,
        tessdata_dir=tessdata_dir,
    )
    require_ocr_preprocess_backend(preprocess)
    offsets = frame_offsets or [0.0]

    cached: dict[int, OcrSegment] = {}
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        cached = _load_checkpoint(checkpoint_path)

    pending_segments = [s for s in transcript_segments if s.index not in cached]
    if progress is not None and cached:
        progress.advance(len(cached))

    if not pending_segments:
        return _ordered_segments(transcript_segments, cached)

    artifact_context = (
        nullcontext(output_dir / "artifacts" / "crops")
        if save_artifacts
        else TemporaryDirectory(prefix="burnsub-ocr-")
    )

    checkpoint_lock = threading.Lock()

    with artifact_context as artifact_root:
        artifact_dir = Path(artifact_root)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        def process(segment: TranscriptSegment) -> OcrSegment:
            built = _build_segment(
                segment,
                video_path=video_path,
                artifact_dir=artifact_dir,
                offsets=offsets,
                languages=languages,
                crop_bottom_percent=crop_bottom_percent,
                crop_box=crop_box,
                preprocess=preprocess,
                upscale_factor=upscale_factor,
                ocr_callable=ocr_callable,
                save_artifacts=save_artifacts,
                engine=engine,
            )
            if checkpoint_path is not None:
                with checkpoint_lock:
                    _append_checkpoint(checkpoint_path, built)
            if progress is not None:
                progress.advance(1)
            return built

        if workers <= 1:
            for segment in pending_segments:
                cached[segment.index] = process(segment)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process, seg): seg for seg in pending_segments}
                for future in as_completed(futures):
                    seg = futures[future]
                    cached[seg.index] = future.result()

    return _ordered_segments(transcript_segments, cached)


def _build_segment(
    segment: TranscriptSegment,
    *,
    video_path: Path,
    artifact_dir: Path,
    offsets: list[float],
    languages: str,
    crop_bottom_percent: float,
    crop_box: tuple[int, int, int, int] | None,
    preprocess: str,
    upscale_factor: float,
    ocr_callable: OcrCallable,
    save_artifacts: bool,
    engine: str,
) -> OcrSegment:
    candidates: list[_OcrCandidate] = []
    errors: list[str] = []
    sampled = []
    for offset in offsets:
        timestamp = max(segment.midpoint + offset, 0)
        sampled.append(timestamp)
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
            text = ocr_callable(crop_path, languages, segment.index)
            candidates.append(
                _OcrCandidate(
                    segment_index=segment.index,
                    timestamp=timestamp,
                    crop_path=crop_path,
                    text=text,
                    error=None,
                )
            )
        except ProcessingError as exc:
            errors.append(f"{timestamp:.3f}s: {exc}")

    best = _choose_best_candidate(candidates, reference_text=segment.text)
    if best is None:
        if not save_artifacts:
            for candidate in candidates:
                _safe_unlink(candidate.crop_path)
        return OcrSegment(
            index=segment.index,
            start=segment.start,
            end=segment.end,
            timestamp=segment.midpoint,
            text="",
            language=languages,
            sampled_timestamps=sampled,
            errors=errors or ["OCR produced no candidates."],
            engine=engine,
        )

    if not save_artifacts:
        for candidate in candidates:
            _safe_unlink(candidate.crop_path)

    return OcrSegment(
        index=segment.index,
        start=segment.start,
        end=segment.end,
        timestamp=best.timestamp,
        text=best.text,
        language=languages,
        crop_path=str(best.crop_path) if save_artifacts else None,
        sampled_timestamps=sampled,
        errors=errors,
        engine=engine,
    )


def resolve_tesseract_data_dir(
    languages: str,
    *,
    requested_dir: Path | None = None,
) -> Path | None:
    if requested_dir is not None:
        return requested_dir
    return find_indic_tessdata_dir(languages)


def _resolve_engine(
    engine: str,
    *,
    languages: str,
    psm: int,
    tessdata_dir: Path | None = None,
) -> OcrCallable:
    if engine == OCR_TESSERACT_ENGINE:
        data_dir = resolve_tesseract_data_dir(languages, requested_dir=tessdata_dir)
        require_executable("tesseract", "sudo apt install tesseract-ocr")
        if data_dir is None:
            require_tesseract_languages(languages)
        else:
            require_tesseract_languages(languages, tessdata_dir=data_dir)
        parse_language_spec(languages)

        def call(image_path: Path, langs: str, _segment_index: int) -> str:
            return run_tesseract(
                image_path,
                languages=langs,
                psm=psm,
                tessdata_dir=data_dir,
            )

        return call
    if engine == OCR_EASYOCR_ENGINE:
        from .easyocr_engine import map_languages, run_easyocr

        codes = map_languages(languages)
        if not codes:
            raise ConfigError(
                "EasyOCR could not map any of the requested languages to its codes."
            )

        def call(image_path: Path, langs: str, _segment_index: int) -> str:
            return run_easyocr(image_path, languages=langs)

        return call
    if engine == OCR_PADDLE_VL_ENGINE:
        from .paddleocr_vl_engine import run_paddleocr_vl

        def call(image_path: Path, langs: str, _segment_index: int) -> str:
            return run_paddleocr_vl(image_path, languages=langs)

        return call
    if engine == OCR_AI4BHARAT_ENGINE:
        from .ai4bharat_ocr_engine import run_ai4bharat_ocr

        def call(image_path: Path, langs: str, _segment_index: int) -> str:
            return run_ai4bharat_ocr(image_path, languages=langs)

        return call
    raise ConfigError(f"Unsupported --ocr-engine: {engine}")


def run_tesseract(
    image_path: Path,
    *,
    languages: str,
    psm: int = 6,
    tessdata_dir: Path | None = None,
) -> str:
    command = [
        "tesseract",
        str(image_path),
        "stdout",
        "-l",
        languages,
        "--psm",
        str(psm),
        "-c",
        "preserve_interword_spaces=1",
    ]
    if tessdata_dir is not None:
        command.extend(["--tessdata-dir", str(tessdata_dir)])
    completed = run_command(
        command,
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


def _choose_best_candidate(
    candidates: Iterable[_OcrCandidate],
    *,
    reference_text: str = "",
) -> _OcrCandidate | None:
    candidates = list(candidates)
    if not candidates:
        return None
    reference = normalize_text(reference_text)
    if not reference:
        return max(candidates, key=lambda item: len(normalize_text(item.text)))
    return max(
        candidates,
        key=lambda item: (similarity_score(reference, item.text), len(normalize_text(item.text))),
    )


def _validate_ocr_options(
    *,
    crop_bottom_percent: float,
    psm: int,
    preprocess: str,
    upscale_factor: float,
    workers: int,
    engine: str,
) -> None:
    if not 0 < crop_bottom_percent <= 100:
        raise ConfigError("--crop-bottom-percent must be greater than 0 and at most 100")
    if psm < 0:
        raise ConfigError("--tesseract-psm must be non-negative")
    if preprocess not in {"none", "grayscale", "threshold"}:
        raise ConfigError("--ocr-preprocess must be one of: none, grayscale, threshold")
    if upscale_factor <= 0:
        raise ConfigError("--ocr-upscale-factor must be greater than 0")
    if workers < 1:
        raise ConfigError("--workers must be at least 1")
    if engine not in OCR_ENGINE_CHOICES:
        raise ConfigError("--ocr-engine must be one of: " + ", ".join(OCR_ENGINE_CHOICES))


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _ordered_segments(
    transcript_segments: list[TranscriptSegment],
    cached: dict[int, OcrSegment],
) -> list[OcrSegment]:
    return [cached[segment.index] for segment in transcript_segments if segment.index in cached]


def _append_checkpoint(checkpoint_path: Path, segment: OcrSegment) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        json.dump(segment.to_dict(), handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:  # pragma: no cover - best effort durability
            pass


def _load_checkpoint(checkpoint_path: Path) -> dict[int, OcrSegment]:
    cached: dict[int, OcrSegment] = {}
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            cached[int(payload["index"])] = OcrSegment(
                index=int(payload["index"]),
                start=float(payload["start"]),
                end=float(payload["end"]),
                timestamp=float(payload["timestamp"]),
                text=str(payload.get("text") or ""),
                language=str(payload.get("language", "")),
                crop_path=payload.get("crop_path"),
                frame_path=payload.get("frame_path"),
                sampled_timestamps=[float(v) for v in payload.get("sampled_timestamps", [])],
                errors=[str(v) for v in payload.get("errors", [])],
                engine=str(payload.get("engine", "tesseract")),
            )
    return cached
