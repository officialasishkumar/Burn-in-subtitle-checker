"""Media helpers backed by ffmpeg and ffprobe."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from tempfile import TemporaryDirectory

from .dependencies import python_module_available, require_executable, run_command
from .exceptions import ConfigError, MissingDependencyError, ProcessingError


def parse_crop_box(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    pieces = [piece.strip() for piece in value.split(",")]
    if len(pieces) != 4:
        raise ConfigError("--crop-box must use x,y,w,h pixel coordinates")
    try:
        x, y, width, height = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise ConfigError("--crop-box values must be integers") from exc
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ConfigError("--crop-box must use non-negative x/y and positive width/height")
    return x, y, width, height


def validate_video_path(path: Path) -> None:
    if not path.exists():
        raise ConfigError(f"Input video does not exist: {path}")
    if not path.is_file():
        raise ConfigError(f"Input video is not a file: {path}")


def probe_media(video_path: Path) -> dict:
    require_executable("ffprobe", "sudo apt install ffmpeg")
    validate_video_path(video_path)
    completed = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ],
        timeout=60,
    )
    if completed.returncode != 0:
        raise ProcessingError(f"ffprobe failed for {video_path}: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProcessingError(f"ffprobe returned invalid JSON for {video_path}") from exc


def media_duration_seconds(video_path: Path) -> float | None:
    metadata = probe_media(video_path)
    raw = metadata.get("format", {}).get("duration")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def video_resolution(video_path: Path) -> tuple[int, int] | None:
    metadata = probe_media(video_path)
    for stream in metadata.get("streams", []):
        if stream.get("codec_type") == "video":
            try:
                return int(stream["width"]), int(stream["height"])
            except (KeyError, TypeError, ValueError):
                continue
    return None


def ensure_audio_stream(video_path: Path) -> None:
    metadata = probe_media(video_path)
    streams = metadata.get("streams", [])
    if not any(stream.get("codec_type") == "audio" for stream in streams):
        raise ConfigError(f"No audio stream found in {video_path}")


def extract_audio(video_path: Path, output_wav: Path) -> Path:
    require_executable("ffmpeg", "sudo apt install ffmpeg")
    validate_video_path(video_path)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    completed = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(output_wav),
        ]
    )
    if completed.returncode != 0:
        raise ProcessingError(f"ffmpeg audio extraction failed: {completed.stderr.strip()}")
    return output_wav


def capture_frame_region(
    video_path: Path,
    timestamp: float,
    output_path: Path,
    *,
    crop_bottom_percent: float = 15.0,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Path:
    require_executable("ffmpeg", "sudo apt install ffmpeg")
    validate_video_path(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_expression = _crop_filter(crop_bottom_percent=crop_bottom_percent, crop_box=crop_box)
    completed = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(timestamp, 0):.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            filter_expression,
            str(output_path),
        ],
        timeout=120,
    )
    if completed.returncode != 0:
        quoted = " ".join(shlex.quote(arg) for arg in completed.args)
        raise ProcessingError(
            f"ffmpeg frame capture failed at {timestamp:.3f}s: {completed.stderr.strip()}\n{quoted}"
        )
    return output_path


def detect_subtitle_band(
    video_path: Path,
    *,
    sample_timestamps: list[float],
    expand_percent: float = 5.0,
) -> tuple[int, int, int, int] | None:
    """Sample frames and locate the most consistent bright text band.

    Returns a (x, y, width, height) crop box covering the detected band, or
    ``None`` if OpenCV is unavailable or no band could be located.
    """

    if not sample_timestamps:
        return None
    if not python_module_available("cv2"):
        raise MissingDependencyError(
            "Adaptive crop detection requires Python package 'opencv-python-headless'. "
            "Install with: python -m pip install -e '.[ocr-preprocess]'"
        )
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    resolution = video_resolution(video_path)
    if resolution is None:
        return None
    width, height = resolution

    band_scores: list[np.ndarray] = []
    with TemporaryDirectory(prefix="burnsub-band-") as tmp:
        tmp_dir = Path(tmp)
        for index, timestamp in enumerate(sample_timestamps):
            sample_path = tmp_dir / f"sample-{index:04d}.png"
            try:
                capture_frame_region(
                    video_path,
                    timestamp,
                    sample_path,
                    crop_bottom_percent=100.0,
                    crop_box=None,
                )
            except ProcessingError:
                continue
            image = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            sobel = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            row_energy = np.mean(np.abs(sobel), axis=1)
            band_scores.append(row_energy)

    if not band_scores:
        return None
    profile = np.mean(np.stack(band_scores), axis=0)
    if profile.size == 0:
        return None

    threshold = profile.mean() + profile.std()
    active = profile >= threshold
    if not active.any():
        return None

    band_start, band_end = _largest_active_band(active)
    if band_start is None or band_end is None:
        return None

    expand_pixels = max(1, int(round(height * expand_percent / 100)))
    band_start = max(0, band_start - expand_pixels)
    band_end = min(height, band_end + expand_pixels)
    band_height = max(1, band_end - band_start)

    return 0, band_start, width, band_height


def _largest_active_band(active) -> tuple[int | None, int | None]:
    best_start: int | None = None
    best_end: int | None = None
    best_size = 0
    current_start: int | None = None
    for index, flag in enumerate(active.tolist()):
        if flag:
            if current_start is None:
                current_start = index
        elif current_start is not None:
            size = index - current_start
            if size > best_size:
                best_size = size
                best_start = current_start
                best_end = index
            current_start = None
    if current_start is not None:
        size = len(active) - current_start
        if size > best_size:
            best_start = current_start
            best_end = len(active)
    return best_start, best_end


def _crop_filter(
    *,
    crop_bottom_percent: float,
    crop_box: tuple[int, int, int, int] | None,
) -> str:
    if crop_box is not None:
        x, y, width, height = crop_box
        return f"crop={width}:{height}:{x}:{y}"
    if not 0 < crop_bottom_percent <= 100:
        raise ConfigError("--crop-bottom-percent must be greater than 0 and at most 100")
    fraction = crop_bottom_percent / 100
    return f"crop=iw:ih*{fraction:.6f}:0:ih*(1-{fraction:.6f})"
