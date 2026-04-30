"""Media helpers backed by ffmpeg and ffprobe."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from .dependencies import require_executable, run_command
from .exceptions import ConfigError, ProcessingError


def parse_crop_box(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    pieces = [piece.strip() for piece in value.split(",")]
    if len(pieces) != 4:
        raise ConfigError("--crop-box must use x,y,w,h pixel coordinates")
    try:
        x, y, width, height = [int(piece) for piece in pieces]
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
