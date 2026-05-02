#!/usr/bin/env python3
"""Build a synthetic burn-in subtitle video from fixtures/regression/spec.json.

The script renders one PNG per segment (via Pillow) using a TrueType font that
ships with most desktop and CI environments, then stitches the frames together
with ffmpeg using the per-segment timing. A silent sine-wave audio track is
attached so the resulting MP4 has both video and audio streams.

The output bundle contains:
  - ``video.mp4``         the generated video with burned-in text
  - ``transcript.json``   precomputed transcript matching ``audio_text``
  - ``reference.srt``     reference SRT also matching ``audio_text``

The companion regression test (``tests/test_regression_pipeline.py``) builds
this bundle into a tmp directory and runs ``burnsub check`` against it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - fixture script only
    raise SystemExit("This script requires Pillow. Install with: pip install Pillow") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO_ROOT / "fixtures" / "regression" / "spec.json"

CANDIDATE_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="Spec JSON path.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write video.mp4, transcript.json, reference.srt into.",
    )
    parser.add_argument(
        "--font", type=Path, default=None, help="Optional TrueType font override."
    )
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_settings = spec["video"]
    segments = spec["segments"]
    font_path = args.font or _locate_font()
    if font_path is None:
        raise SystemExit(
            "Could not find a usable TrueType font. Pass --font /path/to/font.ttf."
        )
    font = _load_font(font_path, video_settings["font_size"])

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to assemble the regression video.")

    video_path = output_dir / "video.mp4"
    transcript_path = output_dir / "transcript.json"
    reference_path = output_dir / "reference.srt"

    with TemporaryDirectory(prefix="burnsub-regression-") as tmp:
        tmp_dir = Path(tmp)
        frame_paths = _render_segment_frames(
            segments,
            tmp_dir,
            font=font,
            width=video_settings["width"],
            height=video_settings["height"],
            background=video_settings["background"],
            text_color=video_settings["text_color"],
        )
        _assemble_video(
            video_path,
            tmp_dir=tmp_dir,
            frame_paths=frame_paths,
            segments=segments,
            fps=video_settings["fps"],
            audio_frequency=video_settings["audio_frequency"],
        )

    transcript_path.write_text(_render_transcript(segments), encoding="utf-8")
    reference_path.write_text(_render_srt(segments), encoding="utf-8")
    print(f"Wrote {video_path}")
    print(f"Wrote {transcript_path}")
    print(f"Wrote {reference_path}")
    return 0


def _locate_font() -> Path | None:
    for candidate in CANDIDATE_FONT_PATHS:
        if Path(candidate).is_file():
            return Path(candidate)
    return None


def _load_font(font_path: Path, size: int):
    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError as exc:  # pragma: no cover - depends on environment fonts
        raise SystemExit(f"Could not load font {font_path}: {exc}") from exc


def _render_segment_frames(
    segments: Iterable[dict[str, Any]],
    tmp_dir: Path,
    *,
    font: Any,
    width: int,
    height: int,
    background: str,
    text_color: str,
) -> list[Path]:
    frame_paths: list[Path] = []
    for index, segment in enumerate(segments):
        image = Image.new("RGB", (width, height), color=background)
        burn_text: str = segment.get("burn_text", "")
        if burn_text:
            draw = ImageDraw.Draw(image)
            bbox = draw.textbbox((0, 0), burn_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = (width - text_w) // 2
            y = height - text_h - 80
            draw.text((x, y), burn_text, fill=text_color, font=font)
        path = tmp_dir / f"segment-{index:03d}.png"
        image.save(path)
        frame_paths.append(path)
    return frame_paths


def _assemble_video(
    output_path: Path,
    *,
    tmp_dir: Path,
    frame_paths: list[Path],
    segments: list[dict[str, Any]],
    fps: int,
    audio_frequency: int,
) -> None:
    last_segment = segments[-1]
    total_duration = float(last_segment["end"]) + 0.5
    concat_list = tmp_dir / "concat.txt"
    with concat_list.open("w", encoding="utf-8") as handle:
        for index, segment in enumerate(segments):
            start = float(segment["start"])
            end = float(segment["end"])
            duration = max(end - start, 0.05)
            handle.write(f"file '{frame_paths[index].as_posix()}'\n")
            handle.write(f"duration {duration:.3f}\n")
        handle.write(f"file '{frame_paths[-1].as_posix()}'\n")
    silent_video = tmp_dir / "video_only.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-fps_mode",
            "cfr",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(silent_video),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(silent_video),
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={audio_frequency}:duration={total_duration:.3f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )


def _render_transcript(segments: list[dict[str, Any]]) -> str:
    payload = {
        "source": "fixtures/regression",
        "segments": [
            {
                "index": index,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["audio_text"],
            }
            for index, segment in enumerate(segments)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _render_srt(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(
            f"{_format_timestamp(float(segment['start']))} --> "
            f"{_format_timestamp(float(segment['end']))}"
        )
        lines.append(segment["audio_text"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, rest = divmod(total_ms, 3600 * 1000)
    minutes, rest = divmod(rest, 60 * 1000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


if __name__ == "__main__":
    sys.exit(main())
