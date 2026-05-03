#!/usr/bin/env python3
"""Build a synthetic multilingual burn-in subtitle stress fixture.

The generated bundle is intentionally editable:

  - ``reference.srt`` contains the audio/truth text.
  - ``burned_subtitles.srt`` contains the text burned into the video.
  - ``transcript.json`` precomputes the transcript so stress runs exercise OCR
    and comparison without needing ASR model weights.
  - ``expected.json`` records the intended OK/REVIEW/NO_SUBTITLE mix.

Edit ``fixtures/stress/spec.json`` or the generated ``burned_subtitles.srt``
and rerun this script to create small-mismatch, timing-drift, missing-subtitle,
and long-video cases.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO_ROOT / "fixtures" / "stress" / "spec.json"

TESSERACT_LANGUAGE_BY_ISO = {
    "as": "asm",
    "bn": "ben",
    "en": "eng",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}

GENERAL_FONT_PATHS = (
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)

FONT_PATHS_BY_LANGUAGE = {
    "as": (
        "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Bangla Sangam MN.ttc",
    ),
    "bn": (
        "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Bangla Sangam MN.ttc",
    ),
    "gu": (
        "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Gujarati Sangam MN.ttc",
    ),
    "hi": (
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
    ),
    "kn": (
        "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Kannada Sangam MN.ttc",
    ),
    "ml": (
        "/usr/share/fonts/truetype/noto/NotoSansMalayalam-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Malayalam Sangam MN.ttc",
    ),
    "mr": (
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
    ),
    "or": (
        "/usr/share/fonts/truetype/noto/NotoSansOriya-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Oriya Sangam MN.ttc",
    ),
    "pa": (
        "/usr/share/fonts/truetype/noto/NotoSansGurmukhi-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Gurmukhi Sangam MN.ttc",
    ),
    "sa": (
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
    ),
    "ta": (
        "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc",
    ),
    "te": (
        "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Telugu Sangam MN.ttc",
    ),
    "ur": (
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/System/Library/Fonts/Supplemental/DecoTypeNaskh.ttc",
    ),
}


@dataclass(frozen=True, slots=True)
class StressSegment:
    index: int
    source_index: int
    repeat_index: int
    start: float
    end: float
    language: str
    ocr_language: str
    audio_text: str
    burn_text: str
    expected_status: str
    kind: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory to write the stress bundle.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="Stress spec JSON path.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Repeat the spec segment list N times. Overrides spec.repeat.",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help="Repeat enough segments to reach at least this many seconds.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="Override default per-segment duration when a segment has no duration.",
    )
    parser.add_argument(
        "--only-languages",
        default=None,
        help="Comma/plus-separated ISO language codes to include, e.g. kn,hi,en.",
    )
    parser.add_argument(
        "--font-name",
        default=None,
        help="ASS/libass font family used by ffmpeg's subtitles filter.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="TrueType/OpenType font path used by the Pillow renderer.",
    )
    parser.add_argument(
        "--fonts-dir",
        type=Path,
        default=None,
        help="Optional directory for libass/fontconfig to search for fonts.",
    )
    parser.add_argument(
        "--renderer",
        default="auto",
        choices=["auto", "ffmpeg-subtitles", "pillow"],
        help=(
            "Video renderer. auto uses ffmpeg subtitles/libass when available, "
            "then falls back to Pillow frame rendering."
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only write JSON/SRT/manifest files; skip ffmpeg video rendering.",
    )
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    languages = _parse_language_filter(args.only_languages)
    segments = build_segment_cases(
        spec,
        repeat=args.repeat,
        target_duration=args.target_duration,
        only_languages=languages,
        segment_seconds=args.segment_seconds,
    )
    if not segments:
        raise SystemExit("No stress segments selected.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_srt = output_dir / "reference.srt"
    burned_srt = output_dir / "burned_subtitles.srt"
    transcript_json = output_dir / "transcript.json"
    expected_json = output_dir / "expected.json"
    manifest_json = output_dir / "manifest.json"
    ocr_languages_txt = output_dir / "ocr_languages.txt"
    video_path = output_dir / "video.mp4"

    video_settings = spec.get("video", {})
    max_chars = int(video_settings.get("max_chars_per_line", 42))
    reference_srt.write_text(render_srt(segments, text_field="audio", max_chars=max_chars), "utf-8")
    burned_srt.write_text(render_srt(segments, text_field="burn", max_chars=max_chars), "utf-8")
    transcript_json.write_text(render_transcript_json(segments), "utf-8")
    expected_json.write_text(render_expected_json(segments), "utf-8")
    ocr_languages = ocr_language_string(segments)
    ocr_languages_txt.write_text(ocr_languages + "\n", "utf-8")

    if not args.no_video:
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg is required to render video.mp4")
        render_video(
            video_path,
            burned_srt=burned_srt,
            segments=segments,
            duration=max(segment.end for segment in segments) + 0.25,
            video_settings=video_settings,
            font_name=args.font_name,
            font_path=args.font,
            fonts_dir=args.fonts_dir,
            renderer=args.renderer,
        )

    manifest = build_manifest(
        spec=spec,
        output_dir=output_dir,
        segments=segments,
        video_path=video_path if not args.no_video else None,
        ocr_languages=ocr_languages,
        video_rendered=not args.no_video,
    )
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8")

    print(f"Wrote {transcript_json}")
    print(f"Wrote {reference_srt}")
    print(f"Wrote {burned_srt}")
    print(f"Wrote {expected_json}")
    print(f"Wrote {manifest_json}")
    if not args.no_video:
        print(f"Wrote {video_path}")
    print(f"OCR languages: {ocr_languages}")
    return 0


def build_segment_cases(
    spec: dict[str, Any],
    *,
    repeat: int | None = None,
    target_duration: float | None = None,
    only_languages: set[str] | None = None,
    segment_seconds: float | None = None,
) -> list[StressSegment]:
    video_settings = spec.get("video", {})
    default_duration = float(
        segment_seconds
        if segment_seconds is not None
        else video_settings.get("segment_seconds", 2.4)
    )
    if default_duration <= 0:
        raise ValueError("segment duration must be positive")

    base_segments = [
        item
        for item in spec.get("segments", [])
        if only_languages is None or str(item.get("language", "")).lower() in only_languages
    ]
    if not base_segments:
        return []

    base_duration = sum(float(item.get("duration", default_duration)) for item in base_segments)
    if base_duration <= 0:
        raise ValueError("base stress duration must be positive")

    spec_repeat = int(spec.get("repeat", 1))
    repeat_count = repeat if repeat is not None else spec_repeat
    if target_duration is not None:
        repeat_count = max(repeat_count, math.ceil(float(target_duration) / base_duration))
    if repeat_count < 1:
        raise ValueError("repeat must be at least 1")

    cursor = 0.0
    built: list[StressSegment] = []
    for repeat_index in range(repeat_count):
        for source_index, raw in enumerate(base_segments):
            duration = float(raw.get("duration", default_duration))
            if duration <= 0:
                raise ValueError(f"segment {source_index} duration must be positive")
            language = str(raw.get("language", "")).strip().lower()
            if not language:
                raise ValueError(f"segment {source_index} is missing language")
            audio_text = str(raw.get("audio_text", "")).strip()
            burn_text = str(raw.get("burn_text", audio_text)).strip()
            built.append(
                StressSegment(
                    index=len(built),
                    source_index=source_index,
                    repeat_index=repeat_index,
                    start=cursor,
                    end=cursor + duration,
                    language=language,
                    ocr_language=str(
                        raw.get("ocr_language")
                        or TESSERACT_LANGUAGE_BY_ISO.get(language)
                        or language
                    ),
                    audio_text=audio_text,
                    burn_text=burn_text,
                    expected_status=str(raw.get("expected_status", "OK")).upper(),
                    kind=str(raw.get("kind", "match")),
                )
            )
            cursor += duration
    return built


def render_transcript_json(segments: Sequence[StressSegment]) -> str:
    payload = {
        "source": "fixtures/stress",
        "segments": [
            {
                "index": segment.index,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.audio_text,
            }
            for segment in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def render_expected_json(segments: Sequence[StressSegment]) -> str:
    counts: dict[str, int] = {}
    for segment in segments:
        counts[segment.expected_status] = counts.get(segment.expected_status, 0) + 1
    payload = {
        "summary": {"total": len(segments), "by_status": counts},
        "segments": [asdict(segment) for segment in segments],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def render_srt(
    segments: Sequence[StressSegment],
    *,
    text_field: str,
    max_chars: int = 42,
) -> str:
    if text_field not in {"audio", "burn"}:
        raise ValueError("text_field must be 'audio' or 'burn'")
    lines: list[str] = []
    cue_index = 1
    for segment in segments:
        text = segment.audio_text if text_field == "audio" else segment.burn_text
        text = text.strip()
        if text_field == "burn" and not text:
            continue
        lines.append(str(cue_index))
        lines.append(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}")
        lines.extend(wrap_srt_text(text, max_chars=max_chars))
        lines.append("")
        cue_index += 1
    return "\n".join(lines).rstrip() + "\n"


def wrap_srt_text(text: str, *, max_chars: int = 42) -> list[str]:
    wrapped: list[str] = []
    for raw_line in text.splitlines() or [text]:
        line = raw_line.strip()
        if not line:
            wrapped.append("")
            continue
        pieces = textwrap.wrap(
            line,
            width=max(8, max_chars),
            break_long_words=False,
            replace_whitespace=False,
        )
        wrapped.extend(pieces or [line])
    return wrapped


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, rest = divmod(total_ms, 3600 * 1000)
    minutes, rest = divmod(rest, 60 * 1000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def ocr_language_string(segments: Iterable[StressSegment]) -> str:
    ordered: list[str] = []
    for segment in segments:
        if segment.ocr_language not in ordered:
            ordered.append(segment.ocr_language)
    return "+".join(ordered)


def render_video_with_burned_subtitles(
    output_path: Path,
    *,
    burned_srt: Path,
    duration: float,
    video_settings: dict[str, Any],
    font_name: str | None = None,
    fonts_dir: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = int(video_settings.get("width", 1280))
    height = int(video_settings.get("height", 720))
    fps = int(video_settings.get("fps", 25))
    background = str(video_settings.get("background", "#111111"))
    audio_frequency = int(video_settings.get("audio_frequency", 440))
    style = ass_force_style(video_settings, font_name=font_name)
    subtitle_filter = subtitles_filter(burned_srt, force_style=style, fonts_dir=fonts_dir)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={background}:s={width}x{height}:r={fps}:d={duration:.3f}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={audio_frequency}:sample_rate=16000:duration={duration:.3f}",
        "-vf",
        subtitle_filter,
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(output_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            "ffmpeg failed to render burned subtitles. Check that ffmpeg was built "
            f"with libass/subtitles support.\n{completed.stderr.strip()}"
        )


def render_video(
    output_path: Path,
    *,
    burned_srt: Path,
    segments: Sequence[StressSegment],
    duration: float,
    video_settings: dict[str, Any],
    font_name: str | None = None,
    font_path: Path | None = None,
    fonts_dir: Path | None = None,
    renderer: str = "auto",
) -> None:
    if renderer in {"auto", "ffmpeg-subtitles"} and _ffmpeg_has_filter("subtitles"):
        render_video_with_burned_subtitles(
            output_path,
            burned_srt=burned_srt,
            duration=duration,
            video_settings=video_settings,
            font_name=font_name,
            fonts_dir=fonts_dir,
        )
        return
    if renderer == "ffmpeg-subtitles":
        raise SystemExit(
            "This ffmpeg build does not provide the subtitles filter. Install an "
            "ffmpeg build with libass support, or use --renderer pillow."
        )
    try:
        render_video_with_pillow_frames(
            output_path,
            segments=segments,
            video_settings=video_settings,
            font_path=font_path,
        )
    except ImportError as exc:
        raise SystemExit(
            "Could not render video: ffmpeg has no subtitles filter and Pillow is "
            "not installed. Install one of: ffmpeg with libass, or python -m pip "
            "install Pillow. Use --no-video to write only JSON/SRT files."
        ) from exc


def render_video_with_pillow_frames(
    output_path: Path,
    *,
    segments: Sequence[StressSegment],
    video_settings: dict[str, Any],
    font_path: Path | None = None,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = int(video_settings.get("width", 1280))
    height = int(video_settings.get("height", 720))
    fps = int(video_settings.get("fps", 25))
    background = str(video_settings.get("background", "#111111"))
    text_color = str(video_settings.get("text_color", "#ffffff"))
    outline_color = str(video_settings.get("outline_color", "#000000"))
    outline = int(video_settings.get("outline", 3))
    margin_v = int(video_settings.get("margin_v", 48))
    audio_frequency = int(video_settings.get("audio_frequency", 440))
    font_size = int(video_settings.get("font_size", 42))
    max_chars = int(video_settings.get("max_chars_per_line", 42))

    font_cache: dict[str, Any] = {}
    with TemporaryDirectory(prefix="burnsub-stress-") as tmp:
        tmp_dir = Path(tmp)
        frame_paths = []
        for segment in segments:
            image = Image.new("RGB", (width, height), color=background)
            if segment.burn_text:
                draw = ImageDraw.Draw(image)
                font = _font_for_language(
                    segment.language,
                    font_size=font_size,
                    font_path=font_path,
                    image_font=ImageFont,
                    cache=font_cache,
                )
                lines = wrap_srt_text(segment.burn_text, max_chars=max_chars)
                _draw_bottom_text(
                    draw,
                    lines,
                    font=font,
                    width=width,
                    height=height,
                    margin_v=margin_v,
                    fill=text_color,
                    stroke_fill=outline_color,
                    stroke_width=outline,
                )
            frame_path = tmp_dir / f"frame-{segment.index:05d}.png"
            image.save(frame_path)
            frame_paths.append(frame_path)
        silent_video = tmp_dir / "video-only.mp4"
        _assemble_frame_video(
            silent_video,
            tmp_dir=tmp_dir,
            frame_paths=frame_paths,
            segments=segments,
            fps=fps,
        )
        total_duration = max(segment.end for segment in segments) + 0.25
        _mux_sine_audio(
            silent_video,
            output_path,
            frequency=audio_frequency,
            duration=total_duration,
        )


def _draw_bottom_text(
    draw: Any,
    lines: Sequence[str],
    *,
    font: Any,
    width: int,
    height: int,
    margin_v: int,
    fill: str,
    stroke_fill: str,
    stroke_width: int,
) -> None:
    line_bboxes = [
        draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width) for line in lines
    ]
    line_heights = [bbox[3] - bbox[1] for bbox in line_bboxes]
    line_spacing = max(4, int(max(line_heights or [24]) * 0.25))
    total_height = sum(line_heights) + line_spacing * max(0, len(lines) - 1)
    y = max(0, height - margin_v - total_height)
    for line, bbox, line_height in zip(lines, line_bboxes, line_heights, strict=True):
        text_width = bbox[2] - bbox[0]
        x = max(0, (width - text_width) // 2)
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_fill=stroke_fill,
            stroke_width=stroke_width,
        )
        y += line_height + line_spacing


def _font_for_language(
    language: str,
    *,
    font_size: int,
    font_path: Path | None,
    image_font: Any,
    cache: dict[str, Any],
) -> Any:
    cache_key = f"{language}:{font_path or ''}:{font_size}"
    if cache_key in cache:
        return cache[cache_key]
    candidates = []
    if font_path is not None:
        candidates.append(font_path)
    candidates.extend(Path(path) for path in FONT_PATHS_BY_LANGUAGE.get(language, ()))
    candidates.extend(Path(path) for path in GENERAL_FONT_PATHS)
    for candidate in candidates:
        if candidate.is_file():
            try:
                font = image_font.truetype(str(candidate), font_size)
            except OSError:
                continue
            cache[cache_key] = font
            return font
    font = image_font.load_default()
    cache[cache_key] = font
    return font


def _assemble_frame_video(
    output_path: Path,
    *,
    tmp_dir: Path,
    frame_paths: Sequence[Path],
    segments: Sequence[StressSegment],
    fps: int,
) -> None:
    concat_list = tmp_dir / "frames.concat.txt"
    with concat_list.open("w", encoding="utf-8") as handle:
        for frame_path, segment in zip(frame_paths, segments, strict=True):
            handle.write(f"file '{frame_path.as_posix()}'\n")
            handle.write(f"duration {max(segment.duration, 0.05):.3f}\n")
        handle.write(f"file '{frame_paths[-1].as_posix()}'\n")
    completed = subprocess.run(
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
            str(output_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"ffmpeg failed to assemble stress frames:\n{completed.stderr.strip()}")


def _mux_sine_audio(
    video_path: Path,
    output_path: Path,
    *,
    frequency: int,
    duration: float,
) -> None:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:sample_rate=16000:duration={duration:.3f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"ffmpeg failed to mux stress audio:\n{completed.stderr.strip()}")


def _ffmpeg_has_filter(name: str) -> bool:
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0 and any(
        line.split()[1:2] == [name] for line in completed.stdout.splitlines()
    )


def subtitles_filter(
    burned_srt: Path,
    *,
    force_style: str,
    fonts_dir: Path | None = None,
) -> str:
    pieces = [f"filename={_escape_filter_value(str(burned_srt))}"]
    if fonts_dir is not None:
        pieces.append(f"fontsdir={_escape_filter_value(str(fonts_dir))}")
    pieces.append(f"force_style={_escape_filter_value(force_style)}")
    return "subtitles=" + ":".join(pieces)


def ass_force_style(video_settings: dict[str, Any], *, font_name: str | None = None) -> str:
    selected_font = font_name or str(video_settings.get("font_name", "Noto Sans"))
    font_size = int(video_settings.get("font_size", 42))
    margin_v = int(video_settings.get("margin_v", 48))
    outline = int(video_settings.get("outline", 3))
    shadow = int(video_settings.get("shadow", 0))
    primary = _css_to_ass_colour(str(video_settings.get("text_color", "#ffffff")))
    outline_colour = _css_to_ass_colour(str(video_settings.get("outline_color", "#000000")))
    return ",".join(
        [
            f"FontName={selected_font}",
            f"FontSize={font_size}",
            f"PrimaryColour={primary}",
            f"OutlineColour={outline_colour}",
            "BorderStyle=1",
            f"Outline={outline}",
            f"Shadow={shadow}",
            "Alignment=2",
            f"MarginV={margin_v}",
        ]
    )


def build_manifest(
    *,
    spec: dict[str, Any],
    output_dir: Path,
    segments: Sequence[StressSegment],
    video_path: Path | None,
    ocr_languages: str,
    video_rendered: bool,
) -> dict[str, Any]:
    duration = max(segment.end for segment in segments) if segments else 0.0
    report_dir = output_dir / "report"
    command = [
        "PYTHONPATH=src",
        "python3 -m burnin_subtitle_checker.cli --quiet check",
        str(video_path or output_dir / "video.mp4"),
        "--transcript-json",
        str(output_dir / "transcript.json"),
        "--reference-srt",
        str(output_dir / "reference.srt"),
        "--output-dir",
        str(report_dir),
        "--ocr-languages",
        ocr_languages,
        "--crop-bottom-percent",
        str(spec.get("video", {}).get("recommended_crop_bottom_percent", 32)),
        "--frame-offsets",
        "0",
        "--threshold",
        str(spec.get("comparison", {}).get("threshold", 0.82)),
        "--wer-threshold",
        str(spec.get("comparison", {}).get("wer_threshold", 0.25)),
        "--formats",
        "html,json,csv",
        "--save-artifacts",
    ]
    return {
        "description": spec.get("description", ""),
        "video_rendered": video_rendered,
        "duration_seconds": round(duration, 3),
        "segment_count": len(segments),
        "languages": sorted({segment.language for segment in segments}),
        "ocr_languages": ocr_languages,
        "expected_status_counts": _status_counts(segments),
        "recommended_command": " ".join(command),
        "notes": [
            "Edit burned_subtitles.srt or fixtures/stress/spec.json, then rerun this script.",
            "Install matching Tesseract traineddata packs before running OCR over every language.",
            "Use --target-duration 3600 to create an hour-long stress video.",
        ],
    }


def _status_counts(segments: Sequence[StressSegment]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for segment in segments:
        counts[segment.expected_status] = counts.get(segment.expected_status, 0) + 1
    return counts


def _parse_language_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    languages = {
        item.strip().lower()
        for chunk in value.split(",")
        for item in chunk.split("+")
        if item.strip()
    }
    return languages or None


def _css_to_ass_colour(value: str) -> str:
    raw = value.strip().lstrip("#")
    if len(raw) != 6:
        raise ValueError(f"Expected #RRGGBB colour, got {value!r}")
    red = int(raw[0:2], 16)
    green = int(raw[2:4], 16)
    blue = int(raw[4:6], 16)
    return f"&H00{blue:02X}{green:02X}{red:02X}"


def _escape_filter_value(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
    )


if __name__ == "__main__":
    sys.exit(main())
