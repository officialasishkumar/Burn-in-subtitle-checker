#!/usr/bin/env python3
"""Build a real-speech burn-in subtitle video for end-to-end regression.

For each segment in ``fixtures/realspeech/spec.json`` this script:

  1. Synthesises the ``audio_text`` to a wav clip via the system TTS
     (``espeak-ng`` on Linux, ``say`` on macOS).
  2. Renders the (intentionally different) ``burn_text`` into a PNG frame
     using Pillow.
  3. Stitches per-segment audio and frames into a single MP4 with
     deterministic timing using ffmpeg.

The output bundle contains:

  - ``video.mp4``        the final video with TTS audio + burned-in text
  - ``timeline.json``    per-segment start/end timestamps and metadata
  - ``audio_only.wav``   the concatenated TTS audio (kept for debugging)

Unlike :mod:`scripts.build_regression_fixture` (which uses a sine-wave
audio track and a precomputed transcript), this script feeds Whisper
real speech so the ASR side of the pipeline gets exercised too.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - fixture script only
    raise SystemExit("This script requires Pillow. Install with: pip install Pillow") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO_ROOT / "fixtures" / "realspeech" / "spec.json"

CANDIDATE_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)

SAMPLE_RATE = 16_000


@dataclass
class _SegmentClip:
    audio_path: Path
    duration: float
    frame_path: Path
    start: float
    end: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--font", type=Path, default=None)
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    video_settings = spec["video"]
    tts_settings = spec.get("tts", {})

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to assemble the regression video.")
    tts = _detect_tts(tts_settings)

    font_path = args.font or _locate_font()
    if font_path is None:
        raise SystemExit(
            "Could not find a usable TrueType font. Pass --font /path/to/font.ttf."
        )
    font = _load_font(font_path, video_settings["font_size"])

    with TemporaryDirectory(prefix="burnsub-realspeech-") as tmp:
        tmp_dir = Path(tmp)
        clips = _build_segment_clips(
            spec["segments"],
            tmp_dir,
            font=font,
            video_settings=video_settings,
            tts=tts,
            gap_seconds=float(video_settings.get("gap_between_segments_seconds", 0.0)),
            trailing_silence=float(video_settings.get("trailing_silence_seconds", 0.0)),
        )
        audio_path = output_dir / "audio_only.wav"
        _concatenate_audio(clips, audio_path)
        video_path = output_dir / "video.mp4"
        _assemble_video(
            video_path,
            tmp_dir=tmp_dir,
            clips=clips,
            audio_path=audio_path,
            fps=int(video_settings["frame_rate"]),
        )

    timeline_path = output_dir / "timeline.json"
    _write_timeline(spec, clips, timeline_path)
    print(f"Wrote {video_path}")
    print(f"Wrote {audio_path}")
    print(f"Wrote {timeline_path}")
    return 0


def _detect_tts(tts_settings: dict[str, Any]) -> dict[str, Any]:
    if shutil.which("espeak-ng"):
        return {
            "kind": "espeak",
            "voice": tts_settings.get("language_espeak", "en"),
            "rate": int(tts_settings.get("rate_words_per_minute", 175)),
        }
    if shutil.which("say"):
        return {
            "kind": "say",
            "voice": tts_settings.get("voice_macos", "Alex"),
            "rate": int(tts_settings.get("rate_words_per_minute", 175)),
        }
    raise SystemExit(
        "Need espeak-ng (Linux) or 'say' (macOS) to synthesize speech."
    )


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


def _build_segment_clips(
    segments: list[dict[str, Any]],
    tmp_dir: Path,
    *,
    font: Any,
    video_settings: dict[str, Any],
    tts: dict[str, Any],
    gap_seconds: float,
    trailing_silence: float,
) -> list[_SegmentClip]:
    clips: list[_SegmentClip] = []
    cursor = 0.0
    for index, segment in enumerate(segments):
        speech_text: str = segment["audio_text"]
        speech_path = tmp_dir / f"speech-{index:03d}.wav"
        _synthesize_speech(speech_text, speech_path, tts=tts)
        speech_duration = _probe_duration(speech_path)
        gap_after = gap_seconds if index < len(segments) - 1 else trailing_silence
        padded_path = tmp_dir / f"padded-{index:03d}.wav"
        _pad_audio(
            speech_path,
            padded_path,
            speech_duration=speech_duration,
            silence_after=gap_after,
        )
        total_duration = _probe_duration(padded_path)

        burn_text: str = segment.get("burn_text", "")
        frame_path = tmp_dir / f"frame-{index:03d}.png"
        _render_frame(
            burn_text,
            frame_path,
            font=font,
            width=int(video_settings["width"]),
            height=int(video_settings["height"]),
            background=video_settings["background"],
            text_color=video_settings["text_color"],
        )

        clips.append(
            _SegmentClip(
                audio_path=padded_path,
                duration=total_duration,
                frame_path=frame_path,
                start=cursor,
                end=cursor + total_duration,
            )
        )
        cursor += total_duration
    return clips


def _synthesize_speech(text: str, output_wav: Path, *, tts: dict[str, Any]) -> None:
    if tts["kind"] == "espeak":
        subprocess.run(
            [
                "espeak-ng",
                "-v",
                str(tts["voice"]),
                "-s",
                str(tts["rate"]),
                "-w",
                str(output_wav),
                text,
            ],
            check=True,
            capture_output=True,
        )
    elif tts["kind"] == "say":
        aiff_path = output_wav.with_suffix(".aiff")
        subprocess.run(
            [
                "say",
                "-v",
                str(tts["voice"]),
                "-r",
                str(tts["rate"]),
                "-o",
                str(aiff_path),
                text,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(aiff_path),
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                str(output_wav),
            ],
            check=True,
            capture_output=True,
        )
        aiff_path.unlink(missing_ok=True)
    else:  # pragma: no cover - guarded above
        raise SystemExit(f"Unknown TTS backend: {tts['kind']}")


def _pad_audio(
    input_wav: Path,
    output_wav: Path,
    *,
    speech_duration: float,
    silence_after: float,
) -> None:
    if silence_after <= 0:
        # Standardise sample rate and channel count.
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_wav),
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                str(output_wav),
            ],
            check=True,
            capture_output=True,
        )
        return
    target_duration = speech_duration + silence_after
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_wav),
            "-af",
            f"apad=whole_dur={target_duration:.3f}",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(output_wav),
        ],
        check=True,
        capture_output=True,
    )


def _probe_duration(audio_path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def _render_frame(
    burn_text: str,
    frame_path: Path,
    *,
    font: Any,
    width: int,
    height: int,
    background: str,
    text_color: str,
) -> None:
    image = Image.new("RGB", (width, height), color=background)
    if burn_text:
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), burn_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (width - text_w) // 2
        y = height - text_h - 80
        draw.text((x, y), burn_text, fill=text_color, font=font)
    image.save(frame_path)


def _concatenate_audio(clips: list[_SegmentClip], output_wav: Path) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_wav.with_suffix(".concat.txt")
    with list_path.open("w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(f"file '{clip.audio_path.as_posix()}'\n")
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
            str(list_path),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(output_wav),
        ],
        check=True,
    )
    list_path.unlink(missing_ok=True)


def _assemble_video(
    output_path: Path,
    *,
    tmp_dir: Path,
    clips: list[_SegmentClip],
    audio_path: Path,
    fps: int,
) -> None:
    concat_list = tmp_dir / "frames.concat.txt"
    with concat_list.open("w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(f"file '{clip.frame_path.as_posix()}'\n")
            handle.write(f"duration {max(clip.duration, 0.05):.3f}\n")
        handle.write(f"file '{clips[-1].frame_path.as_posix()}'\n")
    silent_video = tmp_dir / "silent.mp4"
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
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )


def _write_timeline(
    spec: dict[str, Any],
    clips: list[_SegmentClip],
    timeline_path: Path,
) -> None:
    payload = {
        "tts_backend_hint": spec.get("tts", {}),
        "segments": [
            {
                "index": index,
                "audio_text": segment["audio_text"],
                "burn_text": segment.get("burn_text", ""),
                "kind": segment.get("kind", "match"),
                "start": clip.start,
                "end": clip.end,
                "speech_duration": clip.duration,
            }
            for index, (segment, clip) in enumerate(zip(spec["segments"], clips, strict=True))
        ],
    }
    timeline_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
