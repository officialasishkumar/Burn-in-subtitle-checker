"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .asr import transcribe_video
from .compare import compare_segments, count_review_rows
from .dependencies import collect_doctor_results
from .exceptions import BurnSubError, ConfigError
from .io import load_ocr, load_transcript, ocr_payload, transcript_payload, write_json
from .media import ensure_audio_stream, parse_crop_box, validate_video_path
from .ocr import ocr_video_segments, parse_frame_offsets
from .report import write_reports


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BurnSubError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="burnsub",
        description="Flag mismatches between audio dialogue and burned-in subtitles.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check native tools, OCR packs, and ASR backend.")
    doctor.add_argument("--ocr-languages", default="hin+kan+eng", help="Tesseract language spec.")
    doctor.add_argument(
        "--ocr-preprocess",
        default="none",
        choices=["none", "grayscale", "threshold"],
        help="Optional OpenCV OCR crop preprocessing mode to validate.",
    )
    doctor.add_argument(
        "--asr-backend",
        default="whisper",
        choices=["whisper", "faster-whisper", "none"],
        help="ASR backend to validate.",
    )
    doctor.add_argument("--video", type=Path, help="Optional input video to probe.")
    doctor.set_defaults(func=cmd_doctor)

    check = subparsers.add_parser("check", help="Run the full mismatch-checking pipeline.")
    _add_common_pipeline_args(check)
    check.add_argument("video", type=Path, help="Input video file.")
    check.add_argument("--transcript-json", type=Path, help="Use precomputed transcript JSON.")
    check.add_argument("--ocr-json", type=Path, help="Use precomputed OCR JSON.")
    check.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Return exit code 1 when any row is not OK.",
    )
    check.set_defaults(func=cmd_check)

    transcribe = subparsers.add_parser(
        "transcribe",
        help="Transcribe video audio to transcript JSON.",
    )
    transcribe.add_argument("video", type=Path, help="Input video file.")
    transcribe.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Transcript JSON path.",
    )
    _add_asr_args(transcribe)
    transcribe.set_defaults(func=cmd_transcribe)

    ocr = subparsers.add_parser("ocr", help="OCR subtitle crops for transcript segment timestamps.")
    ocr.add_argument("video", type=Path, help="Input video file.")
    ocr.add_argument("segments_json", type=Path, help="Transcript JSON path.")
    ocr.add_argument("--output", "-o", type=Path, required=True, help="OCR JSON path.")
    _add_ocr_args(ocr)
    ocr.set_defaults(func=cmd_ocr)

    compare = subparsers.add_parser("compare", help="Compare transcript JSON and OCR JSON.")
    compare.add_argument("transcript_json", type=Path, help="Transcript JSON path.")
    compare.add_argument("ocr_json", type=Path, help="OCR JSON path.")
    compare.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Report output directory.",
    )
    compare.add_argument("--threshold", type=float, default=0.75, help="Mismatch threshold.")
    compare.add_argument(
        "--formats",
        default="html,json,csv",
        help="Comma-separated report formats.",
    )
    compare.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Return exit code 1 when any row is not OK.",
    )
    compare.set_defaults(func=cmd_compare)
    return parser


def _add_common_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Report output directory.",
    )
    parser.add_argument("--threshold", type=float, default=0.75, help="Mismatch threshold.")
    parser.add_argument(
        "--formats",
        default="html,json,csv",
        help="Comma-separated report formats.",
    )
    _add_asr_args(parser)
    _add_ocr_args(parser)


def _add_asr_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--asr-backend",
        default="whisper",
        choices=["whisper", "faster-whisper"],
        help="ASR backend.",
    )
    parser.add_argument("--asr-model", default="base", help="Whisper model name.")
    parser.add_argument("--asr-language", default="auto", help="ASR language code, or auto.")
    parser.add_argument("--device", default="auto", help="ASR device: auto, cpu, cuda, etc.")


def _add_ocr_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ocr-languages", default="hin+kan+eng", help="Tesseract languages.")
    parser.add_argument(
        "--crop-bottom-percent",
        type=float,
        default=15.0,
        help="Bottom-frame percentage to OCR when --crop-box is not provided.",
    )
    parser.add_argument("--crop-box", help="Explicit OCR crop box as x,y,w,h pixels.")
    parser.add_argument(
        "--frame-offsets",
        default="0",
        help="Comma-separated offsets around segment midpoint, in seconds.",
    )
    parser.add_argument(
        "--tesseract-psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode.",
    )
    parser.add_argument(
        "--ocr-preprocess",
        default="none",
        choices=["none", "grayscale", "threshold"],
        help="Optional OpenCV preprocessing for OCR crops.",
    )
    parser.add_argument(
        "--ocr-upscale-factor",
        type=float,
        default=2.0,
        help="Scale factor used before OCR when --ocr-preprocess is enabled.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Keep OCR crops and link them from the HTML report.",
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    results = collect_doctor_results(
        ocr_languages=args.ocr_languages,
        ocr_preprocess=args.ocr_preprocess,
        asr_backend=args.asr_backend,
        video_path=args.video,
    )
    width = max(len(result.name) for result in results)
    for result in results:
        mark = "OK" if result.ok else "MISSING"
        print(f"{result.name:<{width}}  {mark:<7}  {result.detail}")
    return 0 if all(result.ok for result in results) else 3


def cmd_check(args: argparse.Namespace) -> int:
    _validate_threshold(args.threshold)
    validate_video_path(args.video)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.transcript_json:
        transcript_segments = load_transcript(args.transcript_json)
    else:
        ensure_audio_stream(args.video)
        transcript_segments = transcribe_video(
            args.video,
            backend=args.asr_backend,
            model_name=args.asr_model,
            language=args.asr_language,
            device=args.device,
        )
        write_json(
            output_dir / "transcript.json",
            transcript_payload(transcript_segments, str(args.video)),
        )

    if args.ocr_json:
        ocr_segments = load_ocr(args.ocr_json)
    else:
        ocr_segments = _run_ocr(args, transcript_segments, output_dir)
        write_json(output_dir / "ocr.json", ocr_payload(ocr_segments, str(args.video)))

    rows = compare_segments(transcript_segments, ocr_segments, threshold=args.threshold)
    paths = _write_cli_reports(args, rows, source_video=str(args.video))
    _print_report_summary(rows, paths)
    if args.fail_on_mismatch and count_review_rows(rows) > 0:
        return 1
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    validate_video_path(args.video)
    ensure_audio_stream(args.video)
    segments = transcribe_video(
        args.video,
        backend=args.asr_backend,
        model_name=args.asr_model,
        language=args.asr_language,
        device=args.device,
    )
    write_json(args.output, transcript_payload(segments, str(args.video)))
    print(f"Wrote transcript JSON: {args.output}")
    return 0


def cmd_ocr(args: argparse.Namespace) -> int:
    validate_video_path(args.video)
    transcript_segments = load_transcript(args.segments_json)
    output_dir = args.output.parent
    ocr_segments = _run_ocr(args, transcript_segments, output_dir)
    write_json(args.output, ocr_payload(ocr_segments, str(args.video)))
    print(f"Wrote OCR JSON: {args.output}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    _validate_threshold(args.threshold)
    transcript_segments = load_transcript(args.transcript_json)
    ocr_segments = load_ocr(args.ocr_json)
    rows = compare_segments(transcript_segments, ocr_segments, threshold=args.threshold)
    paths = write_reports(
        rows,
        output_dir=args.output_dir,
        source_video=None,
        threshold=args.threshold,
        formats=_formats(args.formats),
        config={"mode": "compare"},
    )
    _print_report_summary(rows, paths)
    if args.fail_on_mismatch and count_review_rows(rows) > 0:
        return 1
    return 0


def _run_ocr(args: argparse.Namespace, transcript_segments: list, output_dir: Path):
    return ocr_video_segments(
        args.video,
        transcript_segments,
        output_dir=output_dir,
        languages=args.ocr_languages,
        crop_bottom_percent=args.crop_bottom_percent,
        crop_box=parse_crop_box(args.crop_box),
        frame_offsets=parse_frame_offsets(args.frame_offsets),
        psm=args.tesseract_psm,
        preprocess=args.ocr_preprocess,
        upscale_factor=args.ocr_upscale_factor,
        save_artifacts=args.save_artifacts,
    )


def _write_cli_reports(
    args: argparse.Namespace,
    rows: list,
    *,
    source_video: str | None,
) -> dict[str, Path]:
    config: dict[str, Any] = {
        "ocr_languages": getattr(args, "ocr_languages", None),
        "crop_bottom_percent": getattr(args, "crop_bottom_percent", None),
        "crop_box": getattr(args, "crop_box", None),
        "frame_offsets": getattr(args, "frame_offsets", None),
        "ocr_preprocess": getattr(args, "ocr_preprocess", None),
        "ocr_upscale_factor": getattr(args, "ocr_upscale_factor", None),
        "asr_backend": getattr(args, "asr_backend", None),
        "asr_model": getattr(args, "asr_model", None),
        "asr_language": getattr(args, "asr_language", None),
    }
    return write_reports(
        rows,
        output_dir=args.output_dir,
        source_video=source_video,
        threshold=args.threshold,
        formats=_formats(args.formats),
        config=config,
    )


def _formats(value: str) -> list[str]:
    formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(formats) - {"html", "json", "csv"})
    if invalid:
        raise ConfigError(f"Unsupported report format(s): {', '.join(invalid)}")
    return formats


def _validate_threshold(value: float) -> None:
    if not 0 <= value <= 1:
        raise ConfigError("--threshold must be between 0 and 1")


def _print_report_summary(rows: list, paths: dict[str, Path]) -> None:
    review_count = count_review_rows(rows)
    print(f"Compared {len(rows)} segment(s); {review_count} need review.")
    for fmt, path in sorted(paths.items()):
        print(f"Wrote {fmt.upper()} report: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
