#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-"$ROOT_DIR/reports/stress-fixture"}"
PYTHON_BIN="${PYTHON:-python3}"
DURATION="${BURNSUB_STRESS_DURATION:-180}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/build_stress_fixture.py" \
  "$OUT_DIR/bundle" \
  --target-duration "$DURATION"

OCR_LANGS="$(cat "$OUT_DIR/bundle/ocr_languages.txt")"

PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m burnin_subtitle_checker.cli --quiet doctor \
  --asr-backend none \
  --ocr-engine tesseract \
  --ocr-languages "$OCR_LANGS"

PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m burnin_subtitle_checker.cli --quiet check \
  "$OUT_DIR/bundle/video.mp4" \
  --transcript-json "$OUT_DIR/bundle/transcript.json" \
  --reference-srt "$OUT_DIR/bundle/reference.srt" \
  --output-dir "$OUT_DIR/report" \
  --ocr-languages "$OCR_LANGS" \
  --crop-bottom-percent 34 \
  --frame-offsets 0 \
  --threshold 0.82 \
  --wer-threshold 0.25 \
  --formats html,json,csv \
  --save-artifacts

echo "Stress fixture report written to $OUT_DIR/report"
