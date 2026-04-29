#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
WORK_DIR="${1:-"/tmp/burnsub-native-smoke"}"
VIDEO_PATH="$WORK_DIR/synthetic_burned_subs.mp4"
REPORT_DIR="$WORK_DIR/report"
FONT_PATH="${FONT_PATH:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}"
IMAGE_TOOL="${IMAGE_TOOL:-$(command -v magick || command -v convert || true)}"

if [[ ! -f "$FONT_PATH" ]] && command -v fc-match >/dev/null 2>&1; then
  FONT_PATH="$(fc-match -f '%{file}\n' 'DejaVu Sans:style=Bold' | head -n 1)"
fi

if [[ ! -f "$FONT_PATH" ]]; then
  echo "Could not find a usable font. Set FONT_PATH=/path/to/font.ttf" >&2
  exit 2
fi

if [[ -z "$IMAGE_TOOL" ]]; then
  echo "Could not find ImageMagick. Install magick/convert to build the smoke video." >&2
  exit 2
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

"$IMAGE_TOOL" -size 1280x720 xc:black \
  -font "$FONT_PATH" -fill white -gravity South -pointsize 72 \
  -annotate +0+60 "hello world" "$WORK_DIR/hello.png"
"$IMAGE_TOOL" -size 1280x720 xc:black \
  -font "$FONT_PATH" -fill white -gravity South -pointsize 72 \
  -annotate +0+60 "good night" "$WORK_DIR/good-night.png"

cp "$WORK_DIR/hello.png" "$WORK_DIR/frame000.png"
cp "$WORK_DIR/hello.png" "$WORK_DIR/frame001.png"
cp "$WORK_DIR/good-night.png" "$WORK_DIR/frame002.png"
cp "$WORK_DIR/good-night.png" "$WORK_DIR/frame003.png"

ffmpeg -hide_banner -loglevel error -y \
  -framerate 1 -i "$WORK_DIR/frame%03d.png" \
  -f lavfi -i "sine=frequency=440:duration=4" \
  -shortest \
  -r 25 \
  -c:v libx264 -pix_fmt yuv420p \
  -c:a aac \
  "$VIDEO_PATH"

PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m burnin_subtitle_checker.cli check \
  "$VIDEO_PATH" \
  --transcript-json "$ROOT_DIR/fixtures/e2e/english_transcript.json" \
  --output-dir "$REPORT_DIR" \
  --ocr-languages eng \
  --crop-bottom-percent 25 \
  --frame-offsets 0 \
  --threshold 0.75 \
  --formats html,json,csv \
  --save-artifacts

test -s "$REPORT_DIR/report.html"
test -s "$REPORT_DIR/report.json"
test -s "$REPORT_DIR/report.csv"

"$PYTHON_BIN" - "$REPORT_DIR/report.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

summary = payload["summary"]
assert summary["total"] == 2, summary
assert summary["ok"] == 1, summary
assert summary["review"] == 1, summary
PY

echo "Native ffmpeg/Tesseract smoke reports written to $REPORT_DIR"
