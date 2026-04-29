#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-"$ROOT_DIR/reports/fixture-e2e"}"
PYTHON_BIN="${PYTHON:-python3}"

rm -rf "$OUT_DIR"
PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m burnin_subtitle_checker.cli compare \
  "$ROOT_DIR/fixtures/e2e/transcript.json" \
  "$ROOT_DIR/fixtures/e2e/ocr.json" \
  --output-dir "$OUT_DIR" \
  --threshold 0.75 \
  --formats html,json,csv

test -s "$OUT_DIR/report.html"
test -s "$OUT_DIR/report.json"
test -s "$OUT_DIR/report.csv"

echo "Fixture end-to-end reports written to $OUT_DIR"
