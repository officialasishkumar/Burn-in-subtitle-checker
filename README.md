# Burn-in Subtitle Checker

Lightweight Python CLI for flagging mismatches between spoken dialogue and burned-in subtitles in video files. It can run a full pipeline with Whisper ASR, Tesseract OCR, timestamp-aware comparison, and HTML/JSON/CSV reports.

This project was built for the PlanetRead issue: **[DMP 2026]: Create Lightweight Audio-Subtitle Mismatch Flagging Tool**.

## What It Does

- Transcribes a video audio track into timed text segments with Whisper or faster-whisper.
- Samples video frames at each segment midpoint, with optional nearby timestamp offsets.
- Crops the subtitle region, defaulting to the bottom 15% of the frame.
- Runs Tesseract OCR with Hindi, Kannada, and English language packs by default.
- Normalizes Indic text safely before comparison.
- Scores audio text vs subtitle text and flags low-confidence rows for review.
- Generates deterministic HTML, JSON, and CSV reports.
- Provides split-stage commands so expensive ASR/OCR work can be debugged and reused.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[similarity]"
```

For the full video/OCR/ASR path, install native tools and ASR extras:

```bash
sudo apt update
sudo apt install -y ffmpeg tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-kan
python -m pip install -e ".[asr,similarity]"
```

If you prefer faster-whisper:

```bash
python -m pip install -e ".[asr-fast,similarity]"
```

Optional OCR preprocessing uses OpenCV to upscale, grayscale, and threshold subtitle crops before Tesseract:

```bash
python -m pip install -e ".[ocr-preprocess]"
```

## Quick Start

Check the local machine first:

```bash
burnsub doctor --ocr-languages hin+kan+eng --asr-backend whisper
```

Run the full pipeline:

```bash
burnsub check input.mp4 \
  --output-dir reports/input-check \
  --ocr-languages hin+kan+eng \
  --asr-backend whisper \
  --asr-model base \
  --asr-language auto \
  --threshold 0.75 \
  --formats html,json,csv
```

Open `reports/input-check/report.html` and inspect rows marked `REVIEW`.

## Split-Stage Workflow

Use split commands when ASR/OCR is slow or you need to inspect intermediate data.

```bash
burnsub transcribe input.mp4 --output transcript.json --asr-model base
burnsub ocr input.mp4 transcript.json --output ocr.json --ocr-languages hin+kan+eng
burnsub compare transcript.json ocr.json --output-dir reports/input-check
```

You can also run `check` with precomputed data:

```bash
burnsub check input.mp4 \
  --transcript-json transcript.json \
  --ocr-json ocr.json \
  --output-dir reports/input-check
```

## Useful Options

- `--crop-bottom-percent 15`: OCR the bottom 15% of the frame.
- `--crop-box x,y,w,h`: use an explicit pixel crop box instead.
- `--frame-offsets 0,-0.25,0.25`: OCR multiple nearby frames and keep the strongest text.
- `--ocr-preprocess threshold`: use OpenCV preprocessing before Tesseract.
- `--ocr-upscale-factor 2`: upscale OCR crops before preprocessing.
- `--threshold 0.75`: rows below this score are marked `REVIEW`.
- `--save-artifacts`: preserve OCR crops for report links and debugging.
- `--fail-on-mismatch`: return exit code `1` when review rows are found.
- `--formats html,json,csv`: choose one or more report formats.

## Exit Codes

- `0`: command completed.
- `1`: mismatches found and `--fail-on-mismatch` was used.
- `2`: invalid input or configuration.
- `3`: missing dependency, ASR backend, or OCR language pack.
- `4`: processing failed.

## Hindi and Kannada Support

Tesseract must have the corresponding traineddata installed:

- Hindi: `hin`
- Kannada: `kan`
- English fallback: `eng`

The default OCR language string is `hin+kan+eng`. Use `burnsub doctor` to verify language packs before processing real videos.

## Reliability Notes

Burned-in subtitle OCR is sensitive to font, contrast, resolution, compression, placement, and timing. If the report shows empty or poor OCR, use `--save-artifacts` and inspect the crop images. If subtitles are not in the bottom band, use `--crop-box` or adjust `--crop-bottom-percent`.

For long videos, prefer the split-stage workflow so transcript and OCR JSON can be reused without rerunning every stage.

The comparison stage indexes OCR rows by segment index and timestamp, so large transcript/OCR tables do not require a full OCR scan for every audio segment. When multiple OCR frame offsets are sampled, the chosen OCR text is ranked against the audio segment text instead of only choosing the longest OCR result.

If Tesseract struggles on compressed or low-contrast subtitles, install `.[ocr-preprocess]` and try `--ocr-preprocess threshold --ocr-upscale-factor 2`. Reports include optional jiwer-backed word error rate and character error rate fields when `.[similarity]` is installed, which gives reviewers a second signal beyond the fuzzy match score.

## Development Checks

Run the fast checks before opening a PR:

```bash
ruff check .
python3 -m pytest
```

Run fixture and native smoke checks when touching CLI, report, ffmpeg, or OCR behavior:

```bash
scripts/run_fixture_e2e.sh /tmp/burnsub-fixture-e2e
scripts/run_native_smoke.sh /tmp/burnsub-native-smoke
```
