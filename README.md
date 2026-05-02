# Burn-in Subtitle Checker

Lightweight Python CLI for flagging mismatches between spoken dialogue and burned-in subtitles in video files. It can run a full pipeline with Whisper ASR, Tesseract OCR, timestamp-aware comparison, and HTML/JSON/CSV reports.

This project was built for the PlanetRead issue: **[DMP 2026]: Create Lightweight Audio-Subtitle Mismatch Flagging Tool**.

## What It Does

- Transcribes a video audio track into timed text segments with Whisper or faster-whisper, with optional VAD and hallucination filtering.
- Samples video frames at each segment midpoint, with optional nearby timestamp offsets.
- Crops the subtitle region, defaulting to the bottom 15% of the frame, with optional adaptive band detection (`--crop-mode auto`).
- Runs Tesseract OCR (default) or EasyOCR with Hindi, Kannada, and English language packs by default.
- Normalizes Indic text safely before comparison and ranks audio vs subtitle text with character, token, partial, and jiwer-backed word/character error rates.
- Optionally compares both audio and OCR against a reference SRT for a 3-way mismatch view.
- Runs OCR work in parallel across worker threads and persists progress to a checkpoint so long jobs can resume after a crash.
- Generates deterministic HTML, JSON, and CSV reports. The HTML report supports search, status filters, and sortable columns.
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

For an alternative deep-learning OCR backend (helpful when Tesseract struggles on stylised fonts), install EasyOCR:

```bash
python -m pip install -e ".[ocr-easy]"
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

- `--workers N`: run OCR on N worker threads (default: `min(8, CPU count)`). Frame extraction and Tesseract calls run concurrently.
- `--resume`: pick up where a previous run left off using `<output-dir>/ocr.partial.jsonl`.
- `--crop-bottom-percent 15`: OCR the bottom 15% of the frame.
- `--crop-mode auto`: detect the burned-in subtitle band automatically (requires `.[ocr-preprocess]`).
- `--crop-box x,y,w,h`: use an explicit pixel crop box instead.
- `--frame-offsets 0,-0.25,0.25`: OCR multiple nearby frames and keep the strongest text.
- `--ocr-engine easyocr`: use EasyOCR instead of Tesseract.
- `--ocr-preprocess threshold`: use OpenCV preprocessing before OCR.
- `--ocr-upscale-factor 2`: upscale OCR crops before preprocessing.
- `--asr-vad`: enable faster-whisper VAD to skip silent stretches.
- `--asr-no-speech-threshold 0.6`: drop ASR segments whose no-speech probability exceeds this value.
- `--asr-keep-hallucinations`: opt out of dropping known Whisper hallucination phrases.
- `--asr-initial-prompt "..."`: bias ASR with an initial prompt.
- `--reference-srt path.srt`: also compare each audio/OCR pair to a reference SRT.
- `--threshold 0.75`: rows below this fuzzy score are marked `REVIEW`.
- `--wer-threshold 0.3`: extra check; rows above this WER are flagged even when the score passes.
- `--save-artifacts`: preserve OCR crops for report links and debugging.
- `--fail-on-mismatch`: return exit code `1` when review rows are found.
- `--formats html,json,csv`: choose one or more report formats.
- `--quiet`: silence the stderr progress output.

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

## Reliability and Throughput Notes

Burned-in subtitle OCR is sensitive to font, contrast, resolution, compression, placement, and timing. If the report shows empty or poor OCR, use `--save-artifacts` and inspect the crop images. If subtitles are not in the bottom band, use `--crop-box`, adjust `--crop-bottom-percent`, or try `--crop-mode auto`.

For long videos, prefer the split-stage workflow so transcript and OCR JSON can be reused without rerunning every stage. Within a single OCR run, the pipeline parallelises ffmpeg frame extraction and Tesseract calls across `--workers` threads (defaulting to `min(8, CPU count)`); a per-segment JSONL checkpoint at `<output-dir>/ocr.partial.jsonl` is written after each segment so `--resume` can pick up after a crash.

The comparison stage indexes OCR rows by segment index and timestamp, so large transcript/OCR tables do not require a full OCR scan for every audio segment. When multiple OCR frame offsets are sampled, the chosen OCR text is ranked against the audio segment text instead of only choosing the longest OCR result.

If Tesseract struggles on compressed or low-contrast subtitles, install `.[ocr-preprocess]` and try `--ocr-preprocess threshold --ocr-upscale-factor 2`, or fall back to `--ocr-engine easyocr` (deep learning) after installing `.[ocr-easy]`. Reports include optional jiwer-backed word error rate and character error rate fields when `.[similarity]` is installed, plus a composite score that blends character, token, and partial similarity. Faster-whisper users can pass `--asr-vad` to skip silent stretches, and the pipeline drops common Whisper hallucination phrases by default.

When a reference SRT is available, pass `--reference-srt subs.srt` to add 3-way comparison columns (`reference_vs_audio_score`, `reference_vs_subtitle_score`) so reviewers can tell whether a mismatch comes from the audio leg, the subtitle leg, or both.

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

End-to-end regression test (renders a real video with deliberate burned-in mismatches via Pillow + ffmpeg, runs the full pipeline, and asserts each segment's status). Requires Pillow, ffmpeg, and Tesseract with the `eng` pack:

```bash
pytest -q tests/test_regression_pipeline.py
```

The fixture spec lives in `fixtures/regression/spec.json`; you can edit it to add new mismatch scenarios. To inspect the bundle that powers the test:

```bash
python scripts/build_regression_fixture.py /tmp/burnsub-regression
burnsub check /tmp/burnsub-regression/video.mp4 \
  --transcript-json /tmp/burnsub-regression/transcript.json \
  --reference-srt /tmp/burnsub-regression/reference.srt \
  --output-dir /tmp/burnsub-regression/report \
  --ocr-languages eng --workers 4 \
  --threshold 0.75 --wer-threshold 0.2
open /tmp/burnsub-regression/report/report.html
```
