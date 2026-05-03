# Burn-in Subtitle Checker

Lightweight Python CLI for finding mismatches between spoken dialogue and burned-in subtitles in video files. It was built for PlanetRead's DMP 2026 issue: [Create Lightweight Audio-Subtitle Mismatch Flagging Tool](https://github.com/PlanetRead/Burn-in-subtitle-checker/issues/3).

The pipeline extracts timed speech with ASR, OCRs subtitle crops at those timestamps, compares the text, and writes reviewer-friendly HTML, JSON, and CSV reports.

## What It Does

- Transcribes video audio into timed text segments.
- Uses Indic-specialised ASR by default for Hindi and Kannada when installed.
- Captures subtitle-region frame crops at each segment midpoint, with optional nearby offsets.
- Runs Tesseract OCR with Hindi, Kannada, and English by default, preferring Indic-OCR traineddata when available.
- Falls back to stock Whisper, faster-whisper, stock Tesseract, and EasyOCR without changing the CLI shape.
- Compares normalized Indic text with fuzzy, token, WER, CER, and composite scores.
- Optionally compares audio and OCR against a reference SRT for a 3-way review.
- Saves resumable OCR checkpoints and deterministic HTML/JSON/CSV reports.

## Install

Create an environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[similarity]"
```

Install native tools:

```bash
sudo apt update
sudo apt install -y ffmpeg tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-kan
```

Recommended PlanetRead setup for Hindi/Kannada:

```bash
python -m pip install -e ".[asr-indic,asr,similarity,ocr-preprocess]"
```

Optional backend extras:

```bash
# faster Whisper fallback
python -m pip install -e ".[asr-fast]"

# AI4Bharat IndicConformer; heavier ASR backend
python -m pip install -e ".[asr-conformer]"

# EasyOCR fallback
python -m pip install -e ".[ocr-easy]"

# AI4Bharat Indic OCR model adapter
python -m pip install -e ".[ocr-indic]"

# PaddleOCR-VL for stylised subtitle fonts
python -m pip install -e ".[ocr-paddle]"
```

Indic-OCR traineddata is not auto-downloaded. Install it into the default BurnSub path, then add any non-Indic packs you need, such as `eng.traineddata`:

```bash
mkdir -p ~/.local/share/burnsub/indic-ocr
git clone https://github.com/indic-ocr/tessdata ~/.local/share/burnsub/indic-ocr/tessdata
cp "$(tesseract --list-langs | sed -n 's/^List of available languages in "\(.*\)":/\1/p')"/eng.traineddata \
  ~/.local/share/burnsub/indic-ocr/tessdata/
```

You can also point at another directory:

```bash
export BURNSUB_INDIC_TESSDATA=/path/to/indic-ocr/tessdata
```

## Quick Start

Check the machine first:

```bash
burnsub doctor --ocr-languages hin+kan+eng --asr-backend auto
```

Run the full pipeline with the new defaults:

```bash
burnsub check input.mp4 \
  --output-dir reports/input-check \
  --ocr-languages hin+kan+eng \
  --asr-language auto \
  --threshold 0.75 \
  --formats html,json,csv
```

The CLI logs the resolved ASR backend and OCR data path. Open `reports/input-check/report.html` and review rows that are not `OK`.

## Backend Selection

ASR default is `--asr-backend auto`:

- Hindi/Kannada/Indic language hints use `indicwhisper` when `.[asr-indic]` is installed.
- If IndicWhisper dependencies are missing, auto falls back to `whisper`.
- Non-Indic language hints such as `en` use `whisper`.
- User-selected backends are never rewritten.

ASR options:

```bash
burnsub check input.mp4 --output-dir reports/hi --asr-language hi
burnsub check input.mp4 --output-dir reports/whisper --asr-backend whisper --asr-model base
burnsub check input.mp4 --output-dir reports/fast --asr-backend faster-whisper --asr-vad
burnsub check input.mp4 --output-dir reports/conformer --asr-backend indic-conformer --asr-language kn
burnsub check input.mp4 --output-dir reports/conformer-rnnt --asr-backend indic-conformer --asr-conformer-decoder rnnt
```

`--asr-vad`, `--asr-no-speech-threshold`, hallucination filtering, and `--asr-initial-prompt` continue to work where the backend exposes the needed signals. IndicConformer does not expose VAD or no-speech probabilities, so those controls are ignored for that backend.

OCR default is Tesseract:

- If a complete Indic-OCR tessdata directory is detected, Tesseract uses it.
- Otherwise Tesseract uses the stock system language packs.
- `--ocr-engine` can still be set explicitly.

OCR options:

```bash
burnsub check input.mp4 --output-dir reports/tess --ocr-engine tesseract
burnsub check input.mp4 --output-dir reports/easy --ocr-engine easyocr
burnsub check input.mp4 --output-dir reports/paddle --ocr-engine paddleocr-vl
burnsub check input.mp4 --output-dir reports/ai4bharat --ocr-engine ai4bharat
```

For AI4Bharat OCR weights hosted under a different model id:

```bash
export BURNSUB_AI4BHARAT_OCR_MODEL_ID=your-org/your-indic-ocr-model
```

## Split-Stage Workflow

Use split commands when ASR/OCR is slow or you need reusable intermediate JSON:

```bash
burnsub transcribe input.mp4 --output transcript.json --asr-language hi
burnsub ocr input.mp4 transcript.json --output ocr.json --ocr-languages hin+kan+eng
burnsub compare transcript.json ocr.json --output-dir reports/input-check
```

You can also skip expensive stages in `check`:

```bash
burnsub check input.mp4 \
  --transcript-json transcript.json \
  --ocr-json ocr.json \
  --output-dir reports/input-check
```

## Useful Options

- `--workers N`: OCR worker threads, defaulting to `min(8, CPU count)`.
- `--resume`: reuse `<output-dir>/ocr.partial.jsonl`.
- `--crop-bottom-percent 15`: OCR the bottom 15% of the frame.
- `--crop-mode auto`: detect the subtitle band automatically; requires `.[ocr-preprocess]`.
- `--crop-box x,y,w,h`: use an explicit pixel crop box.
- `--frame-offsets 0,-0.25,0.25`: OCR nearby frames and keep the best text.
- `--ocr-preprocess threshold --ocr-upscale-factor 2`: improve low-contrast crops.
- `--tessdata-dir path`: override Tesseract traineddata discovery.
- `--reference-srt path.srt`: add reference-vs-audio and reference-vs-subtitle columns.
- `--threshold 0.75`: fuzzy score below this becomes `REVIEW`.
- `--wer-threshold 0.3`: flag rows with high WER even when fuzzy score passes.
- `--save-artifacts`: preserve OCR crops for report links.
- `--fail-on-mismatch`: exit `1` when review rows are found.
- `--quiet`: silence progress and backend-selection logs.

## Exit Codes

- `0`: command completed.
- `1`: mismatches found and `--fail-on-mismatch` was used.
- `2`: invalid input or configuration.
- `3`: missing dependency, ASR backend, model, or OCR language pack.
- `4`: processing failed.

## Hindi and Kannada Notes

The default OCR language string is `hin+kan+eng`. Stock Tesseract packs work, but Indic-OCR traineddata is recommended for Indic scripts. `burnsub doctor` reports whether it found stock packs or the improved Indic-OCR data and prints the install command when an upgrade is available.

IndicWhisper model ids can be overridden without changing CLI flags:

```bash
export BURNSUB_INDICWHISPER_HI_SMALL_MODEL_ID=/models/indicwhisper-hi-small
export BURNSUB_INDICWHISPER_KN_MEDIUM_MODEL_ID=/models/indicwhisper-kn-medium
export BURNSUB_INDICWHISPER_MODEL_ID=ai4bharat/indicwhisper
```

## Reliability Notes

Burned-in subtitle OCR is sensitive to font, contrast, resolution, compression, placement, and timing. If OCR is empty or noisy, rerun with `--save-artifacts` and inspect the crop images. Then try `--crop-box`, `--crop-mode auto`, `--ocr-preprocess threshold`, or a stylised-font backend such as `paddleocr-vl`.

For long videos, prefer the split-stage workflow so transcript and OCR JSON can be reused. OCR writes a JSONL checkpoint after each segment, so `--resume` can continue after a crash.

When reference subtitles are available, pass `--reference-srt subs.srt`. The report then shows whether the audio leg, subtitle OCR leg, or both drift from the reference.

## Development

Run the main checks:

```bash
ruff check .
python3 -m pytest
```

Run native fixture smoke checks when touching CLI, media, OCR, or reporting:

```bash
scripts/run_fixture_e2e.sh /tmp/burnsub-fixture-e2e
scripts/run_native_smoke.sh /tmp/burnsub-native-smoke
```

Regression tests use committed fixture bundles:

```bash
pytest -q tests/test_regression_pipeline.py
BURNSUB_RUN_REAL_SPEECH=1 pytest -q tests/test_real_speech_pipeline.py
```

To rebuild fixtures after editing their specs:

```bash
python scripts/build_regression_fixture.py fixtures/regression/bundle
python scripts/build_real_speech_fixture.py fixtures/realspeech/bundle
```

Build multilingual stress fixtures without committing third-party media:

```bash
python -m pip install -e '.[fixtures]'
python scripts/build_stress_fixture.py /tmp/burnsub-stress/bundle --target-duration 900
BURNSUB_STRESS_DURATION=900 scripts/run_stress_fixture.sh /tmp/burnsub-stress
```

Edit `fixtures/stress/spec.json` or the generated `burned_subtitles.srt` to add
small mismatches, missing subtitles, timing drift, and long-video cases. See
`docs/test-media-sources.md` for Kannada/Indic captioned source videos and
download commands. The stress generator uses ffmpeg's subtitles filter when
available, and falls back to Pillow frame rendering with the `fixtures` extra.
