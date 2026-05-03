"""Shared backend metadata for CLI choices, dispatch, and doctor checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ASR_AUTO_BACKEND = "auto"
ASR_WHISPER_BACKEND = "whisper"
ASR_FAST_BACKEND = "faster-whisper"
ASR_INDICWHISPER_BACKEND = "indicwhisper"
ASR_INDIC_CONFORMER_BACKEND = "indic-conformer"

ASR_BACKEND_CHOICES = (
    ASR_AUTO_BACKEND,
    ASR_INDICWHISPER_BACKEND,
    ASR_WHISPER_BACKEND,
    ASR_FAST_BACKEND,
    ASR_INDIC_CONFORMER_BACKEND,
)
DOCTOR_ASR_BACKEND_CHOICES = (*ASR_BACKEND_CHOICES, "none")

ASR_BACKEND_MODULES = {
    ASR_WHISPER_BACKEND: ("whisper", ".[asr]"),
    ASR_FAST_BACKEND: ("faster_whisper", ".[asr-fast]"),
    ASR_INDICWHISPER_BACKEND: ("transformers", ".[asr-indic]"),
    ASR_INDIC_CONFORMER_BACKEND: ("nemo.collections.asr", ".[asr-conformer]"),
}

INDICWHISPER_LANGUAGES = frozenset(
    {
        "bn",
        "gu",
        "hi",
        "kn",
        "ml",
        "mr",
        "or",
        "pa",
        "sa",
        "ta",
        "te",
        "ur",
    }
)

INDIC_CONFORMER_LANGUAGES = frozenset(
    {
        "as",
        "bn",
        "brx",
        "doi",
        "gu",
        "hi",
        "kn",
        "kok",
        "ks",
        "mai",
        "ml",
        "mni",
        "mr",
        "ne",
        "or",
        "pa",
        "sa",
        "sat",
        "sd",
        "ta",
        "te",
        "ur",
    }
)

OCR_TESSERACT_ENGINE = "tesseract"
OCR_EASYOCR_ENGINE = "easyocr"
OCR_PADDLE_VL_ENGINE = "paddleocr-vl"
OCR_AI4BHARAT_ENGINE = "ai4bharat"

OCR_ENGINE_CHOICES = (
    OCR_TESSERACT_ENGINE,
    OCR_EASYOCR_ENGINE,
    OCR_PADDLE_VL_ENGINE,
    OCR_AI4BHARAT_ENGINE,
)

OCR_ENGINE_MODULES = {
    OCR_EASYOCR_ENGINE: ("easyocr", ".[ocr-easy]"),
    OCR_PADDLE_VL_ENGINE: ("paddleocr", ".[ocr-paddle]"),
    OCR_AI4BHARAT_ENGINE: ("transformers", ".[ocr-indic]"),
}

TESSERACT_TO_ISO_LANGUAGES = {
    "eng": "en",
    "hin": "hi",
    "kan": "kn",
    "tam": "ta",
    "tel": "te",
    "ben": "bn",
    "mar": "mr",
    "guj": "gu",
    "pan": "pa",
    "mal": "ml",
    "ori": "or",
    "asm": "as",
    "urd": "ur",
    "san": "sa",
    "nep": "ne",
}

INDIC_OCR_TESSERACT_LANGUAGES = frozenset(
    {
        "ben",
        "guj",
        "hin",
        "kan",
        "mal",
        "mni",
        "ori",
        "pan",
        "sat",
        "tam",
        "tel",
    }
)

DEFAULT_INDIC_TESSDATA_DIR = (
    Path.home() / ".local" / "share" / "burnsub" / "indic-ocr" / "tessdata"
)

INDIC_TESSDATA_ENV_VARS = ("BURNSUB_INDIC_TESSDATA", "BURNSUB_TESSDATA_DIR")

INDIC_CONFORMER_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
INDICWHISPER_MODEL_FAMILY = "ai4bharat/indicwhisper"
INDICWHISPER_MODEL_SIZES = frozenset({"small", "medium", "large"})
INDICWHISPER_DEFAULT_SIZE = "medium"
AI4BHARAT_OCR_MODEL_ID = "ai4bharat/Indic-OCR"


@dataclass(frozen=True, slots=True)
class BackendResolution:
    requested: str
    selected: str
    reason: str


def normalize_asr_language(language: str | None) -> str:
    if language is None:
        return "auto"
    normalized = language.strip().lower().replace("_", "-")
    aliases = {
        "hindi": "hi",
        "kannada": "kn",
        "english": "en",
        "auto-detect": "auto",
        "automatic": "auto",
    }
    return aliases.get(normalized, normalized)


def map_tesseract_language_to_iso(language: str) -> str:
    return TESSERACT_TO_ISO_LANGUAGES.get(language, language)


def indic_tessdata_install_command() -> str:
    parent = DEFAULT_INDIC_TESSDATA_DIR.parent
    return (
        f"mkdir -p {parent} && "
        f"git clone https://github.com/indic-ocr/tessdata "
        f"{DEFAULT_INDIC_TESSDATA_DIR} && "
        "stock=$(tesseract --list-langs | sed -n "
        "'s/^List of available languages in \"\\(.*\\)\":/\\1/p') && "
        f"[ -f \"$stock/eng.traineddata\" ] && "
        f"cp \"$stock/eng.traineddata\" {DEFAULT_INDIC_TESSDATA_DIR}/ || true"
    )


def candidate_indic_tessdata_dirs() -> list[Path]:
    candidates: list[Path] = []
    for env_var in (*INDIC_TESSDATA_ENV_VARS, "TESSDATA_PREFIX"):
        value = os.environ.get(env_var)
        if value:
            path = Path(value).expanduser()
            candidates.append(path)
            candidates.append(path / "tessdata")
    candidates.extend(
        [
            DEFAULT_INDIC_TESSDATA_DIR,
            Path.home() / ".local" / "share" / "indic-ocr" / "tessdata",
            Path.home() / "indic-ocr" / "tessdata",
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique
