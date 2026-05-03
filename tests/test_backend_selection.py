from burnin_subtitle_checker.asr import (
    resolve_asr_backend,
    resolve_indic_conformer_model_id,
    resolve_indicwhisper_model_id,
)
from burnin_subtitle_checker.backend_config import (
    ASR_INDIC_CONFORMER_BACKEND,
    ASR_INDICWHISPER_BACKEND,
    ASR_WHISPER_BACKEND,
    INDIC_CONFORMER_MODEL_ID,
    OCR_AI4BHARAT_ENGINE,
    OCR_PADDLE_VL_ENGINE,
)
from burnin_subtitle_checker.dependencies import collect_doctor_results, find_indic_tessdata_dir
from burnin_subtitle_checker.ocr import _validate_ocr_options, resolve_tesseract_data_dir


def test_auto_asr_prefers_indicwhisper_for_hindi_when_installed():
    result = resolve_asr_backend(
        "auto",
        language="hi",
        module_available=lambda module: module == "transformers",
    )

    assert result.selected == ASR_INDICWHISPER_BACKEND
    assert "hi" in result.reason


def test_auto_asr_falls_back_to_whisper_when_indicwhisper_missing():
    result = resolve_asr_backend("auto", language="kn", module_available=lambda module: False)

    assert result.selected == ASR_WHISPER_BACKEND
    assert "falling back" in result.reason


def test_auto_asr_uses_whisper_for_non_indic_language_even_if_indicwhisper_installed():
    result = resolve_asr_backend(
        "auto",
        language="en",
        module_available=lambda module: module == "transformers",
    )

    assert result.selected == ASR_WHISPER_BACKEND
    assert "not in the IndicWhisper support set" in result.reason


def test_user_selected_asr_backend_is_not_rewritten():
    result = resolve_asr_backend("faster-whisper", language="hi")

    assert result.selected == "faster-whisper"
    assert "user selected" in result.reason


def test_indicwhisper_model_size_resolution_uses_env_override(monkeypatch):
    monkeypatch.setenv("BURNSUB_INDICWHISPER_HI_SMALL_MODEL_ID", "local/hi-small")

    assert resolve_indicwhisper_model_id("small", "hi") == "local/hi-small"


def test_indic_conformer_model_resolution_uses_default_for_size_aliases():
    assert resolve_indic_conformer_model_id("large") == INDIC_CONFORMER_MODEL_ID


def test_indic_tessdata_dir_requires_all_requested_packs(tmp_path, monkeypatch):
    data_dir = tmp_path / "tessdata"
    data_dir.mkdir()
    (data_dir / "hin.traineddata").write_text("hin", encoding="utf-8")
    (data_dir / "kan.traineddata").write_text("kan", encoding="utf-8")
    monkeypatch.setattr(
        "burnin_subtitle_checker.backend_config.candidate_indic_tessdata_dirs",
        lambda: [data_dir],
    )
    monkeypatch.setattr(
        "burnin_subtitle_checker.dependencies.candidate_indic_tessdata_dirs",
        lambda: [data_dir],
    )

    assert find_indic_tessdata_dir("hin+kan") == data_dir
    assert resolve_tesseract_data_dir("hin+kan") == data_dir
    assert find_indic_tessdata_dir("hin+kan+eng") is None


def test_new_ocr_engines_are_valid_choices():
    _validate_ocr_options(
        crop_bottom_percent=15,
        psm=6,
        preprocess="none",
        upscale_factor=2,
        workers=1,
        engine=OCR_PADDLE_VL_ENGINE,
    )
    _validate_ocr_options(
        crop_bottom_percent=15,
        psm=6,
        preprocess="none",
        upscale_factor=2,
        workers=1,
        engine=OCR_AI4BHARAT_ENGINE,
    )


def test_doctor_reports_explicit_missing_indic_conformer_dependency(monkeypatch):
    monkeypatch.setattr("burnin_subtitle_checker.dependencies.executable_path", lambda name: name)
    monkeypatch.setattr("burnin_subtitle_checker.dependencies.tesseract_languages", lambda: {"eng"})
    monkeypatch.setattr(
        "burnin_subtitle_checker.dependencies.python_module_available",
        lambda module: False,
    )

    results = collect_doctor_results(
        ocr_languages="eng",
        asr_backend=ASR_INDIC_CONFORMER_BACKEND,
    )

    asr = next(result for result in results if result.name == "asr backend")
    assert not asr.ok
    assert ".[asr-conformer]" in asr.detail


def test_doctor_reports_new_ocr_engine_dependency(monkeypatch):
    monkeypatch.setattr("burnin_subtitle_checker.dependencies.executable_path", lambda name: name)
    monkeypatch.setattr("burnin_subtitle_checker.dependencies.tesseract_languages", lambda: {"eng"})
    monkeypatch.setattr(
        "burnin_subtitle_checker.dependencies.python_module_available",
        lambda module: False,
    )

    results = collect_doctor_results(
        ocr_languages="eng",
        ocr_engine=OCR_PADDLE_VL_ENGINE,
        asr_backend="none",
    )

    ocr = next(result for result in results if result.name == "ocr engine: paddleocr-vl")
    assert not ocr.ok
    assert ".[ocr-paddle]" in ocr.detail


def test_doctor_non_tesseract_ocr_engine_does_not_require_tesseract(monkeypatch):
    def fake_executable_path(name):
        return name if name in {"ffmpeg", "ffprobe"} else None

    monkeypatch.setattr(
        "burnin_subtitle_checker.dependencies.executable_path",
        fake_executable_path,
    )
    monkeypatch.setattr(
        "burnin_subtitle_checker.dependencies.python_module_available",
        lambda module: module == "easyocr",
    )

    results = collect_doctor_results(
        ocr_languages="hin+kan+eng",
        ocr_engine="easyocr",
        asr_backend="none",
    )

    assert not any(result.name == "tesseract" for result in results)
    assert not any(result.name == "tesseract languages" for result in results)
    assert any(result.name == "ocr engine: easyocr" and result.ok for result in results)


def test_custom_tessdata_dir_is_passed_to_doctor(tmp_path, monkeypatch):
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "hin.traineddata").write_text("hin", encoding="utf-8")
    monkeypatch.setattr("burnin_subtitle_checker.dependencies.executable_path", lambda name: name)
    monkeypatch.setattr(
        "burnin_subtitle_checker.dependencies.tesseract_languages",
        lambda path=None: {"hin"} if path == tessdata else set(),
    )

    results = collect_doctor_results(
        ocr_languages="hin",
        asr_backend="none",
        tessdata_dir=tessdata,
    )

    assert any(
        result.name == "tesseract languages" and result.ok and "hin" in result.detail
        for result in results
    )
