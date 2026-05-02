import sys
from types import SimpleNamespace

import pytest

from burnin_subtitle_checker.easyocr_engine import map_languages, run_easyocr
from burnin_subtitle_checker.exceptions import MissingDependencyError


def test_map_languages_translates_tesseract_codes():
    assert map_languages("hin+kan+eng") == ["hi", "kn", "en"]
    assert map_languages("eng,hin") == ["en", "hi"]


def test_run_easyocr_invokes_reader_when_module_present(monkeypatch, tmp_path):
    image = tmp_path / "crop.png"
    image.write_text("img", encoding="utf-8")

    captured = {}

    class FakeReader:
        def __init__(self, languages, gpu, verbose):
            captured["languages"] = languages

        def readtext(self, path, detail, paragraph):
            captured["path"] = path
            captured["detail"] = detail
            captured["paragraph"] = paragraph
            return ["hello", " world "]

    fake_easyocr = SimpleNamespace(Reader=FakeReader)
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)
    # Reset cache between tests
    monkeypatch.setattr(
        "burnin_subtitle_checker.easyocr_engine._READER_CACHE",
        {},
    )

    text = run_easyocr(image, languages="hin+eng")
    assert text == "hello world"
    assert captured["languages"] == ["hi", "en"]
    assert captured["paragraph"] is True
    assert captured["detail"] == 0


def test_run_easyocr_raises_when_module_missing(monkeypatch, tmp_path):
    image = tmp_path / "crop.png"
    image.write_text("img", encoding="utf-8")
    monkeypatch.setattr(
        "burnin_subtitle_checker.easyocr_engine._READER_CACHE",
        {},
    )
    if "easyocr" in sys.modules:
        monkeypatch.delitem(sys.modules, "easyocr")
    monkeypatch.setattr(
        "burnin_subtitle_checker.easyocr_engine.parse_language_spec",
        lambda spec: ["en"],
    )
    monkeypatch.setattr("builtins.__import__", _import_raises_for("easyocr"))
    with pytest.raises(MissingDependencyError):
        run_easyocr(image, languages="eng")


def _import_raises_for(name: str):
    real_import = __import__

    def fake_import(target, *args, **kwargs):
        if target == name:
            raise ImportError(f"No module named {name}")
        return real_import(target, *args, **kwargs)

    return fake_import
