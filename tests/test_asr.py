import sys
from types import ModuleType, SimpleNamespace

import pytest

from burnin_subtitle_checker.asr import (
    _segments_from_transformers_payload,
    _segments_from_whisper_payload,
    transcribe_video,
)
from burnin_subtitle_checker.exceptions import MissingDependencyError


def test_segments_from_whisper_payload_normalizes_timed_segments():
    rows = _segments_from_whisper_payload(
        {
            "segments": [
                {"start": 1, "end": 2.5, "text": "  hello world  "},
                {"start": "3.0", "end": "4.25", "text": None},
            ]
        }
    )

    assert rows[0].index == 0
    assert rows[0].start == 1.0
    assert rows[0].end == 2.5
    assert rows[0].text == "hello world"
    assert rows[1].text == ""


def test_transcribe_video_rejects_unknown_backend(tmp_path):
    with pytest.raises(MissingDependencyError):
        transcribe_video(tmp_path / "video.mp4", backend="unknown")


def test_segments_from_transformers_payload_uses_chunk_timestamps():
    rows = _segments_from_transformers_payload(
        {
            "chunks": [
                {"timestamp": (0.0, 1.25), "text": " नमस्ते "},
                {"timestamp": (1.25, 2.5), "text": "hello"},
            ]
        }
    )

    assert [row.text for row in rows] == ["नमस्ते", "hello"]
    assert rows[0].start == 0.0
    assert rows[1].end == 2.5


def test_indicwhisper_backend_invokes_transformers_pipeline(monkeypatch, tmp_path):
    captured = {}

    class FakeTokenizer:
        def get_decoder_prompt_ids(self, language, task):
            captured["forced_language"] = language
            captured["task"] = task
            return [(1, 2)]

    class FakePipeline:
        tokenizer = FakeTokenizer()
        processor = None

        def __call__(self, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return {"chunks": [{"timestamp": (0, 1), "text": "नमस्ते"}]}

    fake_transformers = ModuleType("transformers")
    fake_transformers.pipeline = lambda *args, **kwargs: captured.setdefault(
        "pipeline_args", (args, kwargs)
    ) and FakePipeline()
    fake_torch = ModuleType("torch")
    fake_torch.float16 = object()
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("burnin_subtitle_checker.asr._safe_media_duration", lambda path: 1.0)

    rows = transcribe_video(
        tmp_path / "video.mp4",
        backend="indicwhisper",
        model_name="small",
        language="hi",
        device="cpu",
    )

    assert rows[0].text == "नमस्ते"
    assert captured["pipeline_args"][0] == ("automatic-speech-recognition",)
    assert captured["forced_language"] == "hi"
    assert captured["kwargs"]["return_timestamps"] is True


def test_indic_conformer_backend_invokes_decoder(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "burnin_subtitle_checker.asr.extract_audio",
        lambda video_path, output_wav: output_wav.write_text("wav", encoding="utf-8"),
    )
    monkeypatch.setattr("burnin_subtitle_checker.asr._safe_media_duration", lambda path: 3.0)
    captured = {}

    def fake_run(model_id, wav_path, language, decoder, *, device):
        captured.update(
            {
                "model_id": model_id,
                "language": language,
                "decoder": decoder,
                "device": device,
                "wav_exists": wav_path.exists(),
            }
        )
        return "नमस्ते"

    monkeypatch.setattr("burnin_subtitle_checker.asr._run_indic_conformer", fake_run)

    rows = transcribe_video(
        tmp_path / "video.mp4",
        backend="indic-conformer",
        model_name="large",
        language="kn",
        device="cpu",
        conformer_decoder="rnnt",
    )

    assert rows[0].text == "नमस्ते"
    assert rows[0].end == 3.0
    assert captured["language"] == "kn"
    assert captured["decoder"] == "rnnt"
    assert captured["wav_exists"] is True
