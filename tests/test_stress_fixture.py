from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_stress_fixture.py"
SPEC = ROOT / "fixtures" / "stress" / "spec.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_stress_fixture", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stress_segment_builder_filters_and_repeats_languages():
    builder = _load_builder()
    fixture_spec = json.loads(SPEC.read_text(encoding="utf-8"))

    segments = builder.build_segment_cases(
        fixture_spec,
        repeat=2,
        target_duration=None,
        only_languages={"kn", "hi"},
        segment_seconds=None,
    )

    assert len(segments) == 8
    assert {segment.language for segment in segments} == {"kn", "hi"}
    assert segments[0].start == 0.0
    assert segments[1].start == segments[0].end
    assert segments[-1].repeat_index == 1
    assert builder.ocr_language_string(segments) == "kan+hin"


def test_stress_segment_builder_scales_to_target_duration():
    builder = _load_builder()
    fixture_spec = json.loads(SPEC.read_text(encoding="utf-8"))

    segments = builder.build_segment_cases(
        fixture_spec,
        repeat=1,
        target_duration=100.0,
        only_languages=None,
        segment_seconds=None,
    )

    assert segments[-1].end >= 100.0
    assert len({segment.repeat_index for segment in segments}) > 1


def test_stress_srt_and_transcript_outputs_are_editable():
    builder = _load_builder()
    fixture_spec = json.loads(SPEC.read_text(encoding="utf-8"))
    segments = builder.build_segment_cases(
        fixture_spec,
        repeat=1,
        target_duration=None,
        only_languages={"pa", "en"},
        segment_seconds=1.5,
    )

    burned_srt = builder.render_srt(segments, text_field="burn", max_chars=20)
    reference_srt = builder.render_srt(segments, text_field="audio", max_chars=20)
    transcript = json.loads(builder.render_transcript_json(segments))
    expected = json.loads(builder.render_expected_json(segments))

    assert "The train leaves at" in burned_srt
    assert "ten" in burned_srt
    assert "ਉਹ ਅੱਜ ਕੰਮ ਤੇ ਗਿਆ" not in burned_srt
    assert "ਉਹ ਅੱਜ ਕੰਮ ਤੇ ਗਿਆ" in reference_srt
    assert transcript["segments"][0]["text"] == segments[0].audio_text
    assert expected["summary"]["by_status"]["NO_SUBTITLE"] == 1
