import json

from burnin_subtitle_checker.compare import ReferenceWindow, compare_segments
from burnin_subtitle_checker.models import OcrSegment, TranscriptSegment
from burnin_subtitle_checker.report import write_reports


def test_html_report_includes_filter_controls(tmp_path):
    rows = compare_segments(
        [TranscriptSegment(index=0, start=0.0, end=1.0, text="hello")],
        [OcrSegment(index=0, start=0.0, end=1.0, timestamp=0.5, text="hello", language="eng")],
    )
    paths = write_reports(
        rows,
        output_dir=tmp_path,
        source_video=None,
        threshold=0.75,
        formats=["html"],
        config={},
    )
    html_text = paths["html"].read_text(encoding="utf-8")
    assert "row-search" in html_text
    assert 'data-status="ok"' in html_text
    assert "Composite" in html_text


def test_reports_include_reference_columns_when_data_present(tmp_path):
    transcript = [TranscriptSegment(index=0, start=0.0, end=1.0, text="hello world")]
    ocr = [
        OcrSegment(
            index=0,
            start=0.0,
            end=1.0,
            timestamp=0.5,
            text="hello world",
            language="eng",
        )
    ]
    references = [ReferenceWindow(start=0.0, end=1.0, text="reference text")]

    rows = compare_segments(transcript, ocr, reference_windows=references)
    paths = write_reports(
        rows,
        output_dir=tmp_path,
        source_video=None,
        threshold=0.75,
        formats=["html", "json", "csv"],
        config={"mode": "test"},
    )

    html_text = paths["html"].read_text(encoding="utf-8")
    csv_text = paths["csv"].read_text(encoding="utf-8")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))

    assert "Reference Text" in html_text
    assert "reference_vs_audio_score" in csv_text
    assert payload["segments"][0]["reference_text"] == "reference text"
