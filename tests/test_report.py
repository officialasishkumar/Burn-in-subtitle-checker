import json

from burnin_subtitle_checker.compare import compare_segments
from burnin_subtitle_checker.models import OcrSegment, TranscriptSegment
from burnin_subtitle_checker.report import write_reports


def test_write_reports_creates_html_json_csv(tmp_path):
    rows = compare_segments(
        [TranscriptSegment(index=0, start=1.0, end=3.0, text="ವಾಕ್ಯ")],
        [OcrSegment(index=0, start=1.0, end=3.0, timestamp=2.0, text="ವಾಕ್ಯ", language="kan")],
    )

    paths = write_reports(
        rows,
        output_dir=tmp_path,
        source_video="fixture.mp4",
        threshold=0.75,
        formats=["html", "json", "csv"],
        config={"mode": "test"},
    )

    assert set(paths) == {"html", "json", "csv"}
    assert "Burn-in Subtitle Check Report" in paths["html"].read_text(encoding="utf-8")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["summary"]["ok"] == 1
    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "timestamp,audio_text,subtitle_text" in csv_text
    assert "word_error_rate,character_error_rate" in csv_text
