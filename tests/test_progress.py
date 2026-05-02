import io

from burnin_subtitle_checker.progress import ProgressReporter


def test_progress_reporter_writes_summary_when_disabled_finish():
    stream = io.StringIO()
    reporter = ProgressReporter(total=0, label="ocr", stream=stream, enabled=False)
    reporter.advance()  # no-op when total is 0
    reporter.finish(message="done")
    assert "done" in stream.getvalue()


def test_progress_reporter_renders_final_line_to_non_tty():
    stream = io.StringIO()
    reporter = ProgressReporter(total=4, label="ocr", stream=stream, min_interval=0.0)
    reporter.advance(2)
    reporter.advance(2)
    reporter.finish()
    output = stream.getvalue()
    assert "ocr 4/4" in output
    assert "(100.0%)" in output


def test_progress_reporter_eta_formatting_for_long_runs():
    assert ProgressReporter._format_eta(3700) == "1h01m40s"
    assert ProgressReporter._format_eta(75) == "1m15s"
    assert ProgressReporter._format_eta(45) == "45s"
