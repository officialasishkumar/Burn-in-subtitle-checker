"""Report generation."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ComparedSegment


def write_reports(
    rows: list[ComparedSegment],
    *,
    output_dir: Path,
    source_video: str | None,
    threshold: float,
    formats: Iterable[str],
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = {item.strip().lower() for item in formats if item.strip()}
    if not requested:
        requested = {"html"}

    paths: dict[str, Path] = {}
    payload = report_payload(
        rows,
        source_video=source_video,
        threshold=threshold,
        config=config or {},
    )
    if "json" in requested:
        path = output_dir / "report.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        paths["json"] = path
    if "csv" in requested:
        path = output_dir / "report.csv"
        _write_csv(path, rows)
        paths["csv"] = path
    if "html" in requested:
        path = output_dir / "report.html"
        path.write_text(render_html(payload, rows, output_dir=output_dir), encoding="utf-8")
        paths["html"] = path
    return paths


def report_payload(
    rows: list[ComparedSegment],
    *,
    source_video: str | None,
    threshold: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter(row.status for row in rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_video": source_video,
        "threshold": threshold,
        "summary": {
            "total": len(rows),
            "ok": counts.get("OK", 0),
            "review": counts.get("REVIEW", 0),
            "no_subtitle": counts.get("NO_SUBTITLE", 0),
            "no_audio": counts.get("NO_AUDIO", 0),
            "no_text": counts.get("NO_TEXT", 0),
            "non_ok": sum(count for status, count in counts.items() if status != "OK"),
        },
        "config": config,
        "segments": [row.to_dict() for row in rows],
    }


def render_html(payload: dict[str, Any], rows: list[ComparedSegment], *, output_dir: Path) -> str:
    summary = payload["summary"]
    generated_at = html.escape(payload["generated_at"])
    source_video = html.escape(str(payload.get("source_video") or "precomputed inputs"))
    threshold = payload["threshold"]
    rows_html = "\n".join(_render_row(row, output_dir=output_dir) for row in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Burn-in Subtitle Check Report</title>
  <style>{_report_css()}</style>
</head>
<body>
  <main>
    <header>
      <h1>Burn-in Subtitle Check Report</h1>
      <p class="meta">Generated {generated_at} | Threshold {threshold:.2f}</p>
      <p class="meta">Source: {source_video}</p>
    </header>
    <section class="summary" aria-label="summary">
      <div><strong>{summary["total"]}</strong><span>Total</span></div>
      <div><strong>{summary["ok"]}</strong><span>OK</span></div>
      <div><strong>{summary["non_ok"]}</strong><span>Needs Review</span></div>
      <div><strong>{summary["no_subtitle"]}</strong><span>No Subtitle</span></div>
    </section>
    <section class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Audio Text</th>
            <th>Subtitle Text</th>
            <th>Match Score</th>
            <th>WER</th>
            <th>CER</th>
            <th>Status</th>
            <th>Evidence</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def _write_csv(path: Path, rows: list[ComparedSegment]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "start",
                "end",
                "timestamp",
                "audio_text",
                "subtitle_text",
                "score",
                "word_error_rate",
                "character_error_rate",
                "status",
                "crop_path",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": row.index,
                    "start": f"{row.start:.3f}",
                    "end": f"{row.end:.3f}",
                    "timestamp": f"{row.timestamp:.3f}",
                    "audio_text": row.audio_text,
                    "subtitle_text": row.subtitle_text,
                    "score": f"{row.score:.4f}",
                    "word_error_rate": _format_optional_rate(row.word_error_rate),
                    "character_error_rate": _format_optional_rate(row.character_error_rate),
                    "status": row.status,
                    "crop_path": row.crop_path or "",
                    "notes": " | ".join(row.notes),
                }
            )


def _render_row(row: ComparedSegment, *, output_dir: Path) -> str:
    status_class = row.status.lower().replace("_", "-")
    evidence = _evidence_link(row.crop_path, output_dir=output_dir)
    notes = "<br>".join(html.escape(note) for note in row.notes)
    return f"""<tr class="{status_class}">
  <td>{row.timestamp:.2f}s</td>
  <td>{html.escape(row.audio_text)}</td>
  <td>{html.escape(row.subtitle_text)}</td>
  <td>{row.score:.2f}</td>
  <td>{html.escape(_format_optional_rate(row.word_error_rate, empty="n/a"))}</td>
  <td>{html.escape(_format_optional_rate(row.character_error_rate, empty="n/a"))}</td>
  <td><span class="status">{html.escape(row.status)}</span></td>
  <td>{evidence}</td>
  <td>{notes}</td>
</tr>"""


def _format_optional_rate(value: float | None, *, empty: str = "") -> str:
    if value is None:
        return empty
    return f"{value:.4f}"


def _evidence_link(crop_path: str | None, *, output_dir: Path) -> str:
    if not crop_path:
        return ""
    path = Path(crop_path)
    try:
        href = path.relative_to(output_dir)
    except ValueError:
        href = path
    escaped = html.escape(str(href))
    return f'<a href="{escaped}">crop</a>'


def _report_css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #172033;
  --muted: #5c667a;
  --border: #d9dee8;
  --ok: #0f766e;
  --review: #b42318;
  --warn: #9a6700;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}
main {
  width: min(1280px, calc(100% - 32px));
  margin: 32px auto;
}
header {
  margin-bottom: 20px;
}
h1 {
  margin: 0 0 8px;
  font-size: 28px;
  line-height: 1.2;
}
.meta {
  margin: 4px 0;
  color: var(--muted);
}
.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin: 20px 0;
}
.summary div {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
}
.summary strong {
  display: block;
  font-size: 26px;
}
.summary span {
  color: var(--muted);
}
.table-wrap {
  overflow-x: auto;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th, td {
  padding: 12px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}
th {
  background: #eef2f7;
  color: #273449;
  font-size: 13px;
}
td {
  min-width: 120px;
}
td:nth-child(2),
td:nth-child(3) {
  min-width: 240px;
  font-size: 16px;
}
.status {
  display: inline-block;
  min-width: 86px;
  text-align: center;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 700;
}
.ok .status {
  color: #ffffff;
  background: var(--ok);
}
.review .status,
.no-subtitle .status {
  color: #ffffff;
  background: var(--review);
}
.no-audio .status,
.no-text .status {
  color: #ffffff;
  background: var(--warn);
}
a {
  color: #0f4c81;
}
"""
