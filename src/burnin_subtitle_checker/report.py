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
    include_reference: bool | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = {item.strip().lower() for item in formats if item.strip()}
    if not requested:
        requested = {"html"}

    if include_reference is None:
        include_reference = any(row.reference_text for row in rows)

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
        _write_csv(path, rows, include_reference=include_reference)
        paths["csv"] = path
    if "html" in requested:
        path = output_dir / "report.html"
        path.write_text(
            render_html(
                payload,
                rows,
                output_dir=output_dir,
                include_reference=include_reference,
            ),
            encoding="utf-8",
        )
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


def render_html(
    payload: dict[str, Any],
    rows: list[ComparedSegment],
    *,
    output_dir: Path,
    include_reference: bool = False,
) -> str:
    summary = payload["summary"]
    generated_at = html.escape(payload["generated_at"])
    source_video = html.escape(str(payload.get("source_video") or "precomputed inputs"))
    threshold = payload["threshold"]
    rows_html = "\n".join(
        _render_row(row, output_dir=output_dir, include_reference=include_reference)
        for row in rows
    )
    reference_header = (
        '<th data-sort="reference">Reference Text</th>' if include_reference else ""
    )
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
    <section class="controls" aria-label="filters">
      <label class="search">
        <span>Search</span>
        <input type="search" id="row-search"
          placeholder="audio, subtitle, or note text" aria-controls="rows">
      </label>
      <fieldset class="filters">
        <legend>Show</legend>
        <label><input type="checkbox" data-status="ok" checked> OK</label>
        <label><input type="checkbox" data-status="review" checked> Review</label>
        <label><input type="checkbox" data-status="no-subtitle" checked> No Subtitle</label>
        <label><input type="checkbox" data-status="no-audio" checked> No Audio</label>
        <label><input type="checkbox" data-status="no-text" checked> No Text</label>
      </fieldset>
      <p class="row-count" id="row-count" aria-live="polite"></p>
    </section>
    <section class="table-wrap">
      <table id="rows">
        <thead>
          <tr>
            <th data-sort="number">Timestamp</th>
            <th data-sort="text">Audio Text</th>
            <th data-sort="text">Subtitle Text</th>
            {reference_header}
            <th data-sort="number">Match Score</th>
            <th data-sort="number">Composite</th>
            <th data-sort="number">WER</th>
            <th data-sort="number">CER</th>
            <th data-sort="text">Status</th>
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
  <script>{_report_js()}</script>
</body>
</html>
"""


def _write_csv(
    path: Path,
    rows: list[ComparedSegment],
    *,
    include_reference: bool,
) -> None:
    fieldnames = [
        "index",
        "start",
        "end",
        "timestamp",
        "audio_text",
        "subtitle_text",
        "score",
        "composite_score",
        "word_error_rate",
        "character_error_rate",
        "status",
        "crop_path",
        "notes",
    ]
    if include_reference:
        fieldnames.extend(
            [
                "reference_text",
                "reference_vs_audio_score",
                "reference_vs_subtitle_score",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = {
                "index": row.index,
                "start": f"{row.start:.3f}",
                "end": f"{row.end:.3f}",
                "timestamp": f"{row.timestamp:.3f}",
                "audio_text": row.audio_text,
                "subtitle_text": row.subtitle_text,
                "score": f"{row.score:.4f}",
                "composite_score": _format_optional_rate(row.composite_score),
                "word_error_rate": _format_optional_rate(row.word_error_rate),
                "character_error_rate": _format_optional_rate(row.character_error_rate),
                "status": row.status,
                "crop_path": row.crop_path or "",
                "notes": " | ".join(row.notes),
            }
            if include_reference:
                data["reference_text"] = row.reference_text or ""
                data["reference_vs_audio_score"] = _format_optional_rate(
                    row.reference_vs_audio_score
                )
                data["reference_vs_subtitle_score"] = _format_optional_rate(
                    row.reference_vs_subtitle_score
                )
            writer.writerow(data)


def _render_row(
    row: ComparedSegment,
    *,
    output_dir: Path,
    include_reference: bool,
) -> str:
    status_class = row.status.lower().replace("_", "-")
    evidence = _evidence_link(row.crop_path, output_dir=output_dir)
    notes = "<br>".join(html.escape(note) for note in row.notes)
    reference_cell = ""
    if include_reference:
        reference_text = row.reference_text or ""
        reference_cell = (
            f'<td data-sort-value="{html.escape(reference_text)}">'
            f"{html.escape(reference_text)}</td>"
        )
    composite = _format_optional_rate(row.composite_score, empty="n/a")
    wer_text = _format_optional_rate(row.word_error_rate, empty="n/a")
    cer_text = _format_optional_rate(row.character_error_rate, empty="n/a")
    haystack_raw = (
        row.audio_text + " " + row.subtitle_text + " " + " ".join(row.notes)
    ).casefold()
    haystack = html.escape(haystack_raw)
    audio_escaped = html.escape(row.audio_text)
    subtitle_escaped = html.escape(row.subtitle_text)
    status_escaped = html.escape(row.status)
    return (
        f'<tr class="{status_class}" data-status="{status_class}"'
        f' data-search="{haystack}">\n'
        f'  <td data-sort-value="{row.timestamp:.4f}">{row.timestamp:.2f}s</td>\n'
        f'  <td data-sort-value="{audio_escaped}">{audio_escaped}</td>\n'
        f'  <td data-sort-value="{subtitle_escaped}">{subtitle_escaped}</td>\n'
        f"  {reference_cell}\n"
        f'  <td data-sort-value="{row.score:.4f}">{row.score:.2f}</td>\n'
        f'  <td data-sort-value="{_sort_value(row.composite_score)}">'
        f"{html.escape(composite)}</td>\n"
        f'  <td data-sort-value="{_sort_value(row.word_error_rate)}">'
        f"{html.escape(wer_text)}</td>\n"
        f'  <td data-sort-value="{_sort_value(row.character_error_rate)}">'
        f"{html.escape(cer_text)}</td>\n"
        f'  <td data-sort-value="{status_escaped}">'
        f'<span class="status">{status_escaped}</span></td>\n'
        f"  <td>{evidence}</td>\n"
        f"  <td>{notes}</td>\n"
        "</tr>"
    )


def _sort_value(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "-1"


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
  width: min(1320px, calc(100% - 32px));
  margin: 32px auto;
}
header { margin-bottom: 20px; }
h1 { margin: 0 0 8px; font-size: 28px; line-height: 1.2; }
.meta { margin: 4px 0; color: var(--muted); }
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
.summary strong { display: block; font-size: 26px; }
.summary span { color: var(--muted); }
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-end;
  margin-bottom: 12px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
}
.controls label.search {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1 1 240px;
}
.controls .search input {
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 14px;
}
.controls fieldset.filters {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 10px;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin: 0;
}
.controls fieldset.filters legend {
  padding: 0 4px;
  color: var(--muted);
  font-size: 12px;
}
.controls fieldset label {
  font-size: 13px;
  display: inline-flex;
  gap: 6px;
  align-items: center;
}
.row-count { margin: 0; color: var(--muted); font-size: 13px; }
.table-wrap {
  overflow-x: auto;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
}
table { width: 100%; border-collapse: collapse; }
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
  cursor: pointer;
  user-select: none;
  position: sticky;
  top: 0;
}
th[data-sort]:after { content: " ↕"; opacity: .35; font-size: 11px; }
th.sorted-asc:after { content: " ↑"; opacity: 1; }
th.sorted-desc:after { content: " ↓"; opacity: 1; }
td { min-width: 120px; }
td:nth-child(2), td:nth-child(3) { min-width: 240px; font-size: 16px; }
.status {
  display: inline-block;
  min-width: 86px;
  text-align: center;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 700;
}
.ok .status { color: #ffffff; background: var(--ok); }
.review .status, .no-subtitle .status { color: #ffffff; background: var(--review); }
.no-audio .status, .no-text .status { color: #ffffff; background: var(--warn); }
tr.hidden { display: none; }
a { color: #0f4c81; }
"""


def _report_js() -> str:
    return r"""
(function () {
  const search = document.getElementById('row-search');
  const filterInputs = Array.from(document.querySelectorAll('.filters input[type="checkbox"]'));
  const table = document.getElementById('rows');
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const counter = document.getElementById('row-count');
  const headers = Array.from(table.querySelectorAll('th[data-sort]'));

  function applyFilters() {
    const term = (search.value || '').trim().toLowerCase();
    const allowed = new Set(
      filterInputs
        .filter(function (input) { return input.checked; })
        .map(function (input) { return input.dataset.status; })
    );
    let visible = 0;
    rows.forEach(function (row) {
      const status = row.dataset.status;
      const haystack = row.dataset.search || '';
      const matchesStatus = allowed.has(status);
      const matchesText = !term || haystack.indexOf(term) !== -1;
      const visibleRow = matchesStatus && matchesText;
      row.classList.toggle('hidden', !visibleRow);
      if (visibleRow) visible += 1;
    });
    counter.textContent = visible + ' of ' + rows.length + ' rows shown';
  }

  function sortRows(headerIndex, direction, kind) {
    const factor = direction === 'asc' ? 1 : -1;
    const sorted = rows.slice().sort(function (a, b) {
      const aCell = a.children[headerIndex];
      const bCell = b.children[headerIndex];
      const aValue = aCell.dataset.sortValue || aCell.textContent;
      const bValue = bCell.dataset.sortValue || bCell.textContent;
      if (kind === 'number') {
        return (parseFloat(aValue) - parseFloat(bValue)) * factor;
      }
      return aValue.localeCompare(bValue) * factor;
    });
    sorted.forEach(function (row) { tbody.appendChild(row); });
  }

  search.addEventListener('input', applyFilters);
  filterInputs.forEach(function (input) {
    input.addEventListener('change', applyFilters);
  });

  headers.forEach(function (header, index) {
    header.addEventListener('click', function () {
      const kind = header.dataset.sort;
      const isAsc = !header.classList.contains('sorted-asc');
      headers.forEach(function (other) {
        other.classList.remove('sorted-asc');
        other.classList.remove('sorted-desc');
      });
      header.classList.add(isAsc ? 'sorted-asc' : 'sorted-desc');
      sortRows(index, isAsc ? 'asc' : 'desc', kind);
    });
  });

  applyFilters();
})();
"""
