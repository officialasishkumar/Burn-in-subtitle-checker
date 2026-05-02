"""SRT subtitle parsing for reference comparisons."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ConfigError

_TIMING_LINE = re.compile(
    r"(?P<h1>\d{1,2}):(?P<m1>\d{2}):(?P<s1>\d{2})[,.](?P<ms1>\d{1,3})"
    r"\s*-->\s*"
    r"(?P<h2>\d{1,2}):(?P<m2>\d{2}):(?P<s2>\d{2})[,.](?P<ms2>\d{1,3})"
)
_TAG_RE = re.compile(r"<[^>]+>")
_ASS_RE = re.compile(r"\{[^}]*\}")


@dataclass(slots=True)
class ReferenceCue:
    index: int
    start: float
    end: float
    text: str

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


def parse_srt_text(content: str) -> list[ReferenceCue]:
    cues: list[ReferenceCue] = []
    block_lines: list[str] = []
    next_index = 0
    for raw_line in content.splitlines():
        line = raw_line.lstrip("﻿").rstrip()
        if line:
            block_lines.append(line)
            continue
        cue = _cue_from_block(block_lines, next_index)
        if cue is not None:
            cues.append(cue)
            next_index += 1
        block_lines = []
    cue = _cue_from_block(block_lines, next_index)
    if cue is not None:
        cues.append(cue)
    cues.sort(key=lambda item: (item.start, item.end, item.index))
    return cues


def load_reference_srt(path: Path) -> list[ReferenceCue]:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise ConfigError(f"Reference SRT does not exist: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Reference SRT must be UTF-8 encoded: {path}") from exc
    cues = parse_srt_text(content)
    if not cues:
        raise ConfigError(f"Reference SRT did not contain any cues: {path}")
    return cues


def merge_cue_text(cues: Iterable[ReferenceCue]) -> str:
    return " ".join(cue.text for cue in cues if cue.text).strip()


def _cue_from_block(lines: list[str], index: int) -> ReferenceCue | None:
    if not lines:
        return None
    timing_index = _find_timing_line(lines)
    if timing_index is None:
        return None
    match = _TIMING_LINE.search(lines[timing_index])
    if match is None:
        return None
    text_lines = lines[timing_index + 1 :]
    text = " ".join(line for line in text_lines if line).strip()
    text = _strip_inline_tags(text)
    start = _to_seconds(match["h1"], match["m1"], match["s1"], match["ms1"])
    end = _to_seconds(match["h2"], match["m2"], match["s2"], match["ms2"])
    if end < start:
        return None
    return ReferenceCue(index=index, start=start, end=end, text=text)


def _find_timing_line(lines: list[str]) -> int | None:
    for position, line in enumerate(lines):
        if _TIMING_LINE.search(line):
            return position
    return None


def _to_seconds(hours: str, minutes: str, seconds: str, milliseconds: str) -> float:
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds.ljust(3, "0")[:3]) / 1000.0
    )


def _strip_inline_tags(text: str) -> str:
    text = _ASS_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    return text.strip()
