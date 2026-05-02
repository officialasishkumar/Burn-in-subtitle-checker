"""Lightweight progress reporting that writes to stderr."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import IO


class ProgressReporter:
    """Stderr progress that updates inline when attached to a terminal."""

    def __init__(
        self,
        total: int,
        *,
        label: str = "",
        stream: IO[str] | None = None,
        enabled: bool = True,
        min_interval: float = 0.1,
    ) -> None:
        self._total = max(0, int(total))
        self._label = label
        self._stream = stream if stream is not None else sys.stderr
        self._enabled = enabled and self._total > 0
        self._isatty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._min_interval = max(0.0, float(min_interval))
        self._completed = 0
        self._started_at = time.monotonic()
        self._last_render_at = 0.0

    @property
    def total(self) -> int:
        return self._total

    @property
    def completed(self) -> int:
        return self._completed

    def advance(self, amount: int = 1) -> None:
        if amount <= 0:
            return
        self._completed = min(self._total, self._completed + int(amount))
        self._maybe_render(force=self._completed >= self._total)

    def finish(self, *, message: str | None = None) -> None:
        if not self._enabled:
            if message:
                print(message, file=self._stream)
            return
        self._render(force=True)
        if self._isatty:
            self._stream.write("\n")
            self._stream.flush()
        if message:
            print(message, file=self._stream)

    def _maybe_render(self, *, force: bool) -> None:
        now = time.monotonic()
        if not force and (now - self._last_render_at) < self._min_interval:
            return
        self._render(force=force)

    def _render(self, *, force: bool) -> None:
        if not self._enabled:
            return
        elapsed = max(time.monotonic() - self._started_at, 1e-6)
        rate = self._completed / elapsed if self._completed else 0.0
        remaining = (self._total - self._completed) / rate if rate > 0 else 0.0
        percent = (self._completed / self._total) * 100 if self._total else 100.0
        line = (
            f"{self._label} {self._completed}/{self._total}"
            f" ({percent:5.1f}%) {rate:5.1f}/s eta {self._format_eta(remaining)}"
        )
        if self._isatty:
            self._stream.write("\r" + line.ljust(80))
            self._stream.flush()
        elif force:
            self._stream.write(line + "\n")
            self._stream.flush()
        self._last_render_at = time.monotonic()

    @staticmethod
    def _format_eta(seconds: float) -> str:
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            minutes, secs = divmod(seconds, 60)
            return f"{minutes}m{secs:02d}s"
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        return f"{hours}h{minutes:02d}m{secs:02d}s"


@contextmanager
def progress_reporter(
    total: int,
    *,
    label: str = "",
    enabled: bool = True,
):
    reporter = ProgressReporter(total, label=label, enabled=enabled)
    try:
        yield reporter
    finally:
        reporter.finish()
