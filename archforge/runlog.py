"""The "file" sink for the per-cycle cards + loop summary (improvement #5).

The CLI prints each cycle's compact card to stdout AND appends a JSON form here, so
the run's narrative survives the terminal (scrollback is lossy; this file is not).
It also writes the final loop summary. The whole run is one object, overwritten
each run ("last" — not an ever-growing append; the per-run history is the
traces/attempts already persisted to the store under ``<root>``).

``<root>/runs/last.json`` shape::

    {"schema": "archforge.runlog/v1",
     "started_at": <iso str | None>,        # taken once at run start by the CLI
     "cycles":     [ <cycle-dict>, ... ],    # one per attempted cycle, in order
     "summary":    { <summary-dict> | None }} # written once at the end

The run-log is *fail-soft*: if the directory isn't writable (read-only mount,
permission denied) it disables itself at construction and every method becomes a
no-op — so a run NEVER crashes because the log couldn't be written. The caller
checks ``enabled`` and falls back to stdout-only (printing a one-line notice).
No wall clock is read at import (``started_at`` is supplied by the caller; the
schema's field is optional) — sidesteps any "clock at module load" concern.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA = "archforge.runlog/v1"


class RunLog:
    """A fail-soft writer for ``runs/last.json`` (per-run, overwritten each run).

    Construct with the absolute path to the log file (usually
    ``<root>/runs/last.json``); the parent dir is created if missing. On any
    ``PermissionError``/``OSError`` at construction OR on a later write, the log
    disables itself (``enabled=False``) and all methods become no-ops — never
    raises into the run. ``started_at`` is an optional caller-supplied timestamp
    (the CLI takes it once at run start); the field serialises as ``null`` when
    absent.
    """

    def __init__(self, path: str | Path, *, started_at: str | None = None) -> None:
        self.path = Path(path)
        self.started_at: str | None = started_at
        self.enabled: bool = True
        self.reason: str = ""
        self._cycles: list[dict[str, Any]] = []
        self._summary: dict[str, Any] | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as exc:
            self.enabled = False
            self.reason = f"dir not writable: {exc}"
        # probe-write an empty log so a read-only mount surfaces at construction
        # (not lazily on the first append — cleaner disable notice timing).
        if self.enabled:
            self._flush()

    # ------------------------------------------------------------------ write
    def append_cycle(self, cycle: dict[str, Any]) -> None:
        """Append one cycle's record and flush the full file (overwrite = last)."""
        if not self.enabled:
            return
        self._cycles.append(cycle)
        self._flush()

    def write_summary(self, summary: dict[str, Any]) -> None:
        """Set the summary block and flush (the final write of the run)."""
        if not self.enabled:
            return
        self._summary = summary
        self._flush()

    # ------------------------------------------------------------------ load
    def load(self) -> dict[str, Any] | None:
        """Read the persisted log back (``None`` if disabled/unreadable)."""
        if not self.enabled:
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------ priv
    def _flush(self) -> None:
        obj = {
            "schema": _SCHEMA,
            "started_at": self.started_at,
            "cycles": self._cycles,
            "summary": self._summary,
        }
        try:
            self.path.write_text(
                json.dumps(obj, indent=2, default=str), encoding="utf-8"
            )
        except (PermissionError, OSError) as exc:
            self.enabled = False
            self.reason = f"write failed: {exc}"


__all__ = ["RunLog"]
