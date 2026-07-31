"""TraceStore — append-only Traces, keyed (spec_id, task_id, run_id) (spec §3).

Layout: `traces/<spec_id>.jsonl`, one Trace per line. Grouping by spec_id makes
the two hot queries cheap:
  * `all(spec_id)`        — every trace produced under a Spec (for the Judge /
    SuiteRunner aggregate, and for credit assignment on the incumbent's suite).
  * `latest(spec_id, task_id)` — the most recent run of a task under a Spec.
"""

from __future__ import annotations

from pathlib import Path

from archforge.config import TRACES_DIRNAME
from archforge.models import Trace
from archforge.stores._jsonl import append_jsonl, read_jsonl


class TraceStore:
    """Append-only per-spec trace log (one JSONL file per spec_id)."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.traces_dir = self.root / TRACES_DIRNAME

    def append(self, trace: Trace) -> None:
        append_jsonl(self.traces_dir / f"{trace.spec_id}.jsonl", trace.model_dump(mode="json"))

    def all(self, spec_id: str) -> list[Trace]:
        return [
            Trace.model_validate(rec) for rec in read_jsonl(self.traces_dir / f"{spec_id}.jsonl")
        ]

    def latest(self, spec_id: str, task_id: str) -> Trace | None:
        traces = self.all(spec_id)
        matching = [t for t in traces if t.task_id == task_id]
        return matching[-1] if matching else None

    def for_task(self, spec_id: str, task_id: str) -> list[Trace]:
        """All runs of `task_id` under `spec_id`, in append order (for R repeats)."""

        return [t for t in self.all(spec_id) if t.task_id == task_id]
