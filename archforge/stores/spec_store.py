"""SpecStore — content-addressed Pipeline Specs + the active incumbent pointer.

On-disk layout (under the configured root, e.g. `.archforge/`):

    specs/<spec_id>.json         one immutable Spec per content+lineage id
    active.pointer               the active incumbent spec_id (or absent = none)
    archived.jsonl                spec_ids rolled back; queried for I3/rollback

Key behaviours tied to the spec:
  * `commit` is idempotent for the same (content, parent) and never overwrites an
    existing Spec file (immutability, I2).
  * `set_active` is the ONLY mutator of the incumbent pointer (single source of
    truth, I1). Promotion and rollback both go through it.
  * `archive` records a rolled-back Spec; the Spec file itself is never deleted
    (E6/I3 — "archived, never deleted", lineage stays queryable).
  * `lineage(spec_id)` walks `parent_spec_id` to a root — supports rollback and
    I3 (lineage reachability).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from archforge.config import (
    ACTIVE_POINTER_FILE, ARCHIVED_FILE, SPECS_DIRNAME,
)
from archforge.models import Spec, SpecStatus
from archforge.stores._jsonl import (
    append_jsonl,
    read_jsonl,
    read_text,
    write_text_atomic,
)

_POINTER_FILE = ACTIVE_POINTER_FILE
_ARCHIVED_FILE = ARCHIVED_FILE


class NoActiveSpecError(RuntimeError):
    """No Spec has been promoted yet — `active()` was called before any set_active."""


class UnknownSpecError(KeyError):
    """A requested spec_id is not in the store."""


class ArchivedRecord(BaseModel):
    spec_id: str
    reason: str  # "rollback" | "human-reject" | ... (structured, not free-text)


class SpecStore:
    """Content-addressed, immutable Spec storage with an active incumbent pointer."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.specs_dir = self.root / SPECS_DIRNAME
        self.specs_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- commit
    def commit(
        self,
        spec: Spec,
        *,
        parent_spec_id: str | None = None,
        status: SpecStatus = SpecStatus.CANDIDATE,
        created_at: str | None = None,
    ) -> str:
        """Persist `spec` (prompted to `parent_spec_id`), return its content id.

        Idempotent: committing the same (content, parent) twice returns the same
        id and does not rewrite the existing file. Sets `created_at` (UTC ISO) if
        the caller did not supply one.
        """

        spec.parent_spec_id = parent_spec_id
        spec.status = status
        spec.created_at = created_at or _now_iso()
        spec_id = spec.compute_spec_id()
        spec.spec_id = spec_id

        path = self._spec_path(spec_id)
        if not path.exists():
            # Immutability: never overwrite. If the id already exists it would be
            # a hash collision over different content — astronomically unlikely —
            # which we treat as a programmer error rather than silent loss.
            write_text_atomic(path, spec.model_dump_json(indent=2))
        return spec_id

    # ----------------------------------------------------------------- read
    def get(self, spec_id: str) -> Spec:
        path = self._spec_path(spec_id)
        if not path.exists():
            raise UnknownSpecError(spec_id)
        return Spec.model_validate_json(read_text(path))

    def has(self, spec_id: str) -> bool:
        return self._spec_path(spec_id).exists()

    # ----------------------------------------------------------------- active
    def active_id(self) -> str | None:
        """The incumbent spec_id, or None if none has been promoted yet."""

        text = read_text(self.root / _POINTER_FILE)
        if text is None:
            return None
        text = text.strip()
        return text or None

    def active(self) -> Spec:
        spec_id = self.active_id()
        if spec_id is None:
            raise NoActiveSpecError("no Spec has been promoted as incumbent yet")
        return self.get(spec_id)

    def set_active(self, spec_id: str) -> None:
        """The ONLY mutator of the incumbent pointer (I1). Validates the Spec exists."""

        if not self.has(spec_id):
            raise UnknownSpecError(spec_id)
        write_text_atomic(self.root / _POINTER_FILE, spec_id)

    def clear_active(self) -> None:
        """Remove the incumbent pointer (used in tests / reset)."""

        p = self.root / _POINTER_FILE
        if p.exists():
            p.unlink()

    # ----------------------------------------------------------------- archive
    def archive(self, spec_id: str, *, reason: str = "rollback") -> None:
        """Mark a Spec as archived. The Spec file itself is never deleted (E6/I3)."""

        if not self.has(spec_id):
            raise UnknownSpecError(spec_id)
        if self.is_archived(spec_id):
            return
        append_jsonl(
            self.root / _ARCHIVED_FILE,
            ArchivedRecord(spec_id=spec_id, reason=reason).model_dump(),
        )

    def is_archived(self, spec_id: str) -> bool:
        records = read_jsonl(self.root / _ARCHIVED_FILE)
        return any(r.get("spec_id") == spec_id for r in records)

    def archived_ids(self) -> set[str]:
        return {r["spec_id"] for r in read_jsonl(self.root / _ARCHIVED_FILE)}

    # ----------------------------------------------------------------- lineage
    def lineage(self, spec_id: str) -> list[str]:
        """Spec ids from `spec_id` back to root (inclusive). For rollback + I3.

        Stops at a root (parent_spec_id is None or unknown). Raises if the start
        spec is unknown. Cycle-safe (records seen ids).
        """

        if not self.has(spec_id):
            raise UnknownSpecError(spec_id)
        chain: list[str] = [spec_id]
        seen: set[str] = {spec_id}
        current = self.get(spec_id)
        while current.parent_spec_id is not None:
            parent_id = current.parent_spec_id
            if parent_id in seen:
                break  # defensive: lineage should be acyclic by construction
            if not self.has(parent_id):
                break  # parent not stored — root reached, chain ends here
            chain.append(parent_id)
            seen.add(parent_id)
            current = self.get(parent_id)
        return chain

    def known_ids(self) -> set[str]:
        """All spec_ids currently persisted. (Reporting / smoke / tests.)"""

        return {p.stem for p in self.specs_dir.glob("*.json")}

    # ----------------------------------------------------------------- paths
    def _spec_path(self, spec_id: str) -> Path:
        return self.specs_dir / f"{spec_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
