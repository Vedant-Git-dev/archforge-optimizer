"""AttemptStore — append-only Attempts (spec §3), grouped by parent spec_id.

Layout: `attempts/<parent_spec_id>.jsonl`, one Attempt per line. Grouping by
parent makes the Architect's dedup query (E7) a cheap linear scan of one file:
"has this (parent, kind, target) already been tried and rejected/rolled-back?"

The store assigns `attempt_id` (its own content hash, stable + collision-free)
on first append so Attempts are individually addressable like Specs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from archforge.constants import ATTEMPTS_DIRNAME, SPEC_ID_HASH_LEN
from archforge.models import Attempt, SuiteResult, Verdict
from archforge.stores._jsonl import append_jsonl, read_jsonl, write_jsonl

# Verdicts that should block re-proposing the same (parent, kind, target) —
# the Architect consults `match` to skip known dead ends (spec E7).
_BLOCKING_VERDICTS: frozenset[Verdict] = frozenset({Verdict.REJECTED, Verdict.ROLLED_BACK})


class UnknownAttemptError(KeyError):
    """A requested attempt_id is not in the store."""


class AttemptStore:
    """Append-only Attempt memory, grouped per parent spec_id."""

    ROOT_SUBDIR = ATTEMPTS_DIRNAME

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.attempts_dir = self.root / self.ROOT_SUBDIR

    # ----------------------------------------------------------------- append
    def append(self, attempt: Attempt) -> str:
        """Persist `attempt`, assign its `attempt_id` if unset, return the id."""

        if attempt.attempt_id is None:
            attempt.attempt_id = self._compute_id(attempt)
        existing = self.get(attempt.attempt_id)
        if existing is not None:
            return attempt.attempt_id  # idempotent: same Attempt already persisted
        append_jsonl(
            self._path_for_parent(attempt.parent_spec_id),
            attempt.model_dump(mode="json"),
        )
        return attempt.attempt_id

    # ----------------------------------------------------------------- read
    def get(self, attempt_id: str) -> Attempt | None:
        for att in self.all():
            if att.attempt_id == attempt_id:
                return att
        return None

    def require(self, attempt_id: str) -> Attempt:
        att = self.get(attempt_id)
        if att is None:
            raise UnknownAttemptError(attempt_id)
        return att

    def for_parent(self, parent_spec_id: str) -> list[Attempt]:
        return [
            Attempt.model_validate(rec)
            for rec in read_jsonl(self._path_for_parent(parent_spec_id))
        ]

    def all(self) -> list[Attempt]:
        records: list[Attempt] = []
        if not self.attempts_dir.exists():
            return records
        for path in sorted(self.attempts_dir.glob("*.jsonl")):
            for rec in read_jsonl(path):
                records.append(Attempt.model_validate(rec))
        return records

    # ----------------------------------------------------------------- verdict
    def set_verdict(self, attempt_id: str, verdict: Verdict) -> Attempt:
        """Flip an Attempt's verdict (used by the Gatekeeper / Approval Queue).

        Verdicts are append-only, so this rewrites the attempt's row *in place*
        within its parent file. Idempotent: flipping the same attempt to the same
        verdict is a no-op. Returns the updated Attempt.

        Used for: promotion (PENDING -> PROMOTED/PENDING_HUMAN/REJECTED), human
        approval (PENDING_HUMAN -> PROMOTED), and post-promotion regression
        (PROMOTED -> ROLLED_BACK) (spec E6).
        """

        att = self.require(attempt_id)
        if att.verdict is verdict:
            return att
        att = att.model_copy(update={"verdict": verdict})
        self._rewrite_parent_file(att)
        return att

    def _rewrite_parent_file(self, attempt: Attempt) -> None:
        """Rewrite one parent's file with `attempt` replacing its same-id row."""
        path = self._path_for_parent(attempt.parent_spec_id)
        rows = read_jsonl(path)
        out: list[dict] = []
        replaced = False
        for rec in rows:
            if rec.get("attempt_id") == attempt.attempt_id:
                out.append(attempt.model_dump(mode="json"))
                replaced = True
            else:
                out.append(rec)
        if not replaced:
            out.append(attempt.model_dump(mode="json"))  # path landed here
        write_jsonl(path, out)

    # ----------------------------------------------------------------- result
    def set_result(self, attempt_id: str, result: SuiteResult) -> Attempt:
        """Stamp a candidate's suite result onto its attempt, in place.

        Called by the Engine once the Gatekeeper decides, so the human-facing
        surfaces (`status`, `report`, Approval Queue) show the real delta + cost.
        Like `set_verdict`, this rewrites the attempt's row in place; idempotent
        for the same result. Returns the updated Attempt.
        """

        att = self.require(attempt_id)
        att = att.model_copy(update={"suite_result": result})
        self._rewrite_parent_file(att)
        return att

    # ----------------------------------------------------------------- dedup
    def match(self, parent_spec_id: str, kind: str, target: str) -> list[Attempt]:
        """Prior attempts on the same (parent, change.kind, change.target)."""

        return [
            att
            for att in self.for_parent(parent_spec_id)
            if att.change.kind.value == kind and att.change.target == target
        ]

    def blocking(
        self, parent_spec_id: str, kind: str, target: str
    ) -> list[Attempt]:
        """Prior attempts that should stop the Architect re-proposing this change (E7)."""

        return [
            att for att in self.match(parent_spec_id, kind, target)
            if att.verdict in _BLOCKING_VERDICTS
        ]

    # ----------------------------------------------------------------- paths
    def _path_for_parent(self, parent_spec_id: str) -> Path:
        return self.attempts_dir / f"{parent_spec_id}.jsonl"

    @staticmethod
    def _compute_id(attempt: Attempt) -> str:
        payload = json.dumps(attempt.model_dump(mode="json"), sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:SPEC_ID_HASH_LEN]
