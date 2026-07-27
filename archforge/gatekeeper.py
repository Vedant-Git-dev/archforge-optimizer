"""The Gatekeeper — the P-E-C "Commit" step (spec §3, §4, §6, §8; Phase 8).

Lone enforcer of "fail closed to the incumbent": it is the only thing that ever
moves the `active` pointer (via SpecStore.set_active) or flips an Attempt's
verdict beyond INITIAL. Its `decide(...)` returns a `Decision` naming the action;
`apply(...)` executes it (or you can decide without applying to inspect).

Decision rules (thresholds from `m.Thresholds`):
  * unrunnable (candidate suite >ε crashed) -> DISCARD (before any margin math, E4)
  * cross-rubric/cross-suite      -> DISCARD (the only valid comparison is same
    rubric + same suite; I5 — never silently compare across rubrics)
  * candidate_mean - incumbent_mean <  τ  -> DISCARD        (E1 noise inside τ)
  * win + scope SMALL                          -> AUTO_PROMOTE (active = candidate)
  * win + scope STRUCTURAL                    -> QUEUE_HUMAN   (never auto-promote, I4)
  * promoted-then-regresses (≥ δ on the standard suite) -> ROLLBACK
    (active = parent, candidate archived, verdict ROLLED_BACK; pointer swap, E6)

Structural changes are queued even on a clear win — the human gate the user
locked in. `apply` returns the updated Attempt (verdict set) so the orchestrator
can record the cycle's outcome. Rollback uses the lineage pointer; no Spec is
ever deleted (archived, never deleted).

Human approval (`approve`) is the second path — besides AUTO_PROMOTE/ROLLBACK —
that may move `active`: a queued (PENDING_HUMAN) structural change the human
accepts becomes the incumbent; a rejection (`reject`) leaves the incumbent alone
and archives the candidate as a recorded dead end.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

import archforge.models as m
from archforge.stores.attempt_store import AttemptStore
from archforge.stores.spec_store import SpecStore


class Action(str, Enum):
    """What the Gatekeeper decided to do with this candidate."""

    AUTO_PROMOTE = "auto_promote"     # small win -> active = candidate
    QUEUE_HUMAN = "queue_human"       # structural win -> Approval Queue
    DISCARD = "discard"               # loss / unrunnable / cross-rubric
    ROLLBACK = "rollback"             # promoted-then-regressed (E6)


class Decision(BaseModel):
    """The Gatekeeper's verdict on a candidate. ``Gatekeeper.apply_decision`` executes it."""

    model_config = ConfigDict(extra="allow")

    action: Action
    attempt_id: str | None = None      # the candidate Attempt this decides
    reason: str                         # human-readable, surfaced to the report
    margin: float = 0.0                # candidate_mean - incumbent_mean (signed)
    by_rule: str = ""                  # which rule fired ("auto_promote", "rollback", ...)

    def __repr__(self) -> str:  # pragma: no cover  (debug aid)
        return (f"Decision(action={self.action.value}, margin={self.margin:+.3f}, "
                f"rule={self.by_rule}, reason={self.reason!r})")


class _AttemptNotFoundError(KeyError):
    """The candidate Attempt given to the Gatekeeper was never persisted."""


class Gatekeeper:
    """The lone enforcer of promotion + rollback.

    Constructed once per run with the (SpecStore, AttemptStore) it mutates and a
    `Thresholds` (τ, δ, ε). Stateless across calls otherwise — the incumbent + a
    candidate's SuiteRun are passed in `decide`.
    """

    def __init__(
        self,
        spec_store: SpecStore,
        attempt_store: AttemptStore,
        *,
        thresholds: m.Thresholds | None = None,
    ) -> None:
        self._specs = spec_store
        self._attempts = attempt_store
        self._th = thresholds or m.Thresholds()

    # ----------------------------------------------------------------- decide
    def decide(
        self,
        attempt_id: str,
        candidate: Any,                 # SuiteRun (typed loosely to avoid an import cycle)
        incumbent: "m.SuiteRun | None" = None,
    ) -> Decision:
        """Decide the fate of a candidate `SuiteRun` vs the incumbent `SuiteRun`.

        Both are `SuiteRun` objects (archforge.suite). `incumbent=None` means
        "no incumbent yet" — only AUTO_PROMOTE-like forward progress is valid,
        but a structural candidate still queues for human review.
        """
        # NOTE: `m.SuiteRun` lives in archforge.suite, not archforge.models, so we
        # accept it as `Any` here and pull attributes defensively (duck-typed).

        if candidate.unrunnable:
            return _decision(Action.DISCARD, attempt_id,
                            "candidate suite was unrunnable (>ε tasks crashed/unscored)",
                            rule="unrunnable")

        if incumbent is not None and not _same_geometry(incumbent, candidate):
            return _decision(Action.DISCARD, attempt_id,
                            "candidate vs incumbent differ in rubric or suite "
                            "(cross-geometry comparison blocked, I5)",
                            rule="cross_geometry")

        inc_mean = incumbent.mean if incumbent is not None else 0.0
        margin = candidate.mean - inc_mean
        cand_scope = self._candidate_scope(attempt_id)

        if margin < self._th.tau:
            return _decision(Action.DISCARD, attempt_id,
                            f"candidate did not clear margin τ={self._th.tau} "
                            f"(margin {margin:+.3f}); noise not promoted (E1)",
                            margin=margin, rule="below_margin")

        # win past τ
        if cand_scope is m.Scope.STRUCTURAL:
            return _decision(Action.QUEUE_HUMAN, attempt_id,
                            f"structural win by {margin:+.3f} >= τ but structural "
                            f"changes require human approval (I4)",
                            margin=margin, rule="structural_wins_queue")
        return _decision(Action.AUTO_PROMOTE, attempt_id,
                         f"small win by {margin:+.3f} >= τ={self._th.tau}; promoted",
                         margin=margin, rule="auto_promote")

    # ----------------------------------------------------------------- apply
    def apply_decision(self, decision: Decision) -> m.Attempt:
        """Execute a Decision against the stores and return the updated Attempt."""

        if decision.attempt_id is None:
            raise ValueError("cannot apply a Decision with no attempt_id")
        att = self._attempts.require(decision.attempt_id)

        if decision.action is Action.AUTO_PROMOTE:
            spec_id = att.candidate_spec_id
            self._specs.set_active(spec_id)
            verdict = m.Verdict.PROMOTED
        elif decision.action is Action.QUEUE_HUMAN:
            verdict = m.Verdict.PENDING_HUMAN
        elif decision.action is Action.DISCARD:
            verdict = m.Verdict.REJECTED
        elif decision.action is Action.ROLLBACK:
            self._rollback(att)
            verdict = m.Verdict.ROLLED_BACK
        else:  # pragma: no cover  (enum exhaustive)
            raise ValueError(f"unknown action {decision.action}")

        return self._attempts.set_verdict(decision.attempt_id, verdict)

    # ----------------------------------------------------------------- human gate
    def approve(self, attempt_id: str) -> m.Attempt:
        """Human approves a queued (PENDING_HUMAN) structural change (spec I4).

        This is the one path besides AUTO_PROMOTE/ROLLBACK that may move the
        `active` pointer — the human gate the user locked in: a structural win is
        never auto-promoted, only ever promoted through here. `active` -> the
        candidate, verdict -> PROMOTED. Idempotent for an already-promoted attempt
        (a no-op). Raises `ValueError` if the attempt is not queued, so the CLI
        cannot rewrite history (only decide what the gate queued).
        """

        att = self._attempts.require(attempt_id)
        if att.verdict is m.Verdict.PROMOTED:
            return att                       # already approved (idempotent)
        if att.verdict is not m.Verdict.PENDING_HUMAN:
            raise ValueError(
                f"cannot approve attempt {attempt_id}: verdict is "
                f"{att.verdict.value}; only PENDING_HUMAN (queued) changes can be approved"
            )
        self._specs.set_active(att.candidate_spec_id)
        return self._attempts.set_verdict(attempt_id, m.Verdict.PROMOTED)

    def reject(self, attempt_id: str, *, reason: str = "human-reject") -> m.Attempt:
        """Human rejects a queued structural change: verdict -> REJECTED, active unchanged.

        The incumbent is left alone. The candidate Spec is archived (never
        deleted) with `reason="human-reject"` so the dead end is recorded for
        lineage/dedup queries. Idempotent for an already-rejected attempt.
        Raises `ValueError` if the attempt is not queued.
        """

        att = self._attempts.require(attempt_id)
        if att.verdict is m.Verdict.REJECTED:
            return att
        if att.verdict is not m.Verdict.PENDING_HUMAN:
            raise ValueError(
                f"cannot reject attempt {attempt_id}: verdict is "
                f"{att.verdict.value}; only PENDING_HUMAN (queued) changes can be rejected"
            )
        self._specs.archive(att.candidate_spec_id, reason=reason)
        return self._attempts.set_verdict(attempt_id, m.Verdict.REJECTED)

    # ----------------------------------------------------------------- rollback (E6)
    def rollback(
        self,
        attempt_id: str,
        *,
        regressed_mean: float,
        pre_promotion_mean: float,
    ) -> Decision:
        """Decide+apply a rollback after a promoted Spec regressed (E6).

        Rolls back iff `pre_promotion_mean - regressed_mean >= δ` on the standard
        suite (same rubric+suite, I5). Otherwise the regression is within the
        noise floor and the incumbent is left alone. Returns the Decision (which
        has already been applied if action is ROLLBACK).
        """

        drop = pre_promotion_mean - regressed_mean
        # Idempotency: a repeated regression check on an already-rolled-back
        # attempt must not downgrade its verdict or move the active pointer
        # again. Return a benign DISCARD decision without touching the stores.
        current_verdict = self._attempts.require(attempt_id).verdict
        if current_verdict is m.Verdict.ROLLED_BACK:
            return _decision(Action.DISCARD, attempt_id,
                             "regression check on an already-rolled-back attempt; "
                             "no further action",
                             margin=-drop, rule="already_rolled_back")

        if drop < self._th.delta:
            # Regression is within the δ noise floor: the promotion stands, the
            # incumbent (the promoted Spec) is left in place, and the attempt's
            # verdict is NOT mutated — a real small-won, the dip is just noise.
            return _decision(Action.DISCARD, attempt_id,
                            f"regression {drop:+.3f} < δ={self._th.delta}; within the "
                            f"noise floor, incumbent left alone",
                            margin=-drop, rule="regression_within_floor")

        d = _decision(Action.ROLLBACK, attempt_id,
                      f"regression {drop:+.3f} >= δ={self._th.delta}; rolling back to "
                      f"parent (pointer swap, archived, never deleted)",
                      margin=-drop, rule="rollback")
        att = self._attempts.require(attempt_id)
        self._rollback(att)
        self._attempts.set_verdict(attempt_id, m.Verdict.ROLLED_BACK)
        return d

    def _rollback(self, att: m.Attempt) -> None:
        """Active -> parent; candidate archived. Pointer swap, never deleted (E6/I3)."""

        spec = self._specs.get(att.candidate_spec_id)
        parent_id = spec.parent_spec_id
        if parent_id is None:
            # root incumbent regressed: nothing to roll back to; leave pointer, mark rejected
            self._specs.archive(att.candidate_spec_id, reason="rollback_rootless")
            return
        if not self._specs.has(parent_id):
            raise _AttemptNotFoundError(
                f"rollback target parent {parent_id} not in store"
            )
        self._specs.set_active(parent_id)         # pointer swap — the incumbent reverts
        self._specs.archive(att.candidate_spec_id, reason="rollback")

    # ----------------------------------------------------------------- helpers
    def _candidate_scope(self, attempt_id: str) -> m.Scope:
        att = self._attempts.require(attempt_id)
        return att.change.scope


# --------------------------------------------------------------------------- #
# helpers (module-private)
# --------------------------------------------------------------------------- #


def _decision(
    action: Action, attempt_id: str | None, reason: str,
    *, margin: float = 0.0, rule: str = "",
) -> Decision:
    return Decision(action=action, attempt_id=attempt_id, reason=reason,
                   margin=margin, by_rule=rule)


def _same_geometry(inc: Any, cand: Any) -> bool:
    """Invariant I5: comparisons only valid under identical (rubric, suite)."""

    return (getattr(inc, "rubric_id", None) == getattr(cand, "rubric_id", None)
            and getattr(inc, "suite_id", None) == getattr(cand, "suite_id", None))


__all__ = ["Action", "Decision", "Gatekeeper"]
