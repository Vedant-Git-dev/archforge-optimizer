"""The Architect — the P-E-C "Propose" step (spec §3, §4, §6; Phase 6).

One cycle, one change. `next_attempt(...)` returns an `ArchitectResult`:

  * status="proposed"   — a candidate Spec (incumbent + ONE mutation) + a `Change`
                           record ready for the orchestrator to evaluate/commit.
  * status="lint_rejected" — the candidate the Architect formed is structurally
                           invalid; the linter's reasons are returned. The
                           candidate NEVER reaches the SuiteRunner (spec E5).
  * status="plateau"    — no non-dedup-blocked proposal remains; the loop should
                           stop (spec E7/E8).

Boundaries that keep the optimizer safe:
  * The Architect is WRITE-FREE over stores. It READS AttemptStore for dedup
    (E7) but never appends an Attempt — the orchestrator (Phase 9) does that, so
    proposal and commitment stay cleanly separated.
  * `Change` is the persisted *metadata* (kind/target/diff/rationale/scope); the
    concrete edit lives in the candidate Spec returned alongside it, built with
    `archforge.mutate`. dedup keys off (parent, kind, target).
  * Structural change kinds auto-tag `scope=STRUCTURAL` (Phase 1 source of
    truth) — that tag, not the Architect, drives the hybrid gate (I4).

The real `Architect` spends ONE LLM call per cycle (the locked budget); the
`ScriptedArchitect` decides the proposal deterministically for tests.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

import archforge.models as m
from archforge.lint import LintError, lint
from archforge.llm.base import LLMClient, Message, Role
from archforge.mutate import apply_change
from archforge.stores.attempt_store import AttemptStore


# --------------------------------------------------------------------------- #
# Credit assignment — where did the run lose points?
# --------------------------------------------------------------------------- #


class CreditAssignment(BaseModel):
    """The node/route blamed for the incumbent's rubric loss (Phase 6 input)."""

    model_config = ConfigDict(extra="forbid")

    node_id: str | None = None        # the agent most responsible for the loss
    route: str | None = None           # "from->to" if a route is to blame
    sub_rubric: str | None = None      # the dimension lost
    severity: float = 0.0              # 1.0 - blamed score, in [0, 1]


def credit_assign(
    incumbent: m.Spec,
    worst_task_scores: Sequence[m.RunScore],
) -> CreditAssignment | None:
    """Localize the rubric loss to a node from per-step StepScores.

    Deterministic: across the worst task's repeat scores, average each node's
    sub-rubric scores; return the (node, dimension) with the lowest mean. Returns
    None if no step_scores were recorded (nothing to assign blame to).
    """

    if not worst_task_scores:
        return None
    # sum per (node, dimension) across repeats
    acc: dict[str, dict[str, list[float]]] = {}
    for score in worst_task_scores:
        for ss in score.step_scores:
            dims = acc.setdefault(ss.node_id, {})
            for dim, val in ss.sub_rubrics.items():
                dims.setdefault(dim, []).append(val)
    if not acc:
        return None

    # find the (node, dim) with the lowest mean
    worst_node, worst_dim, worst_mean = None, None, 1.0
    for node_id, dims in acc.items():
        for dim, vals in dims.items():
            mean = sum(vals) / len(vals)
            if mean < worst_mean:
                worst_mean, worst_node, worst_dim = mean, node_id, dim
    if worst_node is None:
        return None
    return CreditAssignment(node_id=worst_node, sub_rubric=worst_dim,
                            severity=1.0 - worst_mean)


# --------------------------------------------------------------------------- #
# Result + Proposal types
# --------------------------------------------------------------------------- #


ArchitectStatus = Literal["proposed", "lint_rejected", "plateau"]


class ArchitectProposal(BaseModel):
    """A concrete candidate (incumbent + one mutation) + its Change metadata."""

    model_config = ConfigDict(extra="forbid")

    candidate: m.Spec
    change: m.Change
    blame: CreditAssignment | None = None


class ArchitectResult(BaseModel):
    """The discriminated outcome of one P-E-C Propose step."""

    model_config = ConfigDict(extra="forbid")

    status: ArchitectStatus
    proposal: ArchitectProposal | None = None
    rejected_reasons: list[LintError] = Field(default_factory=list)
    note: str | None = None  # why we rejected/plateaued (for the report)

    @property
    def proposed(self) -> bool:
        return self.status == "proposed"


# --------------------------------------------------------------------------- #
# Architect protocol — real + scripted both satisfy this
# --------------------------------------------------------------------------- #


@runtime_checkable
class ArchitectProtocol(Protocol):
    def next_attempt(
        self,
        incumbent: m.Spec,
        worst_task_scores: Sequence[m.RunScore],
        *,
        attempt_store: AttemptStore,
    ) -> ArchitectResult: ...


# --------------------------------------------------------------------------- #
# Real Architect — one LLM call per cycle
# --------------------------------------------------------------------------- #


class Architect:
    """LLM-backed P-E-C proposer. One `complete()` call per `next_attempt`."""

    def __init__(self, llm: LLMClient, *, model: str) -> None:
        self._llm = llm
        self._model = model

    def next_attempt(
        self,
        incumbent: m.Spec,
        worst_task_scores: Sequence[m.RunScore],
        *,
        attempt_store: AttemptStore,
    ) -> ArchitectResult:
        blame = credit_assign(incumbent, worst_task_scores)
        proposal = self._propose(incumbent, blame)
        if proposal is None:
            return _plateau("architect produced no proposal")

        change = _change_from_payload(incumbent, proposal)
        if change is None:
            return _plateau("architect returned an unparseable change")

        # The LLM nests the edit under `payload`; mutate reads the inner fields.
        edit = proposal.get("payload")
        if not isinstance(edit, dict):
            edit = {k: v for k, v in proposal.items() if k not in {"kind", "target", "rationale"}}

        # structural mistakes that survive this far -> lint_rejected (E5)
        try:
            candidate = apply_change(incumbent, change, edit)
        except Exception as exc:  # noqa: BLE001 — mutate is defensive
            return _lint_rejected(_as_lint_errors(f"mutation failed: {exc}"))

        errors = lint(candidate)
        if errors:
            return _lint_rejected(errors)

        # dedup: skip a change already rejected/rolled-back on (parent, kind, target) (E7)
        blocked = attempt_store.blocking(
            incumbent.spec_id or incumbent.compute_spec_id(),
            change.kind.value, change.target,
        )
        if blocked:
            return _plateau(
                f"(parent={incumbent.spec_id}, kind={change.kind.value}, "
                f"target={change.target}) already tried: "
                f"{[a.attempt_id for a in blocked]}"
            )

        return ArchitectResult(
            status="proposed",
            proposal=ArchitectProposal(candidate=candidate, change=change, blame=blame),
        )

    # --------------------------------------------------------------- internals
    def _propose(
        self, incumbent: m.Spec, blame: CreditAssignment | None
    ) -> dict[str, Any] | None:
        """One LLM call → a structured change payload. None on no proposal.

        A valid-JSON-but-not-a-change response (no `kind`) is *not* a provider
        outage: the model answered, just without a usable change this cycle. We
        return the dict and let `_change_from_payload` turn the missing `kind`
        into a `None` → plateau (E7/E8). A true outage (non-JSON) raises from the
        client itself and propagates as `LLMError` for the loop to retry (E9).
        """
        messages = self._build_messages(incumbent, blame)
        completion = self._llm.complete(
            messages, model=self._model, temperature=0.2, response_format="json",
        )
        parsed = completion.parsed
        return parsed if isinstance(parsed, dict) else None

    def _build_messages(
        self, incumbent: m.Spec, blame: CreditAssignment | None
    ) -> list[Message]:
        blame_text = (
            f"Blame: node='{blame.node_id}', sub_rubric='{blame.sub_rubric}', "
            f"severity={blame.severity:.2f}.\n"
            if blame else "No clear blame node (runs near-perfect). Be conservative.\n"
        )
        return [
            Message(
                role=Role.SYSTEM,
                content=(
                    "You are the Architect of a multi-agent pipeline optimizer. "
                    "Propose EXACTLY ONE small change to improve the incumbent Spec. "
                    "Return JSON: {kind, target, rationale, payload}. `kind` ∈ "
                    "{prompt_edit, knob, add_node, remove_node, rewire, model_swap}. "
                    "`target` = node_id (or 'from,to' for rewire). `payload` carries "
                    "the edit: {prompt} | {knobs:{...}} | {model} | {node, wiring} | "
                    "{remove:[from,to], add:[from,to,type]}. Make the change address the blame."
                ),
            ),
            Message(
                role=Role.USER,
                content=(
                    f"INCUMBENT SPEC (nodes + edges):\n{_spec_summary(incumbent)}\n\n"
                    + blame_text
                    + "Return the one-change JSON."
                ),
            ),
        ]


# --------------------------------------------------------------------------- #
# ScriptedArchitect — deterministic, for tests
# --------------------------------------------------------------------------- #


class ScriptedArchitect:
    """Deterministic Architect whose proposals are scripted by the test.

    `propose(change, payload)` queues a proposal (popped per next_attempt). Use
    `force_lint_error()` to make the next proposal intentionally malformed (E5).
    `force_plateau()` makes the next call return a plateau. If the queue is empty
    when next_attempt is called, it plateaus.
    """

    def __init__(self) -> None:
        self._queue: list[tuple[m.Change, dict[str, Any]]] = []
        self._lint_error: bool = False
        self._force_plateau: bool = False
        self.calls: list[m.Spec] = []  # incumbents seen, for assertions

    def propose(self, change: m.Change, payload: dict[str, Any]) -> "ScriptedArchitect":
        self._queue.append((change, payload))
        return self

    def force_lint_error(self) -> "ScriptedArchitect":
        self._lint_error = True
        return self

    def force_plateau(self) -> "ScriptedArchitect":
        self._force_plateau = True
        return self

    def next_attempt(
        self,
        incumbent: m.Spec,
        worst_task_scores: Sequence[m.RunScore],
        *,
        attempt_store: AttemptStore,
    ) -> ArchitectResult:
        self.calls.append(incumbent)
        if self._force_plateau:
            self._force_plateau = False
            return _plateau("scripted plateau")
        if not self._queue:
            return _plateau("no scripted proposal left")

        change, payload = self._queue.pop(0)
        if self._lint_error:
            self._lint_error = False
            # build something the linter will reject (orphan edge to a ghost node)
            return _lint_rejected(_as_lint_errors("scripted malformed candidate"))

        blame = credit_assign(incumbent, worst_task_scores)
        try:
            candidate = apply_change(incumbent, change, payload)
        except Exception as exc:  # noqa: BLE001
            return _lint_rejected(_as_lint_errors(f"mutation failed: {exc}"))

        errors = lint(candidate)
        if errors:
            return _lint_rejected(errors)

        blocked = attempt_store.blocking(
            incumbent.spec_id or incumbent.compute_spec_id(),
            change.kind.value, change.target,
        )
        if blocked:
            return _plateau(
                f"scripted change already tried on (parent={incumbent.spec_id}, "
                f"kind={change.kind.value}, target={change.target})"
            )

        return ArchitectResult(
            status="proposed",
            proposal=ArchitectProposal(candidate=candidate, change=change, blame=blame),
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _change_from_payload(incumbent: m.Spec, payload: dict[str, Any]) -> m.Change | None:
    """Build a Change (metadata) from the LLM's structured proposal + rationale."""
    try:
        kind = m.ChangeKind(payload["kind"])
        target = str(payload["target"])
        rationale = str(payload.get("rationale", ""))
    except (KeyError, ValueError):
        return None
    diff = _describe_diff(kind, target, payload.get("payload", payload))
    return m.Change.for_kind(kind, target, diff, rationale)


def _describe_diff(kind: m.ChangeKind, target: str, payload: Any) -> str:
    if kind is m.ChangeKind.PROMPT_EDIT:
        return f"rewrite system_prompt of '{target}'"
    if kind is m.ChangeKind.KNOB:
        return f"tune knobs of '{target}'"
    if kind is m.ChangeKind.MODEL_SWAP:
        return f"swap model on '{target}'"
    if kind is m.ChangeKind.ADD_NODE:
        return f"add node '{target}'"
    if kind is m.ChangeKind.REMOVE_NODE:
        return f"remove node '{target}'"
    if kind is m.ChangeKind.REWIRE:
        return f"rewire edge at '{target}'"
    return f"{kind.value} on '{target}'"


def _spec_summary(spec: m.Spec) -> str:
    nodes = "\n".join(
        f"- {n.node_id} [{n.role}] model={n.model} prompt={n.system_prompt!r}"
        for n in spec.nodes
    )
    edges = "\n".join(f"- {e.from_} -> {e.to} ({e.type.value})" for e in spec.edges)
    return f"NODES:\n{nodes}\nEDGES:\n{edges}"


def _plateau(note: str) -> ArchitectResult:
    return ArchitectResult(status="plateau", note=note)


def _lint_rejected(reasons: list[LintError]) -> ArchitectResult:
    return ArchitectResult(status="lint_rejected", rejected_reasons=reasons,
                           note="candidate failed the Spec Linter")


def _as_lint_errors(text: str) -> list[LintError]:
    return [LintError(code="self_loop", message=text, location=None)]


__all__ = [
    "CreditAssignment", "credit_assign",
    "ArchitectProposal", "ArchitectResult", "ArchitectStatus",
    "ArchitectProtocol", "Architect", "ScriptedArchitect",
]
