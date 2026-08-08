"""Core data model for ArchForge (spec §3).

All records are immutable once written. The Pipeline Spec is the artifact being
optimized; everything else (traces, scores, attempts) is append-only evidence.

`spec_id` is a content hash: the canonical JSON of a Spec (nodes+edges, sorted)
determines its identity, so two Specs that are structurally equal share an id.
This is the basis for content-addressed storage (I2 immutability) and for
rollback being a pointer swap.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from archforge.config import SPEC_ID_HASH_LEN
from archforge import userconfig as ucfg

# --------------------------------------------------------------------------- #
# Enums / literal types
# --------------------------------------------------------------------------- #


class EdgeType(str, Enum):
    """Execution-graph edge kinds (spec §3 Pipeline Spec)."""

    SEQUENCE = "sequence"  # A then B
    FANOUT = "fanout"  # A spawns several, each runs
    JOIN = "join"  # several converge into one
    CONDITIONAL = "conditional"  # branch/loop, gated on `gate`


class NodeKind(str, Enum):
    """What BACKS a node — its tunable surface (the optimizer's non-LLM extension).

    `LLM` nodes are tuned via `system_prompt`/`model`/named knobs as today. Non-LLM
    kinds carry their params as EXTRA knobs (a retriever's `top_k`, a rule's
    `threshold`, a tool's `endpoint`); the Architect may edit only the keys a node's
    `Knobs.tunable` lists (plus the always-editable named knobs). See `Knobs`.
    """

    LLM = "llm"  # an LLM agent (the historical default)
    RULE = "rule"  # heuristic / scorer / classifier threshold — no prompt, no model
    RETRIEVER = "retriever"  # vector / search context fetch
    TOOL = "tool"  # external API / tool call
    SYMBOLIC = "symbolic"  # fixed code transform, no tunable internals


class ChangeKind(str, Enum):
    """What the Architect changed on a candidate Spec (spec §3 Attempt)."""

    PROMPT_EDIT = "prompt_edit"
    KNOB = "knob"
    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    REWIRE = "rewire"
    MODEL_SWAP = "model_swap"


class Scope(str, Enum):
    """Change granularity — drives the hybrid gate (spec §3, I4).

    `SMALL`    = prompt/knob only          -> may auto-promote on a win
    `STRUCTURAL` = roster/graph/model swap -> always queues for human
    """

    SMALL = "small"
    STRUCTURAL = "structural"


class Verdict(str, Enum):
    """Outcome of an Attempt (spec §3 Attempt)."""

    PROMOTED = "promoted"
    REJECTED = "rejected"
    PENDING_HUMAN = "pending_human"
    ROLLED_BACK = "rolled_back"


class SpecStatus(str, Enum):
    INCUMBENT = "incumbent"
    CANDIDATE = "candidate"
    ARCHIVED = "archived"


# Which change kinds are structural (the rest are small). Single source of truth
# for the scope tag and for the Gatekeeper's scope->gate rule (I4).
STRUCTURAL_KINDS: frozenset[ChangeKind] = frozenset(
    {ChangeKind.ADD_NODE, ChangeKind.REMOVE_NODE, ChangeKind.REWIRE, ChangeKind.MODEL_SWAP}
)

# The Knobs model's named fields, as a frozenset of field names. Single source of
# truth shared by the host adapter (excludes these when overlaying state-routed
# extras + when projecting a Spec's live knobs to a deploy sidecar) and by the
# spec-diff module (classifies a knob field as a named knob vs a kind-specific
# extra). Includes ``tunable`` — it is Knobs metadata (which extras are editable),
# NOT an editable knob value itself, so it is excluded from both the state overlay
# and the projected knob map. Kept here (core) so ``archforge/diff.py`` need not
# import the langgraph adapter. NB: ``archforge/mutate.py`` keeps its OWN local
# ``_NAMED_KNOBS`` (without ``tunable``) for the editable-knob allowlist — a
# different set for a different purpose; do not conflate.
_NAMED_KNOBS: frozenset[str] = frozenset(
    {"temperature", "retries", "max_tokens", "tunable"}
)


# --------------------------------------------------------------------------- #
# Pipeline Spec — the optimized artifact
# --------------------------------------------------------------------------- #


class Knobs(BaseModel):
    """Tunable per-agent parameters.

    The three NAMED knobs (temperature/retries/max_tokens) plus an OPEN bag of
    per-kind extras (a retriever's `top_k`, a rule's `threshold`, a tool's
    `endpoint`...) carried via ``extra="allow"``. ``tunable`` lists the EXTRA
    keys the Architect is PERMITTED to edit on this node; the named knobs are
    always editable (back-compat for every legacy LLM Spec, whose ``tunable``
    defaults to ``()``). An empty ``tunable`` means the extras are host-owned /
    manual-only (a human author may still set them in the Spec JSON).
    """

    model_config = ConfigDict(extra="allow")

    temperature: float | None = None
    retries: int | None = None
    max_tokens: int | None = None
    tunable: tuple[str, ...] = Field(default_factory=tuple)


class Node(BaseModel):
    """One agent in the pipeline graph.

    ``kind`` defaults to ``LLM`` so every Spec predating non-LLM nodes loads
    unchanged. ``system_prompt``/``model`` are OPTIONAL (empty for non-LLM
    kinds); the linter requires them non-empty for ``LLM`` nodes — a lint rule,
    not a model constraint, so the closed ``Node`` model stays closed.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str
    role: str
    kind: NodeKind = NodeKind.LLM
    system_prompt: str = ""
    model: str = ""
    knobs: Knobs = Field(default_factory=Knobs)
    tools: list[str] = Field(default_factory=list)  # tool_ids, resolved by the host


class Edge(BaseModel):
    """A directed edge in the execution DAG. `gate` only used for CONDITIONAL.

    The Python attribute is `from_` (``from`` is a reserved keyword); the JSON
    alias is `from`. `populate_by_name=True` lets us construct with `from_=` and
    still serialise to/from `from`.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    type: EdgeType
    gate: str | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        # default to the `from` alias so persisted JSON reads naturally
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)


class Spec(BaseModel):
    """A versioned pipeline: agents + wiring. Immutable; identified by content+lineage.

    `spec_id` is a content hash over the *canonical* (sorted, alias-free) JSON of
    `(nodes, edges, parent_spec_id)`.

      * Content (nodes, edges) is included → "content-addressed."
      * Lineage (parent_spec_id) is included → each lineage node is a *distinct*
        stored object; two Specs with identical structure but different parents get
        different ids (so the store never faces a same-content-different-parent
        file collision). This mirrors a git commit hash, which counts its parent.
      * Runtime role (`status`) and write-time (`created_at`) are *excluded*: a
        Spec's identity is stable as it is promoted or rolled back (only the store's
        `active` pointer and `archived` set move), so immutability (I2) holds.
    """

    model_config = ConfigDict(extra="forbid")

    # `spec_id` is excluded from content (would be circular); assigned at commit.
    spec_id: str | None = None
    parent_spec_id: str | None = None
    status: SpecStatus = SpecStatus.CANDIDATE
    created_at: str | None = None  # set by SpecStore on commit
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    def compute_spec_id(self) -> str:
        """Return the content-addressed id (sha256 hex of canonical content+lineage)."""

        return _content_hash(self._canonical_payload())

    def _canonical_payload(self) -> dict[str, Any]:
        """Nodes + edges + parent, sorted deterministically, with `from` alias restored.

        The new defaulted node fields (`kind=llm`, `tunable=()`) are STRIPPED when
        at their default so a legacy Spec — predating non-LLM nodes — hashes
        byte-identically to its pre-feature id (no store migration, no spec_id
        drift). Explicit non-default `kind`/`tunable` DO appear: a different
        artifact gets a different id, as content-addressing requires.
        """

        def _node_dump(n: Node) -> dict[str, Any]:
            d = n.model_dump(mode="json")
            if n.kind is NodeKind.LLM:
                d.pop("kind", None)
            if not n.knobs.tunable:
                d.pop("tunable", None)
            return d

        nodes = sorted((_node_dump(n) for n in self.nodes), key=lambda n: n["node_id"])
        edges = sorted(
            (e.model_dump(mode="json", by_alias=True) for e in self.edges),
            key=lambda e: (e["from"], e["to"], e["type"]),
        )
        return {"nodes": nodes, "edges": edges, "parent_spec_id": self.parent_spec_id}


# --------------------------------------------------------------------------- #
# Trace — per run, append-only (spec §3)
# --------------------------------------------------------------------------- #


class ToolCall(BaseModel):
    """A tool invocation recorded during a step."""

    model_config = ConfigDict(extra="forbid")

    tool_id: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    ok: bool = True


class StepPerf(BaseModel):
    """Perf/cost signals for a step (E3 cost, E4 errors)."""

    model_config = ConfigDict(extra="allow")

    tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    error: str | None = None


class Step(BaseModel):
    """One agent step: prompt in -> response out, plus tool calls + timing."""

    model_config = ConfigDict(extra="allow")

    node_id: str
    prompt_in: str
    response_out: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    timing: dict[str, Any] = Field(default_factory=dict)
    perf: StepPerf = Field(default_factory=StepPerf)


class Trace(BaseModel):
    """A single run of a task under a Spec."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    spec_id: str
    task_id: str
    steps: list[Step] = Field(default_factory=list)
    final_output: str | None = None
    ok: bool = True
    error: str | None = None


# --------------------------------------------------------------------------- #
# RunScore — judge output (spec §3)
# --------------------------------------------------------------------------- #


class JudgeMeta(BaseModel):
    """Provenance for a judge verdict (E2 rubric drift)."""

    model_config = ConfigDict(extra="allow")

    model: str
    rubric_id: str


class StepScore(BaseModel):
    """A single step's score against each sub-rubric (credit-assignment input)."""

    model_config = ConfigDict(extra="allow")

    node_id: str
    sub_rubrics: dict[str, float] = Field(default_factory=dict)
    note: str | None = None


class RunScore(BaseModel):
    """Judge output for one run: aggregate + per-step rubric breakdown.

    The per-step breakdown (StepScore[]) is the input to credit assignment in
    the Architect. `confidence` lets low-confidence verdicts be down-weighted
    (E1c). `rubric_id` blocks cross-rubric comparison (invariant I5).
    """

    model_config = ConfigDict(extra="allow")

    run_id: str
    spec_id: str
    task_id: str
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    aggregate: float = 0.0
    confidence: float = 1.0
    judge_meta: JudgeMeta
    step_scores: list[StepScore] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Attempt — the optimizer's memory (spec §3)
# --------------------------------------------------------------------------- #


class Change(BaseModel):
    """One mutation of a Spec: what changed, where, why, and at what scope."""

    model_config = ConfigDict(extra="forbid")

    kind: ChangeKind
    target: str  # node_id, edge key, or model name depending on kind
    diff: str  # human-readable description of the edit
    rationale: str
    scope: Scope

    @classmethod
    def for_kind(cls, kind: ChangeKind, target: str, diff: str, rationale: str) -> Change:
        """Auto-tag scope from the change kind (spec: structural kinds gate)."""

        scope = Scope.STRUCTURAL if kind in STRUCTURAL_KINDS else Scope.SMALL
        return cls(kind=kind, target=target, diff=diff, rationale=rationale, scope=scope)


class SuiteResult(BaseModel):
    """Aggregated suite scores for a candidate vs its incumbent.

    Persisted onto the candidate's `Attempt` after the Gatekeeper decides, so the
    human-facing surfaces (`status`, `report`, Approval Queue) can show the real
    delta + cost without re-running the suite. `tokens` is the candidate-side
    marginal cost only — the incumbent baseline is a shared, cacheable cost that
    is not attributed to any single attempt (avoids double-counting across cycles).
    """

    model_config = ConfigDict(extra="allow")

    mean: float
    margin_vs_incumbent: float
    repeats: int
    unrunnable: bool = False
    rubric_id: str
    suite_id: str
    tokens: int = 0


class Attempt(BaseModel):
    """One P-E-C cycle's record: a candidate Spec diff + its suite verdict."""

    model_config = ConfigDict(extra="allow")

    attempt_id: str | None = None  # assigned on persist
    candidate_spec_id: str
    parent_spec_id: str
    change: Change
    suite_result: SuiteResult | None = None
    verdict: Verdict


# --------------------------------------------------------------------------- #
# Shared config (spec §6 thresholds) — referenced by later phases.
# Kept here as the single source of truth for τ/δ/R/ε/K/budgets.
# --------------------------------------------------------------------------- #


class Thresholds(BaseModel):
    """Tunable knobs that govern noise/safety behaviour (spec §6)."""

    model_config = ConfigDict(extra="forbid")

    # Tunable defaults resolve lazily from archforge.userconfig (the active config
    # — the project's .archforge/archforge.py, or the in-memory sane template under
    # the pytest gate) — so this module imports cleanly BEFORE `init` has run. Each
    # read is one ucfg.get() dict lookup after a one-time resolve+cache. See
    # archforge/userconfig.py.
    tau: float = Field(default_factory=lambda: ucfg.get("DEFAULT_TAU"))          # promotion margin
    delta: float = Field(default_factory=lambda: ucfg.get("DEFAULT_DELTA"))     # rollback floor (E6/I3)
    repeats: int = Field(default_factory=lambda: ucfg.get("DEFAULT_REPEATS"))    # R repeats/task (E3)
    max_repeats: int = Field(default_factory=lambda: ucfg.get("MAX_REPEATS"))    # cap on adaptive R
    unrunnable_frac: float = Field(default_factory=lambda: ucfg.get("DEFAULT_UNRUNNABLE_FRAC"))  # ε (E4)
    plateau_cycles: int = Field(default_factory=lambda: ucfg.get("DEFAULT_PLATEAU_CYCLES"))     # K (E8)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _content_hash(payload: dict[str, Any]) -> str:
    """Stable sha256 hex over JSON with sorted keys + separators (canonical)."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:SPEC_ID_HASH_LEN]


def scope_for_kind(kind: ChangeKind) -> Scope:
    """Public scope classifier — used by the Architect and Gatekeeper."""

    return Scope.STRUCTURAL if kind in STRUCTURAL_KINDS else Scope.SMALL


# Re-export the literal-backed aliases for convenience in type hints elsewhere.
SpecStatusLiteral = Literal["incumbent", "candidate", "archived"]
VerdictLiteral = Literal["promoted", "rejected", "pending_human", "rolled_back"]
