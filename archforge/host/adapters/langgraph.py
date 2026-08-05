"""ArchForge adapter for LangGraph apps — drives the REAL compiled graph.

LangGraph (https://github.com/langchain-ai/langgraph) runs a state machine: a
``state -> partial_state`` node per superstep, routed by conditional edges and
runtime loops, driven via ``graph.stream(stream_mode="updates")``. Each stream
chunk is ``{node_name: partial_state_diff}``.

This adapter maps the real graph onto ArchForge's P-E-C loop WITHOUT flattening
it to the kit's flat topological walk (`BasePipeline`) — that would erase the
conditional routing and the runtime loop, the very behaviors worth tuning.
Instead it:

  * drives ``graph.stream`` and records one ArchForge ``Step`` per emitted node
    (surgery-free via ``mw._record_step``, like Lumina — ``archforge/`` untouched);
  * injects the live Spec's knobs into the graph's ``initial_state`` so a
    candidate's knob edits reach the REAL execution (the only viable shape:
    ``graph.stream`` reveals a node only *after* it runs, so per-upcoming-node
    contextvars are impossible — populate up-front, consult at call time);
  * kind-aware cost: LLM nodes cost tokens (`estimate_tokens`); retriever/rule/
    symbolic nodes cost ~0 tokens (their real cost is wall-clock `latency_ms`),
    matching the mixed-non-LLM convention (bounded via ``--max-wall-ms-per-cycle``).

Knob routing (the adapter's one design rule — "describe, don't introspect"):
  * EXTRAS (kind-specific params: a retriever's ``top_k``, a rule node's
    ``threshold``) flow through STATE — the author maps each to a state key via
    ``knob_to_state``, and the node reads ``state.get(key, default)``. ArchForge
    live = inject into ``initial_state``; deploy = edit the node's consts.
  * NAMED LLM knobs (``model``/``temperature``/``max_tokens``/``system_prompt``)
    flow through a CALL-TIME injector the author wires (`apply_llm_config`),
    consulted by the node's LLM call at call time — AEDE's pattern (a module-level
    ``_NODE_CONFIG`` the groq client reads).

This module is **langgraph-FREE**: it imports no langgraph types. The langgraph
dependency enters only when a concrete app's ``graph_factory`` builds the real
graph. So ``import archforge`` stays framework-free and this module loads even
without langgraph installed (you just can't *run* a real app until it is).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import archforge.models as m
from archforge.host.adapters.base import BaseHostAdapter
from archforge.host.adapters.helpers import KnobVote, estimate_tokens, run_id
from archforge.lint import lint
from archforge.middleware import TracingMiddleware

# The named LLM knobs flow through the call-time injector, NOT state. These are
# the Knobs model's defined fields to exclude when overlaying state-routed extras.
_NAMED_KNOBS = frozenset({"temperature", "retries", "max_tokens", "tunable"})


# --------------------------------------------------------------------------- #
# Declarative "description" surface — the MAS author fills this in
# --------------------------------------------------------------------------- #


@dataclass
class Nd:
    """One node in the MAS's static roster. Fields mirror ``m.Node``.

    ``knobs`` carries the SEEDED (incumbent) values — the extras that flow through
    state, plus the named LLM knobs that flow through the call-time injector.
    ``tunable`` (a field on ``Knobs``) names which EXTRAS the Architect may edit;
    the named knobs are always editable (back-compat).
    """
    node_id: str
    role: str
    kind: m.NodeKind = m.NodeKind.LLM
    knobs: m.Knobs = field(default_factory=m.Knobs)
    model: str = ""
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)


@dataclass
class EdgeSpec:
    """One static edge. CONDITIONAL edges carry a non-empty ``gate`` (a label the
    linter checks for truthiness; the Forge does NOT evaluate it — the real graph
    does). Runtime loops are documented in ``LangGraphApp.runtime_loops`` but
    OMITTED from ``edges``: encoding a back-edge would trip the linter's ``cycle``
    rule, and the real graph drives the loop anyway (the adapter sees each loop
    iteration as an emitted node)."""
    from_: str
    to: str
    kind: m.EdgeType = m.EdgeType.SEQUENCE
    gate: str | None = None


class LangGraphApp:
    """The declarative description of one LangGraph MAS — "describe, don't
    introspect." Subclass it, set the class attributes, and override the two
    behavior hooks. A ``LangGraphHostAdapter`` wraps an instance; the CLI's
    ``--adapter module:Class`` instantiates the adapter, which holds the app.

    Class attributes the author sets:
      graph_factory   : ``() -> compiled graph`` (builds the real state machine;
                        may import langgraph). Called once per ``instantiate``.
      nodes           : ``list[Nd]`` — the static roster (kinds + seeded knobs).
      edges           : ``list[EdgeSpec]`` — the STATIC wiring (no runtime loops).
      knob_to_state   : ``dict[str,str]`` — maps an EXTRA knob name to the state
                        key the node reads. Named LLM knobs are NOT here (they
                        flow through ``apply_llm_config``).
      runtime_loops   : ``list[(from, to)]`` — documented runtime back-edges;
                        omitted from ``edges`` (informational — NOT enforced
                        against the compiled graph, whose internals are
                        version-fragile).
      final_output_key: state key holding the run's answer (default ``"answer"``).
      base_prompts    : ``{node_id: seeded_prompt}`` for config-decay (default
                        ``{}`` — prompts live as node-code literals for most
                        LangGraph apps, so the seeded prompt is empty and a
                        ``prompt_edit`` rides through).

    Behavior hooks (override for the MAS; defaults are sensible):
      initialize_state(task_input) -> dict      — the graph's initial_state.
      summarize(node_id, partial, merged) -> str — what each Step records as
        ``response_out`` (the Judge scores this; assertions read it).
      apply_llm_config(node_id, vote) -> None   — push the live named LLM knobs
        into whatever the node's LLM call consults (a module-level registry).
        Default noop; override iff the MAS has LLM nodes whose model/temp you
        want tunable. Called UP-FRONT per run (graph.stream emits a node only
        after it runs, so this pre-population is the only viable shape).
      reset_llm_config() -> None               — clear it before a run (default
        noop).
    """
    # required class attributes (the author MUST set these on the subclass):
    graph_factory: Callable[[], Any]
    nodes: list[Nd]

    # optional (sensible defaults):
    edges: list[EdgeSpec] = []
    knob_to_state: dict[str, str] = {}
    runtime_loops: list[tuple[str, str]] = []
    final_output_key: str = "answer"
    base_prompts: dict[str, str] = {}

    # ---- behavior hooks (override me) -------------------------------------- #
    def initialize_state(self, task_input: str) -> dict[str, Any]:
        f = getattr(type(self), "state_factory", None)
        if f is None:
            raise NotImplementedError(
                "override LangGraphApp.initialize_state(task_input) (or set the "
                "class attribute `state_factory: Callable[[str], dict]`)."
            )
        return f(task_input)

    def summarize(self, node_id: str, partial: dict, merged: dict) -> str:
        """Default: a deterministic JSON snapshot of the node's state diff
        (bounded). Override to emit what the Judge should score per node."""
        blob = json.dumps(partial, sort_keys=True, default=str)
        return blob[:500]

    def apply_llm_config(self, node_id: str, vote: KnobVote) -> None:
        """Push the live named LLM knobs (model/temp/max_tokens/system_prompt)
        into the node's LLM-call site. Default noop — override iff the MAS has
        LLM nodes whose model/temp you want tunable."""
        return None

    def reset_llm_config(self) -> None:
        """Clear the call-time injector before a run (called at run start)."""
        return None

    # ---- derived (the adapter reads these) --------------------------------- #
    def kind_of(self, node_id: str) -> m.NodeKind:
        for nd in self.nodes:
            if nd.node_id == node_id:
                return nd.kind
        return m.NodeKind.LLM   # unknown node -> treat as LLM (costs tokens)

    def llm_node_ids(self) -> list[str]:
        return [nd.node_id for nd in self.nodes if nd.kind is m.NodeKind.LLM]

    def build_spec(self) -> m.Spec:
        """Build the bootstrap (incumbent) ``m.Spec`` from this description and
        lint it. A malformed description breaks loudly at adapter construction
        (mirrors ``python -m archforge lint``)."""
        mnodes = [
            m.Node(node_id=nd.node_id, role=nd.role, kind=nd.kind,
                   system_prompt=nd.system_prompt, model=nd.model,
                   knobs=nd.knobs, tools=list(nd.tools))
            for nd in self.nodes
        ]
        medges = [
            m.Edge(from_=e.from_, to=e.to, type=e.kind, gate=e.gate)
            for e in self.edges
        ]
        spec = m.Spec(nodes=mnodes, edges=medges)
        errs = lint(spec)
        assert not errs, (
            "LangGraph app Spec failed lint: "
            f"{[f'{e.code}@{e.location}: {e.message}' for e in errs]}"
        )
        return spec


# --------------------------------------------------------------------------- #
# The adapter + runnable
# --------------------------------------------------------------------------- #


class LangGraphHostAdapter(BaseHostAdapter):
    """A ``HostMAS`` for a LangGraph app. Reuses the kit's ``cfg_for``/``cfg_decay``
    (via ``base_prompts``) + ``run_id``/``estimate_tokens``; OVERRIDES
    ``instantiate`` to return a ``LangGraphRunnable`` that drives ``graph.stream``
    (NOT the kit's flat ``BasePipeline`` walk — that would flatten the
    conditional routing + runtime loop).

    ``make_agent``/``execution_order``/``resolve_prompt``/``stage_context`` are
    moot under the override (only ``BasePipeline.run`` calls them); leave the kit
    defaults — they're never invoked for a LangGraph host.
    """

    def __init__(self, app: LangGraphApp) -> None:
        self._app = app
        # build + validate the bootstrap Spec once (catches a malformed app early)
        self._spec = app.build_spec()
        self.base_prompts = app.base_prompts

    def app_spec(self) -> m.Spec:
        """The bootstrap (incumbent) Spec — for ``--seed`` / seeding the store."""
        return self._spec

    def instantiate(
        self, spec: m.Spec, middleware: TracingMiddleware
    ) -> "LangGraphRunnable":
        graph = self._app.graph_factory()
        return LangGraphRunnable(spec, middleware, self, graph)


class LangGraphRunnable:
    """Drives the real compiled graph and records one ``Step`` per emitted node.

    Holds the LIVE Spec (candidate or incumbent); its knobs flow into
    ``initial_state`` (extras) and the call-time LLM injector (named LLM knobs),
    both sourced from the live Spec — so a candidate's edits reach the real
    execution. The cost loop is: ``begin_run`` → drive ``graph.stream`` → one
    ``_record_step`` per emitted node → ``end_run`` (assembles + persists the
    ``Trace``; a mid-run crash flushes the Steps that ran, spec E4).
    """

    def __init__(
        self,
        spec: m.Spec,
        middleware: TracingMiddleware,
        adapter: LangGraphHostAdapter,
        graph: Any,
    ) -> None:
        self._spec = spec
        self._mw = middleware
        self._adapter = adapter
        self._app = adapter._app
        self._graph = graph
        self._run_counter = 0

    # ---- one run ----------------------------------------------------------- #
    def run(self, task: Any) -> m.Trace:
        sid = self._spec.spec_id or self._spec.compute_spec_id()
        rid = run_id(sid, task.task_id, self._run_counter)
        self._run_counter += 1
        self._mw.begin_run(rid, self._spec, task.task_id)

        # 1. Call-time LLM config: populate the injector UP-FRONT. graph.stream
        #    emits a node only *after* it runs, so pre-population is the only
        #    viable shape (a per-upcoming-node contextvar is impossible).
        #
        #    Read the LIVE Spec's node config (`self._spec` — candidate or
        #    incumbent), NOT the app's seeded `Nd` roster: a candidate's knob
        #    /model_swap/prompt_edit lives on the live Spec, so the seeded values
        #    would silently mask it. Fall back to `nd` for a node the live Spec
        #    dropped (a remove_node mutation the static graph still compiles) so
        #    the run doesn't crash — structural edits are human-gated regardless.
        live_nodes = {n.node_id: n for n in self._spec.nodes}
        self._app.reset_llm_config()
        for nd in self._app.nodes:
            if nd.kind is not m.NodeKind.LLM:
                continue
            live = live_nodes.get(nd.node_id, nd)
            vote = self._adapter.cfg_for(
                nd.node_id, live.system_prompt, live.model, live.knobs, live.tools
            )
            self._app.apply_llm_config(nd.node_id, vote)

        # 2. initial_state, then overlay the live Spec's EXTRA knobs (state-routed).
        initial: dict[str, Any] = dict(self._app.initialize_state(task.input))
        for nd in self._app.nodes:
            live = live_nodes.get(nd.node_id, nd)
            extras = live.knobs.model_dump(exclude=_NAMED_KNOBS)
            for kname, kval in extras.items():
                if kval is None:
                    continue
                state_key = self._app.knob_to_state.get(kname)
                if state_key is None:
                    continue            # an extra with no state mapping: host-owned
                initial[state_key] = kval

        # 3. Drive the real graph: one Step per emitted node. Wall-clock-around-
        #    transition (mirrors aede/runner.run_with_timings): when a new node's
        #    chunk arrives, the *previous* node is done — record it with the
        #    transition time as its latency; the last node is recorded post-loop.
        merged: dict[str, Any] = dict(initial)
        final_output: str | None = None
        current: str | None = None
        current_partial: dict[str, Any] = {}
        node_started_at = 0.0
        try:
            for chunk in self._graph.stream(initial, stream_mode="updates"):
                # AEDE-style graphs have no parallel branch ⇒ one node/superstep.
                # A multi-key chunk means a parallel branch the adapter doesn't
                # yet model — fail loudly with a clear message, not silently.
                assert len(chunk) == 1, (
                    "LangGraph adapter saw a multi-key stream chunk — a parallel "
                    f"branch it doesn't model yet: {list(chunk)}"
                )
                node_name, partial = next(iter(chunk.items()))
                now = time.perf_counter()
                if current is not None:
                    self._record(current, current_partial, now - node_started_at, merged)
                current = node_name
                current_partial = partial or {}
                node_started_at = now
                if partial:
                    merged.update(partial)
            if current is not None:
                now = time.perf_counter()
                self._record(current, current_partial, now - node_started_at, merged)

            final_output = merged.get(self._app.final_output_key)
            if final_output is not None and not isinstance(final_output, str):
                final_output = str(final_output)
            return self._mw.end_run(final_output, ok=True, error=None)
        except Exception as exc:  # noqa: BLE001 — flush a partial trace (E4)
            return self._mw.end_run(final_output, ok=False, error=repr(exc))

    # ---- one step ---------------------------------------------------------- #
    def _record(
        self, node_name: str, partial: dict, elapsed_s: float, merged: dict
    ) -> None:
        kind = self._app.kind_of(node_name)
        resp = self._app.summarize(node_name, partial, merged)
        # Kind-aware cost: LLM nodes cost tokens; retriever/rule/symbolic nodes
        # cost ~0 tokens (their real cost is wall-clock — `latency_ms`).
        tokens = estimate_tokens(resp) if kind is m.NodeKind.LLM else 0
        self._mw._record_step(
            m.Step(
                node_id=node_name,
                prompt_in="",  # LangGraph nodes read the query from state,
                                # not a threaded text prompt.
                response_out=resp,
                perf=m.StepPerf(
                    tokens=tokens,
                    latency_ms=round(elapsed_s * 1000.0, 3),
                ),
            )
        )


__all__ = [
    "Nd", "EdgeSpec", "LangGraphApp",
    "LangGraphHostAdapter", "LangGraphRunnable",
    "export_spec_sidecar", "load_spec_sidecar",
]


# --------------------------------------------------------------------------- #
# Tier-2 deploy — the winning Spec's knobs as a reviewable sidecar
# --------------------------------------------------------------------------- #
#
# On an AUTO_PROMOTE the engine fires an opt-in ``on_promote`` callback (see
# ``engine.Engine.__init__``) with the promoted Spec. A driver wires that to
# ``export_spec_sidecar`` to write a stable JSON sidecar the MAS overlays onto
# its config consts at startup — so the optimization reaches *production* (the
# ordinary `/optimize`-equivalent run, no ArchForge on the hot path). Rollback
# is deleting the file; the diff is what Git shows between promotes.
#
# General: needs only the Spec. The projection is faithful (the Spec's native
# knob vocabulary), so the per-MAS consumer owns the one-time
# ``node_id.knob_name → config field`` mapping (e.g. AEDE's
# ``Settings.from_env`` reads this and overlays onto ``PipelineConfig``). The
# exporter does NOT guess that mapping — "describe, don't introspect."

def _node_knob_projection(node: m.Node) -> dict[str, Any]:
    """One node → its live knobs as a flat ``{knob: value}`` the consumer overlays.

    Named LLM knobs (model/temperature/max_tokens/retries/system_prompt) are the
    universal vocabulary; extras (top_k/threshold/…) are the kind-specific ones.
    Only *real* values ship (None = unset; blank ``model``/``system_prompt`` =
    "use default"), so the sidecar carries just the knobs the Forge set.
    """
    out: dict[str, Any] = {}
    if node.model:
        out["model"] = node.model
    k = node.knobs
    if k is not None:
        if k.temperature is not None:
            out["temperature"] = k.temperature
        if k.max_tokens is not None:
            out["max_tokens"] = k.max_tokens
        if k.retries is not None:
            out["retries"] = k.retries
        # extras (everything that isn't a named knob / `tunable`)
        for kk, vv in k.model_dump().items():
            if kk in _NAMED_KNOBS:
                continue
            if vv is None:
                continue
            out[kk] = vv
    if node.system_prompt:
        out["system_prompt"] = node.system_prompt
    return out


def export_spec_sidecar(spec: m.Spec, path: str | Path) -> dict[str, dict[str, Any]]:
    """Write the winning Spec's knobs to ``path`` as ``{node_id: {knob: value}}``.

    A full snapshot (not just deltas) so the consumer can delete-then-replace on
    each promote and Git shows the between-promote diff. Returns the sidecar dict
    (for in-process use / testing). Stable: nodes in Spec order; knobs in
    insertion order; JSON ``indent=2 sort_keys=False`` so the diff is readable.
    """
    sidecar = {n.node_id: _node_knob_projection(n) for n in spec.nodes if n.node_id}
    # drop empty nodes — a node whose knobs all defaulted carries nothing to deploy
    sidecar = {nid: knobs for nid, knobs in sidecar.items() if knobs}
    Path(path).write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    return sidecar


def load_spec_sidecar(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read a sidecar written by ``export_spec_sidecar`` → ``{node_id: {knob}}``.

    The per-MAS consumer wraps this with its ``node.knob → config field`` map."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
