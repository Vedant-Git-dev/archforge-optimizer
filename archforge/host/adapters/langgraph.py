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
    consulted by the node's LLM call at call time — the usual pattern is a
    module-level config table the SDK client wrapper reads.

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
from archforge import userconfig as ucfg
from archforge.host.adapters.base import BaseHostAdapter
from archforge.host.adapters.helpers import KnobVote, estimate_tokens, run_id
from archforge.host.adapters.inject import (
    NodeLocator, SdkInjector, apply_knob_settings,
)
from archforge.lint import lint
from archforge.middleware import TracingMiddleware

# Re-export the OTel cooperative seam so a MAS's ``build_graph`` imports
# ``wrapped`` from here (the adapter module it already touches) rather than
# reaching into ``archforge.otel`` directly. ``otel`` is import-lazy (it pulls
# NO OpenTelemetry at its top level), so merely naming it here keeps
# ``import archforge`` / this module OTel-free; its OTel imports run lazily only
# when ``_ensure_tracer`` / ``wrapped`` first execute. ``project`` /
# ``_shed_to_budget`` are imported lazily INSIDE the methods that use them
# (below), so they never load unless a budget is actually set.
from archforge.otel import wrapped as _otel_wrapped  # noqa: E402 (lazy module; no eager OTel)


def wrapped(name: str, fn: Callable) -> Callable:
    """Re-exported cooperative OTel seam: wraps ``fn`` to open an
    ``archforge.node`` span so auto-instrumented SDK calls nest as children
    (correlation by parent-link). A MAS's ``build_graph`` routes node fns
    through ``add(name, fn) -> g.add_node(name, wrapped(name, fn))`` so the id
    string is authored ONCE. No-op passthrough when OTel is unavailable.

    See ``archforge.otel.wrapped`` for the implementation; this thin re-export
    is the seam the MAS imports from (the adapter it already touches)."""
    return _otel_wrapped(name, fn)

# The named LLM knobs flow through the call-time injector, NOT state. These are
# the Knobs model's defined fields to exclude when overlaying state-routed extras.
# Lifted into ``archforge.models._NAMED_KNOBS`` as the single source of truth
# (shared with ``archforge.diff``); referenced here as ``m._NAMED_KNOBS``.
_NAMED_KNOBS = m._NAMED_KNOBS


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


# --------------------------------------------------------------------------- #
# NodeIdMap — keyed id source (single-authoring, fail-fast on rename)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NodeIdMap:
    """Keyed lookup over a compiled graph's real node names — the authority for
    the node id string, so it is written ONCE (in ``build_graph``'s
    ``add(name, fn)`` line) and sourced everywhere else via ``_GID["name"]``.

    ``__getitem__`` validates against the compiled graph's node names (drops
    langgraph's built-in ``__start__`` / ``__end__``) and raises ``KeyError``
    listing the declared names on a miss → a rename in ``build_graph`` fails
    LOUDLY at app import (fail-fast per node), never a silently-mislabeled
    run. Introspection is limited to node NAMES — the most stable surface — one
    time at wiring; "describe, don't introspect" still protects the version-
    fragile internals/routing the rest of the adapter avoids.
    """

    _names: tuple[str, ...]

    def __getitem__(self, name: str) -> str:
        if name not in self._names:
            raise KeyError(
                f"unknown node {name!r}; declared: {list(self._names)}"
            )
        return name            # canonical id == the add_node name

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


def node_ids(graph: Any) -> NodeIdMap:
    """Build a :class:`NodeIdMap` from a compiled langgraph graph's node names
    (drops ``__start__`` / ``__end__``). Instantiating this validates the MAS's
    ``_GID["..."]`` lookups at import time. ``graph`` is the value returned by
    ``graph_factory()`` (i.e. a compiled ``StateGraph``)."""
    names = tuple(n for n in graph.nodes if not str(n).startswith("__"))
    return NodeIdMap(names)


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
      settings_getter : ``() -> settings singleton`` (default ``None``). With
                        ``knob_to_settings``, EXTRA knobs that the host reads
                        from a settings object (not from state) become tunable
                        with zero host edits: the runnable snapshots the mapped
                        fields, mutates them in place for the run, and restores
                        them after. Example: ``lambda: mypkg.config.settings``.
      knob_to_settings: ``{knob_name: (section_attr, field_attr)}`` — e.g.
                        ``{"max_k": ("retrieval", "max_k")}``.
      node_modules    : ``{module_name: node_id}`` attribution overlay for the
                        zero-touch injector (rarely needed; the map is derived
                        from the compiled graph's node callables).

    Behavior hooks (override for the MAS; defaults are sensible):
      initialize_state(task_input) -> dict      — the graph's initial_state.
      summarize(node_id, partial, merged) -> str — what each Step records as
        ``response_out`` (the Judge scores this; assertions read it).
      apply_llm_config(node_id, vote) -> None   — deliver the live named LLM
        knobs. The DEFAULT records the vote for the zero-touch injector
        (``inject.SdkInjector``), which patches the host's LLM SDKs at the
        boundary for the run and rewrites the request per node — the host
        needs NO call-time config seam of its own. Override only to route the
        vote through an explicit seam instead (the injector then stays off).
        Called UP-FRONT per run (graph.stream emits a node only after it runs,
        so this pre-population is the only viable shape).
      reset_llm_config() -> None               — clear before a run (default:
        clears the recorded votes; override to clear an explicit seam).
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
    settings_getter: Callable[[], Any] | None = None
    knob_to_settings: dict[str, tuple[str, str]] = {}
    node_modules: dict[str, str] = {}

    # Tracing budget (the OTel trace-projection gate):
    #   None       -> read the ``DEFAULT_TRACE_TOTAL_BUDGET_TOK`` tunable (the
    #                 default = None = today's lossy ``summarize()`` path ⇒ the
    #                 existing suite stays green). Set this None too for parity.
    #   int        -> cap the per-Judge-prompt total projected tokens; auto-instr
    #                 SDK calls become OTel GenAI spans, :meth:`_record` projects a
    #                 BOUNDED slice (LLM nodes only) into each ``Step`` and sheds
    #                 the largest-evidence-chunk steps first, keeping the final-
    #                 answer step. A per-app override lets one MAS opt in without
    #                 touching config.
    trace_total_budget: int | None = None

    def __init__(self) -> None:
        # Per-run vote store for the zero-touch injector (the default
        # ``apply_llm_config`` records here; ``SdkInjector`` reads it) and the
        # first-observed system prompt per node (the ``cfg_decay`` base for a
        # later ``prompt_edit`` when ``base_prompts`` left the node empty).
        self._pending_votes: dict[str, KnobVote] = {}
        self.captured_prompts: dict[str, str] = {}

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
        """Default: record the vote for the zero-touch SDK injector (see the
        class docstring). Override to route the vote through the MAS's own
        call-time seam instead — overriding disables the injector for the app
        (the runnable detects the override and never patches the SDKs)."""
        self._pending_votes[node_id] = vote

    def reset_llm_config(self) -> None:
        """Default: clear the recorded votes (called at run start)."""
        self._pending_votes.clear()

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
        # Copy (not alias) so prompt-capture merges during runs never rewrite
        # the app's declared base_prompts.
        self.base_prompts = dict(app.base_prompts)
        # Once-per-process OTel setup: build ArchForge's TracerProvider + the
        # in-memory span buffer and register the repo-declared auto-instrumentors.
        # Idempotent; a no-op when OTel isn't installed (the rich path degrades
        # gracefully to ``summarize()``). Done here — before any
        # ``graph.stream`` — so instrumentors patch the SDKs BEFORE the first call.
        # Lazy import keeps ``import archforge`` OTel-free; touching the module
        # only when a host adapter is constructed.
        from archforge.otel import _ensure_tracer
        _ensure_tracer()

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

        # The trace-projection gate. ``None`` (default) = today's lossy
        # ``summarize()`` path ⇒ identical ``Step`` s ⇒ existing suite green;
        # an int = rich per-step Steps. A per-app ``trace_total_budget`` overrides
        # the ``DEFAULT_TRACE_TOTAL_BUDGET_TOK`` tunable (mirrors the lazy
        # ``ucfg.get`` pattern at engine.py:146). Resolved once per run.
        budget = self._app.trace_total_budget
        if budget is None:
            budget = ucfg.get("DEFAULT_TRACE_TOTAL_BUDGET_TOK")
        self._budget = int(budget) if isinstance(budget, int) and budget > 0 else None
        # rich-path bookkeeping: keep a ref to each Step built during the run so
        # the post-loop shed can mutate them IN PLACE (middleware holds the same
        # refs → ``end_run``'s ``list(self._steps)`` sees trimmed values). Cleared
        # per run; only populated when a budget is set.
        self._run_steps: list[m.Step] = []
        self._final_answer_node: str | None = None
        # Real provider usage per node, filled by the SDK injector (one entry
        # per observed call, in call order) and drained by ``_record``: each
        # step consumes everything recorded since the previous step for that
        # node, which is exactly the calls made during that node's execution.
        # Stays empty for explicit-seam apps → the estimate path is unchanged.
        self._usage: dict[str, list[int]] = {}

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

        # 2. initial_state, then overlay the live Spec's EXTRA knobs. Two routes:
        #    state-routed (``knob_to_state``) into initial_state, and
        #    settings-routed (``knob_to_settings``) applied to the host's
        #    settings singleton in place and restored after the run — the
        #    zero-touch path for knobs the host reads from config, not state.
        initial: dict[str, Any] = dict(self._app.initialize_state(task.input))
        settings_items: list[tuple[str, Any]] = []
        for nd in self._app.nodes:
            live = live_nodes.get(nd.node_id, nd)
            extras = live.knobs.model_dump(exclude=_NAMED_KNOBS)
            for kname, kval in extras.items():
                if kval is None:
                    continue
                state_key = self._app.knob_to_state.get(kname)
                if state_key is not None:
                    initial[state_key] = kval
                elif kname in self._app.knob_to_settings:
                    settings_items.append((kname, kval))
                # else: an extra with no mapping at all — host-owned.

        # 3. Drive the real graph: one Step per emitted node. Wall-clock-around-
        #    transition: when a new node's
        #    chunk arrives, the *previous* node is done — record it with the
        #    transition time as its latency; the last node is recorded post-loop.
        # 3. Zero-touch plumbing for the run (both no-ops unless configured):
        #    a. settings-routed knobs: mutate the host's settings singleton in
        #       place, restore in ``finally`` (runs are sequential → race-free).
        #    b. the SDK injector: active ONLY when the app uses the DEFAULT
        #       ``apply_llm_config`` (votes recorded in ``_pending_votes``);
        #       an explicit-seam override disables it, so existing adapters
        #       (and their tests) never get their SDKs patched.
        restore_settings: Callable[[], None] = lambda: None
        injector: SdkInjector | None = None
        merged: dict[str, Any] = dict(initial)
        final_output: str | None = None
        current: str | None = None
        current_partial: dict[str, Any] = {}
        node_started_at = 0.0
        try:
            if settings_items and self._app.settings_getter is not None:
                restore_settings = apply_knob_settings(
                    self._app.settings_getter(),
                    self._app.knob_to_settings,
                    settings_items,
                )
            uses_builtin_injector = (
                type(self._app).apply_llm_config is LangGraphApp.apply_llm_config
            )
            if uses_builtin_injector and self._app._pending_votes:
                locator = NodeLocator.from_graph(
                    self._graph, extra=dict(self._app.node_modules)
                )
                injector = SdkInjector(
                    locator, self._app._pending_votes, self._app.captured_prompts,
                    usage=self._usage,
                )
                injector.install()
            for chunk in self._graph.stream(initial, stream_mode="updates"):
                # Graphs with no parallel branch emit one node per superstep.
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
                    # Note the node that write-sets the final-output key: the
                    # post-loop shed protects it (the answer-bearing step always
                    # reaches the Judge). ``partial`` is this node's state diff,
                    # so its carrying ``final_output_key`` marks the producer.
                    if (
                        self._budget is not None
                        and self._app.final_output_key in partial
                    ):
                        self._final_answer_node = node_name
            if current is not None:
                now = time.perf_counter()
                self._record(current, current_partial, now - node_started_at, merged)

            # Shed to budget ON THE SUCCESS PATH only: mutate the shared Step
            # objects in place so the Judge's total ingested text ≤ budget, the
            # largest-evidence-chunk steps trims first, the final-answer step is
            # protected. Skipped on the except path → a mid-run crash leaves ran
            # Steps at their per-node caps (E4 holds).
            if self._budget is not None and self._run_steps:
                from archforge.otel import _shed_to_budget
                _shed_to_budget(self._run_steps, self._final_answer_node, self._budget)

            final_output = merged.get(self._app.final_output_key)
            if final_output is not None and not isinstance(final_output, str):
                final_output = str(final_output)
            return self._mw.end_run(final_output, ok=True, error=None)
        except Exception as exc:  # noqa: BLE001 — flush a partial trace (E4)
            return self._mw.end_run(final_output, ok=False, error=repr(exc))
        finally:
            if injector is not None:
                injector.uninstall()
            restore_settings()
            # Prompt capture: fill GAPS in the adapter's base_prompts with the
            # system prompts observed on the wire, so a later ``prompt_edit``
            # has a real base even when the app left ``base_prompts`` empty.
            # Declared prompts win (setdefault never overwrites).
            for _nid, _prompt in self._app.captured_prompts.items():
                self._adapter.base_prompts.setdefault(_nid, _prompt)

    # ---- one step ---------------------------------------------------------- #
    def _record(
        self, node_name: str, partial: dict, elapsed_s: float, merged: dict
    ) -> None:
        kind = self._app.kind_of(node_name)
        # The gated OTel trace projection. No budget (the default) → today's lossy
        # ``summarize()`` path byte-identical (parity — the existing suite's
        # substring/shape assertions hold). Budget set → ask ``otel.project`` for
        # the real per-step prompt/completion captured as OTel GenAI spans under
        # the ``archforge.node`` parent; ``None`` (no spans / capture off /
        # non-LLM / OTel absent) → degrade gracefully back to ``summarize()``.
        if self._budget is None:
            resp = self._app.summarize(node_name, partial, merged)
            prompt_in = ""            # LangGraph nodes read the query from state,
            # Kind-aware cost: LLM nodes cost tokens; retriever/rule/symbolic ~0
            # (their real cost is wall-clock — `latency_ms`).
            tokens = self._tokens_for(node_name, resp, kind)
        else:
            from archforge.otel import project as _project
            proj = _project(node_name, kind, partial, merged)
            if proj is None:
                # graceful degrade: OTel absent / no child spans / non-LLM kind
                resp = self._app.summarize(node_name, partial, merged)
                prompt_in = ""
                tokens = self._tokens_for(node_name, resp, kind)
            else:
                resp = proj.response_out
                prompt_in = proj.prompt_in
                tokens = proj.tokens
        step = m.Step(
            node_id=node_name,
            prompt_in=prompt_in,
            response_out=resp,
            perf=m.StepPerf(
                tokens=tokens,
                latency_ms=round(elapsed_s * 1000.0, 3),
            ),
        )
        if self._budget is not None:
            # keep a ref so the post-loop shed can mutate this same object in
            # place (middleware holds the same ref → end_run sees trimmed values).
            self._run_steps.append(step)
        self._mw._record_step(step)

    def _tokens_for(self, node_name: str, resp: str, kind: m.NodeKind) -> int:
        """Step cost: real provider usage when the injector observed this node's
        calls, else the len//4 estimate. Draining (sum + clear) works with
        loops: a node's step consumes exactly the calls made since its previous
        step. Non-LLM kinds stay ~0 (their cost is wall-clock)."""
        if kind is not m.NodeKind.LLM:
            return 0
        rec = self._usage.get(node_name)
        if rec:
            total = sum(rec)
            rec.clear()
            return total
        return estimate_tokens(resp)


__all__ = [
    "Nd", "EdgeSpec", "LangGraphApp",
    "LangGraphHostAdapter", "LangGraphRunnable",
    "node_ids", "NodeIdMap", "wrapped",      # the cooperative OTel seam + keyed id source
    "export_spec_sidecar", "load_spec_sidecar",
    "build_optimized_envelope", "export_optimized", "load_optimized",
    "ENVELOPE_SCHEMA",
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
# ``node_id.knob_name → config field`` mapping (e.g. the host's settings loader
# reads this and overlays onto its config objects). The
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


# --------------------------------------------------------------------------- #
# Unified deployment config — the optimized.json envelope (improvement #4)
# --------------------------------------------------------------------------- #
#
# The bare sidecar (``export_spec_sidecar``) ships only ``{node_id:{knob}}`` —
# enough to overlay the MAS's config consts, but it carries none of the *why*.
# The envelope is the single file production loads to apply the winner AND audit
# it: the knobs (today's sidecar body, nested under ``knobs``) plus the lineage,
# the Gatekeeper decision, and the scores that justified the promote. One artifact
# → Tier-2 deploy: the MAS reads ``envelope["knobs"]`` (a one-line change to its
# sidecar consumer), the rest is human-readable provenance.
# Rollback is deleting the file; the diff is what Git shows between promotes.
#
# `knobs` is NESTED UNDER the key named `knobs` (vs the sidecar's flat top level)
# so the envelope has room for `schema`/`spec_id`/`scores`/… alongside it. The
# per-MAS consumer changes ONE line: `load_spec_sidecar(path)` →
# `load_optimized(path)["knobs"]`. That consumer edit belongs to the host
# project; this module ships the producer.

ENVELOPE_SCHEMA = "archforge.optimized/v1"


def _score_dims(run: Any) -> list[dict[str, Any]]:
    """Reduce a SuiteRun's per-RunScore sub-rubrics to a stable dim summary.

    A ``SuiteRun`` (``archforge.suite``) carries a list of ``m.RunScore`` (one per
    scored repeat), each with a ``rubric_scores: {dim: score}`` map. For the
    envelope we want the mean per rubric dimension across the candidate's scored
    repeats — the same dimensions the Gatekeeper's mean aggregates. Tolerant: any
    duck-typed ``SuiteRun``-like (the engine passes the real one; tests may pass
    a lighter object) with optional ``scores``/``aggregate``.
    """
    scores = getattr(run, "scores", None) or []
    sums: dict[str, float] = {}
    n: dict[str, int] = {}
    for rs in scores:
        dims = getattr(rs, "rubric_scores", None) or {}
        for dim, val in dims.items():
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            sums[dim] = sums.get(dim, 0.0) + v
            n[dim] = n.get(dim, 0) + 1
    return [{"dim": d, "mean": round(sums[d] / n[d], 4)} for d in sums]


def build_optimized_envelope(
    spec: m.Spec,
    *,
    parent: m.Spec | None,
    promoted_at_cycle: int,
    decision: "Decision | None" = None,
    cand_run: "Any | None" = None,
    inc_run: "Any | None" = None,
) -> dict[str, Any]:
    """The unified deployment config — knobs + provenance (improvement #4).

    Shape (``archforge.optimized/v1``)::

        {"schema": "archforge.optimized/v1",
         "spec_id": <candidate id>, "parent_spec_id": <parent id | None>,
         "promoted_at_cycle": <int>,
         "decision": {"action": "auto_promote", "rule": <by_rule>,
                      "margin": <±float>, "reason": <str>} | None,
         "scores": {"mean": <cand mean>, "incumbent_mean": <inc mean | None>,
                    "dims": [{"dim","mean"}], "rubric_id": <str|None>,
                    "suite_id": <str|None>, "tokens": <int>, "latency_ms": <float>} | None,
         "knobs": {node_id: {knob: value}, ...}}   # == today's sidecar body

    ``knobs`` is byte-identical to ``export_spec_sidecar``'s body (same
    ``_node_knob_projection``, same empty-node drop), so a consumer already reading
    the bare sidecar switches by reading ``env["knobs"]`` instead of the top level.
    ``decision``/``scores`` are ``None`` when the caller omits them (an embedder
    building an envelope outside a promote context still gets the knobs + lineage).
    """
    # `knobs` is exactly the sidecar body — the consumer-compat seam.
    knobs = {n.node_id: _node_knob_projection(n) for n in spec.nodes if n.node_id}
    knobs = {nid: kv for nid, kv in knobs.items() if kv}

    env: dict[str, Any] = {
        "schema": ENVELOPE_SCHEMA,
        "spec_id": spec.spec_id or spec.compute_spec_id(),
        "parent_spec_id": getattr(parent, "spec_id", None) or
                          (parent.compute_spec_id() if parent is not None else None),
        "promoted_at_cycle": promoted_at_cycle,
        "decision": None,
        "scores": None,
        "knobs": knobs,
    }
    if decision is not None:
        env["decision"] = {
            "action": getattr(decision.action, "value", str(decision.action)),
            "rule": decision.by_rule,
            "margin": decision.margin,
            "reason": decision.reason,
        }
    if cand_run is not None:
        inc_mean = getattr(inc_run, "mean", None) if inc_run is not None else None
        env["scores"] = {
            "mean": getattr(cand_run, "mean", None),
            "incumbent_mean": inc_mean,
            "dims": _score_dims(cand_run),
            "rubric_id": getattr(cand_run, "rubric_id", None),
            "suite_id": getattr(cand_run, "suite_id", None),
            "tokens": getattr(cand_run, "tokens", 0),
            "latency_ms": getattr(cand_run, "latency_ms", 0.0),
        }
    return env


def export_optimized(
    spec: m.Spec, path: str | Path, *,
    parent: m.Spec | None = None, promoted_at_cycle: int = 0,
    decision: "Decision | None" = None, cand_run: "Any | None" = None,
    inc_run: "Any | None" = None,
) -> dict[str, Any]:
    """Write ``build_optimized_envelope(...)`` to ``path`` as ``indent=2`` JSON.

    Returns the envelope dict (for in-process use / testing). Stable: nodes in
    Spec order, knobs in insertion order; ``indent=2 sort_keys=False`` so the
    between-promote Git diff is readable. The CLI's ``on_deploy`` wires this to
    ``<root>/optimized.json`` on every AUTO_PROMOTE — Tier-2 deploy auto-synced.
    """
    env = build_optimized_envelope(
        spec, parent=parent, promoted_at_cycle=promoted_at_cycle,
        decision=decision, cand_run=cand_run, inc_run=inc_run,
    )
    Path(path).write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    return env


def load_optimized(path: str | Path) -> dict[str, Any]:
    """Read an envelope written by ``export_optimized`` (the unified deploy file).

    The per-MAS consumer reads ``env["knobs"]`` (the sidecar body) and overlays it
    onto its config; the rest is provenance for human review. (Bare sidecar
    consumers keep ``load_spec_sidecar``; this is the envelope's loader.)"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
