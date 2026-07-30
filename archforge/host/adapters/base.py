"""The ArchForge adapter kit — a reusable implementation of the host seam.

`HostMAS` / `Agent` / `Runnable` (in ``host/base.py``) are the *contract*; this
module is a *reusable implementation of it* that owns the scaffolding every MAS
adapter repeats:

  * one ``Agent`` per Spec node;
  * the run loop: ``begin_run`` → ordered walk → ``end_run``, with a monotonic
    run-id (unique across R-repeats), partial-trace flush on any crash (E4);
  * the **content-decouple**: score the agent's *content*, thread its
    *plumbing* onward via an ``AgentResponse`` extra (``extra="allow"`` lets a
    ``thread`` field ride without a model change);
  * the **config-decay** (phase-2 rule): thread model/temperature/max_tokens
    always (base run == agent default → behaviour-preserving) and system_prompt
    only when mutated.

The genuinely-MAS-specific bits are exposed as a few imperative hooks an adapter
overrides when its MAS needs them — *not* a declarative graph interpreter (so
hard data-shaping — file-stage handoffs, multi-input joins — is just imperative
code in a hook, never a drop off a declarative cliff).
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

import archforge.models as m
from archforge.host.adapters.helpers import cfg_decay, estimate_tokens, run_id, topo_order
from archforge.host.base import AgentResponse, HostMAS, Runnable, Task
from archforge.middleware import TracingMiddleware


# --------------------------------------------------------------------------- #
# Run context — what the loop carries between nodes
# --------------------------------------------------------------------------- #


@dataclass
class RunContext:
    """The state threaded through one run, advanced by the loop and read by
    the adapter's ``resolve_prompt``.

    ``last_content`` is what each step's response *recorded* (what the Judge
    scores); ``last_thread`` is what each step *threads onward* (plumbing: a
    file path, a ref… — usually the same as content for text-in/text-out MASes,
    different for file-staged ones like Lumina). ``per_node`` lets a join node
    recall a sibling threaded earlier (the report 3-way join reads sibling
    gap/cross paths here).
    """

    last_content: str
    last_thread: str | None = None
    per_node: dict[str, Any] = field(default_factory=dict)

    def advance(self, node_id: str, resp: AgentResponse) -> None:
        self.last_content = resp.text
        # `thread` rides as an AgentResponse extra (extra="allow"); fall back to
        # the scored content if the adapter didn't set one (text-in/text-out).
        self.last_thread = getattr(resp, "thread", None) or resp.text
        self.per_node[node_id] = self.last_thread


# --------------------------------------------------------------------------- #
# CallResult — the content-decouple made first-class
# --------------------------------------------------------------------------- #


@dataclass
class CallResult:
    """What an agent's ``call`` returns: ``thread`` flows to the next node
    (plumbing), ``content`` is what the trace records and the Judge scores.

    Promoted from the Lumina adapter's ``RunResult`` so every kit-made adapter
    gets content-decouple for free instead of re-inventing it.
    """

    thread: str | None
    content: str


# --------------------------------------------------------------------------- #
# BaseAgent — implements `Agent`; owns invoke; author overrides `call`
# --------------------------------------------------------------------------- #


class BaseAgent:
    """One node's executor. The kit owns ``invoke`` (the content-decouple +
    config-decay + perf measurement); the adapter fills ``call`` — the node's
    real work — and is handed a config map it forwards to its agent call.
    """

    def __init__(self, node: m.Node, adapter: "BaseHostAdapter") -> None:
        self._node = node
        self._adapter = adapter

    @property
    def node_id(self) -> str:
        return self._node.node_id

    @property
    def role(self) -> str:
        return self._node.role

    def call(self, prompt: str, vote: "KnobVote") -> CallResult:  # noqa: ARG002
        """The one abstract hook: run the node, return (thread, content).

        ``vote`` is the DECAYED live config (a `KnobVote`: model / temperature /
        max_tokens / retries / system_prompt, each ``None`` meaning "use your
        default"). The adapter binds exactly the keys its agent accepts — the
        kit owns the decay logic, not the agent's signature. Subclasses override.
        """
        raise NotImplementedError(f"{type(self).__name__}.call() not implemented")

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        model: str | None,
        knobs: m.Knobs | None,
        tools: list[str] | None,
    ) -> AgentResponse:
        """Kit-owned: fulfils the `Agent` protocol. Applies config-decay, calls
        the node's real work, packs an AgentResponse that records *content* and
        carries *plumbing* onward via the `thread` extra.
        """
        vote = self._adapter.cfg_for(self.node_id, system_prompt, model, knobs, tools)
        t0 = time.perf_counter()
        cr = self.call(prompt, vote)
        latency = round((time.perf_counter() - t0) * 1000.0, 3)
        # `text` is what the trace records + the Judge scores (content); the
        # `thread` extra is what flows to the next node (plumbing).
        return AgentResponse(
            text=cr.content,
            tool_calls=[],
            thread=cr.thread if cr.thread is not None else cr.content,
            perf=m.StepPerf(tokens=estimate_tokens(cr.content), latency_ms=latency),
        )


def _estimate(text: str) -> int:  # kept as a thin alias for any external callers
    return estimate_tokens(text)


# --------------------------------------------------------------------------- #
# BaseHostAdapter — implements `HostMAS`; owns instantiate; delegates the run
# --------------------------------------------------------------------------- #


class BaseHostAdapter:
    """A `HostMAS` that builds a `BasePipeline` from a Spec.

    An adapter subclasses this and overrides the hooks that are non-default for
    its MAS:

      * ``make_agent``  — wrap one framework agent per Spec node (the per-node
        factory that chooses the right ``BaseAgent`` subclass).
      * ``execution_order``  — default: topological from edges; override for a
        fixed sequence (Lumina's 7-step).
      * ``resolve_prompt``  — default: the last step's content; override for a
        file path when a node needs upstream plumbing (joins, file-stage handoffs).
      * ``stage_context``  — default: none; override for a per-run scratch area
        (Lumina: a temp dir + chdir, released on exit).

    The run loop, content-decouple, config-decay, partial-trace flush, perf —
    none of those are the adapter's concern; the kit owns them.
    """

    #: The seeded/default prompts, so `cfg_decay` can tell a *mutated* prompt
    #: (an Architect prompt_edit) from the default and only thread the former.
    #: Subclasses populate this (often from their base Spec).
    base_prompts: dict[str, str] = {}

    # ---- hooks (override when the MAS needs non-default behaviour) -------- #

    def make_agent(self, node: m.Node) -> BaseAgent:
        """Pick the `BaseAgent` subclass for `node` (one per Spec node)."""
        raise NotImplementedError("make_agent must return a BaseAgent for each node")

    def execution_order(self, spec: m.Spec) -> list[str]:
        """Order to run nodes in. Default: topological from the Spec's edges."""
        return topo_order(spec)

    def resolve_prompt(self, node_id: str, ctx: RunContext) -> str:
        """What `node_id` receives as its prompt. Default: the last step's
        *content* (text-in/text-out). Override to pass a file path / sibling
        ref when the node needs upstream plumbing (joins, file-stage handoffs)."""
        return ctx.last_content

    def stage_context(self):
        """A context manager wrapping one run. Default: none. Override for a
        per-run scratch area — e.g. a temp dir + chdir (Lumina) — that this
        releases on exit. The loop's `finally` cleanup lives here."""
        return nullcontext()

    # ---- kit-internal seam (rarely overridden) --------------------------- #

    def cfg_for(
        self, node_id: str, system_prompt: str | None, model: str | None,
        knobs: m.Knobs | None, tools: list[str] | None,
    ) -> "KnobVote":
        """Live-config → the DECAYED values the adapter's ``call`` forwards to
        its agent. Defaults to the phase-2 `cfg_decay` rule; returns a
        ``KnobVote`` so each adapter binds EXACTLY the keys its agent accepts
        (the kit owns the decay logic, not the agent's signature). Override only
        if the MAS's config-decay differs."""
        return cfg_decay(node_id, system_prompt, model, knobs, tools,
                         base_prompts=self.base_prompts)

    # ---- kit-owned: the existing HostMAS.instantiate contract, un-touched -- #

    def instantiate(self, spec: m.Spec, middleware: TracingMiddleware) -> Runnable:
        agents: dict[str, BaseAgent] = {}
        for node in spec.nodes:
            agents[node.node_id] = self._agent_for(node)
        return BasePipeline(spec, middleware, agents, self)

    def _agent_for(self, node: m.Node) -> BaseAgent:
        """One agent per Spec node; an unknown node_id (an `add_node` we can't
        run) becomes a _RaisingAgent, surfacing as an errored run that scoring
        rejects — structural changes stay human-gated (I4)."""
        try:
            return self.make_agent(node)
        except NotImplementedError:
            return _RaisingAgent(node, self)


# --------------------------------------------------------------------------- #
# BasePipeline — implements `Runnable`; owns the run loop
# --------------------------------------------------------------------------- #


class BasePipeline:
    """A Spec rendered as an executable pipeline by the kit's run loop.

    The loop only calls the adapter's hooks — it never inspects framework
    specifics — so the same loop ranges over a stateless pipeline MAS
    (text-in/text-out, default hooks) and a file-staged one (Lumina, override
    order/prompt/context). ``run`` returns a full `Trace` (ok or not); a mid-run
    crash flushes a partial trace with the Steps that ran (E4) — never lost.
    """

    def __init__(
        self,
        spec: m.Spec,
        middleware: TracingMiddleware,
        agents: dict[str, BaseAgent],
        adapter: BaseHostAdapter,
    ) -> None:
        self._spec = spec
        self._mw = middleware
        self._agents = agents
        self._adapter = adapter
        self._run_counter = 0

    def run(self, task: Task) -> m.Trace:
        sid = self._spec.spec_id or self._spec.compute_spec_id()
        rid = run_id(sid, task.task_id, self._run_counter)
        self._run_counter += 1
        self._mw.begin_run(rid, self._spec, task.task_id)

        present = {n.node_id for n in self._spec.nodes}
        ctx = RunContext(last_content=task.input)
        final_output: str | None = None
        try:
            with self._adapter.stage_context():
                for node_id in self._adapter.execution_order(self._spec):
                    if node_id not in present or node_id not in self._agents:
                        continue  # a remove_node mutation simply skips a step
                    prompt = self._adapter.resolve_prompt(node_id, ctx)
                    wrapped = self._mw.wrap(self._agents[node_id])
                    resp = wrapped.invoke(prompt)
                    ctx.advance(node_id, resp)
                    final_output = resp.text
            return self._mw.end_run(final_output, ok=True, error=None)
        except Exception as exc:  # noqa: BLE001 — flush a partial trace (E4)
            return self._mw.end_run(final_output, ok=False, error=repr(exc))


# --------------------------------------------------------------------------- #
# Unknown-node agent — structural mutations that can't yet run
# --------------------------------------------------------------------------- #


class _RaisingAgent(BaseAgent):
    """A node the adapter can't execute (an `add_node` it has no entry-point
    for). `invoke` runs (so the trace records the step) but `call` raises,
    surfacing as ok=False — scoring rejects it, structural changes stay
    human-gated (I4)."""

    def call(self, prompt: str, vote: "KnobVote") -> CallResult:  # noqa: ARG002
        raise RuntimeError(
            f"Adapter {type(self._adapter).__name__} can't execute node "
            f"{self.node_id!r}: not in its node map (add_node mutations aren't "
            f"runnable yet; the run errors and is rejected by scoring)."
        )


__all__ = [
    "RunContext", "CallResult", "BaseAgent", "BaseHostAdapter",
    "BasePipeline", "run_id", "topo_order", "cfg_decay",
]
