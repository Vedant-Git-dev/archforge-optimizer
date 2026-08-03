"""TracingMiddleware — the observation + config seam (spec §3, §4).

This is the single point where ArchForge touches a host MAS's agents. Two jobs:

  1. **Observe** — every wrapped agent invocation is recorded as a `Step` and
     assembled into a `Trace` that is appended to the TraceStore. Observation is
     *passive*: the response that flows on to the next agent is returned
     unchanged (offline-only v1 — no mid-run rewriting).
  2. **Configure** — each wrapped agent pulls its (system_prompt, model, knobs,
     tools) from the *live Spec* (set via `report_active_spec` / `begin_run`),
     looked up by `node_id`. So swapping the live Spec reconfigures every agent
     on the next run without rebuilding the host — this is "evolving = swapping
     which Spec the host uses."

`begin_run`/`end_run` bound a single run's trace. If an agent raises mid-run,
the host still calls `end_run(ok=False, error=...)` so the Steps that completed
before the crash are flushed (spec E4 — partial trace retained).
"""

from __future__ import annotations

import time

from archforge.host.base import Agent, AgentResponse
from archforge.models import Node, Spec, Step, StepPerf, Trace
from archforge.stores.trace_store import TraceStore


class NoLiveSpecError(RuntimeError):
    """A wrapped agent was invoked before any Spec was made live (begin_run)."""


class UnknownNodeError(KeyError):
    """A wrapped agent's node_id is not in the live Spec (misconfiguration)."""


class TracingMiddleware:
    """Wraps host agents to record steps and apply the live Spec's config."""

    def __init__(self, trace_store: TraceStore) -> None:
        self._trace_store = trace_store
        self._live_spec: Spec | None = None
        # Per-run buffering (a Trace is appended once, at end_run):
        self._run_id: str | None = None
        self._spec_id: str | None = None
        self._task_id: str | None = None
        self._steps: list[Step] = []

    # --------------------------------------------------------------- live spec
    def report_active_spec(self, spec: Spec) -> None:
        """Set the Spec whose node configs wrapped agents will adopt.

        This is the "active Spec" of spec §3 ("instantiates each node from the
        active Spec"). For an eval run it is the candidate being scored; the
        orchestrator sets it (directly or via `begin_run`) before any invoke.
        """

        self._live_spec = spec

    def active_spec(self) -> Spec | None:
        return self._live_spec

    # --------------------------------------------------------------- run bounds
    def begin_run(self, run_id: str, spec: Spec, task_id: str) -> None:
        """Start a new run: adopt `spec` as live, reset the step buffer."""

        self.report_active_spec(spec)
        self._run_id = run_id
        self._spec_id = spec.compute_spec_id() if spec.spec_id is None else spec.spec_id
        self._task_id = task_id
        self._steps = []

    def end_run(
        self, final_output: str | None, ok: bool, error: str | None = None
    ) -> Trace:
        """Assemble the buffered Steps into a Trace, append it, and return it.

        Called by the host in a `finally` — so even a crashed run flushes its
        partial trace (spec E4). `ok=False` + `error` marks a failed run.
        """

        if self._run_id is None or self._spec_id is None or self._task_id is None:
            raise RuntimeError("end_run called without begin_run")
        trace = Trace(
            run_id=self._run_id,
            spec_id=self._spec_id,
            task_id=self._task_id,
            steps=list(self._steps),
            final_output=final_output,
            ok=ok,
            error=error,
        )
        self._trace_store.append(trace)
        # leave the live spec in place (the next begin_run replaces it)
        self._steps = []
        return trace

    # --------------------------------------------------------------- wrap
    def wrap(self, agent: Agent) -> "WrappedAgent":
        """Wrap `agent` so each invoke records a Step and applies live config."""

        return WrappedAgent(agent, self)

    # --------------------------------------------------------------- internals
    def _node_config(self, node_id: str) -> Node:
        """Look up a node by id in the live Spec; raise on misconfiguration."""

        if self._live_spec is None:
            raise NoLiveSpecError(
                f"agent '{node_id}' invoked with no live Spec (call begin_run first)"
            )
        for node in self._live_spec.nodes:
            if node.node_id == node_id:
                return node
        raise UnknownNodeError(node_id)

    def _record_step(self, step: Step) -> None:
        self._steps.append(step)


class WrappedAgent:
    """An `Agent` whose invocations are traced and config-injected by the middleware.

    Implements the `Agent` protocol: `node_id`/`role` delegate to the underlying
    agent; `invoke(prompt)` reads the live node config, calls the agent, records a
    Step, and returns the agent's response **unchanged** (offline-only v1).
    """

    def __init__(self, agent: Agent, middleware: TracingMiddleware) -> None:
        self._agent = agent
        self._mw = middleware

    @property
    def node_id(self) -> str:
        return self._agent.node_id

    @property
    def role(self) -> str:
        return self._agent.role

    @property
    def unwrapped(self) -> Agent:
        """The underlying agent (tests / introspection)."""

        return self._agent

    def invoke(self, prompt: str) -> AgentResponse:
        cfg = self._mw._node_config(self.node_id)
        start = time.perf_counter()
        response = self._agent.invoke(
            prompt,
            system_prompt=cfg.system_prompt,
            model=cfg.model,
            knobs=cfg.knobs,
            tools=cfg.tools,
            kind=cfg.kind,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Merge measured wall-time into the agent's perf (agent supplies tokens/etc.)
        perf = StepPerf(
            tokens=response.perf.tokens,
            latency_ms=round(elapsed_ms, 3),
            retries=response.perf.retries,
            error=response.perf.error,
        )
        self._mw._record_step(
            Step(
                node_id=self.node_id,
                prompt_in=prompt,
                response_out=response.text,
                tool_calls=list(response.tool_calls),
                timing={},
                perf=perf,
            )
        )
        # Offline-only v1: return the agent's response verbatim — no rewriting.
        return response


__all__ = ["TracingMiddleware", "WrappedAgent", "NoLiveSpecError", "UnknownNodeError"]
