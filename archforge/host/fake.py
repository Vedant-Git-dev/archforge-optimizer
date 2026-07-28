"""FakeHostMAS — a deterministic, scriptable stand-in for a real MAS (Phase 3+).

Used by the SuiteRunner and every E2E scenario / smoke test so the optimizer
runs end-to-end **without any real LLM**. It is deliberately simple but honours
the real seam's contracts:

  * reads the Spec graph and runs nodes in a topological order (the order the
    graph implies — sequence edges chain, fanout/join/conditional resolve)
  * each `FakeAgent` produces a deterministic response derived from its config
    and the task; it can be scripted to raise on a chosen call (spec E4
    mid-run crash), so a candidate's behaviour is fully predictable
  * per-run perf is populated (tokens/latency/retries) so cost tracking works
  * the resulting `Trace` is assembled by the `TracingMiddleware` exactly as a
    real host would; a crash yields `ok=False` + a partial trace (E4)

This is the *only* piece that knows framework execution details; everything
above it (SuiteRunner, Judge, Architect, Gatekeeper) treats it as a `HostMAS`.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable

from archforge.constants import SHORT_HASH_LEN
import archforge.models as m
from archforge.host.base import Agent, AgentResponse, HostMAS, Runnable, Task
from archforge.middleware import TracingMiddleware


# --------------------------------------------------------------------------- #
# Fake agents
# --------------------------------------------------------------------------- #


class CrashOnCall(Exception):
    """Scripted failure of a single agent invocation (spec E4)."""


class FakeAgent:
    """A node's executor: deterministic, optionally scripted to fail.

    Behaviour:
      * `invoke` returns a deterministic response built from the node config +
        the incoming prompt + task input, so the same (spec, task) always yields
        the same output (reproducible runs for R-repeat aggregation).
      * An optional `crash_on` callback lets a test force this agent to raise on
        a chosen invocation index (e.g. "fail on the 4th task of the suite") to
        exercise a mid-run crash.
    """

    def __init__(
        self,
        node: m.Node,
        *,
        crash_on: Callable[[int], bool] | None = None,
        responder: Callable[[m.Node, str, str], str] | None = None,
    ) -> None:
        self._node = node
        self._crash_on = crash_on
        self._responder = responder or _default_responder
        self.invoke_count = 0

    @property
    def node_id(self) -> str:
        return self._node.node_id

    @property
    def role(self) -> str:
        return self._node.role

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str,
        model: str,
        knobs: m.Knobs,
        tools: list[str],
    ) -> AgentResponse:
        if self._crash_on is not None and self._crash_on(self.invoke_count):
            self.invoke_count += 1
            raise CrashOnCall(self.node_id)
        self.invoke_count += 1

        out = self._responder(self._node, prompt, system_prompt)
        # deterministic pseudo-token/perf so cost & dedup behave in tests
        token_count = max(1, len(out) // 4)
        return AgentResponse(
            text=out,
            tool_calls=[],  # fake agents do not call tools in v1
            perf=m.StepPerf(tokens=token_count, latency_ms=float((token_count % 7) + 1),
                             retries=0, error=None),
        )


def _default_responder(node: m.Node, prompt: str, _system: str) -> str:
    """Deterministic text so identical (node, prompt) -> identical output."""

    h = hashlib.sha256(f"{node.node_id}|{prompt}".encode()).hexdigest()[:SHORT_HASH_LEN]
    return f"[{node.role}:{node.model}:{h}] {prompt}"


# --------------------------------------------------------------------------- #
# Runnable pipeline over the Spec graph
# --------------------------------------------------------------------------- #


class FakePipeline:
    """A Spec rendered as an executable pipeline.

    Execution model (a deliberate simplification that covers the spec's edge
    kinds for exercising control flow):
      * nodes run in a topological order derived from the edges
      * the first node consumes the task input as its prompt; every later node
        consumes the previous node's response as its prompt (sequence).
      * a `conditional` edge whose `gate` equals the string form of the prior
        step's response is taken; otherwise the next non-conditional edge is
        followed. (Real hosts will have richer semantics; we only need enough to
        run + trace.)
      * the last node's response is the run's `final_output`.
    """

    def __init__(
        self,
        spec: m.Spec,
        middleware: TracingMiddleware,
        agents: dict[str, FakeAgent],
    ) -> None:
        self._spec = spec
        self._mw = middleware
        self._agents = agents
        self._edges_by_src: dict[str, list[m.Edge]] = defaultdict(list)
        for e in spec.edges:
            self._edges_by_src[e.from_].append(e)
        # monotonic counter so each run() mints a UNIQUE run_id, even at R>1
        # (a spec+task no longer collides across repeats). See spec §7 R-repeats.
        self._run_counter = 0

    def run(self, task: Task) -> m.Trace:
        sid = self._spec.compute_spec_id()
        run_id = _run_id(sid, task.task_id, self._run_counter)
        self._run_counter += 1
        self._mw.begin_run(run_id, self._spec, task.task_id)
        last_text: str | None = None
        final_output: str | None = None
        try:
            order = _topo_order(self._spec)
            if not order:
                # nodeless spec -> nothing to trace; trivial success, empty trace
                return self._mw.end_run(None, ok=True, error=None)
            current_id = order[0]
            prompt = task.input
            visited: set[str] = set()
            while current_id is not None:
                if current_id in visited:
                    break  # defensive against cycles the linter should already catch
                visited.add(current_id)
                agent = self._agents[current_id]
                wrapped = self._mw.wrap(agent)
                response = wrapped.invoke(prompt)
                last_text = response.text
                final_output = response.text
                current_id, prompt = self._next_node(current_id, last_text)
            trace = self._mw.end_run(final_output, ok=True, error=None)
            return trace
        except CrashOnCall as exc:
            # A scripted mid-run crash — flush partial trace with ok=False (E4)
            trace = self._mw.end_run(last_text, ok=False, error=f"crash in {exc}")
            return trace
        except Exception as exc:  # noqa: BLE001 — host errors also flush partial trace
            trace = self._mw.end_run(last_text, ok=False, error=repr(exc))
            return trace

    def _next_node(self, current_id: str, response: str) -> tuple[str | None, str]:
        """Pick the next node given the outgoing edges of `current_id`.

        Returns (next_node_id_or_None, next_prompt). A `conditional` edge is
        taken iff its `gate` equals the current response; otherwise the first
        non-conditional out-edge is followed. If no out-edge, we stop.
        """

        outs = self._edges_by_src.get(current_id, [])
        if not outs:
            return None, response
        for edge in outs:
            if edge.type is m.EdgeType.CONDITIONAL:
                if edge.gate is not None and response.strip() == edge.gate:
                    return edge.to, response
            else:
                return edge.to, response
        # only conditional edges, none taken -> stop
        return None, response


# --------------------------------------------------------------------------- #
# The host
# --------------------------------------------------------------------------- #


class FakeHostMAS:
    """A `HostMAS` that builds a `FakePipeline` from a Spec + node scripts.

    `node_scripts` lets tests inject per-node behaviour (e.g. crash_on) keyed by
    node_id; built per `instantiate` so each candidate Spec gets fresh agents.
    """

    def __init__(
        self,
        node_scripts: dict[str, dict] | None = None,
        responder: Callable[[m.Node, str, str], str] | None = None,
    ) -> None:
        self._node_scripts = node_scripts or {}
        self._responder = responder

    def instantiate(self, spec: m.Spec, middleware: TracingMiddleware) -> Runnable:
        agents: dict[str, FakeAgent] = {}
        for node in spec.nodes:
            script = dict(self._node_scripts.get(node.node_id, {}))
            crash_on = script.pop("crash_on", None) if isinstance(script, dict) else None
            agents[node.node_id] = FakeAgent(
                node, crash_on=crash_on, responder=self._responder,
            )
        return FakePipeline(spec, middleware, agents)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _topo_order(spec: m.Spec) -> list[str]:
    """Kahn's algorithm over the (assumed lint-clean, acyclic) Spec graph.

    Roots (no in-edges) are placed first; frontier ties broken by node insertion
    order so the result is stable across runs — important for deterministic traces.
    """

    ids = [n.node_id for n in spec.nodes]
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {nid: 0 for nid in ids}
    for e in spec.edges:
        if e.from_ == e.to:
            continue
        adjacency[e.from_].append(e.to)
        indegree[e.to] += 1
    order_index = {nid: i for i, nid in enumerate(ids)}
    ready = sorted((nid for nid in ids if indegree[nid] == 0), key=lambda n: order_index[n])
    out: list[str] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for nxt in adjacency.get(n, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=lambda n: order_index[n])
    return out


def _run_id(spec_id: str, task_id: str, counter: int) -> str:
    """A unique, human-readable run_id: <spec hash>-<task>-<seq>.

    Uniqueness comes from the per-pipeline `counter` (one increment per `run()`),
    so two repeats of the same (spec, task) — and two candidates that share a
    spec — never collide. Determinism within a run comes from the spec/task hash.
    """

    h = hashlib.sha256(f"{spec_id}|{task_id}".encode()).hexdigest()[:SHORT_HASH_LEN]
    return f"{h}-{task_id}-{counter:04d}"


__all__ = ["FakeHostMAS", "FakeAgent", "FakePipeline", "CrashOnCall"]
