"""FakeHostMAS — a deterministic, scriptable stand-in for a real MAS.

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

import archforge.models as m
from archforge.config import SHORT_HASH_LEN
from archforge.host.adapters.helpers import run_id as _run_id, topo_order as _topo_order
from archforge.host.base import Agent, AgentResponse, HostMAS, Runnable, Task
from archforge.middleware import TracingMiddleware


# --------------------------------------------------------------------------- #
# Fake agents
# --------------------------------------------------------------------------- #


class CrashOnCall(Exception):
    """Scripted failure of a single agent invocation (spec E4)."""


def _default_responder(node: m.Node, prompt: str, _system: str) -> str:
    """Deterministic text so identical (node, prompt) -> identical output."""

    h = hashlib.sha256(f"{node.node_id}|{prompt}".encode()).hexdigest()[:SHORT_HASH_LEN]
    return f"[{node.role}:{node.model}:{h}] {prompt}"


class _FakeBaseAgent:
    """Shared scaffolding for every fake agent kind (LLM + non-LLM).

    Owns the two pieces common to all kinds, lifted out of the old `FakeAgent`:
      * the `crash_on` hook so a test can force ANY node — including a non-LLM
        one — to raise on a chosen invocation index (spec E4 mid-run crash);
      * `invoke_count` so deterministic behaviour + the crash index line up.

    Each kind subclasses and implements ``_respond`` (what text + tool calls the
    node produces). The base then wraps it with kind-aware perf: an ``llm`` node
    costs deterministic pseudo-tokens (``len//4``, as before); a non-llm node
    costs **ZERO tokens** — its real cost is wall-clock latency, which the
    per-cycle wall cap (engine) measures. That tokens=0 convention is what lets a
    tool/retriever/rule-heavy pipeline stay under the token cap yet be bounded by
    the new wall cap (the design's cost fix). Latency is deterministic + nonzero
    so traces stay reproducible (R-repeat aggregation, E1) and the wall cap is
    exercisable.
    """

    def __init__(
        self,
        node: m.Node,
        *,
        crash_on: Callable[[int], bool] | None = None,
    ) -> None:
        self._node = node
        self._crash_on = crash_on
        self.invoke_count = 0

    @property
    def node_id(self) -> str:
        return self._node.node_id

    @property
    def role(self) -> str:
        return self._node.role

    def _respond(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        knobs: m.Knobs,
        tools: list[str],
        kind: m.NodeKind,
    ) -> tuple[str, list[m.ToolCall]]:
        raise NotImplementedError

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str,
        model: str,
        knobs: m.Knobs,
        tools: list[str],
        kind: m.NodeKind = m.NodeKind.LLM,
    ) -> AgentResponse:
        if self._crash_on is not None and self._crash_on(self.invoke_count):
            self.invoke_count += 1
            raise CrashOnCall(self.node_id)
        self.invoke_count += 1

        out, tool_calls = self._respond(prompt, system_prompt, model, knobs, tools, kind)
        # LLM-shaped tokens (len//4) for cost+dedup; non-LLM kinds cost time, not
        # tokens — perf.tokens=0 is the cost-cap convention (see class docstring).
        token_count = max(1, len(out) // 4) if kind is m.NodeKind.LLM else 0
        latency = float((token_count % 7) + 1)  # deterministic, always nonzero
        return AgentResponse(
            text=out,
            tool_calls=tool_calls,
            perf=m.StepPerf(tokens=token_count, latency_ms=latency,
                             retries=0, error=None),
        )


class FakeAgent(_FakeBaseAgent):
    """An LLM node's executor: deterministic, optionally scripted to fail.

    The historical fake (now the `kind=llm` arm). `invoke` returns a deterministic
    response built from the node config + the incoming prompt, so the same
    (spec, task) always yields the same output (reproducible runs for R-repeat
    aggregation). An optional `crash_on` forces a raise on a chosen invocation
    index. Also used, harmless, for `symbolic` nodes — a `symbolic` node is a
    deterministic transform whose cost is latency not tokens, which the base
    enforces via its kind-aware perf (tokens=0 for non-llm).
    """

    def __init__(
        self,
        node: m.Node,
        *,
        crash_on: Callable[[int], bool] | None = None,
        responder: Callable[[m.Node, str, str], str] | None = None,
    ) -> None:
        super().__init__(node, crash_on=crash_on)
        self._responder = responder or _default_responder

    def _respond(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        knobs: m.Knobs,
        tools: list[str],
        kind: m.NodeKind,
    ) -> tuple[str, list[m.ToolCall]]:
        return self._responder(self._node, prompt, system_prompt), []


class FakeRuleAgent(_FakeBaseAgent):
    """A `rule` node (heuristic/scorer/classifier threshold — no prompt, no model).

    Produces a deterministic label that surfaces its `threshold` knob (the
    `tunable` param the LLM Architect edits on a rule node). Zero tokens.
    """

    def _respond(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        knobs: m.Knobs,
        tools: list[str],
        kind: m.NodeKind,
    ) -> tuple[str, list[m.ToolCall]]:
        threshold = getattr(knobs, "threshold", None)
        h = hashlib.sha256(f"{self._node.node_id}|{prompt}".encode()).hexdigest()[:SHORT_HASH_LEN]
        return f"[rule:{self._node.role}:thr={threshold}:{h}] {prompt}", []


class FakeRetrieverAgent(_FakeBaseAgent):
    """A `retriever` node — fetches search/vector context, parameterized by `top_k`.

    Emits `top_k` deterministic context chunks (default 3 when `knobs.top_k` is
    unset), so an Architect `knob` edit raising `top_k` visibly changes the step's
    output (a different SuiteRun). Zero tokens.
    """

    def _respond(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        knobs: m.Knobs,
        tools: list[str],
        kind: m.NodeKind,
    ) -> tuple[str, list[m.ToolCall]]:
        top_k = getattr(knobs, "top_k", None) or 3
        h = hashlib.sha256(f"{self._node.node_id}|{prompt}".encode()).hexdigest()[:SHORT_HASH_LEN]
        chunks = [f"chunk-{i}:{h}" for i in range(int(top_k))]
        return f"[retriever:{self._node.role}:top_k={int(top_k)}] " + "; ".join(chunks), []


class FakeToolAgent(_FakeBaseAgent):
    """A `tool` node — an external call recorded as a `ToolCall`.

    Produces a deterministic serialized result AND a `ToolCall(tool_id=node_id)`
    so a tool node's run is distinguishable from an LLM's (the trace carries the
    call). Zero tokens.
    """

    def _respond(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        knobs: m.Knobs,
        tools: list[str],
        kind: m.NodeKind,
    ) -> tuple[str, list[m.ToolCall]]:
        h = hashlib.sha256(f"{self._node.node_id}|{prompt}".encode()).hexdigest()[:SHORT_HASH_LEN]
        result = f"[tool:{self._node.role}:{h}] {prompt}"
        call = m.ToolCall(tool_id=self._node.node_id, args={"query": prompt},
                          result=result, ok=True)
        return result, [call]


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
        agents: dict[str, _FakeBaseAgent],
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

    Dispatches on each node's `kind` (the non-LLM extension) to pick the matching
    fake agent — `llm`/`symbolic` -> `FakeAgent`, `rule` -> `FakeRuleAgent`,
    `retriever` -> `FakeRetrieverAgent`, `tool` -> `FakeToolAgent` — so a mixed
    pipeline (retriever + llm + tool) runs end-to-end **zero-LLM**. An explicit
    `node_scripts[...]` override (e.g. `crash_on`) still applies to whatever kind
    the node is, so E4 mid-run crashes script on non-LLM nodes too.

    `responder` overrides the LLM-arm responder only (`_default_responder`
    otherwise); it is ignored by the non-LLM fakes (they own their deterministic
    outputs). `node_scripts` lets tests inject per-node behaviour keyed by
    node_id; agents are built per `instantiate` so each candidate Spec gets fresh
    ones.
    """

    def __init__(
        self,
        node_scripts: dict[str, dict] | None = None,
        responder: Callable[[m.Node, str, str], str] | None = None,
    ) -> None:
        self._node_scripts = node_scripts or {}
        self._responder = responder

    def _agent_for(
        self, node: m.Node, crash_on: Callable[[int], bool] | None
    ) -> _FakeBaseAgent:
        kind = node.kind
        if kind is m.NodeKind.RULE:
            return FakeRuleAgent(node, crash_on=crash_on)
        if kind is m.NodeKind.RETRIEVER:
            return FakeRetrieverAgent(node, crash_on=crash_on)
        if kind is m.NodeKind.TOOL:
            return FakeToolAgent(node, crash_on=crash_on)
        # llm + symbolic share the generic (deterministic) agent; symbolic costs
        # latency not tokens, enforced by the base's kind-aware perf.
        return FakeAgent(node, crash_on=crash_on, responder=self._responder)

    def instantiate(self, spec: m.Spec, middleware: TracingMiddleware) -> Runnable:
        agents: dict[str, _FakeBaseAgent] = {}
        for node in spec.nodes:
            script = dict(self._node_scripts.get(node.node_id, {}))
            crash_on = script.pop("crash_on", None) if isinstance(script, dict) else None
            agents[node.node_id] = self._agent_for(node, crash_on)
        return FakePipeline(spec, middleware, agents)


# ``topo_order`` and ``run_id`` now live in ``archforge.host.adapters.helpers``
# (single source for the kit + this module). They are re-imported at the top as
# ``_topo_order`` / ``_run_id`` so the call sites below are unchanged.


__all__ = [
    "FakeHostMAS", "FakeAgent", "FakeRuleAgent", "FakeRetrieverAgent",
    "FakeToolAgent", "FakePipeline", "CrashOnCall",
]
