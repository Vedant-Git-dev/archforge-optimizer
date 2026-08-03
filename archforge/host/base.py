"""The host-MAS integration contract (spec §3, §4).

This module defines the *seam* between ArchForge and an arbitrary multi-agent
system. A concrete host (LangGraph/CrewAI/AutoGen/custom) implements `HostMAS`;
ArchForge talks to it only through this protocol, plus the
`TracingMiddleware` it injects.

Why two collaborating roles:
  * **HostMAS** owns *execution*: it knows the framework's agents and how to
    sequence them per the Spec's graph. It is config-agnostic at the agent level.
  * **TracingMiddleware** owns *observation + config*: it records each Step to
    TraceStore and applies the live Spec's (system_prompt, model, knobs, tools)
    to each agent at invoke time — so reconfiguring the pipeline never requires
    rebuilding the host.

`Runnable.run(task)` returns a full `Trace` (ok or not). A mid-run agent crash
surfaces as a `Trace` with `ok=False`, `error` set, and the Steps that *did*
complete retained (spec E4) — the trace is never lost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

import archforge.models as m

if TYPE_CHECKING:  # avoid a runtime import cycle (middleware imports host.base)
    from archforge.middleware import TracingMiddleware


class Task(BaseModel):
    """A unit of work for the host MAS to execute (one eval-suite item)."""

    model_config = ConfigDict(extra="allow")

    task_id: str
    input: str
    # The rubric to score this task against (spec E2/I5 — comparisons stay
    # within a rubric). None means "use whatever the suite defaults to".
    rubric_id: str | None = None
    suite_id: str | None = None


class AgentResponse(BaseModel):
    """What a single agent produced: its text, any tool calls, and perf signals."""

    model_config = ConfigDict(extra="allow")

    text: str
    tool_calls: list[m.ToolCall] = Field(default_factory=list)
    perf: m.StepPerf = Field(default_factory=m.StepPerf)


@runtime_checkable
class Agent(Protocol):
    """Host-provided implementation of one node's behaviour.

    Concrete hosts wrap their framework agent here. `invoke` receives the Spec's
    configuration (`system_prompt`, `model`, `knobs`, `tools`, `kind`) so the same
    agent is reconfigured when the spec evolves — without rebuilding the host. The
    middleware is what actually supplies these arguments at call time.

    `kind` (the node's `NodeKind`) is OPTIONAL and defaults to `LLM` so every
    pre-existing adapter that knows only LLM agents keeps working unchanged; a
    host that dispatches on kind (e.g. the fake kit's rule/retriever/tool agents)
    reads it to choose its behaviour. It carries NO cost on adapters that ignore it.
    """

    node_id: str
    role: str

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str,
        model: str,
        knobs: m.Knobs,
        tools: list[str],
        kind: m.NodeKind = m.NodeKind.LLM,
    ) -> AgentResponse: ...


@runtime_checkable
class Runnable(Protocol):
    """A Spec instantiated into an executable, observable pipeline."""

    def run(self, task: Task) -> m.Trace: ...


@runtime_checkable
class HostMAS(Protocol):
    """The multi-agent system being optimized.

    `instantiate` builds the agents for a Spec and wires each through the
    middleware (which records steps + applies config). The returned `Runnable`
    is what the SuiteRunner calls once per task per repeat.
    """

    def instantiate(self, spec: m.Spec, middleware: "TracingMiddleware") -> Runnable: ...


__all__ = ["Task", "AgentResponse", "Agent", "Runnable", "HostMAS"]
