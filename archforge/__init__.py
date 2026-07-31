"""ArchForge — a self-improving meta-layer over multi-agent systems.

The **adapter kit** (the public API) lets an external project wrap any MAS as a
``HostAdapter`` and evolve it; the core organs (Engine, Architect, Judge,
Gatekeeper, stores) drive the Propose-Evaluate-Commit loop. See
``docs/superpowers/specs/2026-07-30-adapter-kit-design.md`` for the kit design.

Quick adapter sketch:

    from archforge import (
        BaseHostAdapter, BaseAgent, CallResult,
        SpecBuilder, SEQUENCE, JOIN, run_loop, RunnerConfig,
    )

    class MyAgent(BaseAgent):
        def call(self, prompt, cfg) -> CallResult:       # the one per-node hook
            ...

    class MyAdapter(BaseHostAdapter):
        def make_agent(self, node) -> BaseAgent: ...

    spec = (SpecBuilder().node(...).edge("a","b", kind=SEQUENCE).build())
    r = run_loop(MyAdapter(), spec, suite, config=RunnerConfig(provider="gemini"))
"""
from __future__ import annotations

# ── the contract (host/base.py) — already the seam; re-exported so an adapter
#    author imports everything from one place (`archforge`). Unchanged modules.
from archforge.host.base import Agent, AgentResponse, HostMAS, Runnable, Task

# ── the adapter kit (host/adapters) — what every adapter subclasses / uses.
from archforge.host.adapters import (
    BaseAgent,
    BaseHostAdapter,
    BasePipeline,
    CallResult,
    KnobVote,
    RunContext,
    cfg_decay,
    cfg_as_kwargs,
    estimate_tokens,
    run_id,
    topo_order,
)

# ── Spec bootstrap DSL (spec_builder.py) — the universal floor.
from archforge.spec_builder import (
    CONDITIONAL,
    EdgeType,
    FANOUT,
    JOIN,
    SEQUENCE,
    SpecBuildError,
    SpecBuilder,
)

# ── the run entrypoint (runner.py) — wraps the Engine for embedders.
from archforge.runner import RunnerConfig, run_cycle, run_loop

from archforge.config import VERSION as __version__  # noqa: F401  (public API)

__all__ = [
    # contract
    "HostMAS", "Agent", "Runnable", "Task", "AgentResponse",
    # kit
    "BaseHostAdapter", "BaseAgent", "BasePipeline", "CallResult",
    "RunContext", "KnobVote", "cfg_decay", "cfg_as_kwargs",
    "run_id", "topo_order", "estimate_tokens",
    # Spec DSL
    "SpecBuilder", "SpecBuildError", "EdgeType",
    "JOIN", "FANOUT", "SEQUENCE", "CONDITIONAL",
    # runner
    "RunnerConfig", "run_cycle", "run_loop",
    # version
    "__version__",
]
