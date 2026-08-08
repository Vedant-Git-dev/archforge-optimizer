"""archforge.host.adapters — the reusable adapter kit.

The single place a new MAS author looks: subclass `BaseHostAdapter` + a small
`BaseAgent` per node, fill `call()` (the node's real work), and override the
hooks (`execution_order` / `resolve_prompt` / `stage_context`) only when the
MAS's data-shaping is non-default. The kit owns the run loop, content-decouple,
config-decay, perf, and partial-trace flush — the recurring scaffolding every
adapter previously re-derived.

`helpers` is also re-exported so an author extending the loop's primitives
(topo order, run-id, config decay) does so against one source.
"""
from archforge.host.adapters.base import (
    BaseAgent, BaseHostAdapter, BasePipeline, CallResult, RunContext,
)
from archforge.host.adapters.helpers import (
    KnobVote, cfg_as_kwargs, cfg_decay, estimate_tokens, run_id, topo_order,
)
# LangGraph adapter — re-exported eagerly. The module is langgraph-FREE (it
# imports no langgraph types; the dependency enters only when a concrete app's
# `graph_factory` builds the real graph), so importing it here keeps
# `import archforge` framework-free.
from archforge.host.adapters.langgraph import (
    EdgeSpec, LangGraphApp, LangGraphHostAdapter, LangGraphRunnable, Nd,
    build_optimized_envelope, export_optimized, export_spec_sidecar,
    load_optimized, load_spec_sidecar,
)

__all__ = [
    "BaseHostAdapter", "BaseAgent", "BasePipeline", "CallResult", "RunContext",
    "run_id", "topo_order", "cfg_decay", "cfg_as_kwargs", "KnobVote",
    "estimate_tokens",
    "Nd", "EdgeSpec", "LangGraphApp", "LangGraphHostAdapter",
    "LangGraphRunnable",
    "export_spec_sidecar", "load_spec_sidecar",
    "build_optimized_envelope", "export_optimized", "load_optimized",
]
