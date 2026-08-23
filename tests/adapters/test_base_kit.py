"""Unit tests for the adapter kit core (`archforge.host.adapters`).

Pure, deterministic, no SDK, no Lumina: these exercise the reusable scaffolding
the kit owns — run loop, content-decouple, config-decay, partial-trace flush,
unknown-node rejection — so the kit is proven independent of any one adapter.

The Lumina-adapter *reference* proof lives in `LuminaAI/archforge_glue/
test_smoke_offline.py` (outside `testpaths`, since it imports SDK modules);
those 8 tests show a real adapter built on this kit. These kit tests show the
kit itself is correct with tiny in-test adapters.
"""
from __future__ import annotations

import archforge.models as m
from archforge.host.adapters import (
    BaseAgent, BaseHostAdapter, CallResult,
    cfg_decay, estimate_tokens, run_id, topo_order,
)
from archforge.host.base import AgentResponse, Task
from archforge.middleware import TracingMiddleware
from archforge.stores import TraceStore


# ── a tiny spec for the loop tests ───────────────────────────────────────── #
def _node(nid: str) -> m.Node:
    return m.Node(node_id=nid, role="r", system_prompt="p", model="m")


def _mini_spec(a: str = "a", b: str = "b"):
    return m.Spec(nodes=[_node(a), _node(b)],
                  edges=[m.Edge(from_=a, to=b, type=m.EdgeType.SEQUENCE)])


# ── helpers tests ────────────────────────────────────────────────────────── #
def test_topo_order_roots_first_respects_edges():
    assert topo_order(_mini_spec()) == ["a", "b"]


def test_run_id_unique_per_repeat():
    s = "da069e1e753f759a"
    ids = [run_id(s, "t1", i) for i in range(4)]
    assert len(set(ids)) == 4                     # the counter suffix makes each unique
    # suffix carries the counter, format <hash>-<task>-<NNNN>
    assert all(i.endswith(f"-{i_seq:04d}") for i, i_seq in zip(ids, range(4)))


def test_cfg_decay_threads_mutated_prompt_only():
    base = {"t": "DEFAULT"}
    assert cfg_decay("t", "DEFAULT", "m", None, None, base_prompts=base).system_prompt is None
    assert cfg_decay("t", "EDITED", "m", None, None, base_prompts=base).system_prompt == "EDITED"


def test_cfg_decay_always_threads_model_temp_knobs():
    k = m.Knobs(temperature=0.4, retries=2, max_tokens=2000)
    v = cfg_decay("t", None, "m", k, ["tool"])
    assert v.model == "m" and v.temperature == 0.4 and v.max_tokens == 2000 and v.retries == 2


# ── BaseAgent content-decouple ───────────────────────────────────────────── #
class _Echo(BaseAgent):
    def call(self, prompt, vote):
        self.seen_vote = vote
        return CallResult(thread=prompt[::-1], content=prompt.upper())


class _TextThroughAdapter(BaseHostAdapter):
    """All-default-hooks adapter: each node uppercases its prompt (text-in/out)."""
    def make_agent(self, node):
        return _Echo(node, self)


class _NoOpAdapter(_TextThroughAdapter):
    """Adapter used where only the object (for cfg_for) is needed; nodes not run."""


def test_base_agent_invoke_records_content_threads_onward():
    ag = _Echo(_node("a"), _NoOpAdapter())
    resp = ag.invoke("hi", system_prompt=None, model="m", knobs=None, tools=None)
    assert resp.text == "HI"            # content = what the trace records / Judge scores
    assert resp.thread == "ih"          # plumbing onward, decoupled from scored content
    assert isinstance(resp, AgentResponse)
    assert ag.seen_vote.model == "m"    # vote was decayed + forwarded


def test_estimate_tokens_is_char_heuristic():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("a" * 40) == 10


# ── run loop (driven through TracingMiddleware) ──────────────────────────── #
class _Crash(BaseAgent):
    def __init__(self, node, adapter, *, fail_on):
        super().__init__(node, adapter)
        self._fail_on = fail_on
    def call(self, prompt, vote):
        if self.node_id == self._fail_on:
            raise RuntimeError("boom")
        return CallResult(thread=prompt, content=prompt.upper())


class _CrashAdapter(BaseHostAdapter):
    def __init__(self, *, fail_on=None):
        super().__init__()
        self._fail_on = fail_on
    def make_agent(self, node):
        return _Crash(node, self, fail_on=self._fail_on or "")


def _trace(adapter, spec, tmp_path, task_input="query"):
    ts = TraceStore(tmp_path / ".archforge")
    runnable = adapter.instantiate(spec, TracingMiddleware(ts))
    return runnable.run(Task(task_id="t1", input=task_input))


def test_base_pipeline_default_hooks_run_in_sequence(tmp_path):
    trace = _trace(_TextThroughAdapter(), _mini_spec(), tmp_path)
    assert trace.ok is True
    assert [s.node_id for s in trace.steps] == ["a", "b"]
    assert trace.final_output == "QUERY"   # b uppercased the (already uppercased) input


def test_base_pipeline_partial_trace_on_crash(tmp_path):
    """A node raising mid-run flushes a PARTIAL trace (ok=False) with the steps
    that did run — E4: a crashed run is never lost."""
    trace = _trace(_CrashAdapter(fail_on="b"), _mini_spec(), tmp_path)
    assert trace.ok is False
    assert trace.error and "boom" in trace.error
    assert [s.node_id for s in trace.steps] == ["a"]   # 'a' ran; 'b' didn't


def test_unknown_node_errors_run_not_ok(tmp_path):
    """A node_id `make_agent` can't resolve → kit's _RaisingAgent → trace ok=False,
    so an add_node the adapter can't run is rejected by scoring (I4)."""
    class _Missing(BaseHostAdapter):
        def make_agent(self, node):
            raise NotImplementedError(node.node_id)
    trace = _trace(_Missing(), _mini_spec(), tmp_path)
    assert trace.ok is False
    assert trace.error and "can't execute node" in trace.error
