"""Unit tests for the TracingMiddleware seam + FakeHostMAS (Phase 3).

Pins the spec behaviours:
  * every agent step is recorded as a Step, in order (observation)
  * the live Spec's (system_prompt, model, knobs, tools) is applied to each
    wrapped agent at invoke time — no host edits needed (config bridge)
  * the response that flows to the next agent === the agent's response verbatim
    (offline-only v1: no mid-run rewriting)
  * a mid-run crash yields ok=False + a partial trace with the steps that ran
    before the crash (spec E4)
"""

from __future__ import annotations

from pathlib import Path

import pytest

import archforge.models as m
from archforge.host import FakeHostMAS, Task
from archforge.middleware import TracingMiddleware
from archforge.stores import SpecStore, TraceStore


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _node(nid: str, *, prompt: str = "sys", model: str = "gpt", tools: list[str] | None = None) -> m.Node:
    return m.Node(node_id=nid, role=f"role-{nid}", system_prompt=prompt,
                  model=model, knobs=m.Knobs(temperature=0.5, retries=1),
                  tools=tools if tools is not None else ["t0"])


@pytest.fixture
def trace_store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / ".archforge")


@pytest.fixture
def mw(trace_store: TraceStore) -> TracingMiddleware:
    return TracingMiddleware(trace_store)


def _seq_spec() -> m.Spec:
    return m.Spec(
        nodes=[_node("a"), _node("b"), _node("c")],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE),
               m.Edge(from_="b", to="c", type=m.EdgeType.SEQUENCE)],
    )


# --------------------------------------------------------------------------- #
# Observation — steps recorded in order
# --------------------------------------------------------------------------- #


def test_every_step_recorded_in_order(mw: TracingMiddleware, trace_store: TraceStore) -> None:
    spec = _seq_spec()
    host = FakeHostMAS()
    pipeline = host.instantiate(spec, mw)
    task = Task(task_id="t1", input="hello", rubric_id="r1", suite_id="s1")

    trace = pipeline.run(task)

    assert trace.ok is True
    assert [s.node_id for s in trace.steps] == ["a", "b", "c"]
    # the first node's prompt is the task input; later nodes chain off the prior response
    assert trace.steps[0].prompt_in == "hello"
    assert trace.steps[1].prompt_in == trace.steps[0].response_out
    assert trace.steps[2].prompt_in == trace.steps[1].response_out
    # final output is the last node's response
    assert trace.final_output == trace.steps[-1].response_out
    # the whole trace was appended to the store under the spec id
    assert len(trace_store.all(spec.compute_spec_id())) == 1


def test_response_flows_through_unchanged(mw: TracingMiddleware) -> None:
    """Offline-only v1: the middleware returns the agent's response verbatim."""

    spec = _seq_spec()
    host = FakeHostMAS()
    pipeline = host.instantiate(spec, mw)
    trace = pipeline.run(Task(task_id="t", input="q"))
    # the second node was fed exactly the first node's raw response (no rewriting)
    assert trace.steps[1].prompt_in == trace.steps[0].response_out


# --------------------------------------------------------------------------- #
# Config bridge — the live Spec's config is applied to each wrapped agent
# --------------------------------------------------------------------------- #


def test_live_spec_config_applied_to_agents(tmp_path: Path) -> None:
    seen: list[tuple[str, str, str, m.Knobs, list[str]]] = []

    def responder(node: m.Node, prompt: str, system_prompt: str) -> str:
        seen.append((node.node_id, system_prompt, node.model, node.knobs, node.tools))
        return f"resp-{node.node_id}"

    spec = m.Spec(
        nodes=[_node("a", prompt="SYS-A", model="claude", tools=["search"]),
               _node("b", prompt="SYS-B", model="gpt-4o", tools=["write"])],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)],
    )
    mw = TracingMiddleware(TraceStore(tmp_path / ".archforge"))
    host = FakeHostMAS(responder=responder)
    host.instantiate(spec, mw).run(Task(task_id="t", input="q"))

    by_node = {nid: cfg for nid, *cfg in seen}
    assert by_node["a"][0] == "SYS-A"      # system_prompt from the live Spec
    assert by_node["a"][1] == "claude"     # model from the live Spec
    assert by_node["a"][2].temperature == 0.5
    assert by_node["a"][3] == ["search"]
    assert by_node["b"][0] == "SYS-B"
    assert by_node["b"][1] == "gpt-4o"
    assert by_node["b"][3] == ["write"]


def test_swapping_live_spec_reconfigures_without_rebuild(tmp_path: Path) -> None:
    """Evolving = swapping which Spec the host uses (no host edits)."""

    def responder(node: m.Node, prompt: str, system_prompt: str) -> str:
        return f"{system_prompt}|{prompt}"

    mw = TracingMiddleware(TraceStore(tmp_path / ".archforge"))
    host = FakeHostMAS(responder=responder)

    spec_v1 = m.Spec(nodes=[_node("a", prompt="PROMPT1")], edges=[])
    pid1 = host.instantiate(spec_v1, mw)
    out1 = pid1.run(Task(task_id="t", input="q")).final_output
    assert out1.startswith("PROMPT1|")

    spec_v2 = m.Spec(nodes=[_node("a", prompt="PROMPT2")], edges=[])
    out2 = host.instantiate(spec_v2, mw).run(Task(task_id="t", input="q")).final_output
    assert out2.startswith("PROMPT2|")
    # the middleware adopted each Spec as live; no rebuild needed


# --------------------------------------------------------------------------- #
# Crash handling — partial trace retained (spec E4)
# --------------------------------------------------------------------------- #


def test_mid_run_crash_yields_partial_trace(mw: TracingMiddleware) -> None:
    spec = _seq_spec()
    # crash on the SECOND node (index 1)
    host = FakeHostMAS(node_scripts={"b": {"crash_on": lambda i: i == 0}})
    trace = host.instantiate(spec, mw).run(Task(task_id="t", input="q"))

    assert trace.ok is False
    assert trace.error is not None and "crash" in trace.error
    # steps a completed before b crashed; c never ran
    assert [s.node_id for s in trace.steps] == ["a"]
    assert trace.final_output == trace.steps[0].response_out  # last good text


def test_no_live_spec_raises(mw: TracingMiddleware) -> None:
    spec = _seq_spec()
    host = FakeHostMAS()
    pipeline = host.instantiate(spec, mw)
    # run() calls begin_run which sets the live spec, so we test the wrap path
    # directly: wrap then invoke before any spec is live
    from archforge.host.fake import FakeAgent
    from archforge.middleware import NoLiveSpecError

    mw._live_spec = None  # simulate no live spec
    agent = FakeAgent(spec.nodes[0])
    wrapped = mw.wrap(agent)
    with pytest.raises(NoLiveSpecError):
        wrapped.invoke("x")


# --------------------------------------------------------------------------- #
# Graph shapes — fanout/join/conditional run + trace
# --------------------------------------------------------------------------- #


def test_conditional_branch_taken(mw: TracingMiddleware) -> None:
    """A conditional edge whose gate matches the response is followed."""

    def responder(node: m.Node, prompt: str, system_prompt: str) -> str:
        return "yes"  # constant response -> matches gate "yes"

    spec = m.Spec(
        nodes=[_node("a"), _node("yes_path"), _node("no_path")],
        edges=[m.Edge(from_="a", to="yes_path", type=m.EdgeType.CONDITIONAL, gate="yes"),
               m.Edge(from_="a", to="no_path", type=m.EdgeType.CONDITIONAL, gate="no")],
    )
    host = FakeHostMAS(responder=responder)
    trace = host.instantiate(spec, mw).run(Task(task_id="t", input="q"))
    assert [s.node_id for s in trace.steps] == ["a", "yes_path"]


def test_conditional_no_gate_match_stops(mw: TracingMiddleware) -> None:
    def responder(node: m.Node, prompt: str, system_prompt: str) -> str:
        return "maybe"

    spec = m.Spec(
        nodes=[_node("a"), _node("yes_path")],
        edges=[m.Edge(from_="a", to="yes_path", type=m.EdgeType.CONDITIONAL, gate="yes")],
    )
    host = FakeHostMAS(responder=responder)
    trace = host.instantiate(spec, mw).run(Task(task_id="t", input="q"))
    assert [s.node_id for s in trace.steps] == ["a"]  # gate didn't match -> stop


def test_deterministic_repeats_produce_same_trace(mw: TracingMiddleware) -> None:
    """Same (spec, task) -> identical trace: prerequisite for R-repeat aggregation (E1)."""

    spec = _seq_spec()
    host = FakeHostMAS()
    p = host.instantiate(spec, mw)
    t1 = p.run(Task(task_id="t", input="q"))
    t2 = p.run(Task(task_id="t", input="q"))
    assert t1.final_output == t2.final_output
    assert [s.response_out for s in t1.steps] == [s.response_out for s in t2.steps]


def test_nodeless_spec_runs_empty_OK(mw: TracingMiddleware) -> None:
    spec = m.Spec(nodes=[], edges=[])
    trace = FakeHostMAS().instantiate(spec, mw).run(Task(task_id="t", input="q"))
    assert trace.ok is True
    assert trace.steps == []
    assert trace.final_output is None


# --------------------------------------------------------------------------- #
# Stores hold the trace under the content-addressed spec id
# --------------------------------------------------------------------------- #


def test_trace_keyed_by_spec_content_id(mw: TracingMiddleware, trace_store: TraceStore,
                                        tmp_path: Path) -> None:
    spec = _seq_spec()
    # commit it so it has a .spec_id; the middleware uses compute_spec_id() either way
    store = SpecStore(tmp_path / ".archforge")
    sid = store.commit(spec, parent_spec_id=None)
    FakeHostMAS().instantiate(spec, mw).run(Task(task_id="t", input="q"))

    # the trace landed under whichever id the middleware used; both equal
    assert len(trace_store.all(sid)) == 1
