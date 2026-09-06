"""LangGraph adapter E2E — drives a REAL compiled LangGraph through the Forge.

A tiny synthetic LangGraph app with a realistic retrieval-pipeline topology
(retriever → rule router → conditional fan-out → a runtime retrieve_more LOOP
edge) proves the generic ``LangGraphHostAdapter`` end-to-end on the production
organs — REAL ``Engine`` + REAL ``Architect`` (via ``ScriptedLLM``) + REAL
``SuiteRunner``/``Gatekeeper``/stores — at zero billing (no LLM provider, no
vector DB; the demo's nodes are pure Python).

The headline scenario mirrors ``test_mixed_nonllm_pipeline``'s knob-edit loop,
except the host is NOT ``FakeHostMAS`` — it's the real langgraph adapter driving
the real compiled graph:

  * incumbent (seeded ``route.threshold=0.8``): with a fixed ``coverage=0.5``
    input, ``0.5 < 0.8`` ⇒ retrieve_more loop fires (k 4→8→max) ⇒ ``answer``;
  * candidate (Architect ``knob`` edit lowering ``route.threshold`` to ``0.4``):
    ``0.5 >= 0.4`` ⇒ loop skipped ⇒ ``answer`` directly — a structurally
    different Trace (no ``retrieve_more`` step);
  * ScriptedJudge pins inc 0.50 / cand 0.80 (>= τ) ⇒ AUTO_PROMOTE.

Secondary: a ``knob`` raising ``retrieve.top_k`` 4→8 is observable in the
candidate's trace (more chunks); ``retrieve`` costs ZERO tokens with
``latency_ms`` > 0 (kind-aware cost); a ``model_swap`` on the ``answer`` node
reaches the LLM injector (the call-time config hook fires with the live model).
"""
from __future__ import annotations

import pytest

import archforge.models as m
from archforge.architect import Architect
from archforge.engine import Engine
from archforge.host.adapters import (
    EdgeSpec, LangGraphApp, LangGraphHostAdapter, Nd,
    export_spec_sidecar, load_spec_sidecar,
)
from archforge.judge import ScriptedJudge
from archforge.llm import ScriptedLLM
from archforge.stores import AttemptStore, SpecStore, TraceStore
from archforge.suite import Suite

from tests.scenarios.conftest import (
    build_engine, cand_id, llm_proposal, make_suite,
)

# langgraph is the demo app's only framework dep — imported lazily so the kit
# test suite still collects if langgraph isn't installed.
pytest.importorskip("langgraph")
from langgraph.graph import END, StateGraph  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402


# --------------------------------------------------------------------------- #
# The synthetic demo graph — pure Python, zero billing
# --------------------------------------------------------------------------- #

_DOCS = [f"doc-{i} evidence about topic {i % 3}" for i in range(16)]
_MAX_K = 16   # the loop's termination guard (max-retrieval-reached)


class _DemoState(TypedDict, total=False):
    query: str
    coverage: float
    current_top_k: int
    threshold: float
    documents: list[str]
    visits: list[str]
    decision: str
    answer: str


def _retrieve(state: _DemoState) -> dict:
    # k from state (the live knob) — the focused-retriever shape.
    k = state.get("current_top_k", 4)
    return {
        "documents": _DOCS[:k],
        "current_top_k": k,
        "visits": state.get("visits", []) + ["retrieve"],
        "coverage": 0.5,            # fixed input → routing is deterministic by threshold
    }


def _retrieve_more(state: _DemoState) -> dict:
    cur = state.get("current_top_k", 4)
    nk = 8 if cur == 4 else 16
    return {
        "documents": _DOCS[:nk],
        "current_top_k": nk,
        "visits": state.get("visits", []) + ["retrieve_more"],
    }


def _route(state: _DemoState) -> dict:
    # The rule node: coverage vs threshold decides loop-or-skip; a max_k guard
    # bounded the loop (else fixed coverage < threshold loops forever).
    go = state.get("coverage", 0.0) < state.get("threshold", 0.8)
    loop = go and state.get("current_top_k", 4) < _MAX_K
    return {"visits": state.get("visits", []) + ["route"],
            "decision": "retrieve_more" if loop else "answer"}


def _after_route(state: _DemoState) -> str:
    return state.get("decision", "answer")


def _answer(state: _DemoState) -> dict:
    return {
        "answer": f"answered using {len(state.get('documents', []))} chunks",
        "visits": state.get("visits", []) + ["answer"],
    }


def _build_demo_graph():
    g = StateGraph(_DemoState)
    g.add_node("retrieve", _retrieve)
    g.add_node("route", _route)
    g.add_node("retrieve_more", _retrieve_more)
    g.add_node("answer", _answer)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "route")
    g.add_conditional_edges("route", _after_route,
                            {"retrieve_more": "retrieve_more", "answer": "answer"})
    g.add_edge("retrieve_more", "retrieve")     # the runtime LOOP back-edge
    g.add_edge("answer", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# The MAS description (LangGraphApp) + its host adapter
# --------------------------------------------------------------------------- #

# Call-time LLM injector (the module-level config-table pattern, distilled). The
# `answer` node (a real LLM in a real app) would consult this; the demo does NOT
# so the run stays free, but the adapter still populates it so a `model_swap`
# is observably delivered to the injector.
_NODE_CFG: dict[str, dict] = {}


class DemoApp(LangGraphApp):
    graph_factory = staticmethod(_build_demo_graph)
    final_output_key = "answer"
    base_prompts = {"answer": ""}

    nodes = [
        Nd("retrieve", role="retrieve", kind=m.NodeKind.RETRIEVER,
           knobs=m.Knobs(top_k=4, tunable=("top_k",))),
        Nd("route", role="route", kind=m.NodeKind.RULE,
           knobs=m.Knobs(threshold=0.8, tunable=("threshold",))),
        Nd("retrieve_more", role="retrieve_more", kind=m.NodeKind.RETRIEVER),
        Nd("answer", role="answer", kind=m.NodeKind.LLM,
           knobs=m.Knobs(temperature=0.3, max_tokens=1024)),
    ]
    # static wiring: retrieve → route --(decision)--> {retrieve_more, answer}.
    # the retrieve_more->retrieve runtime LOOP is intentionally OMITTED (it
    # would trip lint `cycle`); documented in runtime_loops.
    edges = [
        EdgeSpec("retrieve", "route", kind=m.EdgeType.SEQUENCE),
        EdgeSpec("route", "retrieve_more", kind=m.EdgeType.CONDITIONAL,
                 gate="low_coverage"),
        EdgeSpec("route", "answer", kind=m.EdgeType.CONDITIONAL,
                 gate="sufficient_coverage_or_max"),
    ]
    runtime_loops = [("retrieve_more", "retrieve")]
    knob_to_state = {"top_k": "current_top_k", "threshold": "threshold"}

    def initialize_state(self, task_input: str) -> dict:
        return {"query": task_input, "current_top_k": 4, "threshold": 0.8,
                "coverage": 0.0, "visits": [], "documents": [], "answer": ""}

    def summarize(self, node_id: str, partial: dict, merged: dict) -> str:
        if node_id == "retrieve":
            return f"top_k={merged.get('current_top_k')} chunks={len(merged.get('documents', []))}"
        if node_id == "retrieve_more":
            return f"retrieve_more(k={merged.get('current_top_k')})"
        if node_id == "route":
            return f"decision={merged.get('decision')}"
        if node_id == "answer":
            return merged.get("answer", "")
        return str(partial)

    def apply_llm_config(self, node_id: str, vote) -> None:
        _NODE_CFG[node_id] = {"model": vote.model, "temperature": vote.temperature,
                              "max_tokens": vote.max_tokens,
                              "system_prompt": vote.system_prompt}

    def reset_llm_config(self) -> None:
        _NODE_CFG.clear()


@pytest.fixture
def langgraph_stores(tmp_path):
    root = tmp_path / ".archforge"
    specs = SpecStore(root)
    atts = AttemptStore(root)
    ts = TraceStore(root)
    adapter = LangGraphHostAdapter(DemoApp())
    rid = specs.commit(adapter.app_spec(), parent_spec_id=None,
                       status=m.SpecStatus.INCUMBENT)
    specs.set_active(rid)
    return specs, atts, ts, rid, adapter


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #


def test_langgraph_adapter_renders_a_trace(langgraph_stores) -> None:
    """The adapter drives the real graph and records one Step per run — an
    incumbent run produces a complete, ok trace with the expected node visits."""
    specs, atts, ts, rid, adapter = langgraph_stores
    from archforge.host.base import Task
    from archforge.middleware import TracingMiddleware

    mw = TracingMiddleware(ts)
    spec = specs.get(rid)
    runnable = adapter.instantiate(spec, mw)
    trace = runnable.run(Task(task_id="t1", input="q1"))

    assert trace.ok
    node_ids = [s.node_id for s in trace.steps]
    # threshold=0.8 > coverage=0.5 ⇒ loop fires: retrieve → route → retrieve_more → ...
    assert "retrieve" in node_ids and "route" in node_ids and "answer" in node_ids
    assert node_ids.count("retrieve_more") >= 1
    assert trace.final_output and "chunks" in trace.final_output


def test_langgraph_knob_edit_on_router_runs_and_promotes(langgraph_stores) -> None:
    """Headline: a `knob` edit lowering `route.threshold` 0.8→0.4 flips the
    retrieve_more loop off → structurally different Trace → AUTO_PROMOTE on the
    production organs."""
    specs, atts, ts, rid, adapter = langgraph_stores
    incumbent = specs.get(rid)
    cid = cand_id(incumbent, kind="knob", target="route",
                  payload={"knobs": {"threshold": 0.4}})

    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="knob", target="route",
                     payload={"knobs": {"threshold": 0.4}}))
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cid, "t1", 0.80)

    eng = build_engine((specs, atts, ts, None), judge=judge,
                       architect=Architect(llm, model="architect-1"),
                       suite=make_suite(), host=adapter)
    r = eng.evolve_cycle(0)

    assert r.attempted and r.decision is not None
    assert r.candidate_mean == pytest.approx(0.80)
    assert specs.active_id() == cid

    # The candidate's run SKIPPED the loop (threshold 0.4 <= coverage 0.5):
    # its trace has no `retrieve_more` step, unlike the incumbent's.
    cand_trace = next(t for t in ts.all(cid) if t.ok)
    cand_nodes = {s.node_id for s in cand_trace.steps}
    assert "retrieve_more" not in cand_nodes
    inc_trace = next(t for t in ts.all(rid) if t.ok)
    inc_nodes = {s.node_id for s in inc_trace.steps}
    assert "retrieve_more" in inc_nodes


def test_langgraph_retriever_uses_new_top_k_in_trace(langgraph_stores) -> None:
    """A `knob` raising `retrieve.top_k` 4→8 reaches the real graph: the
    candidate's retrieve step reflects `top_k=8` + 8 chunks (the live Spec's knob
    flowed through `initial_state` into the real node)."""
    specs, atts, ts, rid, adapter = langgraph_stores
    incumbent = specs.get(rid)
    cid = cand_id(incumbent, kind="knob", target="retrieve",
                  payload={"knobs": {"top_k": 8}})

    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="knob", target="retrieve",
                     payload={"knobs": {"top_k": 8}}))
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cid, "t1", 0.80)
    eng = build_engine((specs, atts, ts, None), judge=judge,
                       architect=Architect(llm, model="architect-1"),
                       suite=make_suite(), host=adapter)
    eng.evolve_cycle(0)

    cand_trace = next(t for t in ts.all(cid) if t.ok)
    ret = next(s for s in cand_trace.steps if s.node_id == "retrieve")
    assert "top_k=8" in ret.response_out
    assert "chunks=8" in ret.response_out


def test_langgraph_retriever_costs_zero_tokens(langgraph_stores) -> None:
    """Kind-aware cost: the retriever node costs ZERO tokens (its cost is
    wall-clock `latency_ms`); only the LLM `answer` node costs tokens."""
    specs, atts, ts, rid, adapter = langgraph_stores
    incumbent = specs.get(rid)
    cid = cand_id(incumbent, kind="knob", target="retrieve",
                  payload={"knobs": {"top_k": 8}})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="knob", target="retrieve",
                     payload={"knobs": {"top_k": 8}}))
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cid, "t1", 0.80)
    eng = build_engine((specs, atts, ts, None), judge=judge,
                       architect=Architect(llm, model="architect-1"),
                       suite=make_suite(), host=adapter)
    r = eng.evolve_cycle(0)

    cand_trace = next(t for t in ts.all(cid) if t.ok)
    ret = next(s for s in cand_trace.steps if s.node_id == "retrieve")
    assert ret.perf.tokens == 0
    assert ret.perf.latency_ms > 0.0
    ans = next(s for s in cand_trace.steps if s.node_id == "answer")
    assert ans.perf.tokens > 0
    assert r.latency_ms > 0.0


def test_langgraph_model_swap_reaches_llm_injector(langgraph_stores) -> None:
    """A `model_swap` on the `answer` node flows through the call-time LLM
    injector: `apply_llm_config` was called UP-FRONT with the live model (the
    only viable shape — graph.stream reveals a node only after it runs)."""
    specs, atts, ts, rid, adapter = langgraph_stores
    incumbent = specs.get(rid)
    cid = cand_id(incumbent, kind="model_swap", target="answer",
                  payload={"model": "gpt-4o-mini"})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="model_swap", target="answer",
                     payload={"model": "gpt-4o-mini"}))
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cid, "t1", 0.80)
    eng = build_engine((specs, atts, ts, None), judge=judge,
                       architect=Architect(llm, model="architect-1"),
                       suite=make_suite(), host=adapter)
    eng.evolve_cycle(0)

    # the candidate's run populated the injector with the swapped model
    assert _NODE_CFG.get("answer", {}).get("model") == "gpt-4o-mini"


# --------------------------------------------------------------------------- #
# Tier-2 deploy — the winning Spec as a reviewable sidecar + on_promote trigger
# --------------------------------------------------------------------------- #


def test_export_spec_sidecar_projects_live_knobs() -> None:
    """export_spec_sidecar projects a Spec's live node knobs (named LLM knobs +
    extras, skipping None/blank defaults) into a `{node_id: {knob: value}}` file
    and round-trips through load_spec_sidecar."""
    import tempfile
    adapter = LangGraphHostAdapter(DemoApp())
    spec = adapter.app_spec()
    # mutate retrieve.top_k 4->9 so the sidecar carries a non-default extra
    from archforge.mutate import apply_change
    change = m.Change.for_kind(m.ChangeKind.KNOB, "retrieve", "tune", "scenario")
    spec = apply_change(spec, change, {"knobs": {"top_k": 9}})

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        sidecar = export_spec_sidecar(spec, path)
        # named knobs (answer: temperature/max_tokens) + extras (retrieve: top_k)
        assert sidecar["retrieve"]["top_k"] == 9
        assert sidecar["answer"]["temperature"] == pytest.approx(0.3)
        assert sidecar["answer"]["max_tokens"] == 1024
        # blank model/system_prompt carrying nothing didn't ship
        assert "model" not in sidecar.get("retrieve", {})
        # round-trips through the loader
        loaded = load_spec_sidecar(path)
        assert loaded == sidecar
    finally:
        import os; os.unlink(path)


def test_on_promote_fires_and_sidecar_has_winner(langgraph_stores, tmp_path) -> None:
    """An opt-in `on_promote` callback fires on AUTO_PROMOTE and ONLY then, with
    the promoted Spec — wired to export_spec_sidecar, the written sidecar carries
    the winning knob values (production reads it; rollback = delete)."""
    specs, atts, ts, rid, adapter = langgraph_stores
    incumbent = specs.get(rid)
    cid = cand_id(incumbent, kind="knob", target="retrieve",
                  payload={"knobs": {"top_k": 8}})

    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="knob", target="retrieve",
                     payload={"knobs": {"top_k": 8}}))
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cid, "t1", 0.80)

    sidecar_path = tmp_path / "winner.json"
    fires: list[str] = []

    def on_promote(promoted: m.Spec) -> None:
        fires.append(promoted.spec_id or promoted.compute_spec_id())
        export_spec_sidecar(promoted, sidecar_path)

    eng = build_engine((specs, atts, ts, None), judge=judge,
                       architect=Architect(llm, model="architect-1"),
                       suite=make_suite(), host=adapter)
    eng._on_promote = on_promote           # opt-in (build_engine doesn't expose it)
    r = eng.evolve_cycle(0)

    assert r.promoted
    assert len(fires) == 1                 # fired exactly once, on the promote
    assert fires[0] == cid                 # with the promoted (candidate) Spec
    assert sidecar_path.exists()
    winner = load_spec_sidecar(sidecar_path)
    assert winner["retrieve"]["top_k"] == 8     # the winning value reached the sidecar


def test_on_promote_does_not_fire_on_discard(langgraph_stores) -> None:
    """on_promote does NOT fire when the candidate is discarded (scored below the
    incumbent) — the deploy hook is promote-gated, so a losing change never ships."""
    specs, atts, ts, rid, adapter = langgraph_stores
    incumbent = specs.get(rid)
    cid = cand_id(incumbent, kind="knob", target="retrieve",
                  payload={"knobs": {"top_k": 8}})

    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="knob", target="retrieve",
                     payload={"knobs": {"top_k": 8}}))
    # candidate scored LOWER than incumbent -> not a promote (discard/queue).
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.80).set_aggregate(cid, "t1", 0.50)

    fires: list[str] = []
    eng = build_engine((specs, atts, ts, None), judge=judge,
                       architect=Architect(llm, model="architect-1"),
                       suite=make_suite(), host=adapter)
    eng._on_promote = lambda s: fires.append(s.spec_id or s.compute_spec_id())
    r = eng.evolve_cycle(0)

    assert not r.promoted
    assert fires == []                     # the discard shipped nothing
