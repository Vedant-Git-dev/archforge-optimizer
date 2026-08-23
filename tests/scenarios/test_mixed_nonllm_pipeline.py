"""Mixed-pipeline E2E (non-LLM optimizer extension) — zero-LLM full cycle.

Drives the REAL `Engine` + REAL `Architect` (via `ScriptedLLM`) on a Spec whose
nodes are a mix of LLM and non-LLM archetypes — here a `retriever` node parameterized
by `top_k` (an open-Knobs extra in its `tunable` allowlist). The Architect proposes a
`knob` edit RAISING `top_k`, which:

  * passes the kind gate (a `knob` edit, not the llm-only `prompt_edit`/`model_swap`);
  * passes the `apply_knob` tunable guard (`top_k` IS in the node's `tunable`);
  * materializes a candidate whose `FakeRetrieverAgent` runs with the NEW `top_k`,
    emitting more context chunks -> a DIFFERENT `SuiteRun`;
  * is scored by the ScriptedJudge (set higher than the incumbent) -> promoted.

This proves the whole non-LLM path is live end-to-end: the open Knobs, the tunable
allowlist, the kind-dispatch fakes (a retriever that respects `top_k`), kind-aware
cost (tokens=0 on the retriever), and the cost cap's new `latency_ms` sum — all on
the production organs (real Architect + SuiteRunner + Gatekeeper + real stores).
"""

from __future__ import annotations

import pytest

import archforge.models as m
from archforge.architect import Architect
from archforge.engine import Engine
from archforge.host import FakeHostMAS
from archforge.host.base import Task
from archforge.judge import ScriptedJudge
from archforge.llm import ScriptedLLM
from archforge.stores import AttemptStore, SpecStore, TraceStore
from archforge.suite import Suite

from tests.scenarios.conftest import build_engine, llm_proposal, make_suite


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #


def _mixed_incumbent() -> m.Spec:
    """An llm answer node fed by a retriever; the retriever's `top_k` is a `tunable`
    extra (the knob the Architect may edit)."""
    ret = m.Node(node_id="ret", role="retrieve", kind=m.NodeKind.RETRIEVER,
                 knobs=m.Knobs(top_k=5, tunable=("top_k",)))
    ans = m.Node(node_id="ans", role="answer", kind=m.NodeKind.LLM,
                 system_prompt="ground the answer in retrieved context", model="gpt")
    return m.Spec(
        nodes=[ret, ans],
        edges=[m.Edge(from_="ret", to="ans", type=m.EdgeType.SEQUENCE)],
    )


@pytest.fixture
def mixed_stores(tmp_path):
    """Real filesystem stores seeded with the mixed incumbent, set active."""
    root = tmp_path / ".archforge"
    specs = SpecStore(root)
    atts = AttemptStore(root)
    ts = TraceStore(root)
    rid = specs.commit(_mixed_incumbent(), parent_spec_id=None,
                       status=m.SpecStatus.INCUMBENT)
    specs.set_active(rid)
    return specs, atts, ts, rid


def _candidate_id(incumbent: m.Spec, *, top_k: int) -> str:
    # Replicate Architect+mutate+Engine.commit to know the candidate content's id:
    # a `knob` edit setting `top_k` on the retriever node (its tunable allows it).
    from archforge.mutate import apply_change
    change = m.Change.for_kind(m.ChangeKind.KNOB, "ret", "tune top_k", "more context")
    cand = apply_change(incumbent, change, {"knobs": {"top_k": top_k}})
    cand.parent_spec_id = incumbent.spec_id
    cand = cand.model_copy(update={"spec_id": None})
    return cand.compute_spec_id()


# --------------------------------------------------------------------------- #
# the scenario
# --------------------------------------------------------------------------- #


def test_mixed_pipeline_knob_edit_on_retriever_runs_and_promotes(mixed_stores) -> None:
    specs, atts, ts, rid = mixed_stores
    incumbent = specs.get(rid)
    cid = _candidate_id(incumbent, top_k=8)

    # Real Architect driven by a ScriptedLLM: one `knob` proposal raising top_k.
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="knob", target="ret", payload={"knobs": {"top_k": 8}}))
    # ScriptedJudge: incumbent 0.50, candidate 0.80 -> +0.30 >= τ (clear win).
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cid, "t1", 0.80)

    eng = build_engine((specs, atts, ts, None),
                       judge=judge, architect=Architect(llm, model="architect-1"),
                       suite=make_suite(), host=FakeHostMAS())
    r = eng.evolve_cycle(0)

    assert r.attempted and r.decision is not None
    # the candidate ran with the NEW top_k -> it scored higher -> promoted, active moved
    assert r.candidate_mean == pytest.approx(0.80)
    assert specs.active_id() == cid


def test_mixed_pipeline_retriever_uses_new_top_k_in_trace(mixed_stores) -> None:
    """The candidate's run actually drove FakeRetrieverAgent with top_k=8 (not 5):
    its recorded trace emits 8 context chunks (the retriever's output reflects the
    mutated knob), proving the open-Knobs edit flowed through to the host's agent."""
    specs, atts, ts, rid = mixed_stores
    incumbent = specs.get(rid)
    cid = _candidate_id(incumbent, top_k=8)

    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="knob", target="ret", payload={"knobs": {"top_k": 8}}))
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cid, "t1", 0.80)
    eng = build_engine((specs, atts, ts, None),
                       judge=judge, architect=Architect(llm, model="architect-1"),
                       suite=make_suite(), host=FakeHostMAS())
    eng.evolve_cycle(0)

    # the candidate's trace was sunk: find the run over the candidate spec,
    # and assert its retirever step reflects top_k=8 (8 chunks + the label).
    cand_trace = next(t for t in ts.all(cid) if t.ok)
    ret_step = next(s for s in cand_trace.steps if s.node_id == "ret")
    assert "top_k=8" in ret_step.response_out
    assert ret_step.response_out.count("chunk-") == 8


def test_mixed_pipeline_retriever_costs_zero_tokens(mixed_stores) -> None:
    """Kind-aware cost: the retriever node costs ZERO tokens (its real cost is wall
    clock, captured by `latency_ms`) — so the candidate's token total reflects only
    the llm answer node, the design's cost-cap convention."""
    specs, atts, ts, rid = mixed_stores
    incumbent = specs.get(rid)
    cid = _candidate_id(incumbent, top_k=8)

    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="knob", target="ret", payload={"knobs": {"top_k": 8}}))
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cid, "t1", 0.80)
    eng = build_engine((specs, atts, ts, None),
                       judge=judge, architect=Architect(llm, model="architect-1"),
                       suite=make_suite(), host=FakeHostMAS())
    r = eng.evolve_cycle(0)

    cand_trace = next(t for t in ts.all(cid) if t.ok)
    ret_step = next(s for s in cand_trace.steps if s.node_id == "ret")
    assert ret_step.perf.tokens == 0          # non-llm node: time, not tokens
    # the cycle recorded wall-clock latency (the bounds the non-llm cost)
    assert r.latency_ms > 0.0


def test_mixed_pipeline_non_tunable_knob_is_rejected(mixed_stores) -> None:
    """The open-Knobs GUARD on the production path: the Architect proposes a `knob`
  edit to an extra NOT in the retriever's `tunable` -> apply_knob raises
  MutationError, routed by next_attempt to `lint_rejected` (E5) -> nothing committed,
  the loop is unchanged."""
    specs, atts, ts, rid = mixed_stores
    incumbent = specs.get(rid)

    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="knob", target="ret",
                     payload={"knobs": {"endpoint": "http://x"}}))   # not in tunable
    judge = ScriptedJudge().set_base(0.5)
    eng = build_engine((specs, atts, ts, None),
                       judge=judge, architect=Architect(llm, model="architect-1"),
                       suite=make_suite(), host=FakeHostMAS())
    r = eng.evolve_cycle(0)

    assert not r.attempted                       # surfaced as a rejection, not a proposal
    assert r.architect_status is not None
    assert r.architect_status.status == "lint_rejected"
    assert "endpoint" in r.architect_status.rejected_reasons[0].message
    assert specs.active_id() == rid               # incumbent untouched
