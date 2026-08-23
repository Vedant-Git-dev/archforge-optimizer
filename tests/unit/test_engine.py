"""Unit tests for the evolve engine (Phase 9) — the P-E-C orchestrator + loop.

Pins the orchestration-level behaviours on top of the four organs:
  E1   one cycle: a clear small win promotes the candidate (active pointer moves)
  I4   a structural win queues for human review (active does NOT move, no auto-promote)
  E8   K consecutive no-promotion cycles -> plateau (loop stops, status surfaced)
  E3   per-cycle token cap aborts cleanly with the incumbent untouched
  (baseline caching: re-running a known incumbent reuses its suite run)

Uses the REAL `Engine` with scripted fakes injected directly (spec §9): ScriptedArchitect,
ScriptedJudge, FakeHostMAS, real stores on a tmp dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import archforge.models as m
from archforge.architect import ScriptedArchitect
from archforge.engine import CycleAborted, Engine, EngineConfig, LoopResult
from archforge.gatekeeper import Action
from archforge.host import FakeHostMAS
from archforge.host.base import Task
from archforge.judge import ScriptedJudge
from archforge.stores import AttemptStore, SpecStore, TraceStore
from archforge.suite import Suite


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


def N(nid: str, *, prompt: str = "p0", model: str = "gpt") -> m.Node:
    return m.Node(node_id=nid, role="r", system_prompt=prompt, model=model, tools=["t0"])


def id_with_parent(nodes: list[m.Node], parent: str | None) -> str:
    s = m.Spec(nodes=nodes)
    s.parent_spec_id = parent
    return s.compute_spec_id()


@pytest.fixture
def stores(tmp_path: Path):
    root = tmp_path / ".archforge"
    spec_store = SpecStore(root)
    att_store = AttemptStore(root)
    trace_store = TraceStore(root)
    root_spec = m.Spec(nodes=[N("a", prompt="p0")], edges=[])
    rid = spec_store.commit(root_spec, parent_spec_id=None, status=m.SpecStatus.INCUMBENT)
    spec_store.set_active(rid)
    return spec_store, att_store, trace_store, rid


def suite_() -> Suite:
    return Suite(suite_id="S", rubric_id="default-v1", tasks=[Task(task_id="t1", input="q")])


def _engine(stores, *, architect, judge, thresholds=None, config=None):
    specs, atts, ts, _ = stores
    return Engine(
        host=FakeHostMAS(), judge=judge, architect=architect,
        spec_store=specs, attempt_store=atts, trace_store=ts, suite=suite_(),
        thresholds=thresholds or m.Thresholds(tau=0.05, delta=0.07),
        config=config or EngineConfig(max_cycles=20, repeats=1, plateau_cycles=5),
    )


# --------------------------------------------------------------------------- #
# one cycle
# --------------------------------------------------------------------------- #


def test_small_win_is_auto_promoted_active_moves(stores) -> None:
    specs, atts, _, rid = stores
    cand_id = id_with_parent([N("a", prompt="p1")], rid)
    arch = (ScriptedArchitect()
            .propose(m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "tighten", "grounding"),
                     {"prompt": "p1"}))
    judge = (ScriptedJudge()
             .set_aggregate(rid, "t1", 0.55)
             .set_aggregate(cand_id, "t1", 0.66))   # +0.11 >= τ
    engine = _engine(stores, architect=arch, judge=judge)

    r = engine.evolve_cycle(0)

    assert r.attempted and r.decision is not None
    assert r.decision.action is Action.AUTO_PROMOTE
    assert r.incumbent_mean == pytest.approx(0.55)
    assert r.candidate_mean == pytest.approx(0.66)
    assert specs.active_id() == cand_id               # active moved
    assert atts.require(r.applied_attempt_id).verdict is m.Verdict.PROMOTED


def test_structural_win_queues_human_active_does_not_move(stores) -> None:
    # I4 at the orchestration level: structural queues even on a clear win
    specs, _, _, rid = stores
    cand = m.Spec(nodes=[N("a"), N("v")],
                  edges=[m.Edge(from_="a", to="v", type=m.EdgeType.SEQUENCE)])
    cand.parent_spec_id = rid
    cand_id = cand.compute_spec_id()
    arch = (ScriptedArchitect()
            .propose(m.Change.for_kind(m.ChangeKind.ADD_NODE, "v", "add verifier", "verify"),
                     {"node": N("v"),
                      "wiring": {"in_edges": [("a", m.EdgeType.SEQUENCE)], "out_edges": []}}))
    judge = (ScriptedJudge()
             .set_aggregate(rid, "t1", 0.55)
             .set_aggregate(cand_id, "t1", 0.80))    # big win, but structural
    engine = _engine(stores, architect=arch, judge=judge)

    r = engine.evolve_cycle(0)

    assert r.decision is not None and r.decision.action is Action.QUEUE_HUMAN
    assert specs.active_id() == rid                   # active UNCHANGED
    atts = stores[1]
    assert atts.require(r.applied_attempt_id).verdict is m.Verdict.PENDING_HUMAN


def test_below_margin_discards_active_unchanged(stores) -> None:
    specs, atts, _, rid = stores
    cand_id = id_with_parent([N("a", prompt="p1")], rid)
    arch = (ScriptedArchitect()
            .propose(m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "tweak", "x"),
                     {"prompt": "p1"}))
    judge = (ScriptedJudge()
             .set_aggregate(rid, "t1", 0.60)
             .set_aggregate(cand_id, "t1", 0.63))    # +0.03 < τ
    engine = _engine(stores, architect=arch, judge=judge)

    r = engine.evolve_cycle(0)

    assert r.decision is not None and r.decision.action is Action.DISCARD
    assert specs.active_id() == rid                   # active unchanged
    assert stores[1].require(r.applied_attempt_id).verdict is m.Verdict.REJECTED


def test_no_proposal_is_not_attempted(stores) -> None:
    specs, _, _, _ = stores
    engine = _engine(stores, architect=ScriptedArchitect(), judge=ScriptedJudge())
    r = engine.evolve_cycle(0)
    assert not r.attempted
    assert r.decision is None and r.applied_attempt_id is None
    # active untouched
    assert specs.active_id() is not None


# --------------------------------------------------------------------------- #
# the loop (E8 plateau, E3 budget)
# --------------------------------------------------------------------------- #


def test_loop_plateaus_after_k_no_promotion_cycles(stores) -> None:
    # an architect that never proposes -> every cycle is "not attempted"
    engine = _engine(stores, architect=ScriptedArchitect(), judge=ScriptedJudge(),
                     config=EngineConfig(max_cycles=50, repeats=1, plateau_cycles=5))
    lr = engine.evolve_loop()

    assert isinstance(lr, LoopResult)
    assert lr.plateaued is True
    assert lr.cycles_run == 5                            # plateaued on the 5th
    assert lr.promotions == 0 and lr.aborted is False
    assert lr.final_incumbent_id == stores[3]            # never moved


def test_loop_promotes_then_plateaus_when_architect_runs_dry(stores) -> None:
    specs, _, _, rid = stores
    cand_id = id_with_parent([N("a", prompt="p1")], rid)
    arch = (ScriptedArchitect()
            .propose(m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "t", "r"), {"prompt": "p1"}))
    judge = (ScriptedJudge()
             .set_aggregate(rid, "t1", 0.55)
             .set_aggregate(cand_id, "t1", 0.70))
    engine = _engine(stores, architect=arch, judge=judge,
                     config=EngineConfig(max_cycles=50, repeats=1, plateau_cycles=5))

    lr = engine.evolve_loop()

    # cycle 1 promotes; cycles 2..6 plateau (architect queue empty) -> stop at 6
    assert lr.promotions == 1
    assert lr.plateaued is True
    assert specs.active_id() == cand_id                  # the promotion stuck


def test_per_cycle_token_cap_aborts_cleanly_incumbent_intact(stores) -> None:
    specs, _, _, rid = stores
    cand_id = id_with_parent([N("a", prompt="p1")], rid)
    arch = (ScriptedArchitect()
            .propose(m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "t", "r"), {"prompt": "p1"}))
    judge = ScriptedJudge().set_base(0.5)
    engine = _engine(stores, architect=arch, judge=judge,
                     config=EngineConfig(max_cycles=50, repeats=1, plateau_cycles=5,
                                         max_tokens_per_cycle=0))   # cap 0 -> aborts

    with pytest.raises(CycleAborted) as ei:
        engine.evolve_cycle(0)
    assert "token cap" in ei.value.reason.lower()
    # the abort fires BEFORE apply_decision, so the active pointer never moved (E3)
    assert specs.active_id() == rid


def test_total_token_budget_aborts_loop(stores) -> None:
    specs, _, _, rid = stores
    # never propose -> plateau fast would stop it first; use max so budget governs:
    # set plateau high so only the budget cap stops the loop
    engine = _engine(stores, architect=ScriptedArchitect(), judge=ScriptedJudge(),
                     config=EngineConfig(max_cycles=50, repeats=1, plateau_cycles=50,
                                         max_tokens_total=0))    # 0 budget
    lr = engine.evolve_loop()
    assert lr.aborted is True
    assert "budget" in lr.abort_reason.lower()
    assert specs.active_id() == rid


# --------------------------------------------------------------------------- #
# baseline cache — a stable incumbent reuses its suite run
# --------------------------------------------------------------------------- #


def test_baseline_cache_reuses_incumbent_run(stores) -> None:
    # Two proposing cycles against the SAME incumbent (cycle 1 does NOT promote,
    # so the incumbent is unchanged in cycle 2). Cycle 1 computes+catches the
    # baseline; cycle 2 must REUSE it (not re-score the incumbent).
    #
    # NB: cand2 uses a *different change kind* (model_swap not prompt_edit) on
    # purpose: E7 dedup blocks any (parent, kind, target) already REJECTED. If
    # both candidates were prompt_edit on "a", cand2 would dedup-plateau in
    # cycle 2 and score nothing — masking the cache. Distinct kinds isolate it.
    specs, _, _, rid = stores
    cand1_id = id_with_parent([N("a", prompt="p1")], rid)            # prompt_edit p0->p1
    cand2_id = id_with_parent([N("a", model="claude")], rid)        # model_swap gpt->claude
    # both candidates lose (below τ) so neither promotes -> incumbent stays rid
    judge = (ScriptedJudge()
             .set_aggregate(rid, "t1", 0.60)
             .set_aggregate(cand1_id, "t1", 0.61)
             .set_aggregate(cand2_id, "t1", 0.62))
    arch = ScriptedArchitect().propose(
        m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "t1", "r"), {"prompt": "p1"}
    ).propose(
        m.Change.for_kind(m.ChangeKind.MODEL_SWAP, "a", "t2", "r"), {"model": "claude"}
    )
    engine = _engine(stores, architect=arch, judge=judge)

    engine.evolve_cycle(0)
    after_c1 = len(judge.scored)          # rid baseline (1) + cand1 (1) = 2
    assert after_c1 == 2
    engine.evolve_cycle(1)
    after_c2 = len(judge.scored)
    # cache hit: cycle 2 reuses rid's baseline, scores ONLY cand2 -> +1 = 3.
    # (a broken cache would re-score rid -> +2 = 4.)
    assert after_c2 == 3
    assert specs.active_id() == rid       # neither candidate promoted


def test_baseline_cache_seeded_on_promote_no_rescore(stores) -> None:
    # Cycle 1: candidate1 BEATS the root incumbent and auto-promotes. The
    # engine seeds the just-scored candidate run into the baseline cache so the
    # NEXT cycle — now running against candidate1 AS the incumbent — reuses
    # that run instead of re-scoring it. Without the seed, `_baseline_for` misses
    # the cache (candidate1's run was the CANDIDATE run, never baselined) and
    # re-runs the suite: Judge run-to-run variance would flip the promoted mean
    # (e.g. 0.70 -> 0.55), discarding the score the Gatekeeper promoted on and
    # injecting noise into every margin thereafter. This is the fix that makes a
    # promoted incumbent keep its earned score.
    #
    # Cycle 2: candidate2 (a DIFFERENT change kind — E7 dedup would otherwise
    # block it as a rejected (parent, kind, target) repeat, scoring nothing and
    # masking the baseline question) loses to the new incumbent. The new
    # incumbent's baseline must be a CACHE HIT (0 extra scores); only candidate2
    # is scored.
    specs, _, _, rid = stores
    cand1_id = id_with_parent([N("a", prompt="p1")], rid)            # prompt_edit p0->p1
    # Cycle 2's candidate is a MODEL_SWAP off the NEW incumbent (cand1, prompt=p1),
    # NOT off rid — the Architect proposes against the active incumbent, which after
    # cycle 1's promote is cand1. So cand2 = cand1 with model gpt->claude.
    cand2_id = id_with_parent([N("a", prompt="p1", model="claude")], cand1_id)
    judge = (ScriptedJudge()
             .set_aggregate(rid, "t1", 0.55)       # root incumbent baseline
             .set_aggregate(cand1_id, "t1", 0.70)  # beats rid by +0.15 > tau -> promote
             .set_aggregate(cand2_id, "t1", 0.60))  # loses to cand1 (0.70) -> discard
    arch = (ScriptedArchitect()
            .propose(m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "t1", "r"),
                     {"prompt": "p1"})
            .propose(m.Change.for_kind(m.ChangeKind.MODEL_SWAP, "a", "t2", "r"),
                     {"model": "claude"}))
    engine = _engine(stores, architect=arch, judge=judge)

    r1 = engine.evolve_cycle(0)
    assert r1.decision is not None and r1.decision.action is Action.AUTO_PROMOTE
    assert specs.active_id() == cand1_id             # candidate1 is now incumbent
    after_c1 = len(judge.scored)                     # rid baseline (1) + cand1 (1) = 2
    assert after_c1 == 2

    r2 = engine.evolve_cycle(1)
    after_c2 = len(judge.scored)
    # FIX: cand1's run was seeded as its own baseline on promote, so cycle 2's
    # `_baseline_for(cand1)` is a cache HIT — only cand2 is scored (+1 = 3).
    # A broken/missing seed would re-score cand1 (+2 = 4), injecting Judge
    # noise into the promoted incumbent's mean.
    assert after_c2 == 3, (
        f"promoted incumbent should reuse its seeded baseline (3 scores); "
        f"got {after_c2} (a re-score of the promoted incumbent — the noise bug)")
    # the new incumbent's baseline mean is the score it was PROMOTED on, reused
    # verbatim — not a fresh (noisy) score
    assert r2.incumbent_mean == pytest.approx(0.70)
    assert r2.candidate_mean == pytest.approx(0.60)
    assert r2.decision is not None and r2.decision.action is Action.DISCARD
    assert specs.active_id() == cand1_id             # incumbent unchanged (cand2 lost)


# --------------------------------------------------------------------------- #
# per-cycle wall-clock cap (the cost fix for non-LLM-heavy pipelines — E3)
# --------------------------------------------------------------------------- #


def test_per_cycle_wall_clock_cap_aborts_cleanly_incumbent_intact(stores) -> None:
    # A non-LLM-heavy pipeline costs TIME, not tokens (its retriever/tool/rule nodes
    # emit perf.tokens=0). The token cap can't bound it, so the wall cap aborts a
    # cycle whose summed latency exceeds `max_wall_ms_per_cycle`. Same E3 contract
    # as the token cap: incumbent untouched (the abort fires BEFORE apply_decision).
    specs, _, _, rid = stores
    cand_id = id_with_parent([N("a", prompt="p1")], rid)
    arch = (ScriptedArchitect()
            .propose(m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "t", "r"), {"prompt": "p1"}))
    judge = ScriptedJudge().set_base(0.5)
    engine = _engine(stores, architect=arch, judge=judge,
                     config=EngineConfig(max_cycles=50, repeats=1, plateau_cycles=5,
                                         max_wall_ms_per_cycle=0.0))  # 0ms -> aborts

    with pytest.raises(CycleAborted) as ei:
        engine.evolve_cycle(0)
    assert "wall-clock" in ei.value.reason.lower()
    assert specs.active_id() == rid            # abort is pre-decision: active unmoved


def test_per_cycle_wall_clock_cap_none_never_aborts(stores) -> None:
    # Back-compat: the default (None = no limit) never aborts — existing suites'
    # latency is microseconds and the cap is opt-in. Mirrors `max_tokens_per_cycle=None`.
    specs, _, _, rid = stores
    cand_id = id_with_parent([N("a", prompt="p1")], rid)
    arch = (ScriptedArchitect()
            .propose(m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "t", "r"), {"prompt": "p1"}))
    judge = (ScriptedJudge()
             .set_aggregate(rid, "t1", 0.5)
             .set_aggregate(cand_id, "t1", 0.8))   # a clear win so the cycle completes
    engine = _engine(stores, architect=arch, judge=judge,
                     config=EngineConfig(max_cycles=50, repeats=1, plateau_cycles=5,
                                         max_wall_ms_per_cycle=None))   # no limit
    r = engine.evolve_cycle(0)
    assert r.attempted and r.latency_ms >= 0.0       # cycle completed; latency recorded
    assert specs.active_id() == cand_id             # promoted (the win applied)
