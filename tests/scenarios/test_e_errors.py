"""Phase-10 scripted E2E scenarios — the spec's E1..E10 error guarantees.

Every test drives the REAL organs on fakes (no `ScriptedArchitect`): the Architect
is the real one over a `ScriptedLLM`, the SuiteRunner/Judge/Gatekeeper/stores are
real. Determinism comes from pre-computing each candidate's content-addressed
`spec_id` (helpers in conftest) and scripting the ScriptedJudge/Host against it
before the run.

These are the spec's error-safety guarantees, made executable end-to-end:
  E1 noise/margin boundary · E2 rubric bump · E3 budget cap · E4 mid-run crash +
    unrunnable · E5 malformed spec never reaches the SuiteRunner · E6 promoted-
    then-regresses rollback · E7 architect dedup skip · E8 plateau stops the loop ·
  E9 grader outage: bounded retry, no fabricated score · E10 reload restores the
  incumbent + failed write leaves `active` unchanged.
"""

from __future__ import annotations

import pytest

import archforge.models as mm
from archforge.engine import CycleAborted, EngineConfig
from archforge.gatekeeper import Action, Gatekeeper
from archforge.host import FakeHostMAS
from archforge.judge.base import SuiteAggregate
from archforge.llm import LLMError, ScriptedLLM
from archforge.stores import SpecStore
from archforge.stores.spec_store import UnknownSpecError
from archforge.suite import SuiteRun, SuiteRunner

from tests.scenarios.conftest import (
    SUITE_ID, build_engine, cand_id, candidate_spec,
    judge_for, llm_noop, llm_malformed, llm_proposal, make_suite, real_architect,
)

# --------------------------------------------------------------------------- #
# E1 — noise/margin boundary (τ)
# --------------------------------------------------------------------------- #


def test_e1a_noise_inside_tau_is_discarded(stores, seeded) -> None:
    rid, inc = seeded
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    eng = build_engine(stores, judge=judge_for((rid, 0.60), (cid, 0.62)),
                       architect=real_architect(llm), suite=make_suite())

    r = eng.evolve_cycle(0)

    assert r.attempted and r.decision is not None
    assert r.decision.action is Action.DISCARD          # +0.02 < τ=0.05
    assert r.decision.by_rule == "below_margin"
    assert stores[0].active_id() == rid                 # active unchanged
    assert stores[1].require(r.applied_attempt_id).verdict is mm.Verdict.REJECTED


def test_e1b_clear_win_past_tau_is_promoted(stores, seeded) -> None:
    rid, inc = seeded
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    eng = build_engine(stores, judge=judge_for((rid, 0.60), (cid, 0.66)),
                       architect=real_architect(llm), suite=make_suite())

    r = eng.evolve_cycle(0)

    assert r.decision is not None and r.decision.action is Action.AUTO_PROMOTE
    assert r.decision.margin == pytest.approx(0.06)
    assert stores[0].active_id() == cid                 # active moved (the click)


# --------------------------------------------------------------------------- #
# E5 — a malformed candidate never reaches the SuiteRunner
# --------------------------------------------------------------------------- #


def test_e5_malformed_candidate_lint_rejected_no_suite_run(stores, seeded) -> None:
    rid, inc = seeded
    judge = judge_for((rid, 0.55))                        # incumbent baseline, never used
    before_specs = stores[0].known_ids()
    llm = ScriptedLLM().respond_json(llm_malformed())    # add orphan node 'v'
    eng = build_engine(stores, judge=judge, architect=real_architect(llm),
                       suite=make_suite())

    r = eng.evolve_cycle(0)

    assert not r.attempted
    assert r.architect_status is not None and r.architect_status.status == "lint_rejected"
    assert stores[0].active_id() == rid                 # nothing committed, nothing moved
    assert stores[0].known_ids() == before_specs        # no candidate Spec was persisted
    assert judge.scored == []                            # the SuiteRunner/Judge NEVER ran


# --------------------------------------------------------------------------- #
# E7 — the Architect dedup-skips an already-rejected (parent, kind, target)
# --------------------------------------------------------------------------- #


def test_e7_architect_dedup_skips_repeated_change(stores, seeded) -> None:
    rid, inc = seeded
    # cycle 1: a prompt_edit on 'a' that loses -> REJECTED
    # cycle 2: the SAME change -> the Architect blocks on the prior REJECTED (E7)
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    llm = (ScriptedLLM()
           .respond_json(llm_proposal(kind="prompt_edit", target="a",
                                       payload={"prompt": "p1"}))
           .respond_json(llm_proposal(kind="prompt_edit", target="a",
                                       payload={"prompt": "p1"})))
    judge = judge_for((rid, 0.60), (cid, 0.58))         # +0.02 -> discard
    eng = build_engine(stores, judge=judge, architect=real_architect(llm),
                       suite=make_suite())

    eng.evolve_cycle(0)                                  # cycle 1: discards (REJECTED)
    scored_after_c1 = len(judge.scored)
    specs_after_c1 = len(stores[0].known_ids())
    eng.evolve_cycle(1)                                  # cycle 2: dedup-plateaus

    assert len(stores[0].known_ids()) == specs_after_c1  # no second candidate committed
    assert len(judge.scored) == scored_after_c1          # SuiteRunner never re-ran
    assert len(llm.calls) == 2                            # the Architect WAS asked twice
    assert stores[0].active_id() == rid                   # never moved


# --------------------------------------------------------------------------- #
# E8 — a plateau stops the loop (K consecutive no-promotion cycles)
# --------------------------------------------------------------------------- #


def test_e8_plateau_stops_the_loop_after_k(stores, seeded) -> None:
    rid, inc = seeded
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    llm = (ScriptedLLM()
           .respond_json(llm_proposal(kind="prompt_edit", target="a",
                                       payload={"prompt": "p1"}))      # cycle 1 -> promote
           .respond_json(llm_noop())                                   # cycle 2 -> plateau
           .respond_json(llm_noop())                                   # cycle 3 -> plateau
           .respond_json(llm_noop()))                                  # cycle 4 -> plateau (K=3)
    judge = judge_for((rid, 0.60), (cid, 0.66))
    eng = build_engine(stores, judge=judge, architect=real_architect(llm),
                       suite=make_suite(),
                       config=EngineConfig(max_cycles=50, plateau_cycles=3))

    lr = eng.evolve_loop()

    assert lr.promotions == 1
    assert lr.plateaued is True
    assert lr.cycles_run == 4                            # 1 promote + 3 no-promotion
    assert stores[0].active_id() == cid                  # the promotion stuck


# --------------------------------------------------------------------------- #
# E3 — budget cap aborts cleanly with the incumbent untouched
# --------------------------------------------------------------------------- #


def test_e3a_per_cycle_token_cap_aborts_incumbent_intact(stores, seeded) -> None:
    rid, inc = seeded
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))
    judge = judge_for((rid, 0.60), (cid, 0.66))
    eng = build_engine(stores, judge=judge, architect=real_architect(llm),
                       suite=make_suite(),
                       config=EngineConfig(max_tokens_per_cycle=0))

    with pytest.raises(CycleAborted) as ei:
        eng.evolve_cycle(0)
    assert "token cap" in ei.value.reason.lower()
    assert stores[0].active_id() == rid                  # abort before apply -> intact


def test_e3b_total_budget_aborts_before_any_cycle_no_llm_call(stores, seeded) -> None:
    rid, inc = seeded
    llm = ScriptedLLM()                                 # nothing queued
    eng = build_engine(stores, judge=judge_for(),
                       architect=real_architect(llm), suite=make_suite(),
                       config=EngineConfig(max_cycles=50, plateau_cycles=50,
                                           max_tokens_total=0))

    lr = eng.evolve_loop()

    assert lr.aborted is True
    assert lr.cycles_run == 0
    assert "budget" in lr.abort_reason.lower()
    assert stores[0].active_id() == rid
    assert len(llm.calls) == 0                           # the Architect was never called


# --------------------------------------------------------------------------- #
# E4 — a mid-run crash sinks a partial trace and is rejected as unrunnable
# --------------------------------------------------------------------------- #


def test_e4_mid_run_crash_partial_trace_unrunnable_discarded(stores, seeded) -> None:
    rid, inc = seeded
    # Candidate adds node 'v'; the Host is scripted to CRASH on 'v'. The incumbent
    # (nodes=[a]) runs clean; the candidate's run dies at 'v' -> partial trace,
    # the task fails -> unrunnable -> DISCARD before any margin math (E4).
    vnode = {"node_id": "v", "role": "verifier", "system_prompt": "pv",
             "model": "gpt", "tools": ["t0"]}
    payload = {"node": vnode, "wiring": {"in_edges": [["a", "sequence"]]}}
    cid = cand_id(inc, kind="add_node", target="v", payload=payload)
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="add_node", target="v", payload=payload))
    host = FakeHostMAS(node_scripts={"v": {"crash_on": lambda i: True}})
    judge = judge_for((rid, 0.60), (cid, 0.99))          # incumbent clean @0.60
    eng = build_engine(stores, judge=judge, architect=real_architect(llm),
                       suite=make_suite(), host=host)

    r = eng.evolve_cycle(0)

    assert r.decision is not None and r.decision.action is Action.DISCARD
    assert r.decision.by_rule == "unrunnable"
    assert stores[0].active_id() == rid                 # the incumbent is untouched
    # E4 partial trace: the candidate's run IS sunk, with ok=False.
    cand_traces = stores[2].all(cid)
    assert cand_traces and not cand_traces[-1].ok
    assert cand_traces[-1].error                        # a recorded crash reason
    # no fabricated score for the crashed candidate: only the incumbent was scored
    assert any(s.spec_id == rid for s in judge.scored)
    assert not any(s.spec_id == cid for s in judge.scored)


# --------------------------------------------------------------------------- #
# E6 + I3 — promote, then regress -> rollback pointer swap; lineage preserved
# --------------------------------------------------------------------------- #


def test_e6_promote_then_regress_rolls_back_lineage_preserved(stores, seeded) -> None:
    rid, inc = seeded
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))
    judge = judge_for((rid, 0.55), (cid, 0.66))         # +0.11 -> promote
    eng = build_engine(stores, judge=judge, architect=real_architect(llm),
                       suite=make_suite())
    r = eng.evolve_cycle(0)
    assert r.decision.action is Action.AUTO_PROMOTE
    attempt_id = r.applied_attempt_id
    assert stores[0].active_id() == cid

    # post-promotion regression on the standard suite: drop 0.66 -> 0.55 (>= δ=0.07)
    gk = Gatekeeper(stores[0], stores[1], thresholds=mm.Thresholds(delta=0.07))
    d = gk.rollback(attempt_id, regressed_mean=0.55, pre_promotion_mean=0.66)
    assert d.action is Action.ROLLBACK and d.by_rule == "rollback"

    # pointer swapped to the parent; candidate archived, NEVER deleted (E6/I3)
    assert stores[0].active_id() == rid
    assert stores[0].is_archived(cid) is True
    assert stores[0].has(cid)                            # file still present
    assert stores[1].require(attempt_id).verdict is mm.Verdict.ROLLED_BACK
    # I3: lineage still walks through the archived node to the root
    assert stores[0].lineage(cid)[-1] == rid
    assert rid in stores[0].lineage(cid)


# --------------------------------------------------------------------------- #
# E9 — grader outage: bounded retry; on exhaustion, NO fabricated score
# --------------------------------------------------------------------------- #


def test_e9_grader_outage_folds_to_unrunnable_no_fabricated_score(stores, seeded) -> None:
    rid, inc = seeded
    # The REAL SuiteRunner with judge_retries=0: a single outage on the candidate
    # folds to None -> task failed -> unrunnable -> no fabricated score (E9).
    judge = judge_for((rid, 0.60))                       # incumbent scripted; candidate outages
    runner = SuiteRunner(FakeHostMAS(), judge, stores[2],
                         epsilon=0.25, judge_retries=0, backoff=lambda _j: 0.0)
    inc_run = runner.run_suite(inc, make_suite(), R=1)
    scored_inc = len(judge.scored)

    judge.raise_on_next(LLMError)                         # armed for the candidate's only score
    cand = candidate_spec(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    cand_run = runner.run_suite(cand, make_suite(), R=1)

    assert cand_run.unrunnable is True                   # the only task failed
    assert cand_run.scores == []                         # nothing fabricated
    assert len(judge.scored) == scored_inc                # no candidate RunScore recorded

    # the Gatekeeper discards an unrunnable candidate before any margin math
    specs, atts, _, _ = stores
    att = mm.Attempt(candidate_spec_id=cand.compute_spec_id(), parent_spec_id=rid,
                     change=mm.Change.for_kind(mm.ChangeKind.PROMPT_EDIT, "a", "x", "r"),
                     verdict=mm.Verdict.PROMOTED)
    aid = atts.append(att)
    gk = Gatekeeper(specs, atts, thresholds=mm.Thresholds())
    d = gk.decide(aid, cand_run, inc_run)
    assert d.action is Action.DISCARD and d.by_rule == "unrunnable"


def test_e9b_grader_outage_retries_then_recovers_real_score(stores, seeded) -> None:
    rid, inc = seeded
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    judge = judge_for((rid, 0.60), (cid, 0.66))
    runner = SuiteRunner(FakeHostMAS(), judge, stores[2], epsilon=0.25, judge_retries=2,
                         backoff=lambda _j: 0.0)
    runner.run_suite(inc, make_suite(), R=1)             # incumbent scores fine
    judge.raise_on_next(LLMError)                         # candidate attempt 1 outages
    cand = candidate_spec(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    cand_run = runner.run_suite(cand, make_suite(), R=1)

    # bounded retry recovered: a REAL (scripted) score, not a fabrication/pending
    assert cand_run.unrunnable is False
    assert cand_run.scores and cand_run.mean == pytest.approx(0.66)


# --------------------------------------------------------------------------- #
# E2 + I5 — a rubric bump stamps a fresh baseline; cross-rubric never promotes
# --------------------------------------------------------------------------- #


def test_e2_suite_runner_stamps_suite_rubric_per_run(stores, seeded) -> None:
    rid, inc = seeded
    judge = judge_for((rid, 0.55))
    runner = SuiteRunner(FakeHostMAS(), judge, stores[2], epsilon=0.25)

    run_r0 = runner.run_suite(inc, make_suite(rubric_id="r0"), R=1)
    run_r1 = runner.run_suite(inc, make_suite(rubric_id="r1"), R=1)

    # E2: each run carries its OWN rubric stamp; a rubric bump is never silently
    # matched to a stale baseline.
    assert run_r0.rubric_id == "r0" and run_r1.rubric_id == "r1"
    assert run_r0.suite_id == run_r1.suite_id == SUITE_ID


def test_i5_cross_rubric_comparison_is_discarded_active_kept(stores, seeded) -> None:
    rid, inc = seeded
    # a real prior promotion (same rubric) -> active moves; then a candidate
    # measured under a DIFFERENT rubric is blocked (I5), active stays put.
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))
    eng = build_engine(stores, judge=judge_for((rid, 0.55), (cid, 0.66)),
                       architect=real_architect(llm), suite=make_suite(rubric_id="r0"))
    eng.evolve_cycle(0)
    assert stores[0].active_id() == cid

    inc_run = SuiteRun(spec_id=cid, suite_id=SUITE_ID, rubric_id="r0", repeats=1,
                      n_tasks=1, n_crashed=0, unrunnable=False, scores=[],
                      aggregate=SuiteAggregate(suite_id=SUITE_ID, rubric_id="r0",
                                               mean=0.55, n_runs=1, per_task={},
                                               confidence=1.0), mean=0.55)
    cand_run = SuiteRun(spec_id="cross", suite_id=SUITE_ID, rubric_id="other", repeats=1,
                       n_tasks=1, n_crashed=0, unrunnable=False, scores=[],
                       aggregate=SuiteAggregate(suite_id=SUITE_ID, rubric_id="other",
                                                mean=0.99, n_runs=1, per_task={},
                                                confidence=1.0), mean=0.99)
    att = mm.Attempt(candidate_spec_id="cross", parent_spec_id=cid,
                     change=mm.Change.for_kind(mm.ChangeKind.PROMPT_EDIT, "a", "x", "r"),
                     verdict=mm.Verdict.PROMOTED)
    aid = stores[1].append(att)
    gk = Gatekeeper(stores[0], stores[1], thresholds=mm.Thresholds())
    d = gk.decide(aid, cand_run, inc_run)
    assert d.action is Action.DISCARD and d.by_rule == "cross_geometry"
    assert stores[0].active_id() == cid                  # no cross-rubric delta moves it


# --------------------------------------------------------------------------- #
# E10 — reload restores the incumbent; a failed set_active leaves `active` intact
# --------------------------------------------------------------------------- #


def test_e10_reload_restores_incumbent_and_failed_write_keeps_active(stores, seeded) -> None:
    rid, inc = seeded
    # a real promotion writes a new incumbent pointer to disk
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))
    eng = build_engine(stores, judge=judge_for((rid, 0.55), (cid, 0.66)),
                       architect=real_architect(llm), suite=make_suite())
    eng.evolve_cycle(0)
    assert stores[0].active_id() == cid

    # E10a: a fresh SpecStore over the SAME root restores the incumbent
    reloaded = SpecStore(stores[3])
    assert reloaded.active_id() == cid
    assert reloaded.get(cid).nodes == stores[0].get(cid).nodes

    # E10b: a failed write (set_active on an unknown id) leaves the pointer intact
    with pytest.raises(UnknownSpecError):
        reloaded.set_active("deadbeefdeadbeef")
    assert reloaded.active_id() == cid                   # pointer not corrupted
