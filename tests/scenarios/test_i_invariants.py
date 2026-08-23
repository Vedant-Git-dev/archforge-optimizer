"""Phase-10 scripted E2E scenarios — the spec's I1..I5 invariants.

Companion to `test_e_errors.py`; here we pin the structural invariants end-to-end
on the REAL organs (real Architect over ScriptedLLM, real SuiteRunner/Gatekeeper,
real filesystem stores):

  I1 exactly one active Spec; only the Gatekeeper moves the pointer (+ human
     approve/reject, which live in the Gatekeeper)
  I2 immutability by content hash — a re-commit of the same content returns the
     same id and refuses to overwrite; a mutated Spec gets a new id
  I3 lineage reachability preserved across promotion + rollback + archive
  I4 a structural win ALWAYS queues for human, never auto-promotes (even at huge
     margins); human approve is the only path that then moves active
  I5 covered in test_e_errors.py (cross-rubric/cross-suite discard)
"""

from __future__ import annotations

import pytest

import archforge.models as mm
from archforge.gatekeeper import Action, Gatekeeper
from archforge.llm import ScriptedLLM
from archforge.stores.spec_store import NoActiveSpecError

from tests.scenarios.conftest import (
    build_engine, cand_id, candidate_spec, judge_for, llm_proposal, make_suite,
    real_architect,
)

# --------------------------------------------------------------------------- #
# I1 — exactly one active Spec; only the Gatekeeper moves the pointer
# --------------------------------------------------------------------------- #


def test_i1_one_active_spec_pointer_moves_only_on_promotion(stores, seeded) -> None:
    rid, inc = seeded
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))

    # a BELOW-margin cycle: DISCARD. The Gatekeeper is the only mover of `active`,
    # and discards do not move it -> exactly one active id stays put (I1).
    eng = build_engine(stores, judge=judge_for((rid, 0.60), (cid, 0.61)),
                       architect=real_architect(llm), suite=make_suite())
    eng.evolve_cycle(0)
    assert stores[0].active_id() == rid

    # then a clear win -> AUTO_PROMOTE moves the pointer. Still exactly one active.
    # NB: reuse a DIFFERENT small kind here (knob, not prompt_edit) — cycle 1's
    # REJECTED prompt_edit on (rid, "a") dedup-blocks another (E7), so a same-kind
    # second cycle would plateau instead of promoting, hiding the pointer move.
    llm2 = ScriptedLLM().respond_json(
        llm_proposal(kind="knob", target="a",
                     payload={"knobs": {"temperature": 0.3}}))
    cid2 = cand_id(inc, kind="knob", target="a",
                  payload={"knobs": {"temperature": 0.3}})
    eng2 = build_engine(stores, judge=judge_for((rid, 0.60), (cid2, 0.70)),
                        architect=real_architect(llm2), suite=make_suite())
    eng2.evolve_cycle(0)
    assert stores[0].active_id() == cid2
    # I1 singularity: `active()` returns exactly one Spec (no ambiguity)
    assert stores[0].active().spec_id == cid2


def test_i1_no_active_raises_cleanly(stores) -> None:
    # before any incumbent is set, `active()` raises a distinct error (no silent None)
    with pytest.raises(NoActiveSpecError):
        stores[0].active()
    assert stores[0].active_id() is None


# --------------------------------------------------------------------------- #
# I2 — immutability by content hash
# --------------------------------------------------------------------------- #


def test_i2_recommit_same_content_idempotent_refuses_overwrite(stores, seeded) -> None:
    rid, inc = seeded
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    cand = candidate_spec(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})

    a = stores[0].commit(cand, parent_spec_id=rid, status=mm.SpecStatus.CANDIDATE)
    # re-commit IDENTICAL content+parent -> same id, existing file untouched (I2)
    b = stores[0].commit(cand, parent_spec_id=rid, status=mm.SpecStatus.CANDIDATE)
    assert a == b == cid

    # a DIFFERENT content gets a different id (content-addressed)
    cand_p2 = candidate_spec(inc, kind="prompt_edit", target="a", payload={"prompt": "p2"})
    p2 = stores[0].commit(cand_p2, parent_spec_id=rid, status=mm.SpecStatus.CANDIDATE)
    assert p2 != cid
    # the original is still present and unchanged (immutability — never overwritten)
    assert stores[0].get(cid).spec_id == cid
    assert stores[0].get(cid).nodes[0].system_prompt == "p1"


# --------------------------------------------------------------------------- #
# I3 — lineage reachability preserved across promote + rollback + archive
# --------------------------------------------------------------------------- #


def test_i3_lineage_reachable_through_promoted_then_archived(stores, seeded) -> None:
    rid, inc = seeded
    cid = cand_id(inc, kind="prompt_edit", target="a", payload={"prompt": "p1"})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))
    eng = build_engine(stores, judge=judge_for((rid, 0.55), (cid, 0.66)),
                       architect=real_architect(llm), suite=make_suite())
    r = eng.evolve_cycle(0)                                  # promote
    aid = r.applied_attempt_id

    # lineage before rollback walks candidate -> root
    assert stores[0].lineage(cid) == [cid, rid]

    # rollback: active -> parent, candidate archived (never deleted)
    gk = Gatekeeper(stores[0], stores[1], thresholds=mm.Thresholds(delta=0.07))
    gk.rollback(aid, regressed_mean=0.55, pre_promotion_mean=0.66)
    assert stores[0].is_archived(cid) is True
    assert stores[0].has(cid)                              # never deleted (I3 + E6)

    # I3: lineage is STILL reachable through the archived node — archived specs
    # remain queryable parents, so a later cycle's lineage walk passes through them
    assert stores[0].lineage(cid) == [cid, rid]
    assert rid in stores[0].lineage(cid)


# --------------------------------------------------------------------------- #
# I4 — a structural win ALWAYS queues for human, never auto-promotes
# --------------------------------------------------------------------------- #


def test_i4_structural_win_queues_even_at_huge_margin(stores, seeded) -> None:
    rid, inc = seeded
    vnode = {"node_id": "v", "role": "verifier", "system_prompt": "pv",
             "model": "gpt", "tools": ["t0"]}
    payload = {"node": vnode, "wiring": {"in_edges": [["a", "sequence"]]}}
    cid = cand_id(inc, kind="add_node", target="v", payload=payload)
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="add_node", target="v", payload=payload))
    eng = build_engine(stores, judge=judge_for((rid, 0.50), (cid, 0.99)),
                       architect=real_architect(llm), suite=make_suite())

    r = eng.evolve_cycle(0)

    # never auto-promote a structural change, regardless of the margin (I4)
    assert r.decision is not None and r.decision.action is Action.QUEUE_HUMAN
    assert r.decision.by_rule == "structural_wins_queue"
    assert stores[0].active_id() == rid                  # active did NOT move
    assert stores[1].require(r.applied_attempt_id).verdict is mm.Verdict.PENDING_HUMAN
    assert stores[0].is_archived(cid) is False            # not archived; awaiting human


def test_i4_human_approve_is_the_only_path_that_then_moves_active(stores, seeded) -> None:
    rid, inc = seeded
    vnode = {"node_id": "v", "role": "verifier", "system_prompt": "pv",
             "model": "gpt", "tools": ["t0"]}
    payload = {"node": vnode, "wiring": {"in_edges": [["a", "sequence"]]}}
    cid = cand_id(inc, kind="add_node", target="v", payload=payload)
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="add_node", target="v", payload=payload))
    eng = build_engine(stores, judge=judge_for((rid, 0.50), (cid, 0.99)),
                       architect=real_architect(llm), suite=make_suite())
    r = eng.evolve_cycle(0)
    aid = r.applied_attempt_id
    assert stores[0].active_id() == rid

    # human approve (inside the Gatekeeper) moves active -> the candidate; reject
    # would leave it. Promote then verify a SECOND queued change can only move via
    # approve again (the CLI never mutates the pointer directly).
    gk = Gatekeeper(stores[0], stores[1], thresholds=mm.Thresholds())
    gk.approve(aid)
    assert stores[0].active_id() == cid
    assert stores[1].require(aid).verdict is mm.Verdict.PROMOTED

    # a reject on a freshly-queued structural change leaves active alone + archives
    w_payload = {"node": {"node_id": "w", "role": "critic", "system_prompt": "pw",
                          "model": "gpt", "tools": ["t0"]},
                 "wiring": {"in_edges": [["v", "sequence"]]}}
    cid2 = cand_id(  # a second structural change off the NOW-active cid
        stores[0].get(cid), kind="add_node", target="w", payload=w_payload)
    llm2 = ScriptedLLM().respond_json(
        llm_proposal(kind="add_node", target="w", payload=w_payload))
    eng2 = build_engine(stores, judge=judge_for((cid, 0.50), (cid2, 0.99)),
                        architect=real_architect(llm2), suite=make_suite())
    r2 = eng2.evolve_cycle(0)
    aid2 = r2.applied_attempt_id
    assert r2.decision.action is Action.QUEUE_HUMAN
    assert stores[0].active_id() == cid                  # still the first approvee

    gk.reject(aid2)
    assert stores[0].active_id() == cid                  # reject kept the incumbent
    assert stores[1].require(aid2).verdict is mm.Verdict.REJECTED
    assert stores[0].is_archived(cid2) is True            # archived as a dead end
