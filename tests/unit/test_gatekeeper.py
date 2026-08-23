"""Unit tests for the Gatekeeper + rollback (Phase 8) — the P-E-C *Commit* step.

Pins the hybrid gate's rules and the fail-closed property:
  I4  structural-always-gates (a structural win NEVER auto-promotes)
  E1  noise inside τ is discarded, a clear win is kept (margin boundary)
  E4  an unrunnable candidate is discarded before any margin math
  I5  cross-rubric/cross-suite comparisons are blocked
  E6  a promoted-then-regressed Spec rolls back to its parent (pointer swap),
      is archived (never deleted), verdict ROLLED_BACK
  E6' regression within the δ noise floor leaves the incumbent alone
  I1  only promote/rollback move the `active` pointer (discard/queue do not)
  E10 the verdict-flip mutator is idempotent

Uses the REAL SpecStore + AttemptStore on a tmp dir; SuiteRun is built directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import archforge.models as m
from archforge.gatekeeper import Action, Decision, Gatekeeper
from archforge.judge.base import SuiteAggregate
from archforge.stores import AttemptStore, SpecStore
from archforge.suite import SuiteRun


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


def N(nid: str, *, prompt: str = "p", model: str = "gpt") -> m.Node:
    return m.Node(node_id=nid, role="r", system_prompt=prompt, model=model, tools=["t0"])


def make_run(spec_id: str, mean: float, *, suite_id: str = "S", rubric_id: str = "R",
             unrunnable: bool = False) -> SuiteRun:
    agg = SuiteAggregate(suite_id=suite_id, rubric_id=rubric_id, mean=mean,
                         n_runs=1, per_task={}, confidence=1.0, unrunnable=unrunnable)
    return SuiteRun(spec_id=spec_id, suite_id=suite_id, rubric_id=rubric_id, repeats=1,
                    n_tasks=1, n_crashed=1 if unrunnable else 0,
                    unrunnable=unrunnable, scores=[], aggregate=agg, mean=mean)


def _spec_nodes(prompt: str = "p") -> list[m.Node]:
    return [N("a", prompt=prompt)]


@pytest.fixture
def stores(tmp_path: Path):
    root = tmp_path / ".archforge"
    specs = SpecStore(root)
    atts = AttemptStore(root)
    # root incumbent
    root_spec = m.Spec(nodes=_spec_nodes("p0"))
    rid = specs.commit(root_spec, parent_spec_id=None, status=m.SpecStatus.INCUMBENT)
    specs.set_active(rid)
    return specs, atts, rid


def _commit_candidate(specs: SpecStore, rid: str, prompt: str = "p1") -> str:
    cand = m.Spec(nodes=_spec_nodes(prompt))
    return specs.commit(cand, parent_spec_id=rid, status=m.SpecStatus.CANDIDATE)


def _commit_structural(specs: SpecStore, parent: str) -> str:
    cand = m.Spec(nodes=[N("a"), N("v")],
                  edges=[m.Edge(from_="a", to="v", type=m.EdgeType.SEQUENCE)])
    return specs.commit(cand, parent_spec_id=parent, status=m.SpecStatus.CANDIDATE)


def _put_attempt(atts: AttemptStore, *, candidate_spec_id: str, parent_spec_id: str,
                 change: m.Change, verdict: m.Verdict = m.Verdict.PROMOTED) -> str:
    att = m.Attempt(candidate_spec_id=candidate_spec_id, parent_spec_id=parent_spec_id,
                    change=change, verdict=verdict)
    return atts.append(att)


def _gatekeeper(specs: SpecStore, atts: AttemptStore, **kw) -> Gatekeeper:
    th = m.Thresholds(tau=0.05, delta=0.07)
    th = th.model_copy(update=kw) if kw else th
    return Gatekeeper(specs, atts, thresholds=th)


def _small_change(target: str = "a") -> m.Change:
    return m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, target, "tweak", "tighten")


def _struct_change(target: str = "v") -> m.Change:
    return m.Change.for_kind(m.ChangeKind.ADD_NODE, target, "add node", "verification")


# --------------------------------------------------------------------------- #
# promotion paths
# --------------------------------------------------------------------------- #


def test_small_win_auto_promotes_moves_active(stores) -> None:
    specs, atts, rid = stores
    cid = _commit_candidate(specs, rid, "p1")
    aid = _put_attempt(atts, candidate_spec_id=cid, parent_spec_id=rid, change=_small_change())

    gk = _gatekeeper(specs, atts)
    d = gk.decide(aid, make_run(cid, 0.66), make_run(rid, 0.60))

    assert d.action is Action.AUTO_PROMOTE
    assert d.margin == pytest.approx(0.06) and d.margin >= gk._th.tau
    applied = gk.apply_decision(d)
    assert applied.verdict is m.Verdict.PROMOTED
    assert specs.active_id() == cid                       # active moved (I1 via promote)
    assert specs.is_archived(cid) is False


def test_structural_win_queues_for_human_never_auto_promotes(stores) -> None:
    # invariant I4: structural ALWAYS gates, even on a clear win
    specs, atts, rid = stores
    cid = _commit_structural(specs, rid)
    aid = _put_attempt(atts, candidate_spec_id=cid, parent_spec_id=rid, change=_struct_change())

    gk = _gatekeeper(specs, atts)
    d = gk.decide(aid, make_run(cid, 0.80), make_run(rid, 0.60))

    assert d.action is Action.QUEUE_HUMAN and d.by_rule == "structural_wins_queue"
    applied = gk.apply_decision(d)
    assert applied.verdict is m.Verdict.PENDING_HUMAN
    assert specs.active_id() == rid                       # active UNCHANGED (I1)


# --------------------------------------------------------------------------- #
# discard paths
# --------------------------------------------------------------------------- #


def test_below_margin_is_discarded_active_unchanged(stores) -> None:
    # E1: noise inside τ must not promote. inc 0.60, cand 0.63 (< τ=0.05)
    specs, atts, rid = stores
    cid = _commit_candidate(specs, rid, "p1")
    aid = _put_attempt(atts, candidate_spec_id=cid, parent_spec_id=rid, change=_small_change())

    gk = _gatekeeper(specs, atts)
    d = gk.decide(aid, make_run(cid, 0.63), make_run(rid, 0.60))

    assert d.action is Action.DISCARD and d.by_rule == "below_margin"
    assert d.margin == pytest.approx(0.03) and d.margin < gk._th.tau
    applied = gk.apply_decision(d)
    assert applied.verdict is m.Verdict.REJECTED
    assert specs.active_id() == rid                       # active unchanged


def test_unrunnable_discarded_before_margin_math(stores) -> None:
    # E4: an unrunnable candidate (>ε crashed) never wins on its survivors
    specs, atts, rid = stores
    cid = _commit_candidate(specs, rid, "p1")
    aid = _put_attempt(atts, candidate_spec_id=cid, parent_spec_id=rid, change=_small_change())

    gk = _gatekeeper(specs, atts)
    d = gk.decide(aid, make_run(cid, 0.99, unrunnable=True), make_run(rid, 0.60))

    assert d.action is Action.DISCARD and d.by_rule == "unrunnable"
    assert specs.active_id() == rid


def test_cross_geometry_comparison_blocked(stores) -> None:
    # I5: a candidate scored under a different suite/rubric cannot be a "win"
    specs, atts, rid = stores
    cid = _commit_candidate(specs, rid, "p1")
    aid = _put_attempt(atts, candidate_spec_id=cid, parent_spec_id=rid, change=_small_change())

    gk = _gatekeeper(specs, atts)
    d = gk.decide(aid, make_run(cid, 0.99, suite_id="S"),
                  make_run(rid, 0.60, suite_id="OTHER"))

    assert d.action is Action.DISCARD and d.by_rule == "cross_geometry"


# --------------------------------------------------------------------------- #
# rollback paths (E6 / I3)
# --------------------------------------------------------------------------- #


def test_promoted_then_regresses_rolls_back_to_parent(stores) -> None:
    # E6/I3: regress >= δ -> active reverts to parent, candidate archived,
    #         verdict ROLLED_BACK (archived, never deleted)
    specs, atts, rid = stores
    cid = _commit_candidate(specs, rid, "p1")
    aid = _put_attempt(atts, candidate_spec_id=cid, parent_spec_id=rid, change=_small_change())

    # first: promote it
    gk = _gatekeeper(specs, atts)
    gk.apply_decision(gk.decide(aid, make_run(cid, 0.66), make_run(rid, 0.60)))
    assert specs.active_id() == cid

    # then: it regresses (pre_promotion 0.66, regressed 0.55 -> drop 0.11 >= δ=0.07)
    d = gk.rollback(aid, regressed_mean=0.55, pre_promotion_mean=0.66)
    assert d.action is Action.ROLLBACK and d.by_rule == "rollback"

    assert specs.active_id() == rid                       # pointer swapped to parent
    assert specs.is_archived(cid) is True                  # archived, NOT deleted
    assert specs.has(cid)                                  # file still present (never deleted)
    assert atts.require(aid).verdict is m.Verdict.ROLLED_BACK
    # lineage still walks through the archived node (I3 reachability)
    assert rid in specs.lineage(cid)


def test_regression_within_floor_leaves_incumbent_alone(stores) -> None:
    # E6': a regression below δ is noise — no rollback (incumbent stays)
    specs, atts, rid = stores
    cid = _commit_candidate(specs, rid, "p1")
    aid = _put_attempt(atts, candidate_spec_id=cid, parent_spec_id=rid, change=_small_change())
    gk = _gatekeeper(specs, atts)
    gk.apply_decision(gk.decide(aid, make_run(cid, 0.66), make_run(rid, 0.60)))
    assert specs.active_id() == cid

    # drop 0.03 < δ=0.07 -> within floor, discard (no rollback, active stays)
    d = gk.rollback(aid, regressed_mean=0.63, pre_promotion_mean=0.66)
    assert d.action is Action.DISCARD and d.by_rule == "regression_within_floor"
    assert specs.active_id() == cid
    assert atts.require(aid).verdict is m.Verdict.PROMOTED  # unchanged


def test_rollback_idempotent_does_not_downgrade_verdict(stores) -> None:
    # re-checking an already-rolled-back attempt must not flip ROLLED_BACK->REJECTED
    specs, atts, rid = stores
    cid = _commit_candidate(specs, rid, "p1")
    aid = _put_attempt(atts, candidate_spec_id=cid, parent_spec_id=rid, change=_small_change())
    gk = _gatekeeper(specs, atts)
    gk.apply_decision(gk.decide(aid, make_run(cid, 0.66), make_run(rid, 0.60)))
    gk.rollback(aid, regressed_mean=0.55, pre_promotion_mean=0.66)
    assert atts.require(aid).verdict is m.Verdict.ROLLED_BACK

    again = gk.rollback(aid, regressed_mean=0.50, pre_promotion_mean=0.66)
    assert again.action is Action.DISCARD and again.by_rule == "already_rolled_back"
    assert atts.require(aid).verdict is m.Verdict.ROLLED_BACK  # not downgraded


# --------------------------------------------------------------------------- #
# AttemptStore.set_verdict (the new mutator) — idempotent + round-trip
# --------------------------------------------------------------------------- #


def test_attempt_store_set_verdict_round_trips(tmp_path: Path) -> None:
    atts = AttemptStore(tmp_path / ".archforge")
    att = m.Attempt(candidate_spec_id="c1", parent_spec_id="p1",
                    change=_small_change(), verdict=m.Verdict.PROMOTED)
    aid = atts.append(att)

    flipped = atts.set_verdict(aid, m.Verdict.ROLLED_BACK)
    assert flipped.verdict is m.Verdict.ROLLED_BACK
    assert atts.require(aid).verdict is m.Verdict.ROLLED_BACK

    # idempotent: flipping to the same verdict is a no-op (no duplicate rows)
    atts.set_verdict(aid, m.Verdict.ROLLED_BACK)
    rows = atts.for_parent("p1")
    assert len(rows) == 1
    # a second flip still yields exactly one row
    atts.set_verdict(aid, m.Verdict.REJECTED)
    assert len(atts.for_parent("p1")) == 1
    assert atts.require(aid).verdict is m.Verdict.REJECTED


def test_decision_is_a_decision_record(stores) -> None:
    specs, atts, rid = stores
    cid = _commit_candidate(specs, rid, "p1")
    aid = _put_attempt(atts, candidate_spec_id=cid, parent_spec_id=rid, change=_small_change())
    gk = _gatekeeper(specs, atts)
    d = gk.decide(aid, make_run(cid, 0.66), make_run(rid, 0.60))
    assert isinstance(d, Decision)
    assert d.action is Action.AUTO_PROMOTE
    assert d.reason and d.by_rule   # surfaced to the report (non-empty)
