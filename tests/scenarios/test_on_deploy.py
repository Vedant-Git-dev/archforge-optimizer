"""Scenario tests for the engine's deploy seam — ``on_deploy`` + ``DeployCtx``
(improvement #4) + the ``on_promote`` back-compat guarantee.

``on_deploy`` is the richer deploy hook the CLI wires to write the unified
``optimized.json``: it fires ONLY on an AUTO_PROMOTE (not on a discard / queue)
and carries the parent Spec, the Gatekeeper ``Decision``, both ``SuiteRun``s,
and the cycle index — everything ``build_optimized_envelope`` needs, so the
callback writes the artifact WITHOUT re-reading the stores. It fires ALONGSIDE
``on_promote`` (kept as-is for back-compat), so the existing
``test_on_promote_*`` scenarios keep passing.

These scenarios drive the REAL ``Engine`` end-to-end on the production organs
(REAL ``Architect`` via ``ScriptedLLM`` + REAL ``SuiteRunner``/``Gatekeeper``/
stores) at zero billing — the same harness as ``test_langgraph_adapter``. Hooks
are injected post-construction (``eng._on_deploy = …``), mirroring the existing
``eng._on_promote = …`` injection pattern — ``build_engine`` doesn't expose them.
"""
from __future__ import annotations

import pytest

import archforge.models as m
from archforge.architect import Architect
from archforge.engine import EngineConfig, LoopResult
from archforge.gatekeeper import Action, Decision
from archforge.host.adapters.langgraph import build_optimized_envelope
from archforge.judge import ScriptedJudge
from archforge.llm import ScriptedLLM
from archforge.suite import SuiteRun

# Reuse the scenario harness builders (content-addressed candidate ids, real
# organs on fakes) so these scenarios mirror test_langgraph_adapter exactly.
from tests.scenarios.conftest import (
    build_engine, cand_id, llm_noop, llm_proposal, make_suite,
)

# Optional langgraph dep is irrelevant here — these scenarios use FakeHostMAS (the
# default in build_engine), so they collect even without langgraph installed.

# Fixed thresholds so the promote/discard gating is deterministic regardless of
# the active config's defaults (mirrors test_engine's _engine helper).
_TH = m.Thresholds(tau=0.05, delta=0.07)
_CFG = EngineConfig(max_cycles=50, repeats=1, plateau_cycles=5)


def _wire(engine, deploys, promotes):
    """Inject both deploy hooks post-construction (the eng._on_promote pattern)."""

    engine._on_deploy = lambda spec, dctx: deploys.append((spec, dctx))
    engine._on_promote = lambda spec: promotes.append(spec)
    return engine


# --------------------------------------------------------------------------- #
# on_deploy fires once on AUTO_PROMOTE and carries the full DeployCtx
# --------------------------------------------------------------------------- #


def test_on_deploy_fires_once_on_auto_promote(stores, seeded) -> None:
    """A clear small win (cand 0.80 vs inc 0.50 → AUTO_PROMOTE) fires ``on_deploy``
    exactly once with the promoted candidate Spec — not the parent — and fires
    ALONGSIDE ``on_promote`` (the back-compat seam still lit)."""
    rid, incumbent = seeded
    cand = cand_id(incumbent, kind="prompt_edit", target="a",
                   payload={"prompt": "p1"})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cand, "t1", 0.80)

    deploys: list[tuple] = []
    promotes: list[m.Spec] = []
    engine = _wire(build_engine(stores, judge=judge,
                                architect=Architect(llm, model="architect-1"),
                                suite=make_suite(), thresholds=_TH),
                   deploys, promotes)
    r = engine.evolve_cycle(0)

    assert r.promoted                          # the gating decision was AUTO_PROMOTE
    assert len(deploys) == 1                   # fired exactly once, on the promote
    prom_spec, _dctx = deploys[0]
    assert prom_spec.spec_id == cand           # the promoted candidate, not the parent
    assert len(promotes) == 1                 # on_promote still fired (back-compat)
    assert promotes[0].spec_id == cand         # with the same promoted Spec


def test_on_deploy_deployctx_carries_full_payload(stores, seeded) -> None:
    """``DeployCtx`` carries the parent Spec, the Gatekeeper ``Decision``, both
    ``SuiteRun``s, and the cycle index — everything ``build_optimized_envelope``
    needs to write the artifact WITHOUT re-reading the stores."""
    rid, incumbent = seeded
    cand = cand_id(incumbent, kind="prompt_edit", target="a",
                   payload={"prompt": "p1"})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cand, "t1", 0.80)

    deploys: list[tuple] = []
    engine = _wire(build_engine(stores, judge=judge,
                                architect=Architect(llm, model="architect-1"),
                                suite=make_suite(), thresholds=_TH),
                   deploys, [])
    engine.evolve_cycle(0)

    assert len(deploys) == 1
    _, dctx = deploys[0]
    # DeployCtx fields — the rich payload the envelope needs:
    assert dctx.parent.spec_id == rid                    # the parent Spec (lineage)
    assert isinstance(dctx.decision, Decision)           # the Gatekeeper verdict
    assert dctx.decision.action is Action.AUTO_PROMOTE
    assert dctx.decision.margin == pytest.approx(0.30)   # 0.80 - 0.50
    assert isinstance(dctx.cand_run, SuiteRun)           # the candidate scores
    assert dctx.cand_run.mean == pytest.approx(0.80)
    assert isinstance(dctx.inc_run, SuiteRun)            # the incumbent baseline
    assert dctx.inc_run.mean == pytest.approx(0.50)
    assert dctx.promoted_at_cycle == 0                   # the cycle index


def test_on_deploy_payload_is_envelope_complete(stores, seeded) -> None:
    """The ``DeployCtx`` the hook receives is rich enough to build the whole
    ``optimized.json`` envelope inline — no store re-read needed. Asserts the
    envelope built from the hook's payload is well-formed and carries the
    winning knobs + the promoting decision + the scores."""
    rid, incumbent = seeded
    cand = cand_id(incumbent, kind="prompt_edit", target="a",
                   payload={"prompt": "p1"})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cand, "t1", 0.80)

    deploys: list[tuple] = []
    engine = _wire(build_engine(stores, judge=judge,
                                architect=Architect(llm, model="architect-1"),
                                suite=make_suite(), thresholds=_TH),
                   deploys, [])
    engine.evolve_cycle(0)

    assert len(deploys) == 1
    prom_spec, dctx = deploys[0]
    # Build the envelope straight from the hook payload — the CLI's `_on_deploy`
    # shape. The single-node seed ("a": model gpt, prompt p1) ships real knobs
    # (model + system_prompt), so the envelope carries the winner's config.
    env = build_optimized_envelope(
        prom_spec, parent=dctx.parent, promoted_at_cycle=dctx.promoted_at_cycle,
        decision=dctx.decision, cand_run=dctx.cand_run, inc_run=dctx.inc_run,
    )
    assert env["schema"] == "archforge.optimized/v1"
    assert env["spec_id"] == cand
    assert env["parent_spec_id"] == rid
    assert env["promoted_at_cycle"] == 0
    assert env["decision"]["action"] == "auto_promote"
    assert env["decision"]["margin"] == pytest.approx(0.30)
    assert env["scores"]["mean"] == pytest.approx(0.80)
    assert env["scores"]["incumbent_mean"] == pytest.approx(0.50)
    # winning knobs shipped: the promoted prompt p1 reached the envelope.
    assert env["knobs"]["a"]["system_prompt"] == "p1"
    assert env["knobs"]["a"]["model"] == "gpt"


# --------------------------------------------------------------------------- #
# on_deploy does NOT fire on a discard (a losing change ships nothing)
# --------------------------------------------------------------------------- #


def test_on_deploy_does_not_fire_on_discard(stores, seeded) -> None:
    """A candidate scored BELOW the incumbent (a discard, not a promote) fires
    NEITHER ``on_deploy`` NOR ``on_promote`` — the deploy seam is promote-gated,
    so a losing change never ships."""
    rid, incumbent = seeded
    cand = cand_id(incumbent, kind="prompt_edit", target="a",
                   payload={"prompt": "p1"})
    llm = ScriptedLLM().respond_json(
        llm_proposal(kind="prompt_edit", target="a", payload={"prompt": "p1"}))
    # inc 0.80 / cand 0.50 → discard (margin −0.30 < τ).
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.80).set_aggregate(cand, "t1", 0.50)

    deploys: list[tuple] = []
    promotes: list[m.Spec] = []
    engine = _wire(build_engine(stores, judge=judge,
                                architect=Architect(llm, model="architect-1"),
                                suite=make_suite(), thresholds=_TH),
                   deploys, promotes)
    r = engine.evolve_cycle(0)

    assert not r.promoted                        # the decision was NOT an auto_promote
    assert deploys == []                         # the losing change shipped nothing
    assert promotes == []                         # on_promote also didn't fire


# --------------------------------------------------------------------------- #
# on_deploy across a loop: one fire per promote (both hooks, neither dedupes)
# --------------------------------------------------------------------------- #


def test_on_deploy_fires_once_per_promote_in_loop(stores, seeded) -> None:
    """In a loop, each accepted promote fires ``on_deploy`` exactly once AND
    ``on_promote`` exactly once (both hooked) — the two hooks are not mutually
    exclusive and neither dedupes the other. One deterministic promote (the
    architect then plateaus) → exactly one deploy + one promote.

    The real ``Architect`` calls the LLM once per cycle, so plateau cycles need a
    queued ``llm_noop`` (no usable ``kind`` → ``proposed=False``). With
    ``plateau_cycles=3``: cycle 0 promotes (streak 0); cycles 1-3 no-op (streak
    1,2,3 → ≥3 trips the break) → ``cycles_run=4``, ``promotions=1``. So the
    queue is 1 proposal + 3 noops; both hooks fire exactly once."""
    rid, incumbent = seeded
    cand = cand_id(incumbent, kind="prompt_edit", target="a",
                   payload={"prompt": "p1"})
    llm = (ScriptedLLM()
           .respond_json(llm_proposal(kind="prompt_edit", target="a",
                                      payload={"prompt": "p1"})))
    for _ in range(3):
        llm.respond_json(llm_noop())            # cycles 1-3 plateau (no usable kind)
    judge = ScriptedJudge().set_aggregate(rid, "t1", 0.50).set_aggregate(cand, "t1", 0.80)

    deploys: list[tuple] = []
    promotes: list[m.Spec] = []
    cfg = EngineConfig(max_cycles=50, repeats=1, plateau_cycles=3)
    engine = _wire(build_engine(stores, judge=judge,
                                architect=Architect(llm, model="architect-1"),
                                suite=make_suite(), thresholds=_TH, config=cfg),
                   deploys, promotes)
    lr: LoopResult = engine.evolve_loop()

    assert lr.cycles_run == 4 and lr.promotions == 1
    assert len(deploys) == 1 == len(promotes)    # one deploy per promote, both hooked
