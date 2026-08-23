"""Unit tests for the Architect (Phase 6).

Pins: credit assignment localizes blame to the worst (node, dimension); the
real Architect materializes one valid candidate per cycle from the LLM proposal
(one LLM call); a malformed candidate -> lint_rejected and never reaches the
SuiteRunner (E5); dedup skips a change already rejected on (parent, kind, target)
or returns plateau when none survive (E7); structural change kinds auto-tag
scope=STRUCTURAL (I4); the candidate differs from the incumbent by exactly one
mutation; and the Architect never writes to AttemptStore (write-free). Links
E5, E7, E8, I4.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import archforge.models as m
from archforge.architect import (
    Architect, ArchitectProtocol, ArchitectResult, ScriptedArchitect, credit_assign,
)
from archforge.lint import is_valid
from archforge.llm import LLMError, ScriptedLLM
from archforge.stores import AttemptStore


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def N(nid: str, *, prompt: str = "p", model: str = "gpt", tools: list[str] | None = None) -> m.Node:
    return m.Node(node_id=nid, role="r", system_prompt=prompt, model=model,
                  tools=tools or ["t0"])


@pytest.fixture
def incumbent() -> m.Spec:
    return m.Spec(
        nodes=[N("a"), N("b"), N("c")],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE),
               m.Edge(from_="b", to="c", type=m.EdgeType.SEQUENCE)],
    )


@pytest.fixture
def attempt_store(tmp_path: Path) -> AttemptStore:
    return AttemptStore(tmp_path / ".archforge")


def _rsc(node_id: str, correctness: float) -> m.StepScore:
    return m.StepScore(node_id=node_id, sub_rubrics={"correctness": correctness})


def _one_run(rep: int, mapping: dict[str, float]) -> m.RunScore:
    return m.RunScore(
        run_id=f"r{rep}", spec_id="s.inc", task_id="t",
        aggregate=sum(mapping.values()) / len(mapping), confidence=1.0,
        rubric_scores={"correctness": sum(mapping.values()) / len(mapping)},
        judge_meta=m.JudgeMeta(model="j", rubric_id="default-v1"),
        step_scores=[_rsc(n, v) for n, v in mapping.items()],
    )


# --------------------------------------------------------------------------- #
# credit assignment
# --------------------------------------------------------------------------- #


def test_credit_assign_picks_lowest_scoring_node() -> None:
    scores = [_one_run(0, {"a": 0.9, "b": 0.3, "c": 0.85}),
              _one_run(1, {"a": 0.91, "b": 0.29, "c": 0.84})]
    blame = credit_assign(m.Spec(nodes=[], edges=[]), scores)
    assert blame is not None
    assert blame.node_id == "b"                 # lowest mean
    assert blame.sub_rubric == "correctness"
    assert blame.severity == pytest.approx(1.0 - 0.295)  # mean of 0.3,0.29


def test_credit_assign_returns_none_when_no_step_scores() -> None:
    scores = [m.RunScore(run_id="r", spec_id="s", task_id="t", aggregate=0.5,
                        judge_meta=m.JudgeMeta(model="j", rubric_id="r"), step_scores=[])]
    assert credit_assign(m.Spec(nodes=[], edges=[]), scores) is None


def test_credit_assign_none_on_empty() -> None:
    assert credit_assign(m.Spec(nodes=[], edges=[]), []) is None


# --------------------------------------------------------------------------- #
# Real Architect (via ScriptedLLM)
# --------------------------------------------------------------------------- #


def _judge_scores_for_incumbent(incumbent: m.Spec, values: dict[str, float]) -> list[m.RunScore]:
    return [_one_run(0, values)]


def test_real_architect_proposes_one_valid_candidate(incumbent: m.Spec) -> None:
    # LLM says: rewrite b's prompt
    llm = ScriptedLLM().respond_json({
        "kind": "prompt_edit", "target": "b", "rationale": "b lost correctness",
        "payload": {"prompt": "be more careful"},
    })
    arch = Architect(model="claude", llm=llm)
    scores = _judge_scores_for_incumbent(incumbent, {"a": 0.9, "b": 0.4, "c": 0.8})
    result = arch.next_attempt(incumbent, scores, attempt_store=AttemptStore("/tmp/x"))

    assert result.status == "proposed"
    assert result.proposal is not None
    cand = result.proposal.candidate
    assert cand.nodes[1].system_prompt == "be more careful"
    assert is_valid(cand)                          # passed the linter
    # candidate differs from incumbent by exactly the one mutation (ignoring lineage)
    assert cand.nodes[1].system_prompt != incumbent.nodes[1].system_prompt
    assert cand.nodes[0].system_prompt == incumbent.nodes[0].system_prompt
    # exactly ONE LLM call was made (the locked budget)
    assert len(llm.calls) == 1


def test_real_architect_result_status_values_collapse() -> None:
    r = ArchitectResult(status="proposed")
    assert r.status in ("proposed", "lint_rejected", "plateau")


def test_real_architect_satisfies_protocol() -> None:
    assert isinstance(Architect(model="c", llm=ScriptedLLM()), ArchitectProtocol)


def test_real_architect_malformed_proposal_returns_plateau(incumbent: m.Spec) -> None:
    # LLM returns JSON missing `kind` -> unparseable change -> plateau
    llm = ScriptedLLM().respond_json({"not_kind": "x"})
    arch = Architect(model="c", llm=llm)
    result = arch.next_attempt(incumbent, [], attempt_store=AttemptStore("/tmp/x"))
    assert result.status == "plateau"


def test_real_architect_raises_on_non_json(incumbent: m.Spec) -> None:
    """A provider outage / non-JSON: propagate as LLMError (the SuiteRunner/loop retries, E9)."""
    llm = ScriptedLLM().respond("not json at all")  # text response to a json request
    arch = Architect(model="c", llm=llm)
    with pytest.raises(LLMError):
        arch.next_attempt(incumbent, [], attempt_store=AttemptStore("/tmp/x"))


# --------------------------------------------------------------------------- #
# ScriptedArchitect — E5 (lint rejection), E7 (dedup), I4 (scope), write-free
# --------------------------------------------------------------------------- #


def test_scripted_architect_proposes_candidate(incumbent: m.Spec,
                                                attempt_store: AttemptStore) -> None:
    ch = m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "b", "edit b", "b weak")
    payload = {"prompt": "improve b"}
    arch = ScriptedArchitect().propose(ch, payload)
    result = arch.next_attempt(incumbent, _judge_scores_for_incumbent(incumbent, {"a": 0.9, "b": 0.4, "c": 0.8}),
                                attempt_store=attempt_store)
    assert result.status == "proposed"
    assert result.proposal.candidate.nodes[1].system_prompt == "improve b"
    assert result.proposal.change.scope is m.Scope.SMALL          # I4: small auto-tag


def test_scripted_architect_structural_scope_auto_tag(
    incumbent: m.Spec, attempt_store: AttemptStore,
) -> None:
    new_node = N("verifier", tools=["check"])
    ch = m.Change.for_kind(m.ChangeKind.ADD_NODE, "verifier", "add verifier", "needs verification")
    # canonical payload shape: node + nested `wiring` (matches the real Architect's
    # prompt and the mutate APPLY dispatch — `payload["node"]`, `payload["wiring"]`)
    payload = {"node": new_node,
               "wiring": {"in_edges": [("b", m.EdgeType.SEQUENCE)],
                          "out_edges": [("c", m.EdgeType.SEQUENCE)]}}
    arch = ScriptedArchitect().propose(ch, payload)
    result = arch.next_attempt(incumbent, _judge_scores_for_incumbent(incumbent, {"a": 0.9, "b": 0.4, "c": 0.8}),
                                attempt_store=attempt_store)
    assert result.status == "proposed"
    assert result.proposal.change.scope is m.Scope.STRUCTURAL    # I4
    assert "verifier" in [n.node_id for n in result.proposal.candidate.nodes]
    edges = {(ed.from_, ed.to) for ed in result.proposal.candidate.edges}
    assert ("b", "verifier") in edges and ("verifier", "c") in edges


def test_e5_malformed_candidate_is_lint_rejected(incumbent: m.Spec,
                                                  attempt_store: AttemptStore) -> None:
    ch = m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "b", "edit b", "why")
    arch = ScriptedArchitect().propose(ch, {"prompt": "x"}).force_lint_error()
    result = arch.next_attempt(incumbent, [], attempt_store=attempt_store)
    assert result.status == "lint_rejected"
    assert result.proposal is None                              # never reaches SuiteRunner
    assert result.rejected_reasons                              # structured reason returned


def test_e7_dedup_blocks_already_rejected_change(
    incumbent: m.Spec, attempt_store: AttemptStore,
) -> None:
    # pre-seed a prior REJECTED attempt on (parent, prompt_edit, b)
    incumbent_id = incumbent.compute_spec_id()
    prior = m.Attempt(
        candidate_spec_id="dead", parent_spec_id=incumbent_id,
        change=m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "b", "tried", "nope"),
        verdict=m.Verdict.REJECTED,
    )
    attempt_store.append(prior)

    arch = ScriptedArchitect().propose(
        m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "b", "edit b", "again"), {"prompt": "x"}
    )
    result = arch.next_attempt(incumbent, [], attempt_store=attempt_store)
    assert result.status == "plateau"                          # the only option dedup-blocked
    assert "already tried" in (result.note or "")


def test_e7_dedup_rolled_back_also_blocks(
    incumbent: m.Spec, attempt_store: AttemptStore,
) -> None:
    incumbent_id = incumbent.compute_spec_id()
    prior = m.Attempt(
        candidate_spec_id="dead2", parent_spec_id=incumbent_id,
        change=m.Change.for_kind(m.ChangeKind.MODEL_SWAP, "a", "swap", "rolled"),
        verdict=m.Verdict.ROLLED_BACK,
    )
    attempt_store.append(prior)
    arch = ScriptedArchitect().propose(
        m.Change.for_kind(m.ChangeKind.MODEL_SWAP, "a", "swap", "again"), {"model": "gpt-4o"}
    )
    result = arch.next_attempt(incumbent, [], attempt_store=attempt_store)
    assert result.status == "plateau"


def test_architect_is_write_free(incumbent: m.Spec, attempt_store: AttemptStore) -> None:
    """Architect READS AttemptStore for dedup but never appends (proposal/commitment separation)."""
    before = {a.attempt_id for a in attempt_store.all()}
    ch = m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "b", "edit b", "why")
    arch = ScriptedArchitect().propose(ch, {"prompt": "x"})
    arch.next_attempt(incumbent, [], attempt_store=attempt_store)
    assert {a.attempt_id for a in attempt_store.all()} == before    # nothing appended


def test_scripted_architect_plateau_when_queue_empty(
    incumbent: m.Spec, attempt_store: AttemptStore,
) -> None:
    result = ScriptedArchitect().next_attempt(incumbent, [], attempt_store=attempt_store)
    assert result.status == "plateau"
    result2 = ScriptedArchitect().force_plateau().next_attempt(
        incumbent, [], attempt_store=attempt_store)
    assert result2.status == "plateau"


def test_scripted_architect_satisfies_protocol() -> None:
    assert isinstance(ScriptedArchitect(), ArchitectProtocol)


def test_one_change_per_cycle_candidate_differs_by_one_mutation(
    incumbent: m.Spec, attempt_store: AttemptStore,
) -> None:
    """Candidate == incumbent + exactly the mutation (lineage aside)."""
    ch = m.Change.for_kind(m.ChangeKind.MODEL_SWAP, "a", "swap", "why")
    arch = ScriptedArchitect().propose(ch, {"model": "claude"})
    result = arch.next_attempt(incumbent, [], attempt_store=attempt_store)
    cand = result.proposal.candidate
    # nodes count unchanged (model swap), only model field moved on node a
    assert len(cand.nodes) == len(incumbent.nodes)
    assert [n.node_id for n in cand.nodes] == [n.node_id for n in incumbent.nodes]
    assert cand.nodes[0].model == "claude"
    assert cand.nodes[1].model == incumbent.nodes[1].model  # untouched
    assert cand.nodes[2].model == incumbent.nodes[2].model  # untouched


# --------------------------------------------------------------------------- #
# Kind gate — prompt_edit/model_swap apply ONLY to llm nodes (non-LLM extension)
# --------------------------------------------------------------------------- #


@pytest.fixture
def mixed_spec() -> m.Spec:
    """An llm node (a) feeding a rule node (r) — exercises the kind gate on `r`."""
    rule = m.Node(node_id="r", role="scorer", kind=m.NodeKind.RULE,
                  knobs=m.Knobs(threshold=0.5, tunable=("threshold",)))
    return m.Spec(
        nodes=[N("a"), rule],
        edges=[m.Edge(from_="a", to="r", type=m.EdgeType.SEQUENCE)],
    )


def test_prompt_edit_on_non_llm_node_plateaus(
    mixed_spec: m.Spec, attempt_store: AttemptStore,
) -> None:
    # A prompt_edit aimed at a rule node is valid metadata (not "unparseable") but a
    # kind mismatch — plateau with a precise note, nothing committed. (The gate
    # returns plateau, NOT lint_rejected, per the design's E7/E8 choice.)
    ch = m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "r", "edit prompt", "r")
    result = (ScriptedArchitect().propose(ch, {"prompt": "new"})
              .next_attempt(mixed_spec, [], attempt_store=attempt_store))
    assert result.status == "plateau"
    assert "non-llm" in (result.note or "") and "r" in (result.note or "")


def test_model_swap_on_non_llm_node_plateaus(
    mixed_spec: m.Spec, attempt_store: AttemptStore,
) -> None:
    ch = m.Change.for_kind(m.ChangeKind.MODEL_SWAP, "r", "swap model", "r")
    result = (ScriptedArchitect().propose(ch, {"model": "bigger"})
              .next_attempt(mixed_spec, [], attempt_store=attempt_store))
    assert result.status == "plateau"
    assert "non-llm" in (result.note or "")


def test_prompt_edit_on_llm_node_passes_gate(
    mixed_spec: m.Spec, attempt_store: AttemptStore,
) -> None:
    # An llm target is the legitimate prompt_edit surface -> proposed (passes the
    # gate, then linted).
    ch = m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "edit prompt", "r")
    result = (ScriptedArchitect().propose(ch, {"prompt": "sharpened"})
              .next_attempt(mixed_spec, [], attempt_store=attempt_store))
    assert result.status == "proposed"


def test_knob_on_tunable_extra_passes_on_non_llm(
    mixed_spec: m.Spec, attempt_store: AttemptStore,
) -> None:
    # A `knob` edit to a tunable extra (threshold is in r's tunable) is allowed —
    # the gate is prompt_edit/model_swap-ONLY; tunable-extras gate at `apply_knob`.
    ch = m.Change.for_kind(m.ChangeKind.KNOB, "r", "tune threshold", "r")
    result = (ScriptedArchitect().propose(ch, {"knobs": {"threshold": 0.7}})
              .next_attempt(mixed_spec, [], attempt_store=attempt_store))
    assert result.status == "proposed"


def test_knob_on_non_tunable_extra_is_lint_rejected(
    mixed_spec: m.Spec, attempt_store: AttemptStore,
) -> None:
    # A `knob` edit to an extra NOT in `tunable` (endpoint) -> apply_knob raises
    # MutationError -> routed through next_attempt's try/except -> lint_rejected
    # (E5). Nothing committed; this is the open-Knobs guard, surfaced via the
    # Architect's E5 route — exactly as the design specifies.
    ch = m.Change.for_kind(m.ChangeKind.KNOB, "r", "tune", "r")
    result = (ScriptedArchitect().propose(ch, {"knobs": {"endpoint": "http://x"}})
              .next_attempt(mixed_spec, [], attempt_store=attempt_store))
    assert result.status == "lint_rejected"
    assert result.rejected_reasons
    assert "endpoint" in result.rejected_reasons[0].message
