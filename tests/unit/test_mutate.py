"""Unit tests for the pure Spec mutations (Phase 6 support).

Pins immutability (incumbent untouched), one-concern-per-mutation, and the
dispatch table. Structural changes (add/remove/rewire/model_swap) tag structural
through Change.for_kind — exercised on Change elsewhere; here we assert the
mutation itself produces a valid, distinct Spec.
"""

from __future__ import annotations

import pytest

import archforge.models as m
from archforge.lint import is_valid
from archforge.mutate import (
    MutationError, apply_add_node, apply_change, apply_knob, apply_model_swap,
    apply_prompt_edit, apply_remove_node, apply_rewire,
)


def N(nid: str, *, prompt: str = "p", model: str = "gpt", tools: list[str] | None = None) -> m.Node:
    return m.Node(node_id=nid, role="r", system_prompt=prompt, model=model,
                  tools=tools or ["t0"])


def seq() -> m.Spec:
    return m.Spec(
        nodes=[N("a"), N("b"), N("c")],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE),
               m.Edge(from_="b", to="c", type=m.EdgeType.SEQUENCE)],
    )


# --------------------------------------------------------------------------- #
# immutability + small changes
# --------------------------------------------------------------------------- #


def test_prompt_edit_is_immutable() -> None:
    s = seq()
    e = apply_prompt_edit(s, "b", "NEW")
    assert s.nodes[1].system_prompt == "p"          # incumbent untouched
    assert e.nodes[1].system_prompt == "NEW"
    assert e.compute_spec_id() != s.compute_spec_id()


def test_knob_edit_only_changes_knobs() -> None:
    s = seq()
    e = apply_knob(s, "a", temperature=0.9, retries=4)
    assert e.nodes[0].knobs.temperature == 0.9 and e.nodes[0].knobs.retries == 4
    assert s.nodes[0].knobs.temperature is None                     # untouched
    assert e.nodes[0].system_prompt == s.nodes[0].system_prompt       # prompt untouched


def test_model_swap() -> None:
    s = seq()
    e = apply_model_swap(s, "a", "claude-opus")
    assert e.nodes[0].model == "claude-opus"
    assert s.nodes[0].model == "gpt"


# --------------------------------------------------------------------------- #
# structural changes
# --------------------------------------------------------------------------- #


def test_add_node_wires_in_and_out() -> None:
    s = seq()
    v = N("verifier", tools=["check"])
    e = apply_add_node(s, v, in_edges=[("b", m.EdgeType.SEQUENCE)],
                       out_edges=[("c", m.EdgeType.SEQUENCE)])
    assert [n.node_id for n in e.nodes] == ["a", "b", "c", "verifier"]
    edges = {(ed.from_, ed.to) for ed in e.edges}
    assert ("b", "verifier") in edges and ("verifier", "c") in edges
    assert is_valid(e)


def test_add_duplicate_node_rejected() -> None:
    with pytest.raises(MutationError):
        apply_add_node(seq(), N("a"))


def test_remove_node_bridges_neighbours() -> None:
    s = seq()
    e = apply_remove_node(s, "b")
    assert [n.node_id for n in e.nodes] == ["a", "c"]
    edges = {(ed.from_, ed.to) for ed in e.edges}
    assert ("a", "c") in edges              # path bridged
    assert ("a", "b") not in edges and ("b", "c") not in edges
    assert is_valid(e)


def test_remove_unknown_node_rejected() -> None:
    with pytest.raises(MutationError):
        apply_remove_node(seq(), "ghost")


def test_rewire_remove_and_add_edge() -> None:
    s = seq()
    e = apply_rewire(s, remove=("a", "b"), add=("a", "c", m.EdgeType.SEQUENCE))
    edges = {(ed.from_, ed.to) for ed in e.edges}
    assert ("a", "b") not in edges
    assert ("a", "c") in edges


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


def test_apply_change_dispatches_prompt_edit() -> None:
    s = seq()
    ch = m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "edit", "why")
    e = apply_change(s, ch, {"prompt": "Z"})
    assert e.nodes[0].system_prompt == "Z"
    assert ch.scope is m.Scope.SMALL


def test_apply_change_dispatches_model_swap_structural_scope() -> None:
    s = seq()
    ch = m.Change.for_kind(m.ChangeKind.MODEL_SWAP, "a", "swap", "why")
    e = apply_change(s, ch, {"model": "gpt-4o"})
    assert e.nodes[0].model == "gpt-4o"
    assert ch.scope is m.Scope.STRUCTURAL


def test_apply_change_unknown_node_in_payload_is_lint_rejected_upstream() -> None:
    # mutate raises MutationError; the linter/architect turn that into lint_rejected
    s = seq()
    ch = m.Change.for_kind(m.ChangeKind.REMOVE_NODE, "ghost", "remove", "why")
    with pytest.raises(MutationError):
        apply_change(s, ch, {})


def test_apply_change_add_node_via_dispatch() -> None:
    # canonical payload: node + nested `wiring` (mirrors the real Architect's
    # prompt and the APPLY dispatch `payload["node"]` / `payload["wiring"]`).
    s = seq()
    v = N("v", tools=["k"])
    edit = {"node": v,
            "wiring": {"in_edges": [("b", m.EdgeType.SEQUENCE)],
                       "out_edges": [("c", m.EdgeType.SEQUENCE)]}}
    ch = m.Change.for_kind(m.ChangeKind.ADD_NODE, "v", "add", "why")
    e = apply_change(s, ch, edit)
    assert "v" in [n.node_id for n in e.nodes]
    edges = {(ed.from_, ed.to) for ed in e.edges}
    assert ("b", "v") in edges and ("v", "c") in edges   # wiring actually applied
    assert is_valid(e)


# --------------------------------------------------------------------------- #
# apply_knob tunable guard (non-LLM optimizer extension)
# --------------------------------------------------------------------------- #


def _knob_node(*, kind: m.NodeKind = m.NodeKind.LLM,
               tunable: tuple[str, ...] = (), **extra) -> m.Spec:
    knobs = m.Knobs(tunable=tunable, **extra)
    return m.Spec(nodes=[m.Node(node_id="a", role="r", kind=kind, knobs=knobs)], edges=[])


def test_apply_knob_named_knobs_editable_on_empty_tunable_legacy_node() -> None:
    # Back-compat cardinal: every legacy LLM Spec has tunable=() (the new field's
    # default). The named knobs temperature/retries/max_tokens MUST stay editable
    # so the historical LLM tuning path survives.
    s = _knob_node()
    e = apply_knob(s, "a", temperature=0.9, retries=3, max_tokens=2048)
    k = e.nodes[0].knobs
    assert k.temperature == 0.9 and k.retries == 3 and k.max_tokens == 2048


def test_apply_knob_extra_key_in_tunable_is_editable() -> None:
    s = _knob_node(kind=m.NodeKind.RETRIEVER, tunable=("top_k",), top_k=3)
    e = apply_knob(s, "a", top_k=10)
    assert e.nodes[0].knobs.top_k == 10


def test_apply_knob_extra_key_not_in_tunable_raises_mutationerror() -> None:
    # An extra (open-bag) key the node's tunable does NOT list is host-owned —
    # the LLM Architect must not touch it. This raise routes through next_attempt's
    # try/except -> _lint_rejected (E5): discarded this cycle, loop continues.
    s = _knob_node(kind=m.NodeKind.RETRIEVER, tunable=("top_k",), top_k=3)
    with pytest.raises(MutationError) as ei:
        apply_knob(s, "a", endpoint="http://x")
    assert "endpoint" in str(ei.value) and "a" in str(ei.value)


def test_apply_knob_named_plus_tunable_extra_in_one_call() -> None:
    # A knob edit may mix the always-editable named knobs with a tunable extra.
    s = _knob_node(kind=m.NodeKind.RETRIEVER, tunable=("top_k",), top_k=3)
    e = apply_knob(s, "a", top_k=10, temperature=0.5, retries=2)
    k = e.nodes[0].knobs
    assert k.top_k == 10 and k.temperature == 0.5 and k.retries == 2


def test_apply_knob_unknown_extra_key_alongside_named_still_raises() -> None:
    # A bad extra in the same call as named knobs does not sneak through — `bad`
    # is not in `tunable`, so the whole edit is rejected (nothing mutated).
    s = _knob_node(kind=m.NodeKind.RULE, tunable=("threshold",), threshold=0.5)
    with pytest.raises(MutationError):
        apply_knob(s, "a", threshold=0.7, endpoint="http://x")
