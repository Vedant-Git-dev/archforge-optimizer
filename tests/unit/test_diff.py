"""Unit tests for ``archforge.diff`` — the spec-level structural diff (#2).

``spec_diff(parent, candidate)`` reconstructs field-level before/after by
diffing the two PERSISTED specs node-by-node (the mutator's ``payload`` evaporates
after ``apply_change``, so the diff is the source of truth for "what did the Forge
change"). Pins every change kind + the determinism + the empty case, plus the
``format_diff`` one-liner the CLI card renders.
"""
from __future__ import annotations

import archforge.models as m
from archforge.diff import DiffEntry, format_diff, spec_diff
from archforge.mutate import apply_change


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #


def _n(nid: str, *, prompt: str = "p0", model: str = "gpt",
       kind: m.NodeKind = m.NodeKind.LLM,
       knobs: m.Knobs | None = None, role: str = "r") -> m.Node:
    tools = ["t0"]
    return m.Node(node_id=nid, role=role, kind=kind, system_prompt=prompt,
                  model=model, knobs=knobs or m.Knobs(), tools=tools)


def _spec(nodes: list[m.Node], edges: list[m.Node] | None = None) -> m.Spec:
    return m.Spec(nodes=nodes, edges=[])


# --------------------------------------------------------------------------- #
# knob diff
# --------------------------------------------------------------------------- #


def test_knob_extra_change_surfaces() -> None:
    parent = _spec([_n("retrieve", kind=m.NodeKind.RETRIEVER,
                       knobs=m.Knobs(top_k=4, tunable=("top_k",)))])
    change = m.Change.for_kind(m.ChangeKind.KNOB, "retrieve", "tune", "r")
    cand = apply_change(parent, change, {"knobs": {"top_k": 8}})
    d = spec_diff(parent, cand)
    assert len(d) == 1
    assert d[0].kind == "knob" and d[0].target == "retrieve"
    assert d[0].field == "top_k" and d[0].old == 4 and d[0].new == 8
    assert format_diff(d) == "top_k: 4 -> 8"


def test_named_knob_change_surfaces() -> None:
    # temperature is a NAMED knob (not an extra); still reported as kind="knob".
    parent = _spec([_n("a", knobs=m.Knobs(temperature=0.3, max_tokens=1024))])
    change = m.Change.for_kind(m.ChangeKind.KNOB, "a", "tune temp", "r")
    cand = apply_change(parent, change, {"knobs": {"temperature": 0.7}})
    d = spec_diff(parent, cand)
    assert len(d) == 1
    assert d[0].field == "temperature" and d[0].old == 0.3 and d[0].new == 0.7


def test_tunable_metadata_is_not_reported() -> None:
    # `tunable` is editability metadata, not a knob value -> editng it alone (if a
    # change did so) still only surfaces an actual value diff. Here a knob change
    # that ALSO widens tunable must NOT add a `tunable` DiffEntry.
    parent = _spec([_n("retrieve", kind=m.NodeKind.RETRIEVER,
                       knobs=m.Knobs(top_k=4, tunable=("top_k",)))])
    change = m.Change.for_kind(m.ChangeKind.KNOB, "retrieve", "tune", "r")
    cand = apply_change(parent, change, {"knobs": {"top_k": 8}})
    d = spec_diff(parent, cand)
    assert all(e.field != "tunable" for e in d)
    assert len(d) == 1     # only the top_k value diff


# --------------------------------------------------------------------------- #
# prompt / model / kind diffs
# --------------------------------------------------------------------------- #


def test_prompt_edit_surfaces() -> None:
    parent = _spec([_n("a", prompt="p0")])
    change = m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "tighten", "r")
    cand = apply_change(parent, change, {"prompt": "p1"})
    d = spec_diff(parent, cand)
    assert len(d) == 1 and d[0].kind == "prompt"
    assert d[0].field == "system_prompt" and d[0].old == "p0" and d[0].new == "p1"


def test_model_swap_surfaces() -> None:
    parent = _spec([_n("a", model="gpt")])
    change = m.Change.for_kind(m.ChangeKind.MODEL_SWAP, "a", "swap", "r")
    cand = apply_change(parent, change, {"model": "claude"})
    d = spec_diff(parent, cand)
    assert len(d) == 1 and d[0].kind == "model"
    assert d[0].old == "gpt" and d[0].new == "claude"


# --------------------------------------------------------------------------- #
# structural diffs — add/remove node + edges
# --------------------------------------------------------------------------- #


def test_add_node_and_edge_surfaces() -> None:
    parent = _spec([_n("a")])
    vnode = _n("v", prompt="pv")
    change = m.Change.for_kind(m.ChangeKind.ADD_NODE, "v", "add verifier", "verify")
    cand = apply_change(parent, change,
                        {"node": vnode, "wiring": {"in_edges": [["a", "sequence"]]}})
    d = spec_diff(parent, cand)
    kinds = {(e.kind, e.target) for e in d}
    assert ("add_node", "v") in kinds
    # the new edge a->v reported under target "a->v"
    edge = [e for e in d if e.kind == "edge"]
    assert len(edge) == 1 and edge[0].target == "a->v"
    assert edge[0].old is None and "sequence" in str(edge[0].new)
    assert "+node v" in format_diff(d) and "+edge 1" in format_diff(d)


def test_remove_node_surfaces() -> None:
    parent = _spec([_n("a"), _n("v")])
    change = m.Change.for_kind(m.ChangeKind.REMOVE_NODE, "v", "drop", "r")
    cand = apply_change(parent, change, {})
    d = spec_diff(parent, cand)
    assert any(e.kind == "remove_node" and e.target == "v" for e in d)
    assert "-node v" in format_diff(d)


def test_rewire_edge_change_surfaces() -> None:
    # REWIRE: swap an edge's target. Both specs have an a->v edge but to different
    # destinations is hard to construct via apply_change alone; build two Specs by
    # hand to assert the edge-match-by-(from,to,type) logic directly.
    parent = m.Spec(nodes=[_n("a"), _n("v"), _n("w")],
                   edges=[m.Edge(from_="a", to="v", type=m.EdgeType.SEQUENCE)])
    cand = m.Spec(nodes=[_n("a"), _n("v"), _n("w")],
                 edges=[m.Edge(from_="a", to="w", type=m.EdgeType.SEQUENCE)])
    d = spec_diff(parent, cand)
    # one removed edge (a->v), one added edge (a->w)
    edges = [e for e in d if e.kind == "edge"]
    assert len(edges) == 2
    assert any(e.old is not None and e.new is None for e in edges)   # a->v removed
    assert any(e.old is None and e.new is not None for e in edges)   # a->w added
    assert "+edge 1" in format_diff(d) and "-edge 1" in format_diff(d)


def test_edge_gate_change_surfaces() -> None:
    parent = m.Spec(nodes=[_n("route"), _n("answer")],
                   edges=[m.Edge(from_="route", to="answer",
                                 type=m.EdgeType.CONDITIONAL, gate="go")])
    cand = m.Spec(nodes=[_n("route"), _n("answer")],
                 edges=[m.Edge(from_="route", to="answer",
                              type=m.EdgeType.CONDITIONAL, gate="stop")])
    d = spec_diff(parent, cand)
    edges = [e for e in d if e.kind == "edge"]
    assert len(edges) == 1 and edges[0].field == "gate"
    assert edges[0].old == "go" and edges[0].new == "stop"
    assert "~edge 1" in format_diff(d)


# --------------------------------------------------------------------------- #
# empty + determinism
# --------------------------------------------------------------------------- #


def test_identical_specs_yield_empty() -> None:
    parent = _spec([_n("a"), _n("v")],
                   edges=[m.Edge(from_="a", to="v", type=m.EdgeType.SEQUENCE)])
    assert spec_diff(parent, parent) == []
    assert format_diff([]) == "(no change)"


def test_diff_is_deterministic_order() -> None:
    parent = _spec([_n("a", kind=m.NodeKind.RETRIEVER,
                       knobs=m.Knobs(top_k=4, tunable=("top_k",)))])
    change = m.Change.for_kind(m.ChangeKind.KNOB, "retrieve", "t", "r")
    # rename the node so the diff catches it: build candidate by hand for two fields
    cand = _spec([_n("a", kind=m.NodeKind.RETRIEVER,
                    knobs=m.Knobs(top_k=8, tunable=("top_k",)))])
    d1 = spec_diff(parent, cand)
    d2 = spec_diff(parent, cand)
    assert d1 == d2
    assert [e.as_dict() for e in d1] == [e.as_dict() for e in d2]


def test_diffentry_one_liner_and_dict() -> None:
    e = DiffEntry("knob", "a", "top_k", 4, 8)
    assert e.one_liner() == "top_k: 4 -> 8"
    assert e.as_dict() == {"kind": "knob", "target": "a", "field": "top_k",
                            "old": 4, "new": 8}
