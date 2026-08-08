"""Spec-level structural diff — what changed from the parent Spec to the candidate.

ArchForge's ``Change`` record (``m.Change``) carries the *intent* of a mutation
(kind/target/diff/rationale/scope) but NOT the field-level before/after — the
``payload`` dict evaporates after ``apply_change`` (it is metadata, not
persisted). So an inspectable "what did the Forge actually change" requires
diffing the two persisted Specs node-by-node. Both specs live in the SpecStore
(``spec_store.get(parent_id)`` / ``get(candidate_id)``), so the comparison is
cheap and needs nothing beyond the Specs themselves.

``spec_diff(parent, candidate)`` does exactly that: a pure comparison returning a
deterministic list of ``DiffEntry`` records. It powers the CLI's per-cycle
mutation-diff card (improvement #2) and is reuseable by any embedder that wants
to surface what a candidate changed vs its parent.

Only *real* changes surface — identical Specs yield ``[]``. Knob changes are
classified against ``m._NAMED_KNOBS`` (the single source lifted into
``archforge.models``): the named LLM knobs (temperature/retries/max_tokens) and
the kind-specific extras (top_k/threshold/...) are both reported as
``kind="knob"``; ``tunable`` is editability *metadata*, not a knob value, and is
NOT reported. Deterministic order: parent-spec node order for matched/removed
nodes, candidate-spec node order for added nodes, then edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import archforge.models as m


@dataclass(frozen=True)
class DiffEntry:
    """One field-level change between a parent Spec and its candidate.

    ``kind``    : category of the change
                 (knob|prompt|model|add_node|remove_node|edge|role|kind)
    ``target``  : the node_id (for node fields) or "from->to" (for edges)
    ``field``   : the field name on the target (system_prompt|model|role|kind|
                 <knob_name>|edge|gate)
    ``old``     : the parent's value (None for an addition);
    ``new``     : the candidate's value (None for a removal)
    """

    kind: str
    target: str
    field: str
    old: Any
    new: Any

    def one_liner(self) -> str:
        """Compact ``field: old -> new`` for a card's change line."""
        return f"{self.field}: {_fmt(self.old)} -> {_fmt(self.new)}"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "field": self.field,
                "old": self.old, "new": self.new}


def _fmt(v: Any) -> str:
    if v is None:
        return "(unset)"
    if isinstance(v, bool):
        return str(v).lower()
    return str(v)


# --------------------------------------------------------------------------- #
# the diff
# --------------------------------------------------------------------------- #


def spec_diff(parent: m.Spec, candidate: m.Spec) -> list[DiffEntry]:
    """Field-level diff of ``candidate`` vs ``parent`` (deterministic, pure).

    Node diffs come before edge diffs. Within a matched node the order is
    system_prompt, model, role, kind, then knobs (named knobs in a fixed order,
    then extras). Returns ``[]`` for identical Specs.
    """
    out: list[DiffEntry] = []
    pnodes: dict[str, m.Node] = {n.node_id: n for n in parent.nodes}
    cnodes: dict[str, m.Node] = {n.node_id: n for n in candidate.nodes}

    # matched + removed nodes (parent order), then added nodes (candidate order).
    for pn in parent.nodes:
        cn = cnodes.get(pn.node_id)
        if cn is None:
            out.append(DiffEntry("remove_node", pn.node_id, "node", pn.node_id, None))
            continue
        out.extend(_node_diff(pn, cn))
    for cn in candidate.nodes:
        if cn.node_id not in pnodes:
            out.append(DiffEntry("add_node", cn.node_id, "node", None, cn.node_id))

    out.extend(_edge_diff(parent, candidate))
    return out


def _node_diff(parent: m.Node, candidate: m.Node) -> list[DiffEntry]:
    nid = parent.node_id
    out: list[DiffEntry] = []
    if parent.system_prompt != candidate.system_prompt:
        out.append(DiffEntry("prompt", nid, "system_prompt",
                             parent.system_prompt, candidate.system_prompt))
    if parent.model != candidate.model:
        out.append(DiffEntry("model", nid, "model", parent.model, candidate.model))
    if parent.role != candidate.role:
        out.append(DiffEntry("role", nid, "role", parent.role, candidate.role))
    if parent.kind is not candidate.kind:
        out.append(DiffEntry("kind", nid, "kind", parent.kind.value, candidate.kind.value))
    out.extend(_knob_diff(nid, parent.knobs, candidate.knobs))
    return out


def _knob_diff(nid: str, pk: m.Knobs, ck: m.Knobs) -> list[DiffEntry]:
    out: list[DiffEntry] = []
    pd = pk.model_dump()
    cd = ck.model_dump()
    # named knobs in a fixed stable order, then extras in candidate insertion
    # order, then parent-only extras (a knob dropped from the candidate).
    named = ["temperature", "retries", "max_tokens"]
    extras = [k for k in cd if k not in m._NAMED_KNOBS]
    parent_extras = [k for k in pd if k not in m._NAMED_KNOBS and k not in extras]
    for key in named + extras + parent_extras:
        pv = pd.get(key)
        cv = cd.get(key)
        if pv != cv:
            out.append(DiffEntry("knob", nid, key, pv, cv))
    return out


def _edge_key(e: m.Edge) -> tuple[str, str, str]:
    return (e.from_, e.to, e.type.value)


def _edge_diff(parent: m.Spec, candidate: m.Spec) -> list[DiffEntry]:
    out: list[DiffEntry] = []
    pedge: dict[tuple[str, str, str], m.Edge] = {_edge_key(e): e for e in parent.edges}
    for pe in parent.edges:
        ce = next((c for c in candidate.edges if _edge_key(c) == _edge_key(pe)), None)
        if ce is None:
            out.append(DiffEntry("edge", _edge_label(pe), "edge",
                                _edge_describe(pe), None))
        elif pe.gate != ce.gate:
            out.append(DiffEntry("edge", _edge_label(pe), "gate", pe.gate, ce.gate))
    for ce in candidate.edges:
        if _edge_key(ce) not in pedge:
            out.append(DiffEntry("edge", _edge_label(ce), "edge", None, _edge_describe(ce)))
    return out


def _edge_label(e: m.Edge) -> str:
    return f"{e.from_}->{e.to}"


def _edge_describe(e: m.Edge) -> str:
    s = e.type.value
    if e.gate:
        s += f" gate={e.gate}"
    return s


# --------------------------------------------------------------------------- #
# rendering helper — a compact one-liner for the CLI's change card
# --------------------------------------------------------------------------- #


def format_diff(entries: list[DiffEntry]) -> str:
    """Compact one-liner for a card's change line, from a ``spec_diff`` list.

    Field-level diffs (knob/prompt/model/role/kind on matched nodes) are joined
    with "; " (most changes are a single field). Structural diffs (added/removed
    nodes + edges — ADD_NODE/REMOVE_NODE/REWIRE) are summarized as a roster delta
    (``+node v +edge 1 -edge 1 ~edge 1``) so the card stays one line even when a
    -- proposed change touches several edges/nodes at once. Any field-level diffs
    riding alongside a structural change (e.g. a rewire that also swapped a model)
    are appended so nothing is silently dropped.
    """
    if not entries:
        return "(no change)"
    structural = [e for e in entries if e.kind in ("add_node", "remove_node", "edge")]
    if not structural:
        return "; ".join(e.one_liner() for e in entries)
    parts: list[str] = []
    added = [e.target for e in structural if e.kind == "add_node"]
    removed = [e.target for e in structural if e.kind == "remove_node"]
    added_edges = [e for e in structural if e.kind == "edge" and e.old is None]
    removed_edges = [e for e in structural if e.kind == "edge" and e.new is None]
    changed_edges = [e for e in structural if e.kind == "edge" and e.old is not None and e.new is not None]
    if added:
        parts.append(f"+node {' '.join(added)}")
    if removed:
        parts.append(f"-node {' '.join(removed)}")
    if added_edges:
        parts.append(f"+edge {len(added_edges)}")
    if removed_edges:
        parts.append(f"-edge {len(removed_edges)}")
    if changed_edges:
        parts.append(f"~edge {len(changed_edges)}")
    field_level = [e for e in entries if e.kind not in ("add_node", "remove_node", "edge")]
    if field_level:
        parts.append("; ".join(e.one_liner() for e in field_level))
    return " ".join(parts)


__all__ = ["DiffEntry", "spec_diff", "format_diff"]
