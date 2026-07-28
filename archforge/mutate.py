"""Pure Spec mutations, one per ChangeKind.

The Architect composes these to build a *concrete* candidate Spec from the
incumbent (the `Change` record only carries metadata — kind/target/description —
for dedup + the human gate; the actual edit lives in the candidate Spec these
helpers construct). Everything is pure: a new `Spec` is returned, the input is
never mutated. Targets are validated defensively; structural mistakes that
survive the linter (E5) should not crash mutate with a bare KeyError.

Used by:
  * the Architect (real + scripted) to materialise a candidate Spec per cycle;
  * `archforge lint`/tests directly to exercise each mutation in isolation.
"""

from __future__ import annotations

from typing import Any

import archforge.models as m


class MutationError(ValueError):
    """A change cannot be applied to a Spec (defensive — usually caught by lint)."""


def _node(spec: m.Spec, node_id: str) -> m.Node:
    for n in spec.nodes:
        if n.node_id == node_id:
            return n
    raise MutationError(f"unknown node '{node_id}'")


def _coerce_edges(
    edges: list[tuple[str, m.EdgeType]] | None,
) -> list[tuple[str, m.EdgeType]] | None:
    """Normalize JSON wiring (`[["a","sequence"], ...]`) to typed tuples.

    `m.Edge` already coerces edge-type strings, but the tuple unpacking in the
    caller relies on 2-element rows; accept either form so the real-LLM JSON path
    and the typed test path share one shape.
    """

    if edges is None:
        return None
    out: list[tuple[str, m.EdgeType]] = []
    for row in edges:
        src, etype = row
        if isinstance(etype, str):
            etype = m.EdgeType(etype)
        out.append((src, etype))
    return out


def _replace_node(spec: m.Spec, node_id: str, **overrides: Any) -> m.Spec:
    """Return a new Spec with one node replaced (shallow copy + keyword overrides)."""

    nodes = [
        n.model_copy(update=overrides) if n.node_id == node_id else n
        for n in spec.nodes
    ]
    return spec.model_copy(update={"nodes": nodes})


# --------------------------------------------------------------- small changes


def apply_prompt_edit(spec: m.Spec, node_id: str, new_prompt: str) -> m.Spec:
    return _replace_node(spec, node_id, system_prompt=new_prompt)


def apply_knob(spec: m.Spec, node_id: str, **knob_overrides: Any) -> m.Spec:
    node = _node(spec, node_id)
    knobs = node.knobs.model_copy(update=knob_overrides)
    return _replace_node(spec, node_id, knobs=knobs)


def apply_model_swap(spec: m.Spec, node_id: str, new_model: str) -> m.Spec:
    return _replace_node(spec, node_id, model=new_model)


# --------------------------------------------------------------- structural changes


def apply_add_node(
    spec: m.Spec,
    node: m.Node,
    *,
    in_edges: list[tuple[str, m.EdgeType]] | None = None,
    out_edges: list[tuple[str, m.EdgeType]] | None = None,
) -> m.Spec:
    """Insert `node` and wire it after each `in_edges` source and before each `out_edges` target.

    `in_edges`: list of (from_node_id, edge_type) producing edges INTO the new node.
    `out_edges`: list of (to_node_id, edge_type) producing edges OUT of the new node.
    Gates for conditional edges default to None (the host decides routing).

    `node` may be a dict — the real Architect feeds the LLM's JSON payload straight
    through `apply_change`, so a dict arrives here; coerce it so both the scripted
    (typed `m.Node`) and real-LLM (dict) paths land the same candidate. `etype`s in
    the wiring arrive as strings from JSON and are coerced by `m.Edge` below.
    """

    if isinstance(node, dict):
        node = m.Node.model_validate(node)
    in_edges = _coerce_edges(in_edges)
    out_edges = _coerce_edges(out_edges)

    if any(n.node_id == node.node_id for n in spec.nodes):
        raise MutationError(f"node '{node.node_id}' already exists")
    nodes = list(spec.nodes) + [node]
    edges = list(spec.edges)
    for src, etype in (in_edges or []):
        edges.append(m.Edge(from_=src, to=node.node_id, type=etype))
    for dst, etype in (out_edges or []):
        edges.append(m.Edge(from_=node.node_id, to=dst, type=etype))
    return spec.model_copy(update={"nodes": nodes, "edges": edges})


def apply_remove_node(spec: m.Spec, node_id: str) -> m.Spec:
    """Remove a node and every edge that references it (rewire callers to the target's downstream).

    v1 policy: delete the node and all edges touching it, then reconnect each
    in-neighbour directly to each out-neighbour (preserve the path). If that
    would create a duplicate edge or self-loop, the linter will flag it (E5),
    which is the intended safety net — the Architect's candidate simply won't promote.
    """

    node = _node(spec, node_id)
    in_neighbours = {e.from_ for e in spec.edges if e.to == node_id}
    out_neighbours = {e.to for e in spec.edges if e.from_ == node_id}
    surviving_edges = [e for e in spec.edges if e.from_ != node_id and e.to != node_id]
    # bridge: each in -> each out (preserves a path through the removed node)
    for src in in_neighbours:
        for dst in out_neighbours:
            if src == dst:
                continue
            surviving_edges.append(m.Edge(from_=src, to=dst, type=m.EdgeType.SEQUENCE))
    nodes = [n for n in spec.nodes if n.node_id != node_id]
    return spec.model_copy(update={"nodes": nodes, "edges": surviving_edges})


def apply_rewire(
    spec: m.Spec,
    *,
    remove: tuple[str, str] | None = None,
    add: tuple[str, str, m.EdgeType] | None = None,
) -> m.Spec:
    """Remove an edge (from,to) and/or add an edge (from,to,type)."""

    edges = list(spec.edges)
    if remove is not None:
        frm, to = remove
        edges = [e for e in edges if not (e.from_ == frm and e.to == to)]
    if add is not None:
        frm, to, etype = add
        edges.append(m.Edge(from_=frm, to=to, type=etype))
    return spec.model_copy(update={"edges": edges})


# --------------------------------------------------------------- dispatch


APPLY = {
    m.ChangeKind.PROMPT_EDIT: lambda spec, change, payload: apply_prompt_edit(spec, change.target, payload["prompt"]),
    m.ChangeKind.KNOB: lambda spec, change, payload: apply_knob(spec, change.target, **payload["knobs"]),
    m.ChangeKind.MODEL_SWAP: lambda spec, change, payload: apply_model_swap(spec, change.target, payload["model"]),
    m.ChangeKind.ADD_NODE: lambda spec, change, payload: apply_add_node(spec, payload["node"], **payload.get("wiring", {})),
    m.ChangeKind.REMOVE_NODE: lambda spec, change, payload: apply_remove_node(spec, change.target),
    m.ChangeKind.REWIRE: lambda spec, change, payload: apply_rewire(spec, remove=payload.get("remove"), add=_add_from_payload(payload.get("add"))),
}


def _add_from_payload(add: Any) -> tuple[str, str, m.EdgeType] | None:
    if add is None:
        return None
    frm, to, etype = add
    return (frm, to, m.EdgeType(etype) if isinstance(etype, str) else etype)


def apply_change(spec: m.Spec, change: m.Change, payload: dict[str, Any]) -> m.Spec:
    """Dispatch a (Change, payload) to the matching mutate helper.

    `payload` carries the concrete edit content the `Change` record does not (new
    prompt/knobs/model, a new node, edge wiring). The Architect supplies it.
    """

    handler = APPLY.get(change.kind)
    if handler is None:
        raise MutationError(f"no mutate handler for kind {change.kind}")
    return handler(spec, change, payload)


__all__ = [
    "MutationError",
    "apply_prompt_edit", "apply_knob", "apply_model_swap",
    "apply_add_node", "apply_remove_node", "apply_rewire",
    "apply_change",
]
