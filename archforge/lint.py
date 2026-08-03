"""Spec Linter — pure structural validation of a Pipeline Spec (spec E5, I2).

`lint(spec) -> list[LintError]` is a pure function (no I/O). It is reused in two
places: inside the Architect (reject a candidate before it reaches the
SuiteRunner — E5) and as the standalone `archforge lint` subcommand.

The linter checks *structure*, not host behaviour:

  * DAG-ness (no cycles, no self-loops)
  * every edge endpoint resolves to an existing node
  * edge-type rules (e.g. CONDITIONAL requires a `gate`)
  * no orphan nodes in a multi-node spec; no duplicate node_ids
  * tool ids are non-empty and unique within a node

Tool ids are checked for well-formedness only — resolution against the host's
tool registry is the host's job, not the linter's.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Literal

from pydantic import BaseModel

from archforge.models import EdgeType, NodeKind, Spec

# Lint error codes — stable identifiers so tests and the Architect can branch on
# them. Keep the set small; add codes only when a new structural fault appears.
LintCode = Literal[
    "duplicate_node",
    "orphan_node",
    "edge_unknown_from",
    "edge_unknown_to",
    "self_loop",
    "duplicate_edge",
    "cycle",
    "conditional_no_gate",
    "tool_id_empty",
    "tool_id_duplicate",
    "node_unknown_kind",
]


class LintError(BaseModel):
    """A single structural defect in a Spec. Structured → returnable to humans."""

    code: LintCode
    message: str
    location: str | None = None  # node_id or "from->to" depending on the fault


def lint(spec: Spec) -> list[LintError]:
    """Return all structural defects found in `spec` (empty list == valid)."""

    errors: list[LintError] = []
    node_ids = {n.node_id for n in spec.nodes}

    # --- duplicate node_ids --------------------------------------------------
    seen: dict[str, int] = defaultdict(int)
    for n in spec.nodes:
        seen[n.node_id] += 1
        if seen[n.node_id] == 2:
            errors.append(
                LintError(code="duplicate_node", message=f"node_id '{n.node_id}' appears more than once",
                          location=n.node_id)
            )
        # tool ids per node
        tool_seen: set[str] = set()
        for tid in n.tools:
            if not tid:
                errors.append(LintError(code="tool_id_empty",
                                        message=f"node '{n.node_id}' has an empty tool id",
                                        location=n.node_id))
            elif tid in tool_seen:
                errors.append(LintError(code="tool_id_duplicate",
                                        message=f"node '{n.node_id}' lists tool '{tid}' more than once",
                                        location=n.node_id))
            else:
                tool_seen.add(tid)
        # node-kind rule (non-LLM optimizer extension). A bad `kind` value is
        # normally rejected by pydantic's enum coercion at construction — this is
        # a defense-in-depth guard for a model_construct / hand-edit path that
        # bypassed validation. We deliberately do NOT require non-empty
        # system_prompt/model on LLM nodes: this project's config-decay contract
        # (`host/adapters/helpers.cfg_decay` -> `KnobVote`) treats an empty
        # system_prompt as "the agent owns its prompt" and a None model as "the
        # agent's hardcoded default". Requiring them would regress the established
        # "text-in/text-out node need not declare one" SpecBuilder contract and
        # break every legacy text-in/text-out Spec on lint — violating the design's
        # own "legacy Specs load unchanged" back-compat guarantee.
        if not isinstance(n.kind, NodeKind):
            errors.append(LintError(code="node_unknown_kind",
                                    message=f"node '{n.node_id}' has an unknown kind {n.kind!r}",
                                    location=n.node_id))

    # --- edge endpoints + per-edge rules ------------------------------------
    edge_keys: set[tuple[str, str, str]] = set()
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {nid: 0 for nid in node_ids}

    for edge in spec.edges:
        f, t = edge.from_, edge.to
        key = (f, t, edge.type.value)
        if key in edge_keys:
            errors.append(LintError(code="duplicate_edge",
                                    message=f"duplicate edge {f}->{t} ({edge.type.value})",
                                    location=f"{f}->{t}"))
        edge_keys.add(key)

        if f not in node_ids:
            errors.append(LintError(code="edge_unknown_from",
                                    message=f"edge from unknown node '{f}' (to '{t}')",
                                    location=f"{f}->{t}"))
        if t not in node_ids:
            errors.append(LintError(code="edge_unknown_to",
                                    message=f"edge to unknown node '{t}' (from '{f}')",
                                    location=f"{f}->{t}"))
        if f == t:
            errors.append(LintError(code="self_loop",
                                    message=f"edge {f}->{t} is a self-loop",
                                    location=f"{f}->{t}"))
        if edge.type is EdgeType.CONDITIONAL and not edge.gate:
            errors.append(LintError(code="conditional_no_gate",
                                    message=f"conditional edge {f}->{t} has no `gate`",
                                    location=f"{f}->{t}"))

        # Only count well-formed edges toward the graph (avoids KeyError spam).
        if f in node_ids and t in node_ids and f != t:
            adjacency[f].append(t)
            indegree[t] += 1

    # --- cycle detection (Kahn's algorithm over the well-formed graph) -------
    if _has_cycle(node_ids, adjacency, indegree):
        # If a self-loop already explained one cycle, keep both; both are real.
        errors.append(LintError(code="cycle",
                                message="execution graph contains a cycle",
                                location=None))

    # --- orphan nodes (a multi-node spec must wire every node) ---------------
    if len(spec.nodes) > 1:
        touched: set[str] = set()
        for edge in spec.edges:
            if edge.from_ in node_ids:
                touched.add(edge.from_)
            if edge.to in node_ids:
                touched.add(edge.to)
        for n in spec.nodes:
            if n.node_id not in touched:
                errors.append(LintError(code="orphan_node",
                                        message=f"node '{n.node_id}' is not connected to any edge",
                                        location=n.node_id))

    return errors


def is_valid(spec: Spec) -> bool:
    """True iff the Spec has no structural defects."""

    return not lint(spec)


def _has_cycle(
    node_ids: set[str],
    adjacency: dict[str, list[str]],
    indegree: dict[str, int],
) -> bool:
    """Kahn's: if we can't emit every node, a cycle exists."""

    indeg = dict(indegree)  # copy; Kahn mutates
    q: deque[str] = deque(n for n, d in indeg.items() if d == 0)
    emitted = 0
    while q:
        n = q.popleft()
        emitted += 1
        for nxt in adjacency.get(n, []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    return emitted != len(node_ids)
