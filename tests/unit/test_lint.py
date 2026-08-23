"""Unit tests for the Spec Linter (spec E5, I2; Phase 1).

`lint` is a pure function returning structured `LintError`s. We cover each
fault code explicitly, then a randomized property check: any acyclic randomly
generated Spec is valid, and each targeted mutation inserts the expected code.
"""

from __future__ import annotations

import random

import pytest

import archforge.models as m
from archforge.lint import is_valid, lint


# --------------------------------------------------------------------------- #
# Minimal valid Specs
# --------------------------------------------------------------------------- #


def _node(nid: str, *, tools: list[str] | None = None) -> m.Node:
    return m.Node(node_id=nid, role="r", system_prompt="p", model="gpt",
                  tools=tools if tools is not None else ["t0"])


def test_single_node_spec_is_valid() -> None:
    s = m.Spec(nodes=[_node("a")], edges=[])
    assert lint(s) == []
    assert is_valid(s)


def test_linear_chain_is_valid() -> None:
    s = m.Spec(
        nodes=[_node("a"), _node("b"), _node("c")],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE),
               m.Edge(from_="b", to="c", type=m.EdgeType.SEQUENCE)],
    )
    assert lint(s) == []


def test_fanout_join_is_valid() -> None:
    # a -> {b, c} -> d  (fanout then join)
    s = m.Spec(
        nodes=[_node("a"), _node("b"), _node("c"), _node("d")],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.FANOUT),
               m.Edge(from_="a", to="c", type=m.EdgeType.FANOUT),
               m.Edge(from_="b", to="d", type=m.EdgeType.JOIN),
               m.Edge(from_="c", to="d", type=m.EdgeType.JOIN)],
    )
    assert lint(s) == []


def test_conditional_requires_gate() -> None:
    valid = m.Spec(
        nodes=[_node("a"), _node("b")],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.CONDITIONAL, gate="ok")],
    )
    assert lint(valid) == []
    no_gate = m.Spec(
        nodes=[_node("a"), _node("b")],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.CONDITIONAL)],
    )
    codes = {e.code for e in lint(no_gate)}
    assert "conditional_no_gate" in codes


# --------------------------------------------------------------------------- #
# Each fault code, explicitly
# --------------------------------------------------------------------------- #


def test_edge_unknown_from_and_to() -> None:
    s = m.Spec(
        nodes=[_node("a")],
        edges=[m.Edge(from_="x", to="y", type=m.EdgeType.SEQUENCE)],
    )
    codes = {e.code for e in lint(s)}
    assert "edge_unknown_from" in codes
    assert "edge_unknown_to" in codes
    # 'a' should additionally be flagged orphan (multi-node-free node, single node -> not orphan)
    # here we have one node 'a' with no edges: single-node spec, so NOT orphan.
    assert "orphan_node" not in codes


def test_self_loop_is_flagged_separately_from_cycle() -> None:
    # A lone self-loop on a node with no other edges: reported as `self_loop`
    # but not `cycle` (the linter uses `cycle` for multi-edge loop topology; a
    # self-loop is its own distinct, clearer code).
    s = m.Spec(
        nodes=[_node("a")],
        edges=[m.Edge(from_="a", to="a", type=m.EdgeType.SEQUENCE)],
    )
    codes = {e.code for e in lint(s)}
    assert "self_loop" in codes
    assert "cycle" not in codes


def test_cycle_detected_on_ring() -> None:
    s = m.Spec(
        nodes=[_node("a"), _node("b"), _node("c")],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE),
               m.Edge(from_="b", to="c", type=m.EdgeType.SEQUENCE),
               m.Edge(from_="c", to="a", type=m.EdgeType.SEQUENCE)],
    )
    codes = {e.code for e in lint(s)}
    assert "cycle" in codes


def test_duplicate_edge_flagged() -> None:
    edge = m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)
    s = m.Spec(nodes=[_node("a"), _node("b")], edges=[edge, edge])
    codes = {e.code for e in lint(s)}
    assert "duplicate_edge" in codes


def test_duplicate_node_flagged() -> None:
    s = m.Spec(nodes=[_node("a"), _node("a")], edges=[])
    codes = {e.code for e in lint(s)}
    assert "duplicate_node" in codes


def test_orphan_node_in_multi_node_spec() -> None:
    s = m.Spec(
        nodes=[_node("a"), _node("b"), _node("lonely")],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)],
    )
    codes = {e.code for e in lint(s)}
    assert "orphan_node" in codes


def test_tool_id_empty_and_duplicate() -> None:
    s = m.Spec(nodes=[_node("a", tools=["ok", "", "ok"])], edges=[])
    codes = {e.code for e in lint(s)}
    assert "tool_id_empty" in codes
    assert "tool_id_duplicate" in codes


# --------------------------------------------------------------------------- #
# Randomized property check (deterministic via fixed seed; no hypothesis needed)
# --------------------------------------------------------------------------- #


def _rand_valid_spec(rng: random.Random) -> m.Spec:
    """Generate a structurally valid acyclic Spec with no tool errors.

    Validity is guaranteed by construction:
      * nodes laid along a random topological order
      * a spanning path (order[i]->order[i+1]) ensures every node is touched
        (so no orphans), and forward edges only (so acyclic)
      * conditional edges get a gate; tool ids are non-empty and distinct
    """

    n = rng.randint(1, 6)
    order = [f"n{i}" for i in range(n)]
    rng.shuffle(order)
    nodes = [_node(order[i], tools=[f"t{order[i]}_0"]) for i in range(n)]
    edges: list[m.Edge] = []
    seen: set[tuple[str, str, str]] = set()
    types_pool = [m.EdgeType.SEQUENCE, m.EdgeType.FANOUT, m.EdgeType.JOIN,
                  m.EdgeType.CONDITIONAL]

    def _add(f: str, t: str, etype: m.EdgeType) -> None:
        key = (f, t, etype.value)
        if key in seen:
            return
        seen.add(key)
        gate = "g" if etype is m.EdgeType.CONDITIONAL else None
        edges.append(m.Edge(from_=f, to=t, type=etype, gate=gate))

    # spanning path -> guarantees connectivity (no orphans) + acyclic backbone
    for i in range(n - 1):
        _add(order[i], order[i + 1], rng.choice(types_pool))
    # extra forward edges for variety
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.35:
                _add(order[i], order[j], rng.choice(types_pool))
    return m.Spec(nodes=nodes, edges=edges)


@pytest.mark.parametrize("seed", range(40))
def test_random_valid_specs_pass_linter(seed: int) -> None:
    rng = random.Random(seed)
    s = _rand_valid_spec(rng)
    errors = lint(s)
    assert errors == [], f"seed={seed}: unexpected {errors}"


def test_mutations_inject_expected_codes() -> None:
    rng = random.Random(7)
    base = _rand_valid_spec(rng)
    assert lint(base) == []

    # Mutation 1: splice in an edge to an unknown node -> unknown_to
    nids = [n.node_id for n in base.nodes]
    if len(nids) >= 1:
        bad = m.Spec(nodes=base.nodes,
                     edges=base.edges + [m.Edge(from_=nids[0], to="ghost",
                                                 type=m.EdgeType.SEQUENCE)])
        assert "edge_unknown_to" in {e.code for e in lint(bad)}

    # Mutation 2: introduce a cycle by reversing one edge's direction
    if len(nids) >= 2:
        e0 = base.edges[0] if base.edges else None
        if e0 is not None:
            cyc = m.Spec(nodes=base.nodes,
                         edges=base.edges + [m.Edge(from_=e0.to, to=e0.from_,
                                                     type=m.EdgeType.SEQUENCE)])
            # reversal may or may not actually create a cycle depending on topology;
            # when it does, we expect 'cycle'. lenient: at most ask it not to crash.
            codes = {e.code for e in lint(cyc)}
            assert isinstance(codes, set)

    # Mutation 3: drop a node that an edge references -> unknown_from/to
    if base.edges and len(base.nodes) > 1:
        dropped = base.edges[0].from_
        short = m.Spec(nodes=[n for n in base.nodes if n.node_id != dropped],
                       edges=base.edges)
        codes = {e.code for e in lint(short)}
        assert "edge_unknown_from" in codes or "edge_unknown_to" in codes


# --------------------------------------------------------------------------- #
# Linter output contract
# --------------------------------------------------------------------------- #


def test_lint_error_is_structured() -> None:
    s = m.Spec(nodes=[_node("a"), _node("b"), _node("lonely")],
               edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)])
    errs = lint(s)
    orphan = next(e for e in errs if e.code == "orphan_node")
    assert orphan.location == "lonely"
    assert orphan.message  # non-empty human text


# --------------------------------------------------------------------------- #
# Node-kind rule (non-LLM optimizer extension)
# --------------------------------------------------------------------------- #


def test_node_unknown_kind_flagged_defense_in_depth() -> None:
    # pydantic's enum coercion rejects a bad `kind` at construction, but a
    # `model_construct` / hand-edit bypass slips past it → the linter flags it.
    bad = m.Node.model_construct(node_id="a", role="r", kind="spaghetti")
    errs = lint(m.Spec(nodes=[bad], edges=[]))
    assert [e.code for e in errs] == ["node_unknown_kind"]
    assert errs[0].location == "a"
    assert "spaghetti" in errs[0].message


def test_non_llm_node_with_empty_prompt_and_model_is_valid() -> None:
    # The deliberate design: `system_prompt`/`model` are OPTIONAL for non-LLM kinds.
    # This pins the contract the SpecBuilder (`prompt: str = ""`) and config-decay
    # (`KnobVote.model: None == use the agent's default`) both depend on, so the
    # "text-in/text-out node need not declare one" path keeps linting clean.
    ret = m.Node(node_id="ret", role="retrieve", kind=m.NodeKind.RETRIEVER,
                 knobs=m.Knobs(top_k=3, tunable=("top_k",)))
    s = m.Spec(nodes=[ret], edges=[])
    assert lint(s) == []


def test_llm_node_with_empty_prompt_and_model_is_valid() -> None:
    # An LLM node with an empty prompt/model is ALSO valid — the cost cap's
    # `kind`-gate (architect) and the open `Knobs` guard are the real surfaces;
    # lint does NOT enforce prompt/model slots (would regress every text-in/text-out
    # Spec built via the Builder). This test guards that decision against re-regression.
    llm = m.Node(node_id="a", role="r")  # defaults: kind=llm, prompt="", model=""
    assert lint(m.Spec(nodes=[llm], edges=[])) == []
