"""Unit tests for the core data model (spec §3, Phase 1).

Covers content-addressed `spec_id` determinism + lineage-independence, the
`from`/`from_` alias handling on edges, scope auto-tagging, and threshold
defaults. Links invariant I2 (immutability by content).
"""

from __future__ import annotations

import pytest

import archforge.models as m


# --------------------------------------------------------------------------- #
# Content-addressed identity (I2)
# --------------------------------------------------------------------------- #


def _linear_spec() -> m.Spec:
    edges = [m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)]
    nodes = [
        m.Node(node_id="a", role="plan", system_prompt="p1", model="gpt",
               knobs=m.Knobs(temperature=0.7, retries=2), tools=["search"]),
        m.Node(node_id="b", role="answer", system_prompt="p2", model="gpt", tools=["write"]),
    ]
    return m.Spec(nodes=nodes, edges=edges)


def test_spec_id_is_deterministic() -> None:
    s = _linear_spec()
    assert s.compute_spec_id() == s.compute_spec_id()
    assert len(s.compute_spec_id()) == 16


def test_spec_id_independent_of_node_order() -> None:
    s = _linear_spec()
    rev = m.Spec(nodes=list(reversed(s.nodes)), edges=s.edges)
    assert rev.compute_spec_id() == s.compute_spec_id()


def test_spec_id_independent_of_edge_order() -> None:
    s = m.Spec(
        nodes=[m.Node(node_id="a", role="r", system_prompt="p", model="gpt"),
               m.Node(node_id="b", role="r", system_prompt="p", model="gpt")],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE),
               m.Edge(from_="b", to="a", type=m.EdgeType.SEQUENCE)],  # ring -> not valid, but hashable
    )
    shuffled = m.Spec(nodes=s.nodes, edges=list(reversed(s.edges)))
    assert s.compute_spec_id() == shuffled.compute_spec_id()


def test_spec_id_includes_lineage_excludes_role_and_timestamp() -> None:
    """Lineage (parent_spec_id) is part of the id; role (status) and time are not.

    Same content + same parent -> same id (idempotent commit).
    Same content + DIFFERENT parent -> different id (distinct lineage node).
    Same content + same parent + different status/created_at -> same id (role and
    timestamp never change identity, so promotion/rollback need no spec rewrite).
    """

    s = _linear_spec()
    # same parent, different role + timestamp -> same id
    role_shift = m.Spec(
        nodes=s.nodes, edges=s.edges,
        parent_spec_id=s.parent_spec_id,
        status=m.SpecStatus.INCUMBENT, created_at="2099-01-01T00:00:00Z",
    )
    assert role_shift.compute_spec_id() == s.compute_spec_id()
    # different parent -> different id
    new_lineage = m.Spec(
        nodes=s.nodes, edges=s.edges, parent_spec_id="some-ancestor",
    )
    assert new_lineage.compute_spec_id() != s.compute_spec_id()


def test_spec_id_changes_when_content_changes() -> None:
    s = _linear_spec()
    altered = m.Spec(nodes=s.nodes, edges=[m.Edge(from_="a", to="b", type=m.EdgeType.CONDITIONAL, gate="x")])
    assert altered.compute_spec_id() != s.compute_spec_id()


# --------------------------------------------------------------------------- #
# Edge alias handling
# --------------------------------------------------------------------------- #


def test_edge_constructible_with_from_keyword_attr() -> None:
    e = m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)
    assert e.from_ == "a"  # Python attribute
    assert e.to == "b"


def test_edge_json_uses_from_alias() -> None:
    e = m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)
    dumped = e.model_dump(mode="json")
    assert dumped["from"] == "a"
    assert "from_" not in dumped
    # round-trip back through the alias
    again = m.Edge.model_validate({**dumped})
    assert again.from_ == "a"


# --------------------------------------------------------------------------- #
# Change scope auto-tagging (I4 source of truth)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kind,expected_scope",
    [
        (m.ChangeKind.PROMPT_EDIT, m.Scope.SMALL),
        (m.ChangeKind.KNOB, m.Scope.SMALL),
        (m.ChangeKind.ADD_NODE, m.Scope.STRUCTURAL),
        (m.ChangeKind.REMOVE_NODE, m.Scope.STRUCTURAL),
        (m.ChangeKind.REWIRE, m.Scope.STRUCTURAL),
        (m.ChangeKind.MODEL_SWAP, m.Scope.STRUCTURAL),
    ],
)
def test_scope_classification(kind: m.ChangeKind, expected_scope: m.Scope) -> None:
    assert m.scope_for_kind(kind) is expected_scope
    ch = m.Change.for_kind(kind, "t", "d", "r")
    assert ch.scope is expected_scope


def test_attempt_builder_tags_scope() -> None:
    structural = m.Attempt(
        candidate_spec_id="c", parent_spec_id="p",
        change=m.Change.for_kind(m.ChangeKind.ADD_NODE, "verifier", "add node", "needs check"),
        verdict=m.Verdict.PENDING_HUMAN,
    )
    assert structural.change.scope is m.Scope.STRUCTURAL
    small = m.Attempt(
        candidate_spec_id="c", parent_spec_id="p",
        change=m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a", "edit prompt", "sharpen"),
        verdict=m.Verdict.PENDING_HUMAN,
    )
    assert small.change.scope is m.Scope.SMALL


# --------------------------------------------------------------------------- #
# Thresholds defaults (spec §6) — kept stable; later phases depend on them
# --------------------------------------------------------------------------- #


def test_threshold_defaults() -> None:
    th = m.Thresholds()
    assert th.tau == 0.05
    assert th.delta >= th.tau  # regression floor must be >= promotion margin (E6/I3)
    assert th.repeats == 1
    assert th.max_repeats == 3
    assert th.unrunnable_frac == 0.25
    assert th.plateau_cycles == 5


# --------------------------------------------------------------------------- #
# Round-trip: Spec serialises to JSON and revives, preserving identity
# --------------------------------------------------------------------------- #


def test_spec_json_roundtrip_preserves_identity() -> None:
    s = _linear_spec()
    payload = s.model_dump(mode="json")
    revived = m.Spec.model_validate(payload)
    assert revived.compute_spec_id() == s.compute_spec_id()


def test_edge_type_enum_members_match_spec() -> None:
    # the four edge kinds named in the spec
    assert set(m.EdgeType) == {
        m.EdgeType.SEQUENCE, m.EdgeType.FANOUT, m.EdgeType.JOIN, m.EdgeType.CONDITIONAL
    }
    # SpecStatus carries the three lineage states
    assert {v.value for v in m.SpecStatus} == {"incumbent", "candidate", "archived"}


# --------------------------------------------------------------------------- #
# NodeKind + open Knobs + kind-aware Node (non-LLM optimizer extension)
# --------------------------------------------------------------------------- #


def test_nodekind_enum_members_match_spec() -> None:
    assert {v.value for v in m.NodeKind} == {
        "llm", "rule", "retriever", "tool", "symbolic"
    }


def test_node_defaults_kind_llm_with_empty_prompt_and_model() -> None:
    # Back-compat: a Node predating non-LLM nodes is constructed without kind/prompt/model.
    # `kind` defaults to LLM; `system_prompt`/`model` are OPTIONAL (empty for non-LLM kinds).
    n = m.Node(node_id="a", role="r")
    assert n.kind is m.NodeKind.LLM
    assert n.system_prompt == "" and n.model == ""


def test_knobs_allows_extra_keys_and_carries_tunable() -> None:
    # The closed Knobs bag is OPENED: extra per-kind knobs (top_k/threshold/...)
    # ride via extra="allow", and `tunable` lists the extra keys the Architect may edit.
    pre = m.Knobs(temperature=0.5, retries=2, max_tokens=100)
    assert pre.tunable == ()                       # default: nothing extra is editable
    open_k = m.Knobs(top_k=3, threshold=0.5, tunable=("top_k", "threshold"))
    assert open_k.top_k == 3 and open_k.threshold == 0.5
    assert open_k.tunable == ("top_k", "threshold")
    # extra keys the model doesn't declare survive (extra="allow", not "forbid")
    assert m.Knobs(endpoint="http://x", tunable=("endpoint",)).endpoint == "http://x"


def test_node_model_config_stays_closed_only_knobs_open() -> None:
    # `Node` stays extra="forbid" — only Knobs opens (an unknown Node field must reject;
    # a non-LLM kind is expressed via `kind`, not by inventing fields).
    with pytest.raises(Exception):
        m.Node(node_id="a", role="r", unknown_field="x")  # type: ignore[call-arg]


def test_spec_id_back_compat_legacy_node_matches_explicit_llm() -> None:
    # Cardinal back-compat guarantee (the design's "legacy Specs load unchanged"):
    # a node at the DEFAULT kind (llm) + empty tunable hashes IDENTICALLY to a node
    # that omits them (a pre-feature Spec). Explicit non-default kind/tunable DO appear
    # → a different artifact gets a different id (content-addressing preserved).
    legacy = m.Spec(nodes=[m.Node(node_id="a", role="r", system_prompt="p", model="gpt")],
                    edges=[])
    explicit_llm = m.Spec(nodes=[m.Node(node_id="a", role="r", kind=m.NodeKind.LLM,
                                         system_prompt="p", model="gpt",
                                         knobs=m.Knobs(tunable=()))],
                          edges=[])
    assert legacy.compute_spec_id() == explicit_llm.compute_spec_id()
    # a retriever node IS a different artifact -> different id
    retriever = m.Spec(nodes=[m.Node(node_id="a", role="r", kind=m.NodeKind.RETRIEVER,
                                     knobs=m.Knobs(top_k=5, tunable=("top_k",)))],
                       edges=[])
    assert retriever.compute_spec_id() != legacy.compute_spec_id()
