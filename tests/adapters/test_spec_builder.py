"""Unit tests for the SpecBuilder DSL (`archforge.spec_builder`).

The Builder is the universal floor: any MAS (inspectable or not) is described
in typed Python that round-trips to a lint-clean `Spec`. These prove the DSL
builds valid specs end-to-end and fails fast (raising `SpecBuildError`) on a
malformed one — never handing a broken spec to the engine.
"""
from __future__ import annotations

import pytest

from archforge.lint import is_valid
from archforge.spec_builder import (
    CONDITIONAL, FANOUT, JOIN, SEQUENCE, SpecBuildError, SpecBuilder,
)


def _two_nodes():
    return (SpecBuilder()
            .node("a", role="planner", model="m", temperature=0.3)
            .node("b", role="writer", model="m", temperature=0.5, max_tokens=1000))


def test_builder_round_trips_to_lint_clean_spec():
    spec = (_two_nodes().edge("a", "b", kind=SEQUENCE).build())
    assert is_valid(spec)
    assert [n.node_id for n in spec.nodes] == ["a", "b"]
    assert len(spec.edges) == 1
    # knobs seeded
    b = next(n for n in spec.nodes if n.node_id == "b")
    assert b.knobs is not None
    assert b.knobs.max_tokens == 1000
    assert b.knobs.temperature == 0.5


def test_builder_seed_tools_and_system_prompt():
    spec = (_two_nodes()
            .node("retrieval", role="retriever", model="m", tools=["tavily"],
                  prompt="Extract from {sources}.")
            .edge("a", "retrieval", kind=SEQUENCE)
            .edge("retrieval", "b", kind=SEQUENCE)
            .build())
    assert is_valid(spec)
    r = next(n for n in spec.nodes if n.node_id == "retrieval")
    assert r.tools == ["tavily"]
    assert r.system_prompt == "Extract from {sources}."   # templates carry literal placeholders


def test_builder_all_edge_kinds_accepted():
    spec = (SpecBuilder()
            .node("s", role="r", model="m").node("t", role="r", model="m")
            .node("u", role="r", model="m").node("v", role="r", model="m")
            .node("w", role="r", model="m")
            .edge("s", "t", kind=SEQUENCE)
            .edge("t", "u", kind=FANOUT)
            .edge("t", "v", kind=FANOUT)
            .edge("u", "w", kind=JOIN)
            .edge("v", "w", kind=JOIN)
            .build())
    assert is_valid(spec)


def test_builder_edge_to_unknown_node_raises_specbuild_error():
    with pytest.raises(SpecBuildError) as exc:
        _two_nodes().edge("a", "GHOST", kind=SEQUENCE).build()
    assert "ghost" in str(exc.value).lower() or "unknown" in str(exc.value).lower()


def test_builder_empty_builds_clean():
    # a spec with nodes but no edges is still structurally valid (lint passes)
    spec = SpecBuilder().node("lone", role="r", model="m").build()
    assert is_valid(spec)
