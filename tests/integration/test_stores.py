"""Integration tests for the filesystem stores (Phase 2).

Covers the spec-tied properties: content-addressing + refuse-overwrite (I2),
active-pointer singularity (I1), lineage reachability (I3), idempotent commit,
rollback-as-pointer-swap + archive, reload-after-write (E10), and AttemptStore
dedup (E7). All against a real tmp dir — no fakes for the stores.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import archforge.models as m
from archforge.stores import AttemptStore, SpecStore, TraceStore
from archforge.stores.spec_store import NoActiveSpecError, UnknownSpecError


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _node(nid: str) -> m.Node:
    return m.Node(node_id=nid, role="r", system_prompt="p", model="gpt", tools=["t0"])


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / ".archforge"


@pytest.fixture
def specs(root: Path) -> SpecStore:
    return SpecStore(root)


@pytest.fixture
def traces(root: Path) -> TraceStore:
    return TraceStore(root)


@pytest.fixture
def attempts(root: Path) -> AttemptStore:
    return AttemptStore(root)


def _seed(specs: SpecStore) -> str:
    """Return a committed root spec_id (single-node incumbent)."""

    root_spec = m.Spec(nodes=[_node("a")], edges=[])
    sid = specs.commit(root_spec, parent_spec_id=None, status=m.SpecStatus.INCUMBENT)
    specs.set_active(sid)
    return sid


# --------------------------------------------------------------------------- #
# SpecStore — content-addressing + idempotent commit (I2)
# --------------------------------------------------------------------------- #


def test_commit_returns_content_id_and_persists(specs: SpecStore) -> None:
    s = m.Spec(nodes=[_node("a"), _node("b")],
               edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)])
    sid = specs.commit(s, parent_spec_id=None)
    assert sid and len(sid) == 16
    assert specs.has(sid)
    # the stored Spec carries its id and lineage
    loaded = specs.get(sid)
    assert loaded.spec_id == sid
    assert loaded.parent_spec_id is None


def test_idempotent_commit_same_content_same_parent_one_file(specs: SpecStore) -> None:
    s = m.Spec(nodes=[_node("a"), _node("b")],
               edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)])
    sid1 = specs.commit(s, parent_spec_id=None, created_at="2026-01-01T00:00:00Z")
    sid2 = specs.commit(s, parent_spec_id=None, created_at="2026-01-02T00:00:00Z")
    assert sid1 == sid2  # content + parent determine the id
    assert len(specs.known_ids()) == 1  # one file, not two (refuse-overwrite)


def test_same_content_different_parent_distinct_ids(specs: SpecStore) -> None:
    s = m.Spec(nodes=[_node("a"), _node("b")],
               edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)])
    as_root = specs.commit(s, parent_spec_id=None)
    as_child = specs.commit(s, parent_spec_id="some-ancestor")
    assert as_root != as_child  # lineage differentiates them
    assert len(specs.known_ids()) == 2


def test_known_specs_file_is_immutable_across_reloads(specs: SpecStore, root: Path) -> None:
    s = m.Spec(nodes=[_node("a")], edges=[])
    sid = specs.commit(s, parent_spec_id=None)
    path = root / "specs" / f"{sid}.json"
    bytes_before = path.read_text()
    # re-commit idempotently — must not rewrite
    specs.commit(s, parent_spec_id=None)
    assert path.read_text() == bytes_before


# --------------------------------------------------------------------------- #
# Active pointer — single source of truth, only set_active mutates (I1)
# --------------------------------------------------------------------------- #


def test_no_active_spec_raises(specs: SpecStore) -> None:
    with pytest.raises(NoActiveSpecError):
        specs.active()


def test_set_active_single_source_of_truth(specs: SpecStore) -> None:
    a = specs.commit(m.Spec(nodes=[_node("a")], edges=[]), parent_spec_id=None)
    b = specs.commit(m.Spec(nodes=[_node("b")], edges=[]), parent_spec_id=None)
    specs.set_active(a)
    assert specs.active_id() == a
    specs.set_active(b)
    assert specs.active_id() == b  # only one active at a time
    assert specs.active().nodes[0].node_id == "b"


def test_set_active_rejects_unknown_spec(specs: SpecStore) -> None:
    with pytest.raises(UnknownSpecError):
        specs.set_active("no-such-spec_id")


# --------------------------------------------------------------------------- #
# Lineage reachability (I3) + rollback as pointer swap (E6/I3)
# --------------------------------------------------------------------------- #


def test_lineage_chains_to_root(specs: SpecStore) -> None:
    root_id = _seed(specs)  # committed with parent None
    child = m.Spec(nodes=[_node("a"), _node("b")],
                  edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)])
    child_id = specs.commit(child, parent_spec_id=root_id)
    grandchild = m.Spec(nodes=[_node("a"), _node("b"), _node("c")],
                       edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE),
                              m.Edge(from_="b", to="c", type=m.EdgeType.SEQUENCE)])
    grand_id = specs.commit(grandchild, parent_spec_id=child_id)
    assert specs.lineage(grand_id) == [grand_id, child_id, root_id]
    assert specs.lineage(child_id)[-1] == root_id
    assert specs.lineage(root_id) == [root_id]


def test_rollback_is_pointer_swap_and_archives_never_deleted(specs: SpecStore) -> None:
    root_id = _seed(specs)  # parent None, active
    bad = m.Spec(nodes=[_node("a"), _node("bad")],
                 edges=[m.Edge(from_="a", to="bad", type=m.EdgeType.SEQUENCE)])
    bad_id = specs.commit(bad, parent_spec_id=root_id)
    specs.set_active(bad_id)  # promoted
    assert specs.active_id() == bad_id
    assert not specs.is_archived(bad_id)

    # rollback: pointer back to parent, bad archived (file still present)
    specs.set_active(root_id)
    specs.archive(bad_id, reason="rollback")
    assert specs.active_id() == root_id
    assert specs.is_archived(bad_id)
    assert specs.has(bad_id)  # file NOT deleted -> lineage/trace stays queryable (E6/I3)
    assert specs.get(bad_id).nodes[1].node_id == "bad"


# --------------------------------------------------------------------------- #
# Reload after write — store state survives re-instantiation (E10 recovery)
# --------------------------------------------------------------------------- #


def test_reload_restores_active_and_lineage(specs: SpecStore, root: Path) -> None:
    root_id = _seed(specs)
    child = m.Spec(nodes=[_node("a"), _node("b")],
                  edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)])
    child_id = specs.commit(child, parent_spec_id=root_id)
    specs.set_active(child_id)

    reloaded = SpecStore(root)  # fresh handle, same on-disk state
    assert reloaded.active_id() == child_id
    assert reloaded.active().nodes[0].node_id == "a"
    assert reloaded.lineage(child_id) == [child_id, root_id]


def test_failed_active_write_leaves_incumbent_unchanged(specs: SpecStore, root: Path) -> None:
    # Simulate a crash mid-promote by committing, setting active, then dropping
    # the pointer file on the floor and reloading -> loses only the pointer,
    # never the committed Spec bytes.
    root_id = _seed(specs)
    child = m.Spec(nodes=[_node("a"), _node("b")],
                  edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)])
    child_id = specs.commit(child, parent_spec_id=root_id)
    assert specs.has(child_id)

    (root / "active.pointer").unlink()  # pointer gone, Specs intact
    reloaded = SpecStore(root)
    assert reloaded.active_id() is None  # incumbent pointer lost, NOT the Spec
    assert reloaded.has(root_id) and reloaded.has(child_id)  # bytes survive


# --------------------------------------------------------------------------- #
# TraceStore — append-only, per-spec grouping
# --------------------------------------------------------------------------- #


def _trace(spec_id: str, task_id: str, run_id: str) -> m.Trace:
    return m.Trace(
        run_id=run_id, spec_id=spec_id, task_id=task_id,
        steps=[m.Step(node_id="a", prompt_in="p", response_out="r")],
        final_output="out", ok=True,
    )


def test_trace_append_and_read(traces: TraceStore) -> None:
    t = _trace("spec1", "task-a", "r1")
    traces.append(t)
    assert [x.run_id for x in traces.all("spec1")] == ["r1"]
    assert traces.latest("spec1", "task-a").run_id == "r1"
    assert traces.latest("spec1", "task-x") is None


def test_trace_for_task_returns_repeats_in_order(traces: TraceStore) -> None:
    for i in range(3):
        traces.append(_trace("spec1", "task-a", f"r{i}"))
    runs = [t.run_id for t in traces.for_task("spec1", "task-a")]
    assert runs == ["r0", "r1", "r2"]


def test_trace_groups_per_spec(traces: TraceStore) -> None:
    traces.append(_trace("specA", "t", "r1"))
    traces.append(_trace("specB", "t", "r1"))
    assert len(traces.all("specA")) == 1
    assert len(traces.all("specB")) == 1


# --------------------------------------------------------------------------- #
# AttemptStore — idempotent append + dedup (E7)
# --------------------------------------------------------------------------- #


def _attempt(*, candidate: str, parent: str, kind: m.ChangeKind, target: str,
             verdict: m.Verdict) -> m.Attempt:
    return m.Attempt(
        candidate_spec_id=candidate, parent_spec_id=parent,
        change=m.Change.for_kind(kind, target, "diff", "why"),
        verdict=verdict,
    )


def test_attempt_append_assigns_id_and_is_idempotent(attempts: AttemptStore) -> None:
    a = _attempt(candidate="c", parent="p", kind=m.ChangeKind.PROMPT_EDIT,
                 target="n2", verdict=m.Verdict.REJECTED)
    aid = attempts.append(a)
    assert aid and len(aid) == 16
    again = attempts.append(a)  # same Attempt object -> same id, not duplicated
    assert again == aid
    assert len(attempts.for_parent("p")) == 1


def test_attempt_match_and_blocking_dedup(attempts: AttemptStore) -> None:
    # a prior rejected attempt on (parent p, prompt_edit, target n2)
    rejected = _attempt(candidate="c1", parent="p", kind=m.ChangeKind.PROMPT_EDIT,
                        target="n2", verdict=m.Verdict.REJECTED)
    attempts.append(rejected)
    # a different change on the same parent (not blocking)
    struct = _attempt(candidate="c2", parent="p", kind=m.ChangeKind.ADD_NODE,
                     target="verifier", verdict=m.Verdict.PENDING_HUMAN)
    attempts.append(struct)

    matches = attempts.match("p", "prompt_edit", "n2")
    assert len(matches) == 1
    blocking = attempts.blocking("p", "prompt_edit", "n2")
    assert len(blocking) == 1 and blocking[0].verdict is m.Verdict.REJECTED
    # structural pending_human is not blocking
    assert attempts.blocking("p", "add_node", "verifier") == []
    # rolled_back is also blocking
    rolled = _attempt(candidate="c3", parent="p", kind=m.ChangeKind.PROMPT_EDIT,
                      target="n2", verdict=m.Verdict.ROLLED_BACK)
    attempts.append(rolled)
    assert len(attempts.blocking("p", "prompt_edit", "n2")) == 2


def test_attempt_reload_preserves_memory(attempts: AttemptStore, root: Path) -> None:
    a = _attempt(candidate="c", parent="p9", kind=m.ChangeKind.KNOB, target="n1",
                 verdict=m.Verdict.REJECTED)
    aid = attempts.append(a)
    reloaded = AttemptStore(root)
    assert reloaded.require(aid).change.target == "n1"
    # change.kind survived the JSON round-trip (Pydantic enum)
    assert reloaded.require(aid).change.kind is m.ChangeKind.KNOB
