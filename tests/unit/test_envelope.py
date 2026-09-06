"""Unit tests for the unified deployment envelope — ``build_optimized_envelope`` /
``export_optimized`` / ``load_optimized`` (improvement #4), in
``archforge.host.adapters.langgraph``.

The envelope is the single file production loads to apply the winner AND audit
it: the winning knobs (today's sidecar body, nested under ``knobs`` — the
consumer-compat seam) plus the lineage, the Gatekeeper decision, and the scores
that justified the promote. Pins the settled ``archforge.optimized/v1`` shape,
the ``knobs == sidecar body`` compat guarantee, the ``scores``/``decision`` field
projection, and the JSON round-trip.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import archforge.models as m
from archforge.gatekeeper import Action, Decision
from archforge.host.adapters.langgraph import (
    ENVELOPE_SCHEMA, build_optimized_envelope, export_optimized,
    export_spec_sidecar, load_optimized, load_spec_sidecar,
)
from archforge.mutate import apply_change


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #


def _n(nid: str, *, prompt: str = "p0", model: str = "",
       kind: m.NodeKind = m.NodeKind.LLM,
       knobs: m.Knobs | None = None, role: str = "r", tools=None) -> m.Node:
    return m.Node(node_id=nid, role=role, kind=kind, system_prompt=prompt,
                  model=model, knobs=knobs or m.Knobs(),
                  tools=tools if tools is not None else ["t0"])


def _spec(nodes: list[m.Node], edges: list[m.Edge] | None = None,
          *, spec_id: str | None = None,
          parent_spec_id: str | None = None) -> m.Spec:
    s = m.Spec(nodes=nodes, edges=edges or [])
    if spec_id is not None:
        s.spec_id = spec_id
    if parent_spec_id is not None:
        s.parent_spec_id = parent_spec_id
    return s


class _Run:
    """A duck-typed SuiteRun stand-in — the envelope reads ``mean``/``scores``/
    ``rubric_id``/``suite_id``/``tokens``/``latency_ms`` via ``getattr`` and is
    documented as tolerant of a lighter object (the producer never assumes the
    full ``SuiteRun``). Mirrors the ``_score_dims`` tolerance contract."""

    def __init__(self, *, mean: float, rubric_id: str = "default-v1",
                 suite_id: str = "S", tokens: int = 12, latency_ms: float = 3.0,
                 scores: list[m.RunScore] | None = None) -> None:
        self.mean = mean
        self.rubric_id = rubric_id
        self.suite_id = suite_id
        self.tokens = tokens
        self.latency_ms = latency_ms
        self.scores = scores or []


def _rubric_scores(agg: float) -> dict[str, float]:
    # the default_rubric dims (config_init.py): correctness/completeness/grounding
    return {"correctness": agg, "completeness": agg, "grounding": agg}


def _decision(*, action: Action = Action.AUTO_PROMOTE, margin: float = 0.11,
              rule: str = "auto_promote", reason: str = "clear small win") -> Decision:
    return Decision(action=action, attempt_id="att-1", reason=reason,
                    margin=margin, by_rule=rule)


# --------------------------------------------------------------------------- #
# shape + schema
# --------------------------------------------------------------------------- #


def test_envelope_has_schema_and_lineage() -> None:
    parent = _spec([_n("a")], spec_id="parent-id")
    cand = _spec([_n("a", prompt="p1")], spec_id="cand-id",
                 parent_spec_id="parent-id")
    env = build_optimized_envelope(cand, parent=parent, promoted_at_cycle=3)
    assert env["schema"] == ENVELOPE_SCHEMA == "archforge.optimized/v1"
    assert env["spec_id"] == "cand-id"
    assert env["parent_spec_id"] == "parent-id"
    assert env["promoted_at_cycle"] == 3
    # decision/scores default to None when the caller omits them — an embedder
    # building an envelope outside a promote still gets knobs + lineage.
    assert env["decision"] is None
    assert env["scores"] is None
    assert isinstance(env["knobs"], dict)


def test_envelope_knobs_equals_sidecar_body() -> None:
    """The consumer-compat seam: ``envelope["knobs"]`` is byte-identical to
    ``export_spec_sidecar``'s body — a consumer already reading the bare sidecar
    switches by reading ``env["knobs"]`` instead of the top level."""

    spec = _spec([
        _n("retrieve", kind=m.NodeKind.RETRIEVER,
           knobs=m.Knobs(top_k=9, tunable=("top_k",))),
        _n("answer", kind=m.NodeKind.LLM,
           knobs=m.Knobs(temperature=0.3, max_tokens=1024)),
    ], spec_id="x")
    env = build_optimized_envelope(spec, parent=None, promoted_at_cycle=0)
    sidecar = export_spec_sidecar(spec, "/dev/null")   # path unused — returns the body

    assert env["knobs"] == sidecar
    assert env["knobs"]["retrieve"]["top_k"] == 9
    assert env["knobs"]["answer"]["temperature"] == pytest.approx(0.3)
    assert env["knobs"]["answer"]["max_tokens"] == 1024
    # empty-projection nodes drop: a node with a blank prompt, no model, and no
    # knobs ships nothing (the projection's None/blank drop) → not in the body.
    # The same drop the sidecar applies.

    env2 = build_optimized_envelope(_spec([_n("z", prompt="", model="", tools=[])],
                                          spec_id="y"),
                                    parent=None, promoted_at_cycle=0)

    assert env2["knobs"] == {}


def test_envelope_decision_projects_action_rule_margin_reason() -> None:
    parent = _spec([_n("a")], spec_id="p")
    cand = _spec([_n("a", prompt="p1")], spec_id="c", parent_spec_id="p")
    env = build_optimized_envelope(
        cand, parent=parent, promoted_at_cycle=0, decision=_decision(),
    )
    d = env["decision"]
    assert d["action"] == "auto_promote"      # Action.value, not the enum
    assert d["rule"] == "auto_promote"
    assert d["margin"] == pytest.approx(0.11)
    assert d["reason"] == "clear small win"


def test_envelope_scores_block_from_runs() -> None:
    parent = _spec([_n("a")], spec_id="p")
    cand = _spec([_n("a", prompt="p1")], spec_id="c", parent_spec_id="p")
    cand_run = _Run(mean=0.66, tokens=42, latency_ms=7.0,
                    scores=[m.RunScore(run_id="r1", spec_id="c", task_id="t1",
                                       rubric_scores=_rubric_scores(0.66),
                                       aggregate=0.66, confidence=1.0,
                                       judge_meta=m.JudgeMeta(model="j",
                                                             rubric_id="default-v1"))])
    inc_run = _Run(mean=0.55, tokens=10, latency_ms=2.0)
    env = build_optimized_envelope(
        cand, parent=parent, promoted_at_cycle=2, decision=_decision(),
        cand_run=cand_run, inc_run=inc_run,
    )
    s = env["scores"]
    assert s["mean"] == pytest.approx(0.66)
    assert s["incumbent_mean"] == pytest.approx(0.55)
    assert s["rubric_id"] == "default-v1"
    assert s["suite_id"] == "S"
    assert s["tokens"] == 42
    assert s["latency_ms"] == 7.0
    # dims: mean per rubric dimension across the candidate's scored repeats
    dims = {d["dim"]: d["mean"] for d in s["dims"]}
    assert set(dims) == {"correctness", "completeness", "grounding"}
    assert all(v == pytest.approx(0.66) for v in dims.values())


def test_envelope_scores_incumbent_mean_none_when_inc_run_omitted() -> None:
    # cand_run given but inc_run None → incumbent_mean is None (a promote with no
    # recorded baseline ships a null inc mean, not a fabricated one).
    parent = _spec([_n("a")], spec_id="p")
    cand = _spec([_n("a", prompt="p1")], spec_id="c", parent_spec_id="p")
    env = build_optimized_envelope(
        cand, parent=parent, promoted_at_cycle=0, cand_run=_Run(mean=0.66),
    )
    assert env["scores"]["mean"] == pytest.approx(0.66)
    assert env["scores"]["incumbent_mean"] is None


def test_envelope_parent_spec_id_none_for_root() -> None:
    # parent=None ⇒ parent_spec_id None (a root promote has no lineage).
    root = _spec([_n("a")], spec_id="root-only")
    env = build_optimized_envelope(root, parent=None, promoted_at_cycle=0)
    assert env["parent_spec_id"] is None


# --------------------------------------------------------------------------- #
# round-trip
# --------------------------------------------------------------------------- #


def test_export_optimized_round_trips(tmp_path: Path) -> None:
    """export_optimized writes the envelope to disk; load_optimized reads it
    back intact — the file production reads to apply the winner."""

    parent = _spec([_n("retrieve", kind=m.NodeKind.RETRIEVER,
                       knobs=m.Knobs(top_k=4, tunable=("top_k",)))], spec_id="p")
    cand = _spec([_n("retrieve", kind=m.NodeKind.RETRIEVER,
                    knobs=m.Knobs(top_k=8, tunable=("top_k",)))],
                 spec_id="c", parent_spec_id="p")
    cand_run = _Run(mean=0.80, tokens=5, latency_ms=1.0,
                    scores=[m.RunScore(run_id="r1", spec_id="c", task_id="t1",
                                       rubric_scores=_rubric_scores(0.80),
                                       aggregate=0.80, confidence=1.0,
                                       judge_meta=m.JudgeMeta(model="j",
                                                             rubric_id="default-v1"))])
    inc_run = _Run(mean=0.50)
    path = tmp_path / "optimized.json"
    written = export_optimized(
        cand, path, parent=parent, promoted_at_cycle=4, decision=_decision(),
        cand_run=cand_run, inc_run=inc_run,
    )
    assert path.exists()
    # the returned dict == the on-disk object (return value for in-process use)
    on_disk_text = path.read_text(encoding="utf-8")
    assert json.loads(on_disk_text) == written
    # load round-trips

    loaded = load_optimized(path)

    assert loaded == written
    assert loaded["schema"] == "archforge.optimized/v1"
    assert loaded["knobs"]["retrieve"]["top_k"] == 8


def test_envelope_compat_sidecar_consumer_switches_to_knobs(tmp_path: Path) -> None:
    """A consumer that today reads ``load_spec_sidecar(path)`` (the bare
    ``{node_id:{knob}}``) switches to ``load_optimized(path)["knobs"]`` — same
    body, nested one level. This is the documented one-line consumer edit."""

    spec = _spec([_n("retrieve", kind=m.NodeKind.RETRIEVER,
                    knobs=m.Knobs(top_k=8, tunable=("top_k",)))], spec_id="c")
    sidecar_path = tmp_path / "side.json"
    env_path = tmp_path / "opt.json"
    export_spec_sidecar(spec, sidecar_path)
    export_optimized(spec, env_path, parent=None, promoted_at_cycle=0)

    bare = load_spec_sidecar(sidecar_path)
    wrapped = load_optimized(env_path)["knobs"]
    assert bare == wrapped
