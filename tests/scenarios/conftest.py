"""Shared harness + fixtures for the Phase-10 scripted E2E scenarios.

These scenarios drive the FULL loop with **real** organs on fakes — no
`ScriptedArchitect`. The Architect is the real `Architect` driven by a
`ScriptedLLM` (the plan's "scriptable Architect via ScriptedLLM"), so the
LLM-JSON -> parse -> mutate -> lint -> dedup path is exercised for real, the
SuiteRunner is the real one over `FakeHostMAS`, the Judge is `ScriptedJudge`,
and the stores are the real filesystem stores on a tmp `.archforge`.

Helpers here replicate exactly what the Engine + Architect + mutate do, so a
scenario can pre-compute a candidate's content-addressed `spec_id` (parent
stamped the way `SpecStore.commit` stamps it) and script the ScriptedJudge /
FakeHostMAS against that id BEFORE the run — determinism without a real model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import archforge.models as m
from archforge.architect import Architect
from archforge.engine import Engine, EngineConfig
from archforge.host import FakeHostMAS
from archforge.host.base import Task
from archforge.judge import ScriptedJudge
from archforge.llm import ScriptedLLM
from archforge.mutate import apply_change
from archforge.stores import AttemptStore, SpecStore, TraceStore
from archforge.suite import Suite

SUITE_ID = "S"
RUBRIC = "default-v1"


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #


def N(nid: str, *, prompt: str = "p0", model: str = "gpt",
      role: str = "r", tools: tuple[str, ...] = ("t0",)) -> m.Node:
    return m.Node(node_id=nid, role=role, system_prompt=prompt, model=model,
                  tools=list(tools))


def tasks(n: int = 1, *, prefix: str = "t") -> list[Task]:
    return [Task(task_id=f"{prefix}{i + 1}", input=f"q{i + 1}") for i in range(n)]


def make_suite(*, rubric_id: str = RUBRIC, suite_id: str = SUITE_ID,
               n_tasks: int = 1) -> Suite:
    return Suite(suite_id=suite_id, rubric_id=rubric_id, tasks=tasks(n_tasks))


def real_architect(llm: ScriptedLLM, *, model: str = "architect-1") -> Architect:
    return Architect(llm, model=model)


def build_engine(stores, *, judge, architect, suite, host=None,
                 thresholds=None, config=None) -> Engine:
    specs, atts, ts, _ = stores
    return Engine(
        host=host or FakeHostMAS(), judge=judge, architect=architect,
        spec_store=specs, attempt_store=atts, trace_store=ts, suite=suite,
        thresholds=thresholds or m.Thresholds(),
        config=config or EngineConfig(),
    )


# --------------------------------------------------------------------------- #
# ScriptedLLM proposal payloads (the JSON the Architect parses)
# --------------------------------------------------------------------------- #


def llm_proposal(*, kind: str, target: str, payload: dict,
                 rationale: str = "scenario") -> dict:
    """The dict queued via `ScriptedLLM.respond_json(...)` -> the Architect."""
    return {"kind": kind, "target": target, "rationale": rationale, "payload": payload}


def llm_noop() -> dict:
    """A well-formed JSON response with no usable `kind` -> the Architect plateaus.

    (A missing/invalid `kind` makes `_change_from_payload` return None -> plateau,
    which is how the loop winds down a real Architect deterministically rather
    than under-scripting the LLM queue into an AssertionError.)
    """
    return {"status": "done", "note": "nothing to propose"}


def llm_malformed() -> dict:
    """A JSON response whose.Node+payload survives mutate but fails the linter (E5)."""
    return llm_proposal(
        kind="add_node", target="v",
        payload={"node": N("v", prompt="pv").model_dump(mode="json"), "wiring": {}},
        rationale="add an isolated verifier",
    )


# --------------------------------------------------------------------------- #
# pre-compute the candidate a real Architect + the Engine will commit
# --------------------------------------------------------------------------- #


def candidate_spec(incumbent: m.Spec, *, kind: str, target: str,
                   payload: dict, rationale: str = "scenario") -> m.Spec:
    """Replicate Architect+mutate+Engine.commit to know a proposal's candidate Spec.

    The Engine commits `ArchitectProposal.candidate` with `parent_spec_id` set to
    the incumbent's id; the Architect's candidate is `apply_change(incumbent, change,
    payload)` where `change = Change.for_kind(kind, target, diff, rationale)`. The
    `diff`/`rationale` neither affect `spec_id` (only nodes+edges+parent do) nor
    survive into the candidate Spec, so this matches byte-for-byte what the live run
    produces — letting scenarios script the Judge/Host against the exact id.
    """
    change = m.Change.for_kind(m.ChangeKind(kind), target, "diff", rationale)
    cand = apply_change(incumbent, change, payload)
    cand.parent_spec_id = incumbent.spec_id
    # `apply_change` model_copies the incumbent, so `cand` INHERITS its spec_id
    # (rid). The real Engine sidesteps this via SpecStore.commit (which reloads a
    # clean Spec); but when a scenario passes `cand` straight to a SuiteRunner, a
    # non-None spec_id makes the runner stamp runs with rid instead of
    # recomputing. Reset it so the content-hash path is used — matching cand_id.
    cand = cand.model_copy(update={"spec_id": None})
    return cand


def cand_id(incumbent: m.Spec, *, kind: str, target: str, payload: dict,
            rationale: str = "scenario") -> str:
    return candidate_spec(incumbent, kind=kind, target=target, payload=payload,
                          rationale=rationale).compute_spec_id()


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def stores(tmp_path: Path):
    """Real filesystem stores on a tmp `.archforge` root."""
    root = tmp_path / ".archforge"
    return SpecStore(root), AttemptStore(root), TraceStore(root), root


@pytest.fixture
def seeded(stores):
    """A root incumbent committed + set active; returns (spec_id, Spec)."""
    specs = stores[0]
    spec = m.Spec(nodes=[N("a", prompt="p0")], edges=[])
    rid = specs.commit(spec, parent_spec_id=None, status=m.SpecStatus.INCUMBENT)
    specs.set_active(rid)
    # active() returns a Spec with spec_id populated — the live engine sees this.
    return rid, specs.get(rid)


def judge_for(*pairs: tuple[str, float]) -> ScriptedJudge:
    """A ScriptedJudge with per-(spec_id, task_id='t1') aggregates pinned."""
    j = ScriptedJudge()
    for spec_id, score in pairs:
        j.set_aggregate(spec_id, "t1", score)
    return j
