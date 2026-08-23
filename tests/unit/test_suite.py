"""Unit tests for the SuiteRunner (Phase 7) — the P-E-C *Evaluate* step.

Uses the REAL `SuiteRunner` with fakes for its dependencies (spec §7):
`FakeHostMAS`, `ScriptedJudge`, a real `TracingMiddleware`+`TraceStore` on a tmp
dir. Single-node specs and R=1 keep the crash scripting simplest — `FakeAgent`
keys its `crash_on` off the agent's cumulative `invoke_count`, so one invoke per
task makes the index == the task index.

Links: E1/E3 (R-repeats -> mean), E4 (mid-run crash + partial trace + ε gate),
E9 (grader outage bounded retry -> crash path), I5 (same geometry for deltas).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import archforge.models as m
from archforge.host import FakeHostMAS
from archforge.host.base import Task
from archforge.judge import ScriptedJudge
from archforge.judge.base import aggregate_scores
from archforge.llm import LLMError
from archforge.stores import TraceStore
from archforge.suite import Suite, SuiteRun, SuiteRunner


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


def N(nid: str, *, model: str = "gpt", prompt: str = "p") -> m.Node:
    return m.Node(node_id=nid, role="r", system_prompt=prompt, model=model, tools=["t0"])


def single_spec(*, model: str = "gpt") -> m.Spec:
    """A one-node spec: each task invokes the agent once (invoke_count == task idx)."""
    return m.Spec(nodes=[N("a", model=model)], edges=[])


def task(tid: str, *, input: str = "do it") -> Task:
    return Task(task_id=tid, input=input)


def make_suite(tasks: list[Task], *, suite_id: str = "suite-1",
               rubric_id: str = "default-v1") -> Suite:
    return Suite(suite_id=suite_id, rubric_id=rubric_id, tasks=tasks)


@pytest.fixture
def trace_store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / ".archforge")


def runner(host, judge, trace_store, **kw) -> SuiteRunner:
    return SuiteRunner(host, judge, trace_store, **kw)


# --------------------------------------------------------------------------- #
# R-repeat aggregation (E1/E3)
# --------------------------------------------------------------------------- #


def test_r_repeats_aggregate_to_mean(trace_store: TraceStore) -> None:
    spec = single_spec()
    sid = spec.compute_spec_id()
    judge = (ScriptedJudge()
             .set_aggregate(sid, "t1", 0.4)
             .set_aggregate(sid, "t2", 0.8))
    run = runner(FakeHostMAS(), judge, trace_store).run_suite(
        spec, make_suite([task("t1"), task("t2")]), R=2)

    assert (run.repeats, run.n_tasks, run.n_crashed) == (2, 2, 0)
    assert [s.task_id for s in run.scores] == ["t1", "t1", "t2", "t2"]  # task-major
    assert run.mean == pytest.approx(0.6)
    assert run.aggregate.per_task == {"t1": 0.4, "t2": 0.8}


# --------------------------------------------------------------------------- #
# Mid-run crash — partial trace retained, suite continues (E4)
# --------------------------------------------------------------------------- #


def test_mid_run_crash_retains_partial_trace(trace_store: TraceStore) -> None:
    spec = single_spec()
    sid = spec.compute_spec_id()
    judge = ScriptedJudge().set_base(0.6)
    # crash the 3rd task only (invoke indices 0,1,*2*,3) -> 1 of 4 crashed
    host = FakeHostMAS(node_scripts={"a": {"crash_on": lambda i: i == 2}})

    run = runner(host, judge, trace_store).run_suite(
        spec, make_suite([task(f"t{i}") for i in range(4)]), R=1)

    assert run.n_crashed == 1
    assert run.unrunnable is False                 # 1/4 == ε(0.25), not > ε
    assert len(run.scores) == 3                     # 3 survivors scored
    partial = trace_store.latest(sid, "t2")
    assert partial is not None and partial.ok is False and partial.error is not None
    assert trace_store.latest(sid, "t0").ok is True


# --------------------------------------------------------------------------- #
# ε fraction → unrunnable (E4); strict boundary
# --------------------------------------------------------------------------- #


def test_over_epsilon_crashed_flags_unrunnable(trace_store: TraceStore) -> None:
    spec = single_spec()
    judge = ScriptedJudge().set_base(0.6)
    # crash 2 of 4 (indices 1,2) -> 0.5 > ε=0.25 -> unrunnable
    host = FakeHostMAS(node_scripts={"a": {"crash_on": lambda i: i in (1, 2)}})

    run = runner(host, judge, trace_store).run_suite(
        spec, make_suite([task(f"t{i}") for i in range(4)]), R=1)

    assert run.n_crashed == 2
    assert run.unrunnable is True
    assert len(run.scores) == 2                     # mean over 2 survivors
    assert run.mean == pytest.approx(0.6)


def test_at_epsilon_boundary_is_not_unrunnable(trace_store: TraceStore) -> None:
    # exactly ε crashed (1/4 = 0.25) must NOT trip unrunnable (strict >)
    spec = single_spec()
    judge = ScriptedJudge().set_base(0.5)
    host = FakeHostMAS(node_scripts={"a": {"crash_on": lambda i: i == 0}})

    run = runner(host, judge, trace_store).run_suite(
        spec, make_suite([task(f"t{i}") for i in range(4)]), R=1)

    assert (run.n_crashed, run.unrunnable) == (1, False)


# --------------------------------------------------------------------------- #
# Clear win/loss at R=1 — suite means are comparable (I5 geometry)
# --------------------------------------------------------------------------- #


def test_suite_means_make_win_loss_computable(trace_store: TraceStore) -> None:
    inc = single_spec(model="gpt")
    cand = single_spec(model="claude")
    judge = (ScriptedJudge()
             .set_aggregate(inc.compute_spec_id(), "t1", 0.60)
             .set_aggregate(cand.compute_spec_id(), "t1", 0.66))

    r = runner(FakeHostMAS(), judge, trace_store)
    inc_run = r.run_suite(inc, make_suite([task("t1")]), R=1)
    cand_run = r.run_suite(cand, make_suite([task("t1")]), R=1)

    assert cand_run.mean - inc_run.mean == pytest.approx(0.06)   # >= τ=0.05 -> a win
    assert cand_run.spec_id != inc_run.spec_id
    # same geometry (suite+rubric) so the delta is meaningful (I5)
    assert (cand_run.suite_id, cand_run.rubric_id) == ("suite-1", "default-v1")
    assert (inc_run.suite_id, inc_run.rubric_id) == ("suite-1", "default-v1")


# --------------------------------------------------------------------------- #
# Determinism / trace sink — run_ids unique across repeats
# --------------------------------------------------------------------------- #


def test_trace_sink_run_ids_unique_across_repeats(trace_store: TraceStore) -> None:
    spec = single_spec()
    sid = spec.compute_spec_id()
    judge = ScriptedJudge().set_base(0.5)

    runner(FakeHostMAS(), judge, trace_store).run_suite(
        spec, make_suite([task("t1"), task("t2")]), R=2)

    all_traces = trace_store.all(sid)
    assert len(all_traces) == 4                       # 2 tasks × 2 repeats, all sunk
    run_ids = [t.run_id for t in all_traces]
    assert len(set(run_ids)) == 4                     # unique across repeats
    assert [t.run_id for t in trace_store.for_task(sid, "t1")] == run_ids[:2]


# --------------------------------------------------------------------------- #
# Judge outage — bounded retry, then fold into the crash path (E9)
# --------------------------------------------------------------------------- #


class _AlwaysDownJudge:
    """Grader permanently down — to exercise the bounded-retry + fold path."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, trace, task, rubric_id):
        self.calls += 1
        raise LLMError("grader down")

    def score_suite(self, scores, *, suite_id, rubric_id):
        return aggregate_scores(scores, suite_id=suite_id, rubric_id=rubric_id)


def test_persistent_judge_outage_retries_then_folds_to_unrunnable(
    trace_store: TraceStore,
) -> None:
    judge = _AlwaysDownJudge()
    backoff_calls: list[int] = []
    run = runner(FakeHostMAS(), judge, trace_store,
                 judge_retries=3, backoff=lambda i: backoff_calls.append(i)
                 ).run_suite(single_spec(), make_suite([task("t1"), task("t2")]), R=1)

    # 2 runs × (1 + 3 retries) attempts -> 8 grader calls (bounded retry done)
    assert judge.calls == 2 * (3 + 1)
    assert backoff_calls == [0, 1, 2, 0, 1, 2]       # 3 backoffs per run, indexed 0-based
    assert run.scores == []                          # nothing scored
    assert run.n_crashed == 2 and run.unrunnable is True   # both tasks fully failed


def test_transient_outage_one_unscored_suite_continues(trace_store: TraceStore) -> None:
    judge = ScriptedJudge().set_base(0.7).raise_on_next()
    runner(FakeHostMAS(), judge, trace_store, judge_retries=0).run_suite(
        single_spec(), make_suite([task("t1"), task("t2")]), R=1)

    # t1's single attempt raised (outage consumed); t1 unscored -> counts as failed.
    # t2: outage gone -> scored normally; suite did not abort.
    # The forward-looking assertions live in the SuiteRun; here we just confirm
    # the judge recovered (it scored exactly one run).
    assert len(judge.scored) == 1 and judge.scored[0].aggregate == pytest.approx(0.7)


def test_transient_outage_survivors_and_shape(trace_store: TraceStore) -> None:
    judge = ScriptedJudge().set_base(0.7).raise_on_next()
    run = runner(FakeHostMAS(), judge, trace_store, judge_retries=0).run_suite(
        single_spec(), make_suite([task("t1"), task("t2")]), R=1)

    assert len(run.scores) == 1 and run.scores[0].task_id == "t2"
    assert run.n_crashed == 1
    assert run.scores[0].aggregate == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# Guards + shape
# --------------------------------------------------------------------------- #


def test_run_suite_rejects_zero_repeats(trace_store: TraceStore) -> None:
    with pytest.raises(ValueError):
        runner(FakeHostMAS(), ScriptedJudge(), trace_store).run_suite(
            single_spec(), make_suite([task("t1")]), R=0)


def test_empty_suite_is_safe_not_unrunnable(trace_store: TraceStore) -> None:
    run = runner(FakeHostMAS(), ScriptedJudge(), trace_store).run_suite(
        single_spec(), make_suite([]), R=1)
    assert (run.n_tasks, run.n_crashed, run.unrunnable) == (0, 0, False)
    assert run.mean == 0.0


def test_suite_run_mean_and_unrunnable_properties(trace_store: TraceStore) -> None:
    spec = single_spec()
    judge = ScriptedJudge().set_aggregate(spec.compute_spec_id(), "t1", 0.9)
    run = runner(FakeHostMAS(), judge, trace_store).run_suite(
        spec, make_suite([task("t1")]), R=1)
    assert isinstance(run, SuiteRun)
    assert run.mean == run.aggregate.mean == pytest.approx(0.9)
    assert run.unrunnable is run.aggregate.unrunnable is False


def test_tokens_reported_from_sunk_steps(trace_store: TraceStore) -> None:
    spec = single_spec()
    judge = ScriptedJudge().set_base(0.5)
    run = runner(FakeHostMAS(), judge, trace_store).run_suite(
        spec, make_suite([task("t1"), task("t2")]), R=1)
    # the fake populates non-zero deterministic tokens per step
    assert run.tokens > 0
