"""Unit tests for the Judge + ScriptedJudge (Phase 5).

Covers: real Judge via ScriptedLLM (JSON parsed, rubric_id stamped, per-step
breakdown populated, malformed-JSON -> LLMError); the shared pure aggregation
(confidence-weighted mean + per-task means, equal-weight degenerate case); and
the ScriptedJudge programming hooks (fixed per-task tables, noise band, forced
regression for E6, raise-on-cue for E9). Links: E1, E2, E6, E9, I5.
"""

from __future__ import annotations

import pytest

import archforge.models as m
from archforge.host.base import Task
from archforge.judge import Judge, ScriptedJudge
from archforge.judge.base import JudgeProtocol, aggregate_scores
from archforge.llm import LLMError, ScriptedLLM


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _trace(spec_id: str = "s1", task_id: str = "t1", run_id: str = "r1",
           nodes: list[str] | None = None) -> m.Trace:
    nodes = nodes or ["a", "b"]
    return m.Trace(
        run_id=run_id, spec_id=spec_id, task_id=task_id,
        steps=[m.Step(node_id=n, prompt_in="p", response_out="r") for n in nodes],
        final_output="out", ok=True,
    )


def _task(task_id: str = "t1") -> Task:
    return Task(task_id=task_id, input="do the thing", rubric_id="default-v1", suite_id="suite-1")


def _judgment(aggregate: float = 0.7, steps: list[dict] | None = None,
              confidence: float = 1.0) -> dict:
    if steps is None:
        steps = [{"node_id": "a", "sub_rubrics": {"correctness": 0.8}, "note": "ok"},
                 {"node_id": "b", "sub_rubrics": {"correctness": 0.6}, "note": "weak"}]
    return {
        "aggregate": aggregate, "confidence": confidence,
        "rubric_scores": {"correctness": 0.7, "completeness": 0.7, "grounding": 0.7},
        "step_scores": steps,
    }


# --------------------------------------------------------------------------- #
# Real Judge via ScriptedLLM
# --------------------------------------------------------------------------- #


def test_real_judge_parses_json_and_stamps_rubric_id() -> None:
    llm = ScriptedLLM().respond_json(_judgment(0.7))
    judge = Judge(model="claude", llm=llm)
    rs = judge.score(_trace(), _task(), rubric_id="default-v1")
    assert rs.aggregate == 0.7
    assert rs.judge_meta.rubric_id == "default-v1"
    assert rs.judge_meta.model == "claude"
    assert rs.run_id == "r1" and rs.spec_id == "s1" and rs.task_id == "t1"
    # per-step breakdown preserved — names the agent that lost points
    assert [s.node_id for s in rs.step_scores] == ["a", "b"]
    assert rs.step_scores[1].sub_rubrics["correctness"] == 0.6


def test_real_judge_records_call_params() -> None:
    llm = ScriptedLLM().respond_json(_judgment())
    Judge(model="claude", llm=llm).score(_trace(), _task(), "default-v1")
    assert llm.calls[0].model == "claude"
    assert llm.calls[0].response_format == "json"
    assert llm.calls[0].temperature == 0.0


def test_real_judge_malformed_json_raises_LLMError() -> None:
    llm = ScriptedLLM().respond_json({"aggregate": "not a float"})  # missing/conflicting
    judge = Judge(model="claude", llm=llm)
    with pytest.raises(LLMError):
        judge.score(_trace(), _task(), "default-v1")


def test_real_judge_empty_trace_still_scorable() -> None:
    llm = ScriptedLLM().respond_json({"aggregate": 0.2, "confidence": 0.5, "rubric_scores": {},
                                       "step_scores": []})
    rs = Judge(model="claude", llm=llm).score(_trace(nodes=[]), _task(), "default-v1")
    assert rs.aggregate == 0.2 and rs.step_scores == []


def test_real_judge_satisfies_protocol() -> None:
    assert isinstance(Judge(model="claude", llm=ScriptedLLM()), JudgeProtocol)


# --------------------------------------------------------------------------- #
# Pure aggregation (shared real + scripted)
# --------------------------------------------------------------------------- #


def _rs(task_id: str, aggregate: float, confidence: float = 1.0, rubric_id: str = "r") -> m.RunScore:
    return m.RunScore(
        run_id="r", spec_id="s", task_id=task_id, aggregate=aggregate,
        confidence=confidence, rubric_scores={},
        judge_meta=m.JudgeMeta(model="m", rubric_id=rubric_id), step_scores=[],
    )


def test_aggregate_empty_scores_is_zero() -> None:
    agg = aggregate_scores([], suite_id="s", rubric_id="r")
    assert agg.mean == 0.0 and agg.n_runs == 0


def test_aggregate_confidence_weighted_mean() -> None:
    scores = [_rs("t1", 1.0, confidence=1.0), _rs("t1", 0.0, confidence=3.0)]
    agg = aggregate_scores(scores, suite_id="s", rubric_id="r")
    # weighted: (1*1 + 0*3)/4 = 0.25
    assert agg.mean == pytest.approx(0.25)
    assert agg.confidence == pytest.approx(2.0)  # (1+3)/2


def test_aggregate_equal_weight_when_confidence_zero() -> None:
    scores = [_rs("t1", 0.0, confidence=0.0), _rs("t1", 1.0, confidence=0.0)]
    agg = aggregate_scores(scores, suite_id="s", rubric_id="r")
    assert agg.mean == pytest.approx(0.5)


def test_aggregate_per_task_means() -> None:
    scores = [_rs("t1", 0.4), _rs("t1", 0.6), _rs("t2", 0.8), _rs("t2", 0.8)]
    agg = aggregate_scores(scores, suite_id="s", rubric_id="r")
    assert agg.per_task == {"t1": 0.5, "t2": 0.8}
    assert agg.rubric_id == "r"


# --------------------------------------------------------------------------- #
# ScriptedJudge — hooks
# --------------------------------------------------------------------------- #


def test_scripted_judge_fixed_aggregates() -> None:
    j = ScriptedJudge().set_aggregate("s1", "t1", 0.9).set_aggregate("s1", "t2", 0.3)
    r1 = j.score(_trace(task_id="t1"), _task("t1"), "default-v1")
    r2 = j.score(_trace(task_id="t2"), _task("t2"), "default-v1")
    assert r1.aggregate == 0.9 and r2.aggregate == 0.3
    # per-step breakdown is filled (one StepScore per step)
    assert len(r1.step_scores) == 2 and r1.step_scores[0].node_id == "a"


def test_scripted_judge_base_when_unconfigured() -> None:
    j = ScriptedJudge().set_base(0.5)
    rs = j.score(_trace(), _task(), "default-v1")
    assert rs.aggregate == 0.5


def test_scripted_judge_noise_band_is_deterministic() -> None:
    # Two judges with identical setup produce identical jitter sequence (E1).
    def run() -> list[float]:
        j = ScriptedJudge().set_base(0.6).with_noise_band(0.03)
        out = []
        for i in range(4):
            out.append(j.score(_trace(run_id=f"r{i}"), _task(), "default-v1").aggregate)
        return out

    assert run() == run()


def test_scripted_judge_regress_after_triggers() -> None:
    j = ScriptedJudge().set_base(1.0).regress_after("promoted", drop=0.5)
    # first score the promoted spec (arms the regression)
    a = j.score(_trace(spec_id="promoted"), _task(), "default-v1").aggregate
    assert a == 1.0
    # a subsequent, different spec regresses by `drop`
    b = j.score(_trace(spec_id="new-incumbent"), _task(), "default-v1").aggregate
    assert b == pytest.approx(0.5)


def test_scripted_judge_raise_on_next_signals_outage() -> None:
    j = ScriptedJudge().raise_on_next()
    with pytest.raises(LLMError):
        j.score(_trace(), _task(), "default-v1")
    # consumed; next call works normally
    j.set_base(0.5)
    assert j.score(_trace(), _task(), "default-v1").aggregate == 0.5


def test_scripted_judge_records_scored() -> None:
    # _trace() defaults spec_id="s1", task_id="t1" — match the scripted table
    j = ScriptedJudge().set_aggregate("s1", "t1", 0.4)
    j.score(_trace(), _task(), "default-v1")
    assert len(j.scored) == 1 and j.scored[0].aggregate == 0.4


def test_scripted_judge_score_suite_reuses_aggregator() -> None:
    j = ScriptedJudge()
    scores = [_rs("t1", 0.4), _rs("t1", 0.6)]
    agg = j.score_suite(scores, suite_id="s", rubric_id="default-v1")
    assert agg.mean == pytest.approx(0.5)
    assert agg.rubric_id == "default-v1"
