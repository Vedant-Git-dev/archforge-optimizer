"""The Judge — LLM-as-judge scoring of a run (spec §3, §4, §5).

`Judge.score(trace, task, rubric_id) -> RunScore` is ONE LLM call that turns a
trace into a scored verdict: an aggregate, named rubric dimensions, a
confidence, and a *per-step breakdown* (StepScore[]) naming which agent lost
which points. The per-step breakdown is the credit-assignment raw material the
Architect consumes — but the Judge contains NO diagnosis or proposal
text; per the locked "keep split" decision it scores only.

`aggregate_scores(...)` turns a list of RunScores (R repeats × N tasks) into a
single `SuiteAggregate` (confidence-weighted mean + per-task means). It is pure
math — no LLM call — so the call budget stays at "1 judge call per scored run".
rubric_id is stamped on every RunScore and the aggregate (invariant I5, E2).
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from archforge.config import DEFAULT_RUBRIC_ID, DEFAULT_SUB_RUBRICS
import archforge.models as m
from archforge.host.base import Task
from archforge.llm.base import LLMClient, LLMError, Message, Role


# --------------------------------------------------------------------------- #
# Rubric — the scoring schema, versioned by rubric_id (E2 / I5)
# --------------------------------------------------------------------------- #


class Rubric(BaseModel):
    """A named, versioned scoring schema: sub-rubric -> human description.

    `rubric_id` is the version key recorded on every RunScore, so the Gatekeeper
    never compares scores graded under different rubrics (invariant I5).
    """

    model_config = ConfigDict(extra="forbid")

    rubric_id: str
    sub_rubrics: dict[str, str]  # dimension name -> "what a high score looks like"


default_rubric = Rubric(
    rubric_id=DEFAULT_RUBRIC_ID,
    sub_rubrics=dict(DEFAULT_SUB_RUBRICS),
)


# --------------------------------------------------------------------------- #
# SuiteAggregate — a suite's scores rolled into one comparable number
# --------------------------------------------------------------------------- #


class SuiteAggregate(BaseModel):
    """Confidence-weighted aggregation of a suite's RunScores (spec §5).

    `mean` is what the Gatekeeper compares across incumbent vs candidate (same
    rubric_id + same suite, per I5). `per_task` supports per-task diagnosis and
    suite-aware plateau reasoning.
    """

    model_config = ConfigDict(extra="allow")

    suite_id: str
    rubric_id: str
    mean: float
    n_runs: int
    per_task: dict[str, float] = Field(default_factory=dict)
    confidence: float = 1.0
    # Set by the SuiteRunner when too many tasks crashed (spec E4).
    unrunnable: bool = False


# --------------------------------------------------------------------------- #
# Judge protocol — real Judge and ScriptedJudge both satisfy this
# --------------------------------------------------------------------------- #


@runtime_checkable
class JudgeProtocol(Protocol):
    """What the SuiteRunner depends on: score one run, aggregate a suite."""

    def score(self, trace: m.Trace, task: Task, rubric_id: str) -> m.RunScore: ...

    def score_suite(
        self, scores: Sequence[m.RunScore], *, suite_id: str, rubric_id: str
    ) -> SuiteAggregate: ...


# --------------------------------------------------------------------------- #
# The real Judge — one LLM call per trace, JSON-structured scoring
# --------------------------------------------------------------------------- #


class Judge:
    """Scores runs via an LLM client. Provider-agnostic — depends on `LLMClient` only.

    On any failure (provider error or malformed JSON judgment) `score` raises
    `LLMError`; the caller (SuiteRunner) does bounded retry with backoff (E9)
    rather than fabricating a score.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str,
        rubric: Rubric = default_rubric,
    ) -> None:
        self._llm = llm
        self._model = model
        self._rubric = rubric

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    def score(self, trace: m.Trace, task: Task, rubric_id: str) -> m.RunScore:
        messages = self._build_messages(trace, task, rubric_id)
        completion = self._llm.complete(
            messages, model=self._model, temperature=0.0, response_format="json",
        )
        parsed = completion.parsed
        if not isinstance(parsed, dict):
            raise LLMError("judge returned no parsed JSON judgment")
        return self._to_run_score(parsed, trace, rubric_id)

    def score_suite(
        self, scores: Sequence[m.RunScore], *, suite_id: str, rubric_id: str
    ) -> SuiteAggregate:
        return aggregate_scores(scores, suite_id=suite_id, rubric_id=rubric_id)

    # --------------------------------------------------------------- internals
    def _build_messages(
        self, trace: m.Trace, task: Task, rubric_id: str
    ) -> list[Message]:
        return [
            Message(
                role=Role.SYSTEM,
                content=(
                    "You are a strict LLM-as-judge. Score a multi-agent run against "
                    f"rubric `{rubric_id}`. Each score is a float in [0, 1]. Return "
                    "ONLY JSON with: `aggregate` (overall 0-1), `confidence` (0-1), "
                    "`rubric_scores` (per dimension 0-1), and `step_scores` (one per "
                    "step: {node_id, sub_rubrics: {dim: 0-1}, note})."
                ),
            ),
            Message(
                role=Role.USER,
                content=(
                    f"TASK INPUT:\n{task.input}\n\n"
                    f"RUBRIC ({rubric_id}):\n"
                    + "\n".join(f"- {k}: {v}" for k, v in self._rubric.sub_rubrics.items())
                    + "\n\nRUN TRACE (steps in order):\n"
                    + _trace_summary(trace)
                    + "\n\nReturn the JSON judgment."
                ),
            ),
        ]

    def _to_run_score(
        self, parsed: dict, trace: m.Trace, rubric_id: str
    ) -> m.RunScore:
        try:
            aggregate = float(parsed["aggregate"])
            confidence = float(parsed.get("confidence", 1.0))
            rubric_scores = {k: float(v) for k, v in parsed.get("rubric_scores", {}).items()}
            step_scores = [
                m.StepScore(
                    node_id=str(s["node_id"]),
                    sub_rubrics={k: float(v) for k, v in s.get("sub_rubrics", {}).items()},
                    note=s.get("note"),
                )
                for s in parsed.get("step_scores", [])
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"malformed judge JSON: {exc}") from exc

        return m.RunScore(
            run_id=trace.run_id,
            spec_id=trace.spec_id,
            task_id=trace.task_id,
            rubric_scores=rubric_scores,
            aggregate=aggregate,
            confidence=confidence,
            judge_meta=m.JudgeMeta(model=self._model, rubric_id=rubric_id),
            step_scores=step_scores,
        )


# --------------------------------------------------------------------------- #
# Pure aggregation — shared by the real Judge and ScriptedJudge (1 impl)
# --------------------------------------------------------------------------- #


def aggregate_scores(
    scores: Sequence[m.RunScore], *, suite_id: str, rubric_id: str
) -> SuiteAggregate:
    """Confidence-weighted mean of a suite's RunScores (pure, no LLM).

    Equal weights when total confidence is zero (degenerate but well-defined).
    Per-task means are simple averages — the per-step breakdown is retained on
    each RunScore for the Architect; the aggregate only needs task-level means.
    """

    scores = list(scores)
    n = len(scores)
    if n == 0:
        return SuiteAggregate(suite_id=suite_id, rubric_id=rubric_id, mean=0.0, n_runs=0,
                              per_task={}, confidence=1.0)

    total_w = sum(s.confidence for s in scores)
    if total_w <= 0:
        mean = sum(s.aggregate for s in scores) / n
    else:
        mean = sum(s.aggregate * s.confidence for s in scores) / total_w

    per_task: dict[str, list[float]] = {}
    for s in scores:
        per_task.setdefault(s.task_id, []).append(s.aggregate)
    per_task_mean = {tid: sum(vs) / len(vs) for tid, vs in per_task.items()}

    return SuiteAggregate(
        suite_id=suite_id, rubric_id=rubric_id, mean=mean, n_runs=n,
        per_task=per_task_mean, confidence=total_w / n,
    )


def _trace_summary(trace: m.Trace) -> str:
    if not trace.steps:
        return "(no steps recorded — empty run)"
    lines: list[str] = []
    for i, step in enumerate(trace.steps):
        lines.append(
            f"[{i}] node={step.node_id}\n    prompt_in: {step.prompt_in}\n"
            f"    response_out: {step.response_out}"
        )
    return "\n".join(lines)


__all__ = [
    "Rubric", "default_rubric", "SuiteAggregate", "JudgeProtocol",
    "Judge", "aggregate_scores",
]
