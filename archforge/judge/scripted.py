"""ScriptedJudge — a deterministic fake `JudgeProtocol`.

Used by every scenario test that needs a Judge without a real model. It honors
the same contract as the real Judge and reuses the pure `aggregate_scores`, so
aggregation behaviour is identical real-vs-fake.

Scripting hooks (the things scenario tests need):
  * `set_aggregate(...)` / fixed per-(spec_id, task_id) tables -> deterministic scores
  * `with_noise_band(width)` -> jitter the aggregate within ±width of a base (E1)
  * `regress_after(spec_id, drop)` -> score the next-scoring spec lower by `drop`
    (used to prove promoted-then-regresses rollback, E6)
  * `raise_on_next(...)` -> force the next score() call to raise LLMError (E9)

It records every run it scores, including the per-step breakdown, so the
Architect and tests can assert on credit-assignment input. If a run
is unknown and no base/scripted behaviour covers it, it returns a neutral 0.5
(never silently wrong — tests configure exactly what they need).
"""

from __future__ import annotations

from typing import Sequence

from archforge.constants import SCRIPTED_NOISE_PATTERN
import archforge.models as m
from archforge.host.base import Task
from archforge.judge.base import JudgeProtocol, Rubric, SuiteAggregate, aggregate_scores, default_rubric
from archforge.llm.base import LLMError


class ScriptedJudge:
    """A programmable, deterministic Judge (no LLM)."""

    def __init__(self, *, rubric: Rubric = default_rubric) -> None:
        self._rubric = rubric
        # (spec_id, task_id) -> aggregate
        self._fixed: dict[tuple[str, str], float] = {}
        # base aggregate when not explicitly fixed
        self._base: float = 0.5
        # noise band applied to the aggregate (spec E1), deterministic per run order
        self._noise_width: float = 0.0
        # forced regression: after seeing `trigger_spec_id` once, subtract `drop`
        self._regress: tuple[str, float] | None = None
        self._regress_armed: bool = False
        # raise on the next score() call (spec E9 grader outage)
        self._raise_next: Exception | None = None
        # recorded verdicts for assertions
        self.scored: list[m.RunScore] = []

    # --------------------------------------------------------------- config
    def using(self, rubric: Rubric) -> "ScriptedJudge":
        self._rubric = rubric
        return self

    def set_aggregate(self, spec_id: str, task_id: str, aggregate: float) -> "ScriptedJudge":
        self._fixed[(spec_id, task_id)] = aggregate
        return self

    def set_base(self, aggregate: float) -> "ScriptedJudge":
        self._base = aggregate
        return self

    def with_noise_band(self, width: float) -> "ScriptedJudge":
        self._noise_width = width
        return self

    def regress_after(self, trigger_spec_id: str, drop: float) -> "ScriptedJudge":
        """Arm regression: after `trigger_spec_id` is scored once, drop subsequent by `drop`."""

        self._regress = (trigger_spec_id, drop)
        self._regress_armed = False
        return self

    def raise_on_next(self, exc: Exception | type[Exception] | None = None) -> "ScriptedJudge":
        """Force the next score() call to raise (default LLMError). Consumed once."""

        self._raise_next = (exc if isinstance(exc, Exception) else (exc or LLMError)("scripted outage"))
        return self

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    # --------------------------------------------------------------- contract
    def score(self, trace: m.Trace, task: Task, rubric_id: str) -> m.RunScore:
        if self._raise_next is not None:
            err = self._raise_next
            self._raise_next = None
            raise err

        aggregate = self._resolve_aggregate(trace, task)
        confidence = 1.0

        rs = m.RunScore(
            run_id=trace.run_id,
            spec_id=trace.spec_id,
            task_id=trace.task_id,
            rubric_scores={dim: aggregate for dim in self._rubric.sub_rubrics},
            aggregate=aggregate,
            confidence=confidence,
            judge_meta=m.JudgeMeta(model="scripted-judge", rubric_id=rubric_id),
            step_scores=[
                m.StepScore(
                    node_id=step.node_id,
                    sub_rubrics={dim: aggregate for dim in self._rubric.sub_rubrics},
                    note=f"scripted score for {step.node_id}",
                )
                for step in trace.steps
            ],
        )
        self.scored.append(rs)
        return rs

    def score_suite(
        self, scores: Sequence[m.RunScore], *, suite_id: str, rubric_id: str
    ) -> SuiteAggregate:
        return aggregate_scores(scores, suite_id=suite_id, rubric_id=rubric_id)

    # --------------------------------------------------------------- internals
    def _resolve_aggregate(self, trace: m.Trace, task: Task) -> float:
        # regression hook takes priority over fixed values (it models a real
        # post-promotion regression regardless of scripted per-task scores)
        if self._regress is not None:
            trigger, drop = self._regress
            if trace.spec_id == trigger and not self._regress_armed:
                self._regress_armed = True
            elif self._regress_armed and trace.spec_id != trigger:
                # subsequent non-trigger specs regress
                base = self._fixed.get((trace.spec_id, trace.task_id), self._base)
                return max(0.0, base - drop)

        base = self._fixed.get((trace.spec_id, trace.task_id), self._base)

        # deterministic noise: index of this run within the recorded history,
        # so identical runs -> identical jitter (reproducible E1 boundary tests)
        if self._noise_width > 0:
            idx = len(self.scored)
            # map idx onto a small repeating pattern within [-width, +width]
            pattern = list(SCRIPTED_NOISE_PATTERN)
            jitter = (pattern[idx % len(pattern)]) * self._noise_width
            base = max(0.0, min(1.0, base + jitter))
        return base


__all__ = ["ScriptedJudge"]
