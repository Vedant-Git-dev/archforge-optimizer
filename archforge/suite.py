"""SuiteRunner — the P-E-C "Evaluate" step (spec §3, §4, §7).

The ONLY component that invokes the host MAS. For one (Spec, Suite, R) it runs
each task R times, scores each non-crashed run, and rolls the survivors into a
single comparable `SuiteRun`:

  * crashed host run           -> a partial trace is sunk anyway (E4); not scored
  * grader outage (LLMError)    -> bounded retry w/ backoff, then folded to the
    crash path (never fabricates a score) (E9)
  * a task is "failed"           iff ALL its repeats crashed/unscored
  * `unrunnable` (E4 gate)       iff failed-task fraction > ε  (strictly greater;
    exactly ε is NOT unrunnable). Candidates flagged unrunnable are rejected by
    the Gatekeeper before any margin math.

`run_suite` creates ONE TracingMiddleware over the TraceStore and instantiates
the host ONCE per suite (so a host's cumulative `invoke_count` advances in
suite order — exercise crash scripting predictably). Runs are sequential in v1.
"""

from __future__ import annotations

import time
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from archforge import userconfig as ucfg
import archforge.models as m
from archforge.host.base import HostMAS, Task
from archforge.judge.base import JudgeProtocol, SuiteAggregate
from archforge.middleware import TracingMiddleware
from archforge.stores import TraceStore


# --------------------------------------------------------------------------- #
# A held-out evaluation suite
# --------------------------------------------------------------------------- #


class Suite(BaseModel):
    """A held-out evaluation suite: identifier + rubric + the tasks to run."""

    model_config = ConfigDict(extra="forbid")

    suite_id: str
    rubric_id: str
    tasks: list[Task] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# The outcome of running one Spec over a Suite (R repeats)
# --------------------------------------------------------------------------- #


class SuiteRun(BaseModel):
    """Everything the Gatekeeper needs to decide promote/discard/unrunnable."""

    model_config = ConfigDict(extra="allow")

    spec_id: str
    suite_id: str
    rubric_id: str
    repeats: int
    n_tasks: int
    n_crashed: int                # tasks whose every repeat crashed/unscored
    unrunnable: bool              # failed-task fraction > ε (E4)
    scores: list[m.RunScore]      # all RunScores from scored repeats (survivors)
    aggregate: SuiteAggregate
    mean: float                   # == aggregate.mean (convenience for the Gatekeeper)
    tokens: int = 0               # real cost incurred (sum of sunk step tokens)


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #


def _default_backoff(attempt: int) -> float:
    """Exponential backoff capped at BACKOFF_CAP_SECONDS (production default). Tests override."""
    return min(2.0 ** attempt, ucfg.get("BACKOFF_CAP_SECONDS"))


class SuiteRunner:
    """Runs a Spec over a Suite and returns a scored `SuiteRun`."""

    def __init__(
        self,
        host: HostMAS,
        judge: JudgeProtocol,
        trace_store: TraceStore,
        *,
        epsilon: float | None = None,
        judge_retries: int | None = None,
        backoff: Callable[[int], float] | None = None,
    ) -> None:
        self._host = host
        self._judge = judge
        self._trace_store = trace_store
        # Tunables resolve lazily from the active config (disk post-init, or the
        # in-memory sane template under the pytest gate); None arg ⇒ the default.
        self._epsilon = epsilon if epsilon is not None else ucfg.get("DEFAULT_UNRUNNABLE_FRAC")
        self._judge_retries = judge_retries if judge_retries is not None else ucfg.get("DEFAULT_JUDGE_RETRIES")
        self._backoff = backoff if backoff is not None else _default_backoff

    def run_suite(self, spec: m.Spec, suite: Suite, *, R: int = 1) -> SuiteRun:
        if R < 1:
            raise ValueError(f"R (repeats) must be >= 1, got {R}")

        spec_id = spec.compute_spec_id() if spec.spec_id is None else spec.spec_id
        middleware = TracingMiddleware(self._trace_store)
        runnable = self._host.instantiate(spec, middleware)

        scores: list[m.RunScore] = []
        failed_tasks: set[str] = set()
        tokens = 0

        for task in suite.tasks:                       # task-major order
            rubric_id = task.rubric_id or suite.rubric_id
            any_scored = False
            for _ in range(R):
                trace = runnable.run(task)
                tokens += _sum_tokens(trace)
                rs = self._score_with_retry(trace, task, rubric_id)
                if rs is not None:
                    scores.append(rs)
                    any_scored = True
            if not any_scored:
                failed_tasks.add(task.task_id)

        n_tasks = len(suite.tasks)
        n_crashed = len(failed_tasks)
        unrunnable = n_tasks > 0 and (n_crashed / n_tasks) > self._epsilon

        aggregate = self._judge.score_suite(
            scores, suite_id=suite.suite_id, rubric_id=suite.rubric_id,
        )
        aggregate = aggregate.model_copy(update={"unrunnable": unrunnable})

        return SuiteRun(
            spec_id=spec_id,
            suite_id=suite.suite_id,
            rubric_id=suite.rubric_id,
            repeats=R,
            n_tasks=n_tasks,
            n_crashed=n_crashed,
            unrunnable=unrunnable,
            scores=scores,
            aggregate=aggregate,
            mean=aggregate.mean,
            tokens=tokens,
        )

    # --------------------------------------------------------------- scoring
    def _score_with_retry(
        self, trace: m.Trace, task: Task, rubric_id: str,
    ) -> m.RunScore | None:
        """Score one run; bounded-retry LLMError, fold to None (crash path).

        A crashed host run (trace.ok=False) is not scored; its partial trace was
        sunk by the host already, so the run still counts toward `failed` if it
        never produces a score.
        """

        if not trace.ok:
            return None

        attempts = 1 + self._judge_retries
        for j in range(attempts):
            try:
                return self._judge.score(trace, task, rubric_id)
            except Exception:  # noqa: BLE001 — grader outage of any kind (E9)
                if j < self._judge_retries:
                    delay = self._backoff(j)
                    if isinstance(delay, (int, float)) and delay > 0:
                        time.sleep(delay)
                    continue
                return None        # outage exhausted -> folded to crash path
        return None                # unreachable; for the type checker


def _sum_tokens(trace: m.Trace) -> int:
    total = 0
    for step in trace.steps:
        if step.perf is not None and step.perf.tokens:
            total += int(step.perf.tokens)
    return total


__all__ = ["Suite", "SuiteRun", "SuiteRunner"]
