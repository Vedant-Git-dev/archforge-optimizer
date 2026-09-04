"""The Judge — LLM-as-judge scoring of runs (spec §3, §4).

`Judge.score(trace, task, rubric_id) -> RunScore` turns a run's trace into a
scored verdict: an aggregate + named rubric dimensions + confidence, plus a
*per-step breakdown* (StepScore[]) that names which agent lost which points. The
per-step breakdown is the raw material the Architect uses to credit-assign a
fault to a node/route.

`score_suite(...)` aggregates over R repeats: mean (down-weighted by
confidence) and a stable aggregate used by the Gatekeeper. `rubric_id` is
stamped on every score so cross-rubric comparisons never masquerade as
improvement (invariant I5, spec E2).
"""

from __future__ import annotations

from typing import Sequence

from archforge.config import EVALUATORS
from archforge.judge.base import (
    Judge, JudgeProtocol, SuiteAggregate, default_rubric,
)
from archforge.judge.scripted import ScriptedJudge
from archforge.llm.base import LLMError

# `EVALUATORS` is re-exported (in __all__) straight from archforge.config — the
# single source of truth the CLI also imports for its `--evaluator` choices.


def make_evaluator(
    evaluator: str, *, model: str | None = None, metrics: Sequence[str] | None = None
) -> JudgeProtocol:
    """Build an external evaluation backend by name.

    Resolves `DeepEvalEvaluator` lazily (so importing DeepEval is deferred to here,
    and the optional extra is only required at construction). Raises `LLMError` for
    an unknown evaluator so the CLI surfaces one clear message across the backend
    seam. The `native` evaluator is NOT built here: it is the built-in `Judge`,
    constructed inline by the CLI from the provider's `LLMClient`.
    """

    if evaluator == "native":
        raise LLMError(
            "'native' is the built-in Judge; construct it with Judge(llm, model=...)"
        )
    if evaluator == "deepeval":
        from archforge.judge.deepeval import DeepEvalEvaluator

        return DeepEvalEvaluator(model=model, metrics=metrics)  # type: ignore[return-value]
    raise LLMError(f"unknown evaluator {evaluator!r}; expected one of {EVALUATORS}")


__all__ = [
    "Judge", "JudgeProtocol", "ScriptedJudge", "SuiteAggregate", "default_rubric",
    "make_evaluator", "EVALUATORS",
]
