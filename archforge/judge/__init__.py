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

from archforge.judge.base import Judge, SuiteAggregate, default_rubric
from archforge.judge.scripted import ScriptedJudge

__all__ = ["Judge", "ScriptedJudge", "SuiteAggregate", "default_rubric"]
