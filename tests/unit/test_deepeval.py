"""DeepEval evaluation-backend tests (issue #2) — NO network, NO real DeepEval.

Stubs `deepeval` (+ `deepeval.test_case`, `deepeval.metrics`) in `sys.modules` so
`DeepEvalEvaluator` is exercised hermetic: the trace -> `LLMTestCase` projection,
metric-score -> `RunScore` conversion (rubric_scores keyed by metric slug,
aggregate = mean, empty step_scores), the shared `score_suite` aggregation, the
the sync `measure` call, and the import-lazy `LLMError` when DeepEval is absent.
A multi-evaluator conformance test asserts the native `Judge` and the DeepEval
backend BOTH satisfy `JudgeProtocol` and emit compatible score shapes (issue
criterion 6). The live smoke path is opt-in; this file runs in the default suite
and stays green without the `[deepeval]` extra installed.
"""

from __future__ import annotations

import sys
import types as pytypes
from contextlib import contextmanager

import pytest

import archforge.models as m
from archforge.host.base import Task
from archforge.judge import Judge, make_evaluator
from archforge.judge.base import JudgeProtocol
from archforge.llm import LLMError, ScriptedLLM


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _trace(spec_id: str = "s1", task_id: str = "t1", run_id: str = "r1") -> m.Trace:
    return m.Trace(
        run_id=run_id, spec_id=spec_id, task_id=task_id,
        steps=[m.Step(node_id="a", prompt_in="p", response_out="ra"),
               m.Step(node_id="b", prompt_in="q", response_out="rb")],
        final_output="the answer", ok=True,
    )


def _task(task_id: str = "t1") -> Task:
    return Task(task_id=task_id, input="do the thing", rubric_id="default-v1",
                suite_id="suite-1")


# --------------------------------------------------------------------------- #
# fake deepeval — what DeepEvalEvaluator unpacks
# --------------------------------------------------------------------------- #


class _FakeLLMTestCase:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.input = kwargs.get("input")
        self.actual_output = kwargs.get("actual_output")
        self.context = kwargs.get("context")
        self.retrieval_context = kwargs.get("retrieval_context")
        self.expected_output = kwargs.get("expected_output")


class _FakeMetric:
    """Sync DeepEval-metric stand-in: measure() sets .score/.reason.

    Matches the REAL deepeval API (verified against deepeval 4.1.10):
    `measure()` is synchronous; the async entry point is `a_measure()`. A fake
    with an async measure would mask the asyncio.run bug this guards against.
    """

    score_value = 0.5  # overridden per metric class below

    def __init__(self, *, threshold: float = 0.5, model: str | None = None) -> None:
        self.threshold = threshold
        self.model = model
        self.score: float | None = None
        self.reason: str | None = None
        self.measured_cases: list = []

    def measure(self, test_case) -> None:
        self.measured_cases.append(test_case)
        self.score = self.score_value
        self.reason = "stub reason"


class _AnswerRelevancy(_FakeMetric):
    score_value = 0.8


class _Faithfulness(_FakeMetric):
    score_value = 0.6


class _FakeLiteLLMModel:
    """Stand-in for deepeval.models.LiteLLMModel: records the model string it
    was wrapped around (the real one routes it through LiteLLM by prefix)."""

    def __init__(self, model: str | None = None, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs


@contextmanager
def _stub_deepeval():
    """Install a fake `deepeval` package into sys.modules; restore on exit."""
    deepeval = pytypes.ModuleType("deepeval")
    test_case = pytypes.ModuleType("deepeval.test_case")
    metrics = pytypes.ModuleType("deepeval.metrics")
    models = pytypes.ModuleType("deepeval.models")
    test_case.LLMTestCase = _FakeLLMTestCase  # type: ignore[attr-defined]
    metrics.AnswerRelevancyMetric = _AnswerRelevancy  # type: ignore[attr-defined]
    metrics.FaithfulnessMetric = _Faithfulness  # type: ignore[attr-defined]
    models.LiteLLMModel = _FakeLiteLLMModel  # type: ignore[attr-defined]
    deepeval.test_case = test_case  # type: ignore[attr-defined]
    deepeval.metrics = metrics  # type: ignore[attr-defined]
    deepeval.models = models  # type: ignore[attr-defined]

    saved = {k: sys.modules.get(k) for k in ("deepeval", "deepeval.test_case",
                                             "deepeval.metrics", "deepeval.models")}
    sys.modules["deepeval"] = deepeval
    sys.modules["deepeval.test_case"] = test_case
    sys.modules["deepeval.metrics"] = metrics
    sys.modules["deepeval.models"] = models
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# --------------------------------------------------------------------------- #
# score: trace -> LLMTestCase -> RunScore
# --------------------------------------------------------------------------- #


def test_score_maps_metrics_to_runscore() -> None:
    with _stub_deepeval():
        ev = make_evaluator("deepeval",
                            metrics=["answer_relevancy", "faithfulness"])
        rs = ev.score(_trace(), _task(), "default-v1")
    assert rs.rubric_scores == {"answer_relevancy": 0.8, "faithfulness": 0.6}
    assert rs.aggregate == pytest.approx(0.7)          # mean of metric scores
    assert rs.step_scores == []                         # top-level only
    assert rs.confidence == 1.0
    assert rs.judge_meta.rubric_id == "default-v1"
    assert rs.run_id == "r1" and rs.spec_id == "s1" and rs.task_id == "t1"


def test_score_default_metric_is_answer_relevancy() -> None:
    with _stub_deepeval():
        rs = make_evaluator("deepeval").score(_trace(), _task(), "default-v1")
    assert rs.rubric_scores == {"answer_relevancy": 0.8}
    assert rs.aggregate == pytest.approx(0.8)


def test_score_projects_trace_into_test_case() -> None:
    with _stub_deepeval():
        ev = make_evaluator("deepeval", metrics=["answer_relevancy"])
        rs = ev.score(_trace(), _task(), "default-v1")
        metric = ev._metrics[0][1]
        tc = metric.measured_cases[0]
        assert tc.input == "do the thing"
        assert tc.actual_output == "the answer"          # final_output preferred
        assert tc.context == ["p", "q"]                  # step prompt_in grounding
        assert tc.retrieval_context == ["p", "q"]        # Faithfulness requires it
        assert "expected_output" not in tc.kwargs        # task carries none
        assert rs.judge_meta.model == "deepeval"         # default model label


def test_score_custom_model_stamped_and_passed() -> None:
    with _stub_deepeval():
        ev = make_evaluator("deepeval", model="openai/gpt-4o",
                            metrics=["answer_relevancy"])
        rs = ev.score(_trace(), _task(), "default-v1")
        assert rs.judge_meta.model == "openai/gpt-4o"
        # a string model is wrapped in LiteLLMModel so it routes by prefix
        # (never DeepEval's OpenAI default), then handed to the metric
        wrapped = ev._metrics[0][1].model
        assert isinstance(wrapped, _FakeLiteLLMModel)
        assert wrapped.model == "openai/gpt-4o"


def test_string_model_wrapped_in_litellm_model() -> None:
    """A provider-prefixed string becomes a LiteLLMModel (not a bare string, which
    DeepEval would resolve to its OpenAI default regardless of prefix)."""
    with _stub_deepeval():
        ev = make_evaluator("deepeval", model="gemini/gemini-3.6-flash",
                            metrics=["answer_relevancy"])
        assert isinstance(ev._model, _FakeLiteLLMModel)
        assert ev._model.model == "gemini/gemini-3.6-flash"
        assert ev._judge_label == "gemini/gemini-3.6-flash"


def test_no_model_keeps_deepeval_default() -> None:
    with _stub_deepeval():
        ev = make_evaluator("deepeval", metrics=["answer_relevancy"])
        assert ev._model is None                       # DeepEval's own default
        assert ev._metrics[0][1].model is None


def test_score_unknown_metric_raises_LLMError() -> None:
    with _stub_deepeval():
        with pytest.raises(LLMError):
            make_evaluator("deepeval", metrics=["not_a_metric"])


def test_metric_objects_passed_directly() -> None:
    with _stub_deepeval():
        m_obj = _AnswerRelevancy(threshold=0.5, model="m")
        ev = make_evaluator("deepeval", metrics=[m_obj])
        rs = ev.score(_trace(), _task(), "default-v1")
        assert rs.aggregate == pytest.approx(0.8)
        assert m_obj.measured_cases                      # measure() was called


# --------------------------------------------------------------------------- #
# score_suite: reuse the shared pure aggregation
# --------------------------------------------------------------------------- #


def test_score_suite_aggregates_like_native() -> None:
    with _stub_deepeval():
        ev = make_evaluator("deepeval", metrics=["answer_relevancy"])
        scores = [ev.score(_trace(run_id="r1", task_id="t1"), _task("t1"), "default-v1"),
                  ev.score(_trace(run_id="r2", task_id="t1"), _task("t1"), "default-v1")]
        agg = ev.score_suite(scores, suite_id="suite-1", rubric_id="default-v1")
    assert agg.suite_id == "suite-1"
    assert agg.rubric_id == "default-v1"
    assert agg.n_runs == 2
    assert agg.mean == pytest.approx(0.8)
    assert agg.per_task == {"t1": pytest.approx(0.8)}


# --------------------------------------------------------------------------- #
# import-lazy: missing deepeval -> LLMError, never bare ImportError
# --------------------------------------------------------------------------- #


def test_missing_deepeval_raises_LLMError(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "deepeval", raising=False)
    monkeypatch.setitem(sys.modules, "deepeval", None)  # force ImportError on import
    with pytest.raises(LLMError):
        make_evaluator("deepeval")


def test_factory_unknown_evaluator_raises_LLMError() -> None:
    with pytest.raises(LLMError):
        make_evaluator("nope")


def test_factory_native_rejects_with_guidance() -> None:
    with pytest.raises(LLMError):
        make_evaluator("native")


# --------------------------------------------------------------------------- #
# multi-evaluator conformance (issue criterion 6): both satisfy JudgeProtocol
# --------------------------------------------------------------------------- #


def test_both_backends_satisfy_judge_protocol() -> None:
    with _stub_deepeval():
        deepeval_ev = make_evaluator("deepeval", metrics=["answer_relevancy"])
        native = Judge(model="claude", llm=ScriptedLLM().respond_json({
            "aggregate": 0.7, "confidence": 1.0,
            "rubric_scores": {"correctness": 0.7},
            "step_scores": [{"node_id": "a", "sub_rubrics": {"correctness": 0.7}}],
        }))
        assert isinstance(native, JudgeProtocol)
        assert isinstance(deepeval_ev, JudgeProtocol)

        # both emit structurally compatible RunScore / SuiteAggregate
        d_rs = deepeval_ev.score(_trace(), _task(), "default-v1")
        n_rs = native.score(_trace(), _task(), "default-v1")
        assert 0.0 <= d_rs.aggregate <= 1.0 and 0.0 <= n_rs.aggregate <= 1.0
        assert d_rs.judge_meta.rubric_id == n_rs.judge_meta.rubric_id == "default-v1"

        d_agg = deepeval_ev.score_suite([d_rs], suite_id="s", rubric_id="default-v1")
        n_agg = native.score_suite([n_rs], suite_id="s", rubric_id="default-v1")
        assert d_agg.mean == pytest.approx(d_rs.aggregate)
        assert n_agg.mean == pytest.approx(n_rs.aggregate)
