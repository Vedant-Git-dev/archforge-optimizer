"""The DeepEval evaluation backend — an external `JudgeProtocol` (issue #2).

`DeepEvalEvaluator` implements the SAME protocol the built-in `Judge` does
(`JudgeProtocol` in archforge/judge/base.py), so the SuiteRunner and optimizer
treat it exactly like the native backend: `score(trace, task, rubric_id) ->
RunScore` and `score_suite(...) -> SuiteAggregate`. Nothing downstream knows or
cares which backend produced a score — that is the pluggability the issue asks
for (criterion 5: the optimizer stays independent of the evaluator).

Unlike the native `Judge` (one LLM call returning a whole JSON rubric verdict),
DeepEval scores a run with one or more standalone LLM-judge METRICS
(AnswerRelevancy, Faithfulness, ...), each yielding a score in [0,1]. We project
an ArchForge trace into a DeepEval `LLMTestCase` and convert each metric's score
into a `RunScore`:

  * `rubric_scores` = {metric slug: metric.score}  (the named dimensions)
  * `aggregate`      = mean of the metric scores  (comparable to the native Judge's
                       [0,1] aggregate; invariant I5 still holds: rubric_id is
                       stamped on every score)
  * `step_scores`    = []  (top-level only — DeepEval is run-level, not per-step;
                       credit assignment degrades gracefully to `blame=None`)

DeepEval is imported LAZILY inside `__init__` (it is an optional extra and is
heavy — it pulls an LLM provider SDK + more). A missing install surfaces as a
clear `LLMError` at construction, never a bare ImportError at import time, so
`import archforge` stays deepeval-free.
"""

from __future__ import annotations

from typing import Any, Sequence

from archforge.llm.base import LLMError
from archforge.judge.base import SuiteAggregate, aggregate_scores
from archforge.host.base import Task
import archforge.models as m


# A metric name -> factory. Kept as a mapping (not `if` chains) so a configured
# name resolves to a metric object AND gives us the stable rubric_scores slug.
_METRIC_FACTORIES: dict[str, Any] = {}


def _metric_slug(name: str) -> str:
    """Canonical rubric_scores key for a DeepEval metric name (lower snake)."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


class DeepEvalEvaluator:
    """Scores runs via DeepEval metrics, satisfying `JudgeProtocol`.

    `metrics` are metric NAMES (`"answer_relevancy"`, `"faithfulness"`) or
    already-constructed DeepEval metric objects. Names resolve to metric objects
    lazily at construction (DeepEval stays unimported until then). `model` is a
    LiteLLM-style model string (e.g. `"gemini/gemini-3.6-flash"`; prefix with the
    provider) or an already-built DeepEval model object; a string is wrapped in
    DeepEval's `LiteLLMModel` so scoring routes through LiteLLM with the
    provider's env key, never DeepEval's OpenAI default. When None, DeepEval
    falls back to its own default model.
    """

    def __init__(
        self,
        *,
        metrics: Sequence[str | Any] | None = None,
        model: str | Any | None = None,
        threshold: float = 0.5,
    ) -> None:
        self._deepeval = _import_deepeval()  # LLMError if the extra is missing
        # A string model is wrapped in LiteLLMModel (below) so the provider
        # prefix routes correctly; `_judge_label` is only the provenance stamp
        # on JudgeMeta.
        self._model = self._wrap_model(model)
        if isinstance(model, str):
            self._judge_label = model
        elif model is not None and hasattr(model, "get_model_name"):
            self._judge_label = model.get_model_name()  # DeepEvalBaseLLM API
        else:
            self._judge_label = "deepeval"
        self._threshold = threshold
        self._metrics = self._build_metrics(metrics)  # list[(slug, metric)]

    def _wrap_model(self, model: str | Any | None) -> Any:
        """Wrap a string model id in DeepEval's `LiteLLMModel` (lazy import).

        A bare string handed to a DeepEval metric resolves to DeepEval's OpenAI
        default regardless of any provider prefix, so we wrap it explicitly:
        `LiteLLMModel` is a `DeepEvalBaseLLM`, which `initialize_model` passes
        through untouched, and LiteLLM routes by the model's provider prefix
        using the provider's env key (same as ArchForge's own LiteLLMClient).
        Non-string models (already-built DeepEval models) pass through.
        """
        if model is None or not isinstance(model, str):
            return model
        try:
            from deepeval.models import LiteLLMModel

            return LiteLLMModel(model=model)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 — deepeval/litellm construction errors
            raise LLMError(
                f"could not build the DeepEval judge model {model!r}: {exc}"
            ) from exc

    # --------------------------------------------------------------- protocol
    def score(self, trace: m.Trace, task: Task, rubric_id: str) -> m.RunScore:
        test_case = self._llm_test_case(trace, task)
        rubric_scores: dict[str, float] = {}
        for slug, metric in self._metrics:
            # `measure()` is SYNCHRONOUS in deepeval (the async entry point is
            # `a_measure`), matching the sync SuiteRunner path directly — do NOT
            # wrap it in asyncio.run (that raises TypeError on a non-coroutine).
            # A metric LLM outage raises deepeval's own error; convert to
            # LLMError so the SuiteRunner's bounded retry treats it exactly like
            # a native Judge failure (E9), never a fabricated 0.0.
            try:
                metric.measure(test_case)
            except Exception as exc:  # noqa: BLE001 — deepeval's error type
                raise LLMError(f"DeepEval metric {slug!r} failed: {exc}") from exc
            rubric_scores[slug] = float(metric.score or 0.0)

        aggregate = _mean(rubric_scores.values()) if rubric_scores else 0.0
        return m.RunScore(
            run_id=trace.run_id,
            spec_id=trace.spec_id,
            task_id=trace.task_id,
            rubric_scores=rubric_scores,
            aggregate=aggregate,
            confidence=1.0,
            judge_meta=m.JudgeMeta(model=self._judge_label, rubric_id=rubric_id),
            step_scores=[],  # top-level only (DeepEval is run-level, not per-step)
        )

    def score_suite(
        self, scores: Sequence[m.RunScore], *, suite_id: str, rubric_id: str
    ) -> SuiteAggregate:
        # Reuse the pure shared aggregation (same as the native Judge) so the two
        # backends produce structurally identical, comparable SuiteAggregates.
        return aggregate_scores(scores, suite_id=suite_id, rubric_id=rubric_id)

    # --------------------------------------------------------------- internals
    def _build_metrics(
        self, metrics: Sequence[str | Any] | None
    ) -> list[tuple[str, Any]]:
        built: list[tuple[str, Any]] = []
        for item in (metrics or ["answer_relevancy"]):
            if isinstance(item, str):
                slug = _metric_slug(item)
                factory = _METRIC_FACTORIES.get(slug)
                if factory is None:
                    raise LLMError(
                        f"unknown DeepEval metric {item!r}; expected one of "
                        f"{sorted(_METRIC_FACTORIES)}"
                    )
                # A metric validates its judge model at CONSTRUCTION (deepeval
                # raises its own DeepEvalError if no API key is configured). Catch
                # ANY construction failure and re-raise as LLMError so the CLI /
                # SuiteRunner see the same error type as the native Judge (E9).
                try:
                    metric = factory(model=self._model, threshold=self._threshold)
                except Exception as exc:  # noqa: BLE001 — deepeval's DeepEvalError
                    raise LLMError(
                        f"could not build DeepEval metric {item!r}: {exc}"
                    ) from exc
                built.append((slug, metric))
            else:
                built.append((_metric_slug(type(item).__name__), item))
        return built

    def _llm_test_case(self, trace: m.Trace, task: Task):
        """Project an ArchForge trace into a DeepEval `LLMTestCase`.

        `actual_output` is the run's final answer (the traced final_output, else
        the last step's response). Grounding comes from each step's prompt_in
        and is set on BOTH `context` and `retrieval_context`: Faithfulness
        REQUIRES `retrieval_context` (its `_required_params`) and errors without
        it, while RAG-style metrics read `context`. `expected_output` is carried
        through when the task JSON declares one (Task allows extra fields).
        """
        actual_output = trace.final_output or (
            trace.steps[-1].response_out if trace.steps else ""
        )
        expected_output = getattr(task, "expected_output", None)
        grounding = [s.prompt_in for s in trace.steps]
        kwargs: dict[str, Any] = {
            "input": task.input,
            "actual_output": actual_output,
            "context": grounding,
            "retrieval_context": grounding,  # Faithfulness requires this param
        }
        if expected_output:
            kwargs["expected_output"] = expected_output
        return self._deepeval.test_case.LLMTestCase(**kwargs)


def _import_deepeval():
    """Lazily import the `deepeval` package, or raise a clear LLMError."""
    try:
        import deepeval
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise LLMError(
            "DeepEval is not installed; install it with "
            "`pip install archforge-optimizer[deepeval]` to use --evaluator deepeval"
        ) from exc
    return deepeval


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# Register the metrics ArchForge exposes by name. Factories construct the metric
# objects lazily (deepeval classes are imported inside each factory, not here, so
# this module stays importable before the extra is installed).
def _answer_relevancy_factory(*, model, threshold):
    from deepeval.metrics import AnswerRelevancyMetric

    return AnswerRelevancyMetric(threshold=threshold, model=model)


def _faithfulness_factory(*, model, threshold):
    from deepeval.metrics import FaithfulnessMetric

    return FaithfulnessMetric(threshold=threshold, model=model)


_METRIC_FACTORIES.update(
    {
        "answer_relevancy": _answer_relevancy_factory,
        "faithfulness": _faithfulness_factory,
    }
)


__all__ = ["DeepEvalEvaluator"]