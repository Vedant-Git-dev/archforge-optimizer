"""Scenario: the OTel trace-projection makes the Judge see REAL LLM I/O (spec #1).

Drives AEDE's REAL LangGraph offline — every external call (groq, google-genai,
chroma) faked — with ``trace_total_budget`` set, and asserts the bounded OTel
projection replaces the lossy ``summarize()`` one-liners the Judge used to score:

  * ``reason`` (Gemini) ``response_out`` carries the REAL canned completion
    (the diagnosis's wrong-domain ``"Comprehensive Report on Apple's Quarterly
    Earnings"`` that ``summarize()`` hid as ``answer_len=1189``) — the Goal,
    WITHOUT the deferred Gap-A bugfix (the rich per-step ``response_out`` now
    carries the answer the Judge can compare to the task).
  * ``small_reasoner`` (Groq) ``response_out`` carries its REAL canned text.
  * Retriever nodes KEEP ``summarize()`` (v1 cut: ``project`` enriches LLM
    nodes only — the gate is honest about what it changed).
  * the post-loop shed trims to budget (protecting the answer node).
  * the budget=``None`` variant reproduces today's ``answer_len=…`` one-liners
    BYTE-IDENTICALLY (the toggle, not a fork — the existing suite stays green).

This is the TRUE end-to-end of the cooperative seam: AEDE's ``build_graph``
routes node fns through ``wrapped(name, fn)`` (real — opens an
``archforge.node`` span); the fake SDK calls emit faithful ``gen_ai`` child
spans nesting under that parent (mirroring what the real auto-instrumentors
do, but offline); ``project`` attributes each child to its node by parent-link
and reads a bounded slice into the ``Step``. No network, no keys.
"""
from __future__ import annotations

import sys
import json as _json
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Path bootstrap — AEDE is a sibling subdir of the ArchForge repo, NOT installed.
# --------------------------------------------------------------------------- #
_REPO = Path(__file__).resolve().parents[2]
_AEDE_BACK = _REPO / "AEDE" / "backend"
_AEDE_SRC = _AEDE_BACK / "src"
if not _AEDE_SRC.exists():
    pytest.skip("AEDE backend not present in this checkout",
                allow_module_level=True)
for _p in (str(_AEDE_BACK), str(_AEDE_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")
pytest.importorskip("groq")
pytest.importorskip("google.genai")
# OTel SDK is the rich-path engine; skip the whole module if it isn't installed.
import importlib.util as _ilu  # noqa: E402
if _ilu.find_spec("opentelemetry.sdk") is None:  # pragma: no cover
    pytest.skip("opentelemetry-sdk not installed in this env",
                allow_module_level=True)

import archforge.models as m  # noqa: E402
import archforge.otel as _otel  # noqa: E402
from archforge.host.base import Task  # noqa: E402
from archforge.middleware import TracingMiddleware  # noqa: E402
from archforge.stores import TraceStore  # noqa: E402

from archforge_glue.aede_app import AedeApp  # noqa: E402
from archforge_glue.aede_host import AEDEAdapter  # noqa: E402

import aede.utils.node_config as _nc  # noqa: E402

# NOTE (shadow trap, from test_smoke_offline): ``aede.nodes.__init__`` does
# ``from aede.nodes.X import X`` so the FUNCTION shadows the submodule. We patch
# via ``importlib.import_module`` (reads sys.modules — the real module survives).


# --------------------------------------------------------------------------- #
# Fakes — faithful gen_ai spans, no network, no keys
# --------------------------------------------------------------------------- #

# The SAME scripted docs + analyzer JSON as the offline smoke test, so routing
# is deterministic (extract→analyze→compile→...). Two analyzer profiles drive
# the two terminal LLM nodes:
#   * _ANALYZE_DEEP:   coverage 0.5 < 0.8 ⇒ retrieve_more loop ⇒ max_retrieval
#                      ⇒ reason (Gemini).  required_reasoning="deep".
#   * _ANALYZE_DONE:   coverage 0.9 ≥ 0.8 ⇒ no loop ⇒ direct_answer ⇒ small_reasoner.
#                      required_reasoning="none".
_FAKE_DOCS = [f"doc-{i} evidence about topic {i}" for i in range(16)]

# The wrong-domain reason completion — the diagnosis's over-scoring root cause.
_REASON_COMPLETION = (
    "Comprehensive Report on Apple's Quarterly Earnings — The company posted "
    "record revenue driven by iPhone sales and services growth, with strong "
    "guidance for the next quarter."
)
_SMALL_COMPLETION = "fake small-reasoner answer text (groq)"
_EXTRACT_JSON = _json.dumps({
    "facts": [
        {"claim": f"fact from chunk {i}", "quote": _FAKE_DOCS[i][:40], "chunk_id": i}
        for i in range(4)
    ]
})
_COMPRESS_JSON = _json.dumps({"compressed_evidence": ["merged evidence 1",
                                                      "merged evidence 2"]})


def _emit_child_genai(node_id, prompt, completion, system_prompt="",
                      in_tok=50, out_tok=80):
    """Emit a faithful ``gen_ai`` child span under the CURRENT ``archforge.node``
    span (opened by AEDE's real ``wrapped``), mirroring what the real groq /
    google-genai auto-instrumentors emit when content capture is ON — but with
    NO network (the canned ``completion`` stands in for the SDK response). Uses
    the SAME tracer archforge's buffer is attached to, so the child nests by
    parent-link → ``project`` attributes it to ``node_id``.

    Emits the NEWER STRUCTURED message shape the INSTALLED instrumentors emit
    (``opentelemetry-util-genai`` 1.0b0): text lives in ``parts[].content``, not
    a top-level ``content`` key. (Earlier the fake used the older flat
    ``[{"role","content"}]`` form, diverging from production — which let a real
    ``parts[]``-shape regression hide until the first live ``evolve-loop``.)"""
    tracer = _otel._ensure_tracer()
    if not tracer:
        return
    msgs_in = []
    if system_prompt:
        msgs_in.append({"role": "system",
                         "parts": [{"type": "text", "content": system_prompt}]})
    msgs_in.append({"role": "user",
                    "parts": [{"type": "text", "content": str(prompt)}]})
    with tracer.start_as_current_span(f"gen_ai.chat {node_id}") as child:
        child.set_attribute("gen_ai.system", "archforge-test")
        child.set_attribute("gen_ai.request.model", "test-model")
        child.set_attribute("gen_ai.input.messages", _json.dumps(msgs_in))
        child.set_attribute("gen_ai.output.messages",
                           _json.dumps([{"role": "assistant",
                                        "parts": [{"type": "text",
                                                   "content": completion}],
                                        "finish_reason": "stop"}]))
        child.set_attribute("gen_ai.usage.input_tokens", in_tok)
        child.set_attribute("gen_ai.usage.output_tokens", out_tok)


def _fake_groq(analyzer_json):
    """Span-emitting fake for ``generate_with_groq`` (covers extract/analyze/
    compress/small_reasoner). Keys the canned JSON off ``node_id=`` kwarg and
    emits a faithful gen_ai child span per call. Returns the dict the node code
    expects (``text`` + ``usage``)."""
    def _impl(*args, **kwargs):
        nid = kwargs.get("node_id")
        prompt = kwargs.get("prompt", "")
        sysp = kwargs.get("system_prompt", "")
        if nid == "extract":
            text, it, ot = _EXTRACT_JSON, 120, 40
        elif nid == "analyze":
            text, it, ot = analyzer_json, 100, 60
        elif nid == "compress":
            text, it, ot = _COMPRESS_JSON, 140, 30
        elif nid == "small_reasoner":
            text, it, ot = _SMALL_COMPLETION, 60, 25
        else:
            text, it, ot = "{}", 10, 5
        _emit_child_genai(nid, prompt, text, sysp, it, ot)
        return {"text": text, "usage": {"prompt_tokens": it,
                                        "completion_tokens": ot,
                                        "total_tokens": it + ot}}
    return _impl


class _FakeUsageMeta:
    def __init__(self, in_tok, out_tok):
        self.prompt_token_count = in_tok
        self.candidates_token_count = out_tok


class _FakeGenaiResponse:
    def __init__(self, text, in_tok, out_tok):
        self.text = text
        self.usage_metadata = _FakeUsageMeta(in_tok, out_tok)


class _FakeGenaiModels:
    """``client.models.generate_content(...)`` — emits a faithful gen_ai child
    span (the real instrumentor's analog) and returns canned ``.text`` +
    ``.usage_metadata`` (what the reasoner reads)."""
    def __init__(self, completion, in_tok, out_tok):
        self._c, self._i, self._o = completion, in_tok, out_tok

    def generate_content(self, *, model, contents, **kwargs):
        _emit_child_genai("reason", contents, self._c, "", self._i, self._o)
        return _FakeGenaiResponse(self._c, self._i, self._o)


class _FakeGenaiClient:
    """Replaces ``google.genai.Client``: ``Client(api_key=...)`` → this. Its
    ``.models.generate_content`` emits the span (so reason's completion is
    projected), bypassing the real (networked) SDK + the registered genai
    instrumentor (which wrapped the ORIGINAL Client — our monkeypatch replaces
    the class attr, so only this fake runs)."""
    def __init__(self, *, api_key=None, **kwargs):
        self.models = _FakeGenaiModels(_REASON_COMPLETION, 40, 110)


class _FakeCollection:
    """Chroma collection stand-in. Retriever nodes stay on ``summarize()``
    (v1 cut) — we emit NO span for them (the plan's retriever_dump hook is the
    future), so this returns docs only."""
    def query(self, query_texts, n_results, **kwargs):
        docs = _FAKE_DOCS[:n_results]
        return {
            "documents": [docs],
            "metadatas": [[{"source": f"doc_{i}"} for i in range(len(docs))]],
            "distances": [[0.1 * i for i in range(len(docs))]],
        }


def _patch_externals(monkeypatch, analyzer_json):
    """Replace groq (span-emitting), genai (span-emitting), chroma (no span),
    and set a fake GEMINI key so the reasoner takes the genai branch (emits a
    span) instead of the no-span ``_simple_answer`` cold path."""
    import importlib
    _vk = lambda *a, **k: _FakeCollection()             # noqa: E731
    for modname in ("aede.nodes.retrieval", "aede.nodes.retriever_more"):
        monkeypatch.setattr(importlib.import_module(modname),
                            "get_or_create_vectorstore", _vk)
    _groq = _fake_groq(analyzer_json)
    for modname in ("aede.nodes.extractor", "aede.nodes.analyzer",
                    "aede.nodes.compressor", "aede.nodes.small_reasoner"):
        monkeypatch.setattr(importlib.import_module(modname),
                            "generate_with_groq", _groq)
    # reasoner reads os.getenv("GEMINI_API_KEY") — set a fake key so it takes the
    # genai branch (where our span-emitting fake lives), not _simple_answer.
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-span-emit")
    import google.genai as _ggenai
    monkeypatch.setattr(_ggenai, "Client", _FakeGenaiClient)


@pytest.fixture(autouse=True)
def _fresh_otel():
    """Each test starts from a clean OTel singleton so the in-memory buffer
    holds only THIS test's spans (no cross-run leak). The adapter's
    ``__init__`` re-runs ``_ensure_tracer`` (idempotent) on construct."""
    _otel._reset()
    yield
    _otel._reset()


def _mw(ts):
    return TracingMiddleware(ts)


def _run(adapter, spec, ts, task_input="Why did revenue grow in Q3?"):
    """Instantiate + run one task on the real AEDE graph; return the Trace."""
    return adapter.instantiate(spec, _mw(ts)).run(
        Task(task_id="t1", input=task_input))


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #


def test_reason_rich_carries_real_genai_completion(tmp_path, monkeypatch):
    """Budget ON + ``deep`` routing ⇒ the ``reason`` (Gemini) ``Step`` carries
    the REAL wrong-domain completion, NOT a ``answer_len=…`` one-liner — the
    diagnosis's root cause is gone. Retriever nodes KEEP ``summarize()`` (v1 cut:
    ``project`` enriches LLM only). ``_trace_summary`` (what the Judge ingests)
    carries the real completion next to the task."""
    _patch_externals(monkeypatch, _json.dumps({
        "answered_parts": ["topic 0"], "missing_parts": [],
        "coverage": 0.5, "redundancy": 0.3, "confidence": 0.6,
        "direct_answer_possible": False, "required_reasoning": "deep",
    }))
    monkeypatch.setattr(AedeApp, "trace_total_budget", 8000)
    _nc.reset()

    adapter = AEDEAdapter()
    spec = adapter.app_spec()
    ts = TraceStore(tmp_path / ".archforge")
    trace = _run(adapter, spec, ts)

    assert trace.ok, f"run failed: {trace.error}"
    by = {s.node_id: s for s in trace.steps}
    assert "reason" in by, f"deep routing should reach reason: {list(by)}"

    reason = by["reason"]
    # THE GOAL: the real wrong-domain completion reaches the Judge, not a length
    # stub. Today's lossy path emitted ``answer_len=<n>`` here.
    assert "Comprehensive Report on Apple" in reason.response_out, reason.response_out
    assert not reason.response_out.startswith("answer_len=")
    # the prompt head (PROMPT_SLICE_TOK) is carried too
    assert reason.prompt_in
    # real usage from the gen_ai span (input+output), not an estimate-only path
    assert reason.perf.tokens >= 1

    # v1 CUT: retriever nodes are NOT enriched (project returns None for non-LLM)
    # → they keep the summarize() one-liner. The gate is honest about its scope.
    retrieve = by["retrieve"]
    assert retrieve.response_out.startswith("top_k="), retrieve.response_out
    assert "Comprehensive Report" not in retrieve.response_out

    # total projected text ≤ budget (the shed held it)
    from archforge.host.adapters.helpers import estimate_tokens
    total = sum(estimate_tokens(s.response_out) for s in trace.steps)
    assert total <= 8000, f"total {total} > budget 8000"

    # the Judge's _trace_summary input now carries the real completion text —
    # the load-bearing behavioral change (Judge can compare reason's answer to
    # the task instead of scoring "answer_len=1189").
    from archforge.judge.base import _trace_summary
    summary = _trace_summary(trace)
    assert "Comprehensive Report on Apple" in summary
    assert "[0]" in summary   # the per-step header the Judge reads


def test_small_reasoner_rich_carries_real_groq_completion(tmp_path, monkeypatch):
    """Budget ON + ``none``/high-coverage routing ⇒ the ``small_reasoner``
    (Groq) ``Step`` carries its REAL canned text (the groq span-emitting fake),
    not a ``answer_len=…`` one-liner — the other LLM-reasoner path."""
    _patch_externals(monkeypatch, _json.dumps({
        "answered_parts": ["topic 0"], "missing_parts": [],
        "coverage": 0.9, "redundancy": 0.2, "confidence": 0.9,
        "direct_answer_possible": True, "required_reasoning": "none",
    }))
    monkeypatch.setattr(AedeApp, "trace_total_budget", 8000)
    _nc.reset()

    adapter = AEDEAdapter()
    spec = adapter.app_spec()
    ts = TraceStore(tmp_path / ".archforge")
    trace = _run(adapter, spec, ts)

    assert trace.ok, f"run failed: {trace.error}"
    by = {s.node_id: s for s in trace.steps}
    assert "small_reasoner" in by, (
        f"none/high-coverage routing should reach small_reasoner: {list(by)}")

    sm = by["small_reasoner"]
    assert _SMALL_COMPLETION in sm.response_out, sm.response_out
    assert not sm.response_out.startswith("answer_len=")
    assert sm.perf.tokens >= 1


def test_budget_none_is_parity_summarize_path(tmp_path, monkeypatch):
    """Budget ``None`` (the default) ⇒ today's lossy ``summarize()`` path
    BYTE-IDENTICALLY: ``reason.response_out`` is ``answer_len=…`` (not the real
    completion), ``prompt_in`` empty, retriever one-liners unchanged. The gate
    is a TOGGLE, not a fork — the existing 359/16 suite stays green."""
    _patch_externals(monkeypatch, _json.dumps({
        "answered_parts": ["topic 0"], "missing_parts": [],
        "coverage": 0.5, "redundancy": 0.3, "confidence": 0.6,
        "direct_answer_possible": False, "required_reasoning": "deep",
    }))
    # budget stays None (the default) — do NOT set trace_total_budget.
    _nc.reset()

    adapter = AEDEAdapter()
    spec = adapter.app_spec()
    ts = TraceStore(tmp_path / ".archforge")
    trace = _run(adapter, spec, ts)

    assert trace.ok, f"run failed: {trace.error}"
    by = {s.node_id: s for s in trace.steps}
    assert "reason" in by, list(by)

    reason = by["reason"]
    # PARITY: today's lossy summarize() label — NOT the real completion.
    assert reason.response_out.startswith("answer_len="), reason.response_out
    assert "Comprehensive Report" not in reason.response_out
    # prompt_in empty on the lossy path (LangGraph nodes read the query from state)
    assert reason.prompt_in == ""
    # retriever one-liner unchanged
    assert by["retrieve"].response_out.startswith("top_k=")


def test_shed_protects_reason_under_tiny_budget(tmp_path, monkeypatch):
    """A TINY budget on the deep routing: the post-loop shed trims the largest
    non-answer ``response_out`` steps to the ``[trimmed: …]`` marker while
    PROTECTING the ``reason`` (answer-bearing) step — the answer always reaches
    the Judge, and the total ≤ budget once there's room to shed.

    Iterates the FULL ``trace.steps`` list (with duplicates from the
    ``retrieve_more`` loop) — NOT a ``{node_id: Step}`` dict, which would
    last-wins dedupe and hide an EARLIER trimmed duplicate (the shed mutates
    individual Step objects in the list, not a per-node aggregate)."""
    _patch_externals(monkeypatch, _json.dumps({
        "answered_parts": ["topic 0"], "missing_parts": [],
        "coverage": 0.5, "redundancy": 0.3, "confidence": 0.6,
        "direct_answer_possible": False, "required_reasoning": "deep",
    }))
    # budget BELOW the run's natural total (~440 tok): forces the shed to trim
    # the largest non-answer LLM step(s) — the repeating ``extract`` evidence
    # chunks (~90 tok each) — and keep the protected ``reason`` step.
    monkeypatch.setattr(AedeApp, "trace_total_budget", 400)
    _nc.reset()

    adapter = AEDEAdapter()
    spec = adapter.app_spec()
    ts = TraceStore(tmp_path / ".archforge")
    trace = _run(adapter, spec, ts)

    assert trace.ok, f"run failed: {trace.error}"
    assert "reason" in {s.node_id for s in trace.steps}, list({s.node_id for s in trace.steps})

    # at least one non-answer step was trimmed to its marker (read the FULL list —
    # a node_id dict would hide an earlier trimmed duplicate behind a later copy).
    trimmed = [s for s in trace.steps if s.response_out.startswith("[trimmed:")]
    assert trimmed, (
        f"budget 400 (below natural total) should have trimmed non-answer steps: "
        f"{[(s.node_id, s.response_out[:24]) for s in trace.steps]}"
    )
    # the answer step is PROTECTED — never a trim victim, even under a tiny budget
    assert not any(s.node_id == "reason" and s.response_out.startswith("[trimmed:")
                  for s in trace.steps)
    reason = next(s for s in trace.steps if s.node_id == "reason")
    assert "Comprehensive Report on Apple" in reason.response_out

    # total ≤ budget once the shed had room to shed (it trims just enough)
    from archforge.host.adapters.helpers import estimate_tokens
    total = sum(estimate_tokens(s.response_out) for s in trace.steps)
    assert total <= 400, f"total {total} exceeded budget 400 after shed"
