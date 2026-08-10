"""OpenTelemetry GenAI tracing seam — bounded real I/O to the Judge.

Spec #1 of the three library-delegation specs (design:
``docs/superpowers/specs/2026-08-08-tracing-otel-design.md``). It enriches the
host-STREAMING path's lossy per-step ``Step``: ``LangGraphRunnable._record``
builds ``Step(prompt_in="", response_out=app.summarize(...))`` — a contentless
label like ``"answer_len=1189"`` that hid AEDE's wrong-domain ``reason``
completion from the Judge. This module auto-instruments the MAS's SDK calls as
OTel GenAI spans and projects a BOUNDED slice of the real prompt/completion into
the ``Step`` the Judge reads.

Cooperative attribution: a forge-owned ``wrapped(name, fn)`` opens an
``archforge.node`` span around each node function; the auto-instrumented GenAI
spans then nest as CHILDREN → correlation by parent-link (order-independent,
multi-call-safe) — NOT a temporal "the next chunk's calls belong to this node"
guess. Node functions + SDK call sites stay surgery-free.

Gated by ``DEFAULT_TRACE_TOTAL_BUDGET_TOK``: ``None`` (default) reproduces
today's lossy ``summarize()`` path BYTE-IDENTICALLY (parity — the suite stays
green); a number turns on rich ``Step`` s. A toggle, not a fork.

IMPORT-LAZY: this module's top level imports NO OpenTelemetry. Every OTel import
is inside :func:`_ensure_tracer` (called lazily from
``LangGraphHostAdapter.__init__`` and :func:`wrapped`). So ``import archforge``
and importing this module stay OTel-free; a missing/no-OTel install degrades
gracefully (:func:`project` → ``None`` → the caller keeps ``summarize()``).

v1 scope: :func:`project` enriches LLM nodes ONLY. Retriever/rule/symbolic
return ``None`` → the adapter keeps ``summarize()`` (state-shape-agnostic — no
guessing a retriever's docs state key; respects "describe, don't introspect").
The Goal — the wrong-domain ``reason`` completion reaching the Judge — is met:
``reason`` / ``small_reasoner`` are LLM. Retriever-span enrichment is a future
``retriever_dump`` hook (``partial`` / ``merged`` are accepted on :func:`project`
for that forward-compat and unused in v1).
"""
from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import archforge.models as m
from archforge.host.adapters.helpers import estimate_tokens

_log = logging.getLogger("archforge.otel")

# --------------------------------------------------------------------------- #
# Per-kind projection ceilings — module constants, NOT user config. These are
# internal caps deciding how much of each captured span reaches the Judge.
# --------------------------------------------------------------------------- #
PROMPT_SLICE_TOK = 256        # LLM prompt head    → Step.prompt_in
COMPLETION_SLICE_TOK = 1024   # LLM completion     → Step.response_out
# RETRIEVER_SLICE_TOK = 2048  # reserved for the future retriever_dump hook

# --------------------------------------------------------------------------- #
# Process-singleton state. Built once by _ensure_tracer; cached so repeated
# calls (one per node execution) are a dict-lookup. ``_TRACER`` is either the
# ``opentelemetry.trace.Tracer`` or the ``False`` sentinel (OTel absent).
# --------------------------------------------------------------------------- #
_TRACER: Any = None           # None = uninit; Tracer / False after _ensure_tracer
_BUFFER: Any = None           # the _NodeSpanBuffer SpanProcessor (or None)
_REGISTERED: set[str] = set()  # instrumentor module names already .instrument()-ed


@dataclass
class Proj:
    """The bounded projection of one node's real LLM I/O into a ``Step``.

    ``None`` from :func:`project` means "no rich data — caller keeps
    ``summarize()``"; a ``Proj`` means "build the Step from this". ``tokens`` is
    the real ``gen_ai.usage`` (input+output) when available, else an estimate
    over ``response_out``. ``StepPerf.tokens`` stays flat (no input/output split)
    — a locked invariant; the typed split is a non-blocking follow-up.
    """

    prompt_in: str
    response_out: str
    tokens: int


# --------------------------------------------------------------------------- #
# Repo-declared auto-instrumentors (user-approved wiring: NOT Traceloop.init).
# --------------------------------------------------------------------------- #
# Each entry is the OpenTelemetry GenAI/retriever instrumentor for one SDK, living
# at ``opentelemetry.instrumentation.<name>``. We import it LAZILY; importing it
# raises ``ModuleNotFoundError`` when EITHER the instrumentor OR its underlying
# SDK is missing (verified: ``groq``/``google_genai``/``google_generativeai``/
# ``chromadb``/``weaviate`` import when installed; ``openai``/``anthropic``/
# ``pinecone``/``qdrant``/``bedrock`` raise when their SDK is absent). So the
# common case for a new MAS is ZERO archforge edits — its SDKs are already
# listed; a rare new SDK is one line here, once. Includes BOTH the new
# ``google.genai`` SDK (``google_genai``) and the old ``google.generativeai``
# (``google_generativeai``) so either works with no edit.
_INSTRUMENTORS: tuple[str, ...] = (
    # GenAI / LLM clients
    "openai", "groq", "anthropic", "google_genai", "google_generativeai",
    "vertexai", "bedrock", "mistralai", "cohere", "replicate", "together",
    "ollama", "watsonx",
    # retrievers / vector stores
    "chromadb", "pinecone", "qdrant", "weaviate", "milvus", "lancedb",
    "marqo", "voyageai",
    # meta-client (covers many providers via litellm)
    "litellm",
)


def _ensure_tracer() -> Any:
    """Idempotently build ArchForge's tracer + in-memory buffer and register the
    repo-declared auto-instrumentors. Returns the ``Tracer`` (or ``False`` when
    OTel is not installed → callers degrade gracefully).

    Safe to call repeatedly; builds once per process (the ``_TRACER is not None``
    guard). Sets ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT``
    (before instrumentor registration — some read it at ``.instrument()`` time) so
    the GenAI spans carry ``gen_ai.input.messages`` / ``gen_ai.output.messages``
    (the real prompt/completion); without it the spans hold only metadata and
    :func:`project` degrades to ``summarize()``. No OTLP exporter is configured ⇒
    spans never leave the process (the secrets-in-spans risk is bounded).

    Global-provider handling: OTel's ``set_tracer_provider`` only honors the
    FIRST call per process and silently no-ops (logs, does not raise) on any later
    one. So rather than pre-attaching the buffer to a local provider and hoping
    our set wins, we set best-effort and then attach ``_BUFFER`` to whichever
    provider ``get_tracer_provider()`` returns AFTER the attempt — ours if the set
    took, an existing one if a host/library pre-set it. Done exactly once per init
    (the idempotent guard above), so never doubles spans.
    """
    global _TRACER, _BUFFER
    if _TRACER is not None:           # real Tracer OR the False sentinel
        return _TRACER

    try:
        import opentelemetry.trace as trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
    except Exception as exc:  # noqa: BLE001 — OTel SDK not installed
        _log.debug("OTel SDK unavailable; tracing degrades to summarize(): %s", exc)
        _TRACER = False
        return _TRACER

    # Content capture must be ON before the instrumentors register. Don't clobber
    # an explicit user choice.
    os.environ.setdefault(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )

    provider = TracerProvider(resource=Resource.create({"service.name": "archforge"}))
    try:
        trace.set_tracer_provider(provider)   # honored only on the first call / process
    except Exception as exc:  # noqa: BLE001 — defensive; observed to no-op, not raise
        _log.debug("set_tracer_provider raised (absorbed): %s", exc)
    # Build the buffer and attach it to the ACTIVE provider — ours if the set
    # took, an existing one if a host/library pre-set it (set_tracer_provider
    # silently no-ops on a second call). The idempotent guard above means this
    # runs once per init → never doubles spans.
    _BUFFER = _NodeSpanBuffer()
    try:
        trace.get_tracer_provider().add_span_processor(_BUFFER)
    except Exception as exc:  # noqa: BLE001 — rare; project() degrades to summarize()
        _log.debug("could not attach span buffer to active provider: %s", exc)

    _register_instrumentors()

    _TRACER = trace.get_tracer("archforge")
    return _TRACER


def _register_instrumentors() -> None:
    """Register the repo-declared auto-instrumentors against the current global
    tracer provider. Each lives at ``opentelemetry.instrumentation.<name>``;
    a missing instrumentor OR a missing underlying SDK raises
    ``ModuleNotFoundError`` → silently skipped (the MAS owner's install set
    decides what is active — no archforge edit). One broken instrumentor never
    sinks the run.
    """
    try:
        from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
    except Exception as exc:  # noqa: BLE001
        _log.debug("BaseInstrumentor unavailable; no auto-instrumentation: %s", exc)
        return

    for name in _INSTRUMENTORS:
        if name in _REGISTERED:
            continue
        modname = f"opentelemetry.instrumentation.{name}"
        try:
            mod = importlib.import_module(modname)
            cls = _instrumentor_class(mod, BaseInstrumentor)
            if cls is None:
                continue
            cls().instrument()
            _REGISTERED.add(name)
        except ModuleNotFoundError:
            continue  # instrumentor OR its SDK not installed → skip silently (common case)
        except Exception as exc:  # noqa: BLE001 — dep conflict / instrument failure → skip one, keep going
            _log.debug("instrumentor %s skipped: %s", name, exc)
            continue


def _instrumentor_class(mod: Any, BaseInstrumentor: type) -> Any:
    """The single Instrumentor subclass DEFINED IN ``mod`` (not merely imported
    into it). Returns ``None`` if the module has none."""
    for _name, obj in inspect.getmembers(mod, inspect.isclass):
        try:
            if obj.__module__ == mod.__name__ and issubclass(obj, BaseInstrumentor):
                return obj
        except TypeError:
            continue
    return None


# --------------------------------------------------------------------------- #
# The in-memory span buffer — the reader project() queries.
# --------------------------------------------------------------------------- #


class _NodeSpanBuffer:
    """An in-memory span buffer :func:`project` reads — never exports (no OTLP
    sink ⇒ spans never leave the process).

    Implements the SpanProcessor protocol DUCK-TYPED (NOT subclassing
    ``SpanProcessor``) so this module's top level imports NO OpenTelemetry —
    ``Span.end()`` / ``SynchronousMultiSpanProcessor`` resolve ``_on_ending`` /
    ``on_start`` / ``on_end`` / ``shutdown`` / ``force_flush`` via plain attribute
    dispatch with no ``isinstance`` gate (verified against the installed SDK),
    so a plain class with those methods suffices. Keeps the import-lazy contract:
    an env without OTel still loads :mod:`archforge.otel` and :mod:`archforge`.

    Cooperative attribution: an ``archforge.node`` span (from :func:`wrapped`)
    is the PARENT; auto-instrumented GenAI spans nest as children. Children
    always end BEFORE their parent (OTel semantics), and the parent ends when the
    node fn returns — which is BEFORE ``graph.stream`` emits the node's chunk ⇒
    by the time ``_record(node_name, ...)`` runs, the just-finished
    ``archforge.node`` span AND its GenAI children are already buffered.

    :meth:`take_node` drains the matched node span (+ everything before it — its
    children, plus any earlier/already-consumed spans) on each LLM call, so
    re-entry (the ``retrieve_more→extract`` loop) populates a fresh span set per
    occurrence and memory stays bounded to roughly one superstep's worth.
    """

    def __init__(self) -> None:
        self._ended: list[Any] = []  # ended spans, append order = end order

    # -- SpanProcessor protocol (duck-typed; called by Span.end + composite) - #
    def _on_ending(self, span):  # noqa: ARG002, D401  # called by Span.end() before on_end
        pass

    def on_start(self, span, parent_context=None):  # noqa: ARG002, D401
        pass

    def on_end(self, span):  # noqa: D401
        self._ended.append(span)

    def shutdown(self) -> None:  # noqa: D401
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002, D401
        return True

    # -- project support ---------------------------------------------------- #
    def take_node(self, node_name: str) -> tuple[Any, list[Any]]:
        """Pop + return the most recently ended ``archforge.node`` span whose
        ``archforge.node_id`` == ``node_name``, along with its GenAI child spans.
        Returns ``(node_span, children)`` or ``(None, [])`` if no match.

        Drains everything up to and including the matched span: its children
        always ended before it, and no later node's spans could have ended before
        it (later nodes run later) — so the drained prefix is exactly this
        occurrence's spans plus earlier/already-consumed ones.
        """
        node_idx = None
        node_span = None
        for i in range(len(self._ended) - 1, -1, -1):
            s = self._ended[i]
            if _is_node_span(s, node_name):
                node_idx = i
                node_span = s
                break
        if node_span is None:
            return None, []
        snapshot = self._ended[: node_idx + 1]
        self._ended = self._ended[node_idx + 1:]
        target = node_span.context.span_id
        children = [
            s for s in snapshot
            if s is not node_span and _parent_id(s) == target and _is_genai_span(s)
        ]
        return node_span, children


# --------------------------------------------------------------------------- #
# Span attribute helpers (version-agnostic over gen_ai.* / llm.*).
# --------------------------------------------------------------------------- #
# Content keys are version-sensitive: newer OpenLLMetry emits
# ``gen_ai.input.messages`` / ``gen_ai.output.messages`` (JSON message arrays);
# older instrumentors emit ``gen_ai.prompt`` / ``gen_ai.completion`` (flat
# strings); yet older emit ``llm.prompt`` / ``llm.completion``. :func:`_span_attr`
# tries candidates in order so :func:`project` is instrumentor-version-agnostic.
_INPUT_KEYS = ("gen_ai.input.messages", "gen_ai.prompt", "llm.prompt")
_OUTPUT_KEYS = ("gen_ai.output.messages", "gen_ai.completion", "llm.completion")
_USAGE_INPUT_KEYS = ("gen_ai.usage.input_tokens", "llm.usage.prompt_tokens")
_USAGE_OUTPUT_KEYS = ("gen_ai.usage.output_tokens", "llm.usage.completion_tokens")


def _span_attr(span: Any, *candidates: str) -> Any:
    """First present, non-empty attribute among ``candidates`` (version-agnostic)."""
    attrs = span.attributes or {}
    for k in candidates:
        v = attrs.get(k)
        if v is not None and v != "":
            return v
    return None


def _is_node_span(span: Any, name: str) -> bool:
    if getattr(span, "name", None) != "archforge.node":
        return False
    attrs = span.attributes or {}
    return attrs.get("archforge.node_id") == name


def _is_genai_span(span: Any) -> bool:
    attrs = span.attributes or {}
    return any(
        str(k).startswith("gen_ai.") or str(k).startswith("llm.")
        for k in attrs
    )


def _parent_id(span: Any) -> Any:
    parent = getattr(span, "parent", None)
    return parent.span_id if parent is not None else None


def _messages_text(raw: Any) -> str:
    """Render a span's messages/prompt/completion attribute to text. Handles BOTH
    the newer JSON message-array form (``[{"role","content"}, ...]``) and a legacy
    flat string. Returns ``""`` on any decode failure (caller treats empty →
    degrade to ``summarize()``)."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.strip()
    if not s:
        return ""
    if s[0] in "[{":
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return s  # not JSON after all — return verbatim
        if isinstance(parsed, list):
            parts = []
            for msg in parsed:
                if isinstance(msg, dict):
                    parts.append(str(msg.get("content", "")))
                else:
                    parts.append(str(msg))
            return "\n".join(p for p in parts if p)
        if isinstance(parsed, dict):
            return str(parsed.get("content", parsed))
        return str(parsed)
    return s


def _join_prompt(children: list[Any]) -> str:
    """Concatenate prompt text across the node's child GenAI spans (a node may
    make several calls)."""
    out = []
    for s in children:
        out.append(_messages_text(_span_attr(s, *_INPUT_KEYS)))
    return "\n".join(t for t in out if t)


def _join_completion(children: list[Any]) -> str:
    """Concatenate completion text across the node's child GenAI spans."""
    out = []
    for s in children:
        out.append(_messages_text(_span_attr(s, *_OUTPUT_KEYS)))
    return "\n".join(t for t in out if t)


def _usage_tokens(children: list[Any]) -> int:
    """Sum real ``gen_ai.usage`` across child spans; 0 if absent (caller falls
    back to ``estimate_tokens`` over the completion)."""
    total = 0
    for s in children:
        i = _span_attr(s, *_USAGE_INPUT_KEYS)
        o = _span_attr(s, *_USAGE_OUTPUT_KEYS)
        try:
            total += int(i) if i is not None else 0
            total += int(o) if o is not None else 0
        except (TypeError, ValueError):
            continue
    return total


def _cap(text: str, max_tok: int) -> str:
    """Slice ``text`` to ≤ ``max_tok`` tokens under the len//4 estimate
    (``max_tok * 4`` chars ⇒ ``max_tok`` tokens by that heuristic)."""
    if not text:
        return ""
    n = max_tok * 4
    return text if len(text) <= n else text[:n]


# --------------------------------------------------------------------------- #
# The cooperative seam — owned by the forge, used by the MAS's build_graph.
# --------------------------------------------------------------------------- #


@contextmanager
def archforge_node_span(name: str):
    """Context manager opening an ``archforge.node`` span (attr
    ``archforge.node_id=name``) as the active context. Auto-instrumented SDK
    calls inside ``with`` nest as children → parent-link correlation. No-op (the
    body runs without a span) when OTel is unavailable.

    ``wrapped`` builds on this; tests use it directly to open a node span around
    a stubbed SDK call.
    """
    tracer = _ensure_tracer()
    if not tracer:                 # False/None → no spans; run the body untouched
        yield None
        return
    with tracer.start_as_current_span("archforge.node") as span:
        span.set_attribute("archforge.node_id", name)
        yield span


def wrapped(name: str, fn: Callable) -> Callable:
    """``@wraps(fn)`` wrapper running the node fn inside an ``archforge.node``
    span. Auto-instrumented SDK calls it makes then nest as children →
    :func:`project` attributes them to ``name`` by parent-link.

    No-op passthrough (calls ``fn`` directly) when OTel is unavailable, so a MAS
    whose env lacks OTel runs unchanged. ``build_graph`` imports this from
    ``archforge.host.adapters.langgraph`` — the MAS never imports
    ``archforge.otel`` directly.
    """
    @wraps(fn)
    def _wrapped(*args, **kwargs):
        with archforge_node_span(name):
            return fn(*args, **kwargs)
    return _wrapped


# --------------------------------------------------------------------------- #
# project — the reader the adapter's _record calls instead of summarize().
# --------------------------------------------------------------------------- #


def project(
    node_name: str,
    kind: m.NodeKind,
    partial: dict,
    merged: dict,
) -> Proj | None:
    """The bounded projection of ``node_name``'s real LLM I/O into a ``Step``,
    or ``None`` ⇒ the caller keeps ``summarize()`` (the parity / graceful-degrade
    path).

    v1 scope: enriches LLM nodes ONLY. Retriever/rule/symbolic → ``None``.
    ``partial`` / ``merged`` are accepted for the future ``retriever_dump`` hook
    (state-shape-agnostic — no guessing a retriever's docs state key) and are
    unused in v1.

    LLM: reads the just-closed ``archforge.node`` span's child GenAI span(s):
      * ``gen_ai.output.messages`` (capped ``COMPLETION_SLICE_TOK``) → ``response_out``
      * ``gen_ai.input.messages``  (capped ``PROMPT_SLICE_TOK``)    → ``prompt_in``
      * ``gen_ai.usage.{input,output}_tokens`` → ``tokens`` (``estimate_tokens``
        over the completion as fallback)
    Absent buffer / no children / empty completion (capture off) → ``None``.
    """
    _ = (partial, merged)  # reserved for the future retriever_dump hook; v1 unused
    if kind is not m.NodeKind.LLM:
        return None        # v1 cut: only LLM nodes get a span projection
    if not _ensure_tracer():
        return None        # OTel absent → degrade to summarize()
    if _BUFFER is None:
        return None
    _node, children = _BUFFER.take_node(node_name)
    if not children:
        return None        # no child GenAI spans (instrumentor absent / capture off)
    completion = _join_completion(children)
    if not completion:
        return None        # capture off → empty completion → degrade
    prompt = _join_prompt(children)
    tokens = _usage_tokens(children)
    if tokens <= 0:
        tokens = estimate_tokens(completion)
    return Proj(
        prompt_in=_cap(prompt, PROMPT_SLICE_TOK),
        response_out=_cap(completion, COMPLETION_SLICE_TOK),
        tokens=tokens,
    )


# --------------------------------------------------------------------------- #
# _shed_to_budget — post-loop trim so the Judge's total ingested text ≤ budget.
# --------------------------------------------------------------------------- #


def _shed_to_budget(
    steps: list, final_answer_node: str | None, total_budget: int
) -> None:
    """Post-loop: mutate the given ``Step`` objects' ``response_out`` IN PLACE so
    the Judge's total ingested text ≤ ``total_budget`` tokens. Trim the LARGEST
    ``response_out`` first (the heaviest evidence chunks) while PROTECTING
    ``final_answer_node`` (the answer-bearing step always survives). Mutates the
    SAME objects ``_record_step`` already appended (the middleware holds
    references → ``end_run``'s ``list(self._steps)`` sees trimmed values).

    Marker: ``"[trimmed: <node_id>]"``. Idempotent — already-trimmed steps are
    skipped. Returns nothing (in place). If trimming all non-protected steps
    still exceeds ``total_budget`` (tiny budget), the final-answer step is kept
    anyway — the answer must reach the Judge.

    Called only on the success path (a mid-run crash skips it → ran Steps retain
    per-node caps → spec E4 holds).
    """
    if total_budget is None or total_budget <= 0:
        return
    marker_prefix = "[trimmed:"

    def is_final(s) -> bool:
        return final_answer_node is not None and s.node_id == final_answer_node

    def is_trimmed(s) -> bool:
        return str(s.response_out or "").startswith(marker_prefix)

    def current_total() -> int:
        return sum(estimate_tokens(s.response_out) for s in steps)

    while current_total() > total_budget:
        candidate = None
        candidate_tok = 0
        for s in steps:
            if is_final(s) or is_trimmed(s):
                continue
            t = estimate_tokens(s.response_out)
            if t > candidate_tok:
                candidate = s
                candidate_tok = t
        if candidate is None:
            break  # nothing left to trim (only the protected step remains) → keep it
        candidate.response_out = f"[trimmed: {candidate.node_id}]"


# --------------------------------------------------------------------------- #
# Test-only reset (not in __all__): roll back the process singleton so a fresh
# _ensure_tracer re-builds between tests that stub different SDK sets.
# --------------------------------------------------------------------------- #


def _reset() -> None:
    """Test-only: clear the cached tracer/buffer/registered set so a fresh
    :func:`_ensure_tracer` re-builds (e.g. between tests stubbing different SDKs).
    """
    global _TRACER, _BUFFER
    _TRACER = None
    _BUFFER = None
    _REGISTERED.clear()


__all__ = [
    "Proj",
    "wrapped",
    "archforge_node_span",
    "project",
    "PROMPT_SLICE_TOK",
    "COMPLETION_SLICE_TOK",
]
