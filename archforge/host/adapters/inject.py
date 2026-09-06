"""Zero-touch LLM injection — SDK-boundary patching + node attribution.

The LangGraph adapter's named LLM knobs (model/temperature/max_tokens/
system_prompt) historically needed a call-time seam INSIDE the host MAS (a
``node_config`` registry its call sites consult). This module removes that
requirement: the adapter patches the LLM SDKs' chokepoints for the duration of
one run, attributes each outbound call to a graph node by walking the call
stack, and rewrites the request per that node's ``KnobVote``.

Attribution — a :class:`NodeLocator` maps each node id to the MODULE of the
callable the graph builder passed to ``add_node`` (read off the compiled graph
once, best-effort; a node whose function cannot be found simply never receives
an override). When a patched SDK method fires, the first stack frame whose
``__module__`` is a known node module wins. Helper functions in non-node
modules are skipped transparently, and a call from outside any node (auth
probes, SDK internals) passes through untouched. Stack walking is per-call, so
interleaved/parallel nodes attribute correctly.

Patch targets (each skipped silently when the SDK is not installed):

  * ``groq``      — ``groq.resources.chat.completions.Completions.create``
  * ``openai``    — ``openai.resources.chat.completions.Completions.create``
  * ``google.genai`` — ``google.genai.models.Models.generate_content``

Only the SYNC entry points are patched; an async host keeps the explicit-seam
``apply_llm_config`` override (the injector never activates there — the
activation guard lives in ``langgraph.LangGraphRunnable``).

Prompt capture: the first system prompt observed per node is recorded into a
caller-supplied dict, so a later ``prompt_edit`` has a real base even when the
app left ``base_prompts`` empty.

This module imports NO SDK at module level; every import is inside the patch
functions so ``import archforge`` stays provider-free.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

from archforge.host.adapters.helpers import KnobVote


# --------------------------------------------------------------------------- #
# Node attribution
# --------------------------------------------------------------------------- #


class NodeLocator:
    """Map module names -> node ids; attribute a live SDK call via the stack.

    Built from a compiled graph's node table (best-effort: LangGraph exposes
    each node's bound callable as ``graph.nodes[name].runnable.func``; the attr
    chain is version-fragile, so any node that does not resolve is simply
    absent from the map and never receives an override). ``extra`` is an
    explicit ``{module: node_id}`` overlay for hosts whose shape introspection
    misses (or for tests).
    """

    def __init__(self, module_to_node: dict[str, str]) -> None:
        self._map = dict(module_to_node)

    @classmethod
    def from_graph(
        cls, graph: Any, extra: dict[str, str] | None = None
    ) -> "NodeLocator":
        m: dict[str, str] = {}
        nodes = getattr(graph, "nodes", None) or {}
        for name in nodes:
            if str(name).startswith("__"):
                continue
            spec = nodes[name]
            # The bound callable's home moves between LangGraph versions:
            # newer PregelNode keeps it at ``.bound.func`` (a RunnableCallable),
            # older at ``.runnable.func``. Try both; unresolvable nodes are
            # simply absent from the map (they never receive an override).
            fn = None
            for holder in ("bound", "runnable", None):
                carrier = spec if holder is None else getattr(spec, holder, None)
                fn = getattr(carrier, "func", None)
                if fn is not None:
                    break
            mod = getattr(fn, "__module__", None)
            if mod:
                m.setdefault(mod, str(name))
        if extra:
            m.update(extra)
        return cls(m)

    def locate(self) -> str | None:
        """The node id whose module is nearest on the call stack, else None."""
        frame = inspect.currentframe()
        try:
            if frame is not None:
                frame = frame.f_back  # skip locate() itself
            while frame is not None:
                mod = frame.f_globals.get("__name__")
                if mod in self._map:
                    return self._map[mod]
                frame = frame.f_back
            return None
        finally:
            del frame  # break the reference cycle promptly


# --------------------------------------------------------------------------- #
# Request rewriting per SDK call shape
# --------------------------------------------------------------------------- #


def _system_from_messages(messages: Any) -> str | None:
    """The system prompt of a chat-completions ``messages`` list, if present."""
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict) and first.get("role") == "system":
            content = first.get("content")
            return content if isinstance(content, str) else None
    return None


def _apply_chat_vote(kwargs: dict[str, Any], vote: KnobVote) -> None:
    """Rewrite an OpenAI/Groq chat-completions call's kwargs in place."""
    if vote.model:
        kwargs["model"] = vote.model
    if vote.temperature is not None:
        kwargs["temperature"] = vote.temperature
    if vote.max_tokens is not None:
        kwargs["max_tokens"] = vote.max_tokens
    if vote.system_prompt:
        # truthy, not "is not None": cfg_decay yields "" (not None) when the
        # seeded prompt differs from an EMPTY base — and "" must not erase the
        # host's real system prompt.
        msgs = list(kwargs.get("messages") or [])
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            msgs[0] = {**msgs[0], "content": vote.system_prompt}
        else:
            msgs.insert(0, {"role": "system", "content": vote.system_prompt})
        kwargs["messages"] = msgs


def _apply_genai_vote(kwargs: dict[str, Any], vote: KnobVote) -> None:
    """Rewrite a google.genai ``generate_content`` call's kwargs in place.

    The SDK takes ``model=`` plus a ``config`` object carrying
    ``temperature`` / ``max_output_tokens`` / ``system_instruction``. The config
    is mutated in place (it is the caller's object, but a run-scoped override is
    exactly what the caller asked the injector to deliver). A call with no
    ``config`` gets model-only overrides; constructing a config type here would
    couple us to the SDK's types for a marginal case.
    """
    if vote.model:
        kwargs["model"] = vote.model
    config = kwargs.get("config")
    if config is None:
        return
    if vote.temperature is not None:
        setattr(config, "temperature", vote.temperature)
    if vote.max_tokens is not None:
        setattr(config, "max_output_tokens", vote.max_tokens)
    if vote.system_prompt:  # truthy: "" (cfg_decay's empty-base artifact) is no override
        setattr(config, "system_instruction", vote.system_prompt)


def _capture(captured: dict[str, str] | None, node_id: str, prompt: str | None) -> None:
    """Record the first observed system prompt per node (first write wins)."""
    if captured is None or prompt is None or node_id in captured:
        return
    captured[node_id] = prompt


def _record_usage(
    usage: dict[str, list[int]] | None, node_id: str, response: Any, attr_path: tuple[str, str]
) -> None:
    """Append the response's metered total tokens to ``usage[node_id]``.

    ``attr_path`` is the SDK's usage shape: ``("usage", "total_tokens")`` for
    chat completions, ``("usage_metadata", "total_token_count")`` for
    google.genai. Any deviation (older SDK, streaming chunk, provider quirk)
    records nothing — never break the host's call over accounting.
    """
    if usage is None:
        return
    try:
        holder = getattr(response, attr_path[0], None)
        total = getattr(holder, attr_path[1], None) if holder is not None else None
        if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
            usage.setdefault(node_id, []).append(total)
    except Exception:  # noqa: BLE001 — accounting must never break the host
        pass


# --------------------------------------------------------------------------- #
# The injector
# --------------------------------------------------------------------------- #

# (import-path, class attr chain) per supported SDK. Kept as data so adding an
# SDK is one entry, not a new function.
_CHAT_TARGETS: tuple[tuple[str, str], ...] = (
    ("groq.resources.chat.completions", "Completions"),
    ("openai.resources.chat.completions", "Completions"),
)


class SdkInjector:
    """Patches SDK entry points for one run; ``uninstall`` restores them.

    ``votes`` maps node id -> the ``KnobVote`` the adapter decayed from the live
    Spec (only nodes with a recorded vote are overridden; others pass through).
    ``captured`` (optional) collects first-seen system prompts per node.
    """

    def __init__(
        self,
        locator: NodeLocator,
        votes: dict[str, KnobVote],
        captured: dict[str, str] | None = None,
        usage: dict[str, list[int]] | None = None,
    ) -> None:
        self._locator = locator
        self._votes = votes
        self._captured = captured
        # Real provider usage per node (one entry per observed call, in call
        # order). The adapter drains it per step so the trace's cost column is
        # the metered total, not the len//4 estimate.
        self._usage = usage
        self._originals: list[tuple[type, str, Any]] = []

    # ---- install/uninstall ------------------------------------------------ #
    def install(self) -> None:
        """Patch every importable target SDK. Idempotent-safe: a second install
        after uninstall repatches cleanly (originals were restored)."""
        import importlib

        for mod_path, cls_name in _CHAT_TARGETS:
            try:
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, cls_name)
            except (ImportError, AttributeError):
                continue  # SDK not installed in this host env: nothing to patch
            self._patch(cls, "create", self._wrap_chat)

        try:
            genai_models = importlib.import_module("google.genai.models")
            models_cls = getattr(genai_models, "Models")
        except (ImportError, AttributeError):
            models_cls = None
        if models_cls is not None:
            self._patch(models_cls, "generate_content", self._wrap_genai)

    def uninstall(self) -> None:
        """Restore every patched method to its original (LIFO)."""
        while self._originals:
            cls, attr, original = self._originals.pop()
            setattr(cls, attr, original)

    def _patch(self, cls: type, attr: str, wrap: Callable[[Any], Any]) -> None:
        original = getattr(cls, attr, None)
        if original is None:
            return
        self._originals.append((cls, attr, original))
        setattr(cls, attr, wrap(original))

    # ---- wrappers ---------------------------------------------------------- #
    def _wrap_chat(self, original: Any) -> Any:
        @functools.wraps(original)
        def create(*args: Any, **kwargs: Any) -> Any:
            node_id = self._locator.locate()
            if node_id is not None:
                vote = self._votes.get(node_id)
                if vote is not None:
                    _apply_chat_vote(kwargs, vote)
                _capture(self._captured, node_id,
                         _system_from_messages(kwargs.get("messages")))
            resp = original(*args, **kwargs)
            if node_id is not None:
                _record_usage(self._usage, node_id, resp, ("usage", "total_tokens"))
            return resp

        return create

    def _wrap_genai(self, original: Any) -> Any:
        @functools.wraps(original)
        def generate_content(*args: Any, **kwargs: Any) -> Any:
            node_id = self._locator.locate()
            if node_id is not None:
                vote = self._votes.get(node_id)
                if vote is not None:
                    _apply_genai_vote(kwargs, vote)
                config = kwargs.get("config")
                _capture(self._captured, node_id,
                         getattr(config, "system_instruction", None))
            resp = original(*args, **kwargs)
            if node_id is not None:
                _record_usage(self._usage, node_id, resp,
                              ("usage_metadata", "total_token_count"))
            return resp

        return generate_content


# --------------------------------------------------------------------------- #
# Settings-singleton knobs (the second zero-touch mechanism)
# --------------------------------------------------------------------------- #


def apply_knob_settings(
    settings: Any,
    mapping: dict[str, tuple[str, str]],
    items: list[tuple[str, Any]],
) -> Callable[[], None]:
    """Overlay knob values onto a settings singleton IN PLACE; return a restore.

    ``mapping`` is knob name -> (section attr, field attr). ``items`` are
    ``(knob_name, value)`` pairs in node order; the LAST write to a field wins
    (mirrors the deploy sidecar's overlay order). Fields not named in
    ``mapping`` are untouched. Values are coerced to the current field's type
    (bool/int/float passthrough of JSON scalars). The returned callable restores
    every touched field to its pre-call value; it is idempotent.
    """
    originals: dict[tuple[int, str], tuple[Any, str, Any]] = {}
    for kname, kval in items:
        if kval is None or kname not in mapping:
            continue
        sect, field = mapping[kname]
        section = getattr(settings, sect, None)
        if section is None or not hasattr(section, field):
            continue
        key = (id(section), field)
        if key not in originals:
            originals[key] = (section, field, getattr(section, field))
        setattr(section, field, _coerce(kval, originals[key][2]))

    def restore() -> None:
        for section, field, value in originals.values():
            setattr(section, field, value)

    return restore


def _coerce(value: Any, current: Any) -> Any:
    """Coerce a knob value to the current field's type (minimal, JSON scalars)."""
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


__all__ = ["NodeLocator", "SdkInjector", "apply_knob_settings"]
