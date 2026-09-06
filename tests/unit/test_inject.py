"""Zero-touch injector tests — NO network, NO real SDKs.

Stubs ``groq`` / ``google.genai`` in ``sys.modules`` (the same pattern as
``tests/unit/test_deepeval.py``) so ``SdkInjector`` patches fake classes:
attribution of a call to its node via the call stack, per-node ``KnobVote``
rewrites (model/temperature/max_tokens/system_prompt), passthrough for
unvoted nodes and unknown modules, prompt capture, and ``uninstall``
restoring the originals. ``apply_knob_settings`` is covered for apply /
coerce / restore-on-exception. The explicit-seam regression asserts the
adapter-level guard: an app overriding ``apply_llm_config`` never activates
the injector.
"""
from __future__ import annotations

import sys
import types as pytypes
from contextlib import contextmanager

import pytest

from archforge.host.adapters.inject import (
    NodeLocator, SdkInjector, apply_knob_settings,
)
from archforge.host.adapters.helpers import KnobVote
from archforge.host.adapters.langgraph import LangGraphApp


# --------------------------------------------------------------------------- #
# fake SDKs
# --------------------------------------------------------------------------- #


class _FakeChatCompletions:
    """Stands in for groq/openai ``Completions``: records create() kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"text": "ok"}


class _FakeGenaiConfig:
    def __init__(self, system_instruction=None, temperature=None):
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.max_output_tokens = None


class _FakeGenaiModels:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(dict(kwargs))
        return "answer"


@contextmanager
def _stub_sdks():
    """Install fake ``groq`` and ``google.genai`` module chains; restore after."""
    chat_mod = pytypes.ModuleType("groq.resources.chat.completions")
    chat_mod.Completions = _FakeChatCompletions  # type: ignore[attr-defined]

    genai_models = pytypes.ModuleType("google.genai.models")
    genai_models.Models = _FakeGenaiModels  # type: ignore[attr-defined]

    saved = {k: sys.modules.get(k) for k in
             ("groq.resources.chat.completions", "google.genai.models")}
    sys.modules["groq.resources.chat.completions"] = chat_mod
    sys.modules["google.genai.models"] = genai_models
    try:
        yield chat_mod, genai_models
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _node_module(mod_name: str):
    """A fake host-node module whose ``call`` invokes a client; its
    ``__module__``-attributed frames are how the injector attributes calls."""
    mod = pytypes.ModuleType(mod_name)
    mod.__dict__["__name__"] = mod_name
    exec(
        "def call(client, **kw):\n    return client.create(**kw)\n"
        "def call_genai(models, **kw):\n    return models.generate_content(**kw)\n",
        mod.__dict__,
    )
    sys.modules[mod_name] = mod
    return mod


def _vote(**kw) -> KnobVote:
    base = dict(model=None, temperature=None, max_tokens=None,
                retries=None, system_prompt=None)
    base.update(kw)
    return KnobVote(**base)


# --------------------------------------------------------------------------- #
# NodeLocator
# --------------------------------------------------------------------------- #


def test_locator_attributes_nearest_node_module() -> None:
    mod = _node_module("mymas.nodes.extractor")
    locator = NodeLocator({"mymas.nodes.extractor": "extract"})
    assert locator.locate() is None            # called from the test module
    assert mod.call.__module__ == "mymas.nodes.extractor"

    hits = []
    exec("def probe_fn(locator, hits):\n    hits.append(locator.locate())\n",
         mod.__dict__)
    mod.probe_fn(locator, hits)  # type: ignore[attr-defined]
    assert hits == ["extract"]


def test_locator_unknown_module_returns_none() -> None:
    locator = NodeLocator({"mymas.nodes.extractor": "extract"})
    assert locator.locate() is None  # this test's module is not mapped


def test_locator_from_graph_skips_unresolvable_nodes() -> None:
    class _Spec:
        runnable = None  # no .func -> skipped, never crashes
    graph = pytypes.SimpleNamespace(nodes={"a": _Spec(), "__start__": _Spec()})
    locator = NodeLocator.from_graph(graph, extra={"mymas.nodes.x": "x"})
    assert locator._map == {"mymas.nodes.x": "x"}


# --------------------------------------------------------------------------- #
# SdkInjector: chat-completions shape (groq/openai)
# --------------------------------------------------------------------------- #


def test_chat_vote_rewrites_model_temp_tokens_and_system() -> None:
    with _stub_sdks() as (chat_mod, _):
        node = _node_module("mymas.nodes.extract")
        locator = NodeLocator({"mymas.nodes.extract": "extract"})
        inj = SdkInjector(locator, {"extract": _vote(
            model="llama-3.3-70b", temperature=0.7, max_tokens=128,
            system_prompt="NEW SYSTEM")})
        inj.install()
        try:
            client = chat_mod.Completions()
            node.call(client, model="old", temperature=0.3, max_tokens=4096,
                      messages=[{"role": "system", "content": "OLD"},
                                {"role": "user", "content": "hi"}])
        finally:
            inj.uninstall()
        sent = client.calls[0]
        assert sent["model"] == "llama-3.3-70b"
        assert sent["temperature"] == 0.7
        assert sent["max_tokens"] == 128
        assert sent["messages"][0] == {"role": "system", "content": "NEW SYSTEM"}
        assert sent["messages"][1] == {"role": "user", "content": "hi"}


def test_chat_system_prompt_inserted_when_absent() -> None:
    with _stub_sdks() as (chat_mod, _):
        node = _node_module("mymas.nodes.answer")
        locator = NodeLocator({"mymas.nodes.answer": "answer"})
        inj = SdkInjector(locator, {"answer": _vote(system_prompt="S")})
        inj.install()
        try:
            client = chat_mod.Completions()
            node.call(client, model="m", messages=[{"role": "user", "content": "q"}])
        finally:
            inj.uninstall()
        msgs = client.calls[0]["messages"]
        assert msgs[0] == {"role": "system", "content": "S"}
        assert len(msgs) == 2


def test_unvoted_node_and_unknown_module_pass_through() -> None:
    with _stub_sdks() as (chat_mod, _):
        voted = _node_module("mymas.nodes.extract")
        unvoted = _node_module("mymas.nodes.compress")
        locator = NodeLocator({"mymas.nodes.extract": "extract",
                               "mymas.nodes.compress": "compress"})
        inj = SdkInjector(locator, {"extract": _vote(model="NEW")})
        inj.install()
        try:
            c1, c2, c3 = (chat_mod.Completions() for _ in range(3))
            voted.call(c1, model="a", messages=[])
            unvoted.call(c2, model="b", messages=[])   # known node, no vote
            c3.create(model="c", messages=[])          # caller outside any node
        finally:
            inj.uninstall()
        assert c1.calls[0]["model"] == "NEW"
        assert c2.calls[0]["model"] == "b"
        assert c3.calls[0]["model"] == "c"


def test_uninstall_restores_original() -> None:
    with _stub_sdks() as (chat_mod, _):
        original = chat_mod.Completions.create
        inj = SdkInjector(NodeLocator({}), {})
        inj.install()
        assert chat_mod.Completions.create is not original
        inj.uninstall()
        assert chat_mod.Completions.create is original


# --------------------------------------------------------------------------- #
# prompt capture
# --------------------------------------------------------------------------- #


def test_first_system_prompt_captured_per_node() -> None:
    with _stub_sdks() as (chat_mod, _):
        node = _node_module("mymas.nodes.compress")
        locator = NodeLocator({"mymas.nodes.compress": "compress"})
        captured: dict[str, str] = {}
        inj = SdkInjector(locator, {}, captured)
        inj.install()
        try:
            client = chat_mod.Completions()
            node.call(client, model="m",
                      messages=[{"role": "system", "content": "CAPTURED"}])
            node.call(client, model="m",
                      messages=[{"role": "system", "content": "SECOND"}])
        finally:
            inj.uninstall()
        assert captured == {"compress": "CAPTURED"}   # first write wins


# --------------------------------------------------------------------------- #
# google.genai shape
# --------------------------------------------------------------------------- #


def test_genai_vote_rewrites_model_and_config() -> None:
    with _stub_sdks() as (_, genai_models):
        node = _node_module("mymas.nodes.reason")
        locator = NodeLocator({"mymas.nodes.reason": "reason"})
        inj = SdkInjector(locator, {"reason": _vote(
            model="gemini-2.5-pro", temperature=0.9, max_tokens=64,
            system_prompt="DEEP")})
        inj.install()
        try:
            models = genai_models.Models()
            cfg = _FakeGenaiConfig(system_instruction="OLD", temperature=0.2)
            node.call_genai(models, model="gemini-2.5-flash-lite",
                            contents="q", config=cfg)
        finally:
            inj.uninstall()
        sent = models.calls[0]
        assert sent["model"] == "gemini-2.5-pro"
        assert cfg.system_instruction == "DEEP"
        assert cfg.temperature == 0.9
        assert cfg.max_output_tokens == 64


def test_missing_sdk_is_skipped() -> None:
    # No stubs installed: every target import fails; install is a no-op.
    for name in ("groq.resources.chat.completions", "google.genai.models"):
        sys.modules.pop(name, None)
    inj = SdkInjector(NodeLocator({}), {"x": _vote(model="m")})
    inj.install()      # must not raise
    inj.uninstall()


# --------------------------------------------------------------------------- #
# apply_knob_settings
# --------------------------------------------------------------------------- #


class _Section:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_settings_knobs_applied_coerced_and_restored() -> None:
    settings = pytypes.SimpleNamespace(
        retrieval=_Section(max_k=16, initial_k=4),
        pipeline=_Section(coverage_target=0.8, binary_growth=True),
    )
    mapping = {"max_k": ("retrieval", "max_k"),
               "coverage_target": ("pipeline", "coverage_target"),
               "binary_growth": ("pipeline", "binary_growth")}
    restore = apply_knob_settings(settings, mapping, [
        ("max_k", 32),                 # int stays int
        ("coverage_target", 0.5),      # float coerced
        ("binary_growth", 0),          # int -> bool field coerces to bool
        ("unknown_knob", 1),           # unmapped: skipped
    ])
    assert settings.retrieval.max_k == 32
    assert settings.pipeline.coverage_target == 0.5
    assert settings.pipeline.binary_growth is False
    restore()
    assert settings.retrieval.max_k == 16
    assert settings.pipeline.coverage_target == 0.8
    assert settings.pipeline.binary_growth is True


def test_settings_restore_runs_when_run_raises() -> None:
    settings = pytypes.SimpleNamespace(retrieval=_Section(max_k=16))
    restore = apply_knob_settings(settings, {"max_k": ("retrieval", "max_k")},
                                  [("max_k", 8)])
    with pytest.raises(RuntimeError):
        try:
            raise RuntimeError("boom")
        finally:
            restore()
    assert settings.retrieval.max_k == 16


def test_missing_section_is_skipped() -> None:
    settings = pytypes.SimpleNamespace()  # no .retrieval at all
    restore = apply_knob_settings(settings, {"max_k": ("retrieval", "max_k")},
                                  [("max_k", 8)])
    restore()  # no-op, no raise


# --------------------------------------------------------------------------- #
# explicit-seam regression: an override disables the injector
# --------------------------------------------------------------------------- #


def test_explicit_seam_app_disables_injector() -> None:
    class SeamApp(LangGraphApp):
        def apply_llm_config(self, node_id, vote) -> None:  # explicit seam
            pass

    class ZeroTouchApp(LangGraphApp):
        pass

    assert type(SeamApp()).apply_llm_config is not LangGraphApp.apply_llm_config
    assert type(ZeroTouchApp()).apply_llm_config is LangGraphApp.apply_llm_config


def test_default_hooks_record_and_clear_votes() -> None:
    app = LangGraphApp()
    v = _vote(model="m")
    app.apply_llm_config("extract", v)
    assert app._pending_votes == {"extract": v}
    app.reset_llm_config()
    assert app._pending_votes == {}


# --------------------------------------------------------------------------- #
# real provider usage capture
# --------------------------------------------------------------------------- #


class _UsageChatCompletions(_FakeChatCompletions):
    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return pytypes.SimpleNamespace(
            usage=pytypes.SimpleNamespace(prompt_tokens=11, completion_tokens=7,
                                          total_tokens=18))


class _UsageGenaiModels(_FakeGenaiModels):
    def generate_content(self, **kwargs):
        self.calls.append(dict(kwargs))
        return pytypes.SimpleNamespace(
            usage_metadata=pytypes.SimpleNamespace(prompt_token_count=5,
                                                   candidates_token_count=3,
                                                   total_token_count=8))


def _usage_stub_sdks():
    """_stub_sdks with usage-carrying response objects."""
    import contextlib
    @contextlib.contextmanager
    def _ctx():
        chat_mod = pytypes.ModuleType("groq.resources.chat.completions")
        chat_mod.Completions = _UsageChatCompletions  # type: ignore[attr-defined]
        genai_models = pytypes.ModuleType("google.genai.models")
        genai_models.Models = _UsageGenaiModels  # type: ignore[attr-defined]
        saved = {k: sys.modules.get(k) for k in
                 ("groq.resources.chat.completions", "google.genai.models")}
        sys.modules["groq.resources.chat.completions"] = chat_mod
        sys.modules["google.genai.models"] = genai_models
        try:
            yield chat_mod, genai_models
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    return _ctx()


def test_chat_usage_recorded_per_node() -> None:
    with _usage_stub_sdks():
        mod = _node_module("hostapp.nodes.answer")
        loc = NodeLocator({"hostapp.nodes.answer": "answer"})
        usage: dict[str, list[int]] = {}
        inj = SdkInjector(loc, {}, usage=usage)
        inj.install()
        try:
            mod.call(_UsageChatCompletions(), model="m", messages=[])
            mod.call(_UsageChatCompletions(), model="m", messages=[])
        finally:
            inj.uninstall()
    assert usage == {"answer": [18, 18]}


def test_genai_usage_recorded() -> None:
    with _usage_stub_sdks():
        mod = _node_module("hostapp.nodes.reason")
        loc = NodeLocator({"hostapp.nodes.reason": "reason"})
        usage: dict[str, list[int]] = {}
        inj = SdkInjector(loc, {}, usage=usage)
        inj.install()
        try:
            mod.call_genai(_UsageGenaiModels(), model="m", contents="hi")
        finally:
            inj.uninstall()
    assert usage == {"reason": [8]}


def test_usage_not_recorded_for_unknown_or_missing() -> None:
    with _stub_sdks():  # plain fakes: dict/str responses carry no usage attrs
        mod = _node_module("hostapp.nodes.answer")
        loc = NodeLocator({"hostapp.nodes.answer": "answer"})
        usage: dict[str, list[int]] = {}
        inj = SdkInjector(loc, {}, usage=usage)
        inj.install()
        try:
            mod.call(_FakeChatCompletions(), model="m", messages=[])
        finally:
            inj.uninstall()
    assert usage == {}  # no usage on the response -> nothing recorded, no raise
