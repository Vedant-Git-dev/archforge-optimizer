"""LiteLLM provider-layer tests (issue #1) — NO network, NO real SDK.

Stubs `litellm.completion` in `sys.modules` so the single `LiteLLMClient` is
exercised hermetic: request-building (model prefixing, JSON mode, api_key/base_url,
system-inline), response-unpacking (content + OpenAI-style usage → our `Usage`),
JSON round-trip, and `APIError`→`LLMError` wrapping. The live smoke
(`tests/smoke/`, marker `smoke`) covers the real API; this file runs in the
default suite and stays green without any provider SDK.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import types as pytypes
from contextlib import contextmanager

import pytest

from archforge.llm.base import Completion, LLMError, Message, Role, extract_json
from archforge import userconfig as ucfg


# --------------------------------------------------------------------------- #
# fake litellm response + completion — what LiteLLMClient unpacks
# --------------------------------------------------------------------------- #


class _FMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FChoice:
    def __init__(self, content: str) -> None:
        self.message = _FMsg(content)


class _FUsage:
    # OpenAI-style field names, exactly as LiteLLM exposes them
    def __init__(self) -> None:
        self.prompt_tokens = 10
        self.completion_tokens = 20


class _FResp:
    def __init__(self, content: str) -> None:
        self.choices = [_FChoice(content)]
        self.usage = _FUsage()


class _FakeAPIError(RuntimeError):
    """Stand-in for litellm.exceptions.APIError (uniform provider-error base)."""


class _State:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail_with: Exception | None = None


_state = _State()


def _completion(**kwargs) -> _FResp:
    _state.calls.append(kwargs)
    if _state.fail_with is not None:
        raise _state.fail_with
    return _FResp('{"kind": "prompt_edit", "target": "a"}')


@contextmanager
def _stub_litellm():
    """Install a fake `litellm` module into sys.modules; restore on exit."""
    litellm = pytypes.ModuleType("litellm")
    litellm.completion = _completion
    saved = sys.modules.get("litellm")
    sys.modules["litellm"] = litellm
    _state.calls = []
    _state.fail_with = None
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("litellm", None)
        else:
            sys.modules["litellm"] = saved  # type: ignore[assignment]


def _seed_messages() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="You are an optimizer."),
        Message(role=Role.USER, content="Propose one change."),
    ]


PARAMS = ["openai", "anthropic", "groq", "gemini"]


# --------------------------------------------------------------------------- #
# request-building + response-unpacking + JSON round-trip (all four providers)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", PARAMS)
def test_litellm_complete_returns_parsed_json(provider: str) -> None:
    with _stub_litellm():
        from archforge.llm import make_client

        client = make_client(provider, api_key="stub-key", base_url="http://stub")
        comp = client.complete(
            _seed_messages(), model="m-x", temperature=0.2,
            max_tokens=64, response_format="json",
        )

    assert isinstance(comp, Completion)
    assert comp.parsed == {"kind": "prompt_edit", "target": "a"}
    assert comp.usage.input_tokens == 10 and comp.usage.output_tokens == 20
    # bare model id is prefixed so LiteLLM routes to the provider's API
    assert comp.model == f"{provider}/m-x"


@pytest.mark.parametrize("provider", PARAMS)
def test_litellm_passthrough_request_kwargs(provider: str) -> None:
    """temperature/json-mode/api_key/base_url reach the litellm call; system stays inline."""
    with _stub_litellm():
        from archforge.llm import make_client

        client = make_client(provider, api_key="stub-key", base_url="http://stub")
        client.complete(_seed_messages(), temperature=0.0, max_tokens=32, response_format="json")
        kw = _state.calls[-1]

    assert kw["temperature"] == 0.0
    assert kw["max_tokens"] == 32
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["api_key"] == "stub-key"
    assert kw["base_url"] == "http://stub"
    assert kw["model"].startswith(f"{provider}/")
    # LiteLLM handles per-provider system placement internally; it stays a message
    assert any(m["role"] == "system" for m in kw["messages"])


def test_litellm_does_not_double_prefix_explicit_model() -> None:
    with _stub_litellm():
        from archforge.llm import make_client

        client = make_client("openai")
        client.complete(_seed_messages(), model="openai/gpt-4o")
        assert _state.calls[-1]["model"] == "openai/gpt-4o"


def test_litellm_prefixes_slash_in_native_model_id() -> None:
    """Groq's native id ``openai/gpt-oss-120b`` has a slash but still needs the
    groq/ prefix — otherwise LiteLLM would route it to OpenAI, not Groq."""
    with _stub_litellm():
        from archforge.llm import make_client

        client = make_client("groq")
        client.complete(_seed_messages(), model="openai/gpt-oss-120b")
        assert _state.calls[-1]["model"] == "groq/openai/gpt-oss-120b"


def test_litellm_prefixed_default_model() -> None:
    with _stub_litellm():
        from archforge.llm import make_client

        client = make_client("gemini")
        client.complete(_seed_messages())
        bare = ucfg.get("DEFAULT_ARCHITECT_MODELS")["gemini"]
        assert _state.calls[-1]["model"] == f"gemini/{bare}"


@pytest.mark.parametrize("provider", PARAMS)
def test_litellm_failure_raises_llmerror(provider: str) -> None:
    """Any provider error becomes LLMError (so E9 retry fires at the suite layer)."""
    with _stub_litellm():
        from archforge.llm import make_client

        client = make_client(provider)
        _state.fail_with = _FakeAPIError("rate-limited")
        with pytest.raises(LLMError):
            client.complete(_seed_messages(), response_format="json")


# --------------------------------------------------------------------------- #
# make_client + laziness + missing LiteLLM → a single clear LLMError
# --------------------------------------------------------------------------- #


def test_make_client_unknown_or_scripted_raises() -> None:
    from archforge.llm import make_client

    with pytest.raises(LLMError):
        make_client("madeup")
    with pytest.raises(LLMError):
        make_client("scripted")


def test_litellm_adapter_import_is_lazy(monkeypatch) -> None:
    """Importing the adapter module must NOT pull in the litellm SDK."""
    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    importlib.import_module("archforge.llm.litellm")
    assert "litellm" not in sys.modules


def test_missing_litellm_raises_llmerror(monkeypatch) -> None:
    from archforge.llm.litellm import LiteLLMClient

    client = LiteLLMClient(provider="openai", api_key="k", base_url="b")
    real_import = builtins.__import__

    def fake_import(name: str, *a, **k):  # type: ignore[no-untyped-def]
        if name == "litellm":
            raise ImportError("No module named 'litellm'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(LLMError) as ei:
        client.complete(_seed_messages(), response_format="json")
    assert "install" in str(ei.value).lower()


# --------------------------------------------------------------------------- #
# extract_json helper (now in base)
# --------------------------------------------------------------------------- #


def test_extract_json_strips_fences_and_prose() -> None:
    assert extract_json('```json\n{"kind": "x"}\n```') == {"kind": "x"}
    assert extract_json('Sure! Here it is: {"a": 1} done.') == {"a": 1}
    assert extract_json('{"nested": {"y": 2}}') == {"nested": {"y": 2}}


def test_extract_json_raises_on_no_object() -> None:
    with pytest.raises(LLMError):
        extract_json("not json at all")
    with pytest.raises(LLMError):
        extract_json("```text\nplain\n```")