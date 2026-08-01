"""OpenAI provider — an `LLMClient` over the `openai` SDK.

Lazy SDK import like the other adapters. Uses the native
`response_format={"type": "json_object"}` JSON mode when JSON is requested, with
the shared `extract_json` as the parse backstop. Messages keep their `system`
role inline (OpenAI's Chat Completions accept it).
"""

from __future__ import annotations

from typing import Sequence

from archforge import userconfig as ucfg
from archforge.llm._common import openai_style_complete
from archforge.llm.base import Completion, LLMError, Message


def _default_model() -> str:
    """The provider's default model id, resolved lazily from the active config."""
    return ucfg.get("DEFAULT_MODELS")["openai"]


class OpenAIClient:
    """An `LLMClient` backed by the OpenAI SDK (model family: GPT)."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError("the 'openai' provider needs the SDK: pip install openai") from e
        try:
            kwargs: dict[str, object] = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"could not initialize the openai client (set OPENAI_API_KEY): {exc}"
            ) from exc

    def complete(
        self, messages: Sequence[Message], *, model: str | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
        response_format: str = "text",
    ) -> Completion:
        return openai_style_complete(
            self._client, provider="openai", default_model=_default_model(), messages=messages,
            model=model, temperature=temperature, max_tokens=max_tokens,
            response_format=response_format,
        )


__all__ = ["OpenAIClient", "DEFAULT_MODEL"]


def __getattr__(name: str):  # PEP 562
    if name == "DEFAULT_MODEL":
        return _default_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
