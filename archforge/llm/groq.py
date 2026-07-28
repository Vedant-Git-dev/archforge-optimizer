"""Groq provider — an `LLMClient` over the `groq` SDK (Phase 11).

Groq runs OpenAI-compatible Chat Completions, so this adapter shares
`openai_style_complete` with the OpenAI provider; only the SDK client and default
model differ. `response_format={"type": "json_object"}` is supported on Groq's
flagship llama/mixtral models. Lazy import, `LLMError` on any failure.
"""

from __future__ import annotations

from typing import Sequence

from archforge.constants import DEFAULT_MODELS
from archforge.llm._common import openai_style_complete
from archforge.llm.base import Completion, LLMError, Message

DEFAULT_MODEL = DEFAULT_MODELS["groq"]


class GroqClient:
    """An `LLMClient` backed by the Groq SDK (OpenAI-compatible API)."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            from groq import Groq
        except ImportError as e:
            raise LLMError("the 'groq' provider needs the SDK: pip install groq") from e
        try:
            kwargs: dict[str, object] = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            self._client = Groq(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"could not initialize the groq client (set GROQ_API_KEY): {exc}"
            ) from exc

    def complete(
        self, messages: Sequence[Message], *, model: str | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
        response_format: str = "text",
    ) -> Completion:
        return openai_style_complete(
            self._client, provider="groq", default_model=DEFAULT_MODEL, messages=messages,
            model=model, temperature=temperature, max_tokens=max_tokens,
            response_format=response_format,
        )


__all__ = ["GroqClient", "DEFAULT_MODEL"]
