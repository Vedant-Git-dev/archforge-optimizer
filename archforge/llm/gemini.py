"""Google Gemini provider — an `LLMClient` over the `google-genai` SDK (Phase 11).

Gemini's API is not OpenAI-compatible: the system instruction is a separate
parameter and the message array forbids a `system` role, so this adapter
specializes the request/response mapping rather than sharing
`openai_style_complete`. The SDK is imported lazily inside `__init__`.

Uses `google-genai` (the unified `from google import genai` client) per Google's
current Python SDK. JSON output is requested via
`response_mime_type="application/json"` when `response_format="json"`, with the
shared `extract_json` as a parse backstop.
"""

from __future__ import annotations

from typing import Any, Sequence

from archforge.constants import DEFAULT_MODELS
from archforge.llm._common import extract_json, split_system
from archforge.llm.base import Completion, LLMError, Message, Usage

DEFAULT_MODEL = DEFAULT_MODELS["gemini"]


class GeminiClient:
    """An `LLMClient` backed by the google-genai SDK (model family: Gemini)."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        # `base_url` is accepted for seam uniformity with the other adapters
        # (the CLI's `--base-url` passes through to every provider). The
        # google-genai client keys endpoints per-model rather than via a single
        # base_url in v1, so it is currently unused here — kept in the signature
        # so `make_client("gemini", api_key=, base_url=)` does not TypeError.
        del base_url
        try:
            from google import genai
        except ImportError as e:
            raise LLMError(
                "the 'gemini' provider needs the SDK: pip install google-genai"
            ) from e
        self._genai = genai
        try:
            # genai.Client reads GOOGLE_API_KEY / GEMINI_API_KEY from env if omitted
            self._client = genai.Client(api_key=api_key) if api_key is not None else genai.Client()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                "could not initialize the gemini client "
                "(set GOOGLE_API_KEY or GEMINI_API_KEY): " f"{exc}"
            ) from exc

    def complete(
        self, messages: Sequence[Message], *, model: str | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
        response_format: str = "text",
    ) -> Completion:
        m = model or DEFAULT_MODEL
        system, rest = split_system(messages)
        # Gemini orders contents as alternating user/model turns; flatten our
        # messages into its single-string "text" content. (v1 holds a linear chat;
        # richer turns can arrive once a host supplies structured prompts.)
        parts = [msg.content for msg in rest]
        if not parts:
            parts = [""]
        try:
            from google.genai import types
        except ImportError as e:
            raise LLMError("gemini provider needs the full google-genai SDK") from e

        config_kwargs: dict[str, Any] = {}
        if system:
            config_kwargs["system_instruction"] = system
        if temperature is not None:
            config_kwargs["temperature"] = float(temperature)
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = int(max_tokens)
        if response_format == "json":
            config_kwargs["response_mime_type"] = "application/json"

        try:
            resp = self._client.models.generate_content(
                model=m, contents="\n\n".join(parts),
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:  # noqa: BLE001 — any provider failure → LLMError (E9)
            raise LLMError(f"gemini request failed: {exc}") from exc

        text = (resp.text or "") if getattr(resp, "text", None) else ""
        parsed = extract_json(text) if (response_format == "json" and text) else None

        in_tok = out_tok = 0
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
            out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
        return Completion(
            text=text, usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
            model=m, parsed=parsed,
        )


__all__ = ["GeminiClient", "DEFAULT_MODEL"]
