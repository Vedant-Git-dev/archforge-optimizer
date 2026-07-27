"""Shared helpers for the real-LLM provider adapters (Phase 11).

Provider-agnostic plumbing so the four adapters (anthropic/openai/groq/gemini)
share one JSON-extraction + message-shape logic. None of this touches an SDK at
import time — adapters import their SDK lazily inside `__init__`, so importing
`archforge.llm` is free of provider dependencies.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from archforge.llm.base import Completion, LLMError, Message, Role, Usage


def split_system(messages: Sequence[Message]) -> tuple[str, list[Message]]:
    """Pull SYSTEM messages out of the list.

    Anthropic and Gemini take the system instruction as a separate parameter
    (their message/contents arrays forbid a `system` role); OpenAI/Groq accept
    it inline. Returns (joined-system-text, non-system-messages).
    """

    sys_parts = [m.content for m in messages if m.role is Role.SYSTEM]
    rest = [m for m in messages if m.role is not Role.SYSTEM]
    return "\n\n".join(sys_parts), rest


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction: strip code fences + surrounding prose, parse.

    The Judge/Architect prompts already demand "Return ONLY JSON"; this is the
    safety net that survives a model that wraps its object in a ```json fence or
    adds a leading sentence. Raises `LLMError` on a genuine parse failure (the
    caller-must-not-fabricate contract), never returns a half-formed dict.
    """

    t = (text or "").strip()
    if t.startswith("```"):
        # drop an opening ```lang fence, and its matching closer if present
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t[3:]
        end = t.rfind("```")
        if end != -1:
            t = t[:end]
        t = t.strip()
    start, stop = t.find("{"), t.rfind("}")
    if start == -1 or stop == -1 or stop < start:
        raise LLMError(f"expected a JSON object, found none in model output: {text[:200]!r}")
    try:
        return json.loads(t[start:stop + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"model output was not valid JSON ({exc}): {text[:200]!r}") from exc


def openai_style_complete(
    sdk_client, *, provider: str, default_model: str,
    messages: Sequence[Message], model: str | None, temperature: float | None,
    max_tokens: int | None, response_format: str,
) -> Completion:
    """Shared body for OpenAI-compatible Chat Completions APIs (OpenAI + Groq).

    Both SDKs mirror the OpenAI Python client shape, so the create-call,
    response unpacking, and usage mapping are identical. `response_format="json"`
    maps to the providers' native `{"type": "json_object"}` mode where present.
    """

    msgs = [{"role": m.role.value, "content": m.content} for m in messages]
    kwargs: dict[str, Any] = {"model": model or default_model, "messages": msgs}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = sdk_client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 — surface any SDK error as LLMError (E9)
        raise LLMError(f"{provider} request failed: {exc}") from exc

    text = (resp.choices[0].message.content or "") if getattr(resp, "choices", None) else ""
    usage_obj = getattr(resp, "usage", None)
    in_tok = getattr(usage_obj, "prompt_tokens", 0) or 0
    out_tok = getattr(usage_obj, "completion_tokens", 0) or 0
    parsed = extract_json(text) if (response_format == "json" and text) else None
    return Completion(
        text=text,
        usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
        model=model or default_model, parsed=parsed,
    )


__all__ = ["split_system", "extract_json", "openai_style_complete"]
