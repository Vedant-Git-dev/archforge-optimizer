"""Anthropic provider — an `LLMClient` over the `anthropic` SDK.

The SDK is imported lazily inside `__init__`, so importing this module (and thus
`archforge.llm`) never requires the package to be installed — only an actual run
with `--provider anthropic` does, at which point a missing/misconfigured SDK
surfaces as a clear `LLMError` rather than a bare `ImportError`.

`Judge`/`Architect` already instruct the model to "Return ONLY JSON"; Anthropic's
messages API has no native JSON mode, so we parse the text with the shared
`extract_json` fence/prose stripper. `max_tokens` is required by Anthropic and
defaults to 1024 when the caller omits it; `temperature` is clamped to [0, 1].
"""

from __future__ import annotations

from typing import Any, Sequence

from archforge import userconfig as ucfg
from archforge.config import ANTHROPIC_DEFAULT_MAX_TOKENS
from archforge.llm._common import extract_json, split_system
from archforge.llm.base import Completion, LLMError, Message, Usage


def _default_model() -> str:
    """The provider's default model id, resolved lazily from the active config."""
    return ucfg.get("DEFAULT_MODELS")["anthropic"]


class AnthropicClient:
    """An `LLMClient` backed by the Anthropic SDK (model family: Claude)."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise LLMError(
                "the 'anthropic' provider needs the SDK: pip install anthropic"
            ) from e
        try:
            self._client = Anthropic(api_key=api_key, base_url=base_url)
        except Exception as exc:  # noqa: BLE001 — construction (e.g. no key) → LLMError
            raise LLMError(
                "could not initialize the anthropic client (set ANTHROPIC_API_KEY): "
                f"{exc}"
            ) from exc

    def complete(
        self, messages: Sequence[Message], *, model: str | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
        response_format: str = "text",
    ) -> Completion:
        m = model or _default_model()
        system, rest = split_system(messages)
        body = [
            {"role": ("user" if msg.role.value == "user" else "assistant"), "content": msg.content}
            for msg in rest
        ] or [{"role": "user", "content": ""}]
        kwargs: dict[str, Any] = {
            "model": m, "max_tokens": max_tokens if max_tokens is not None else ANTHROPIC_DEFAULT_MAX_TOKENS,
            "messages": body,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = max(0.0, min(1.0, float(temperature)))
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — any provider failure → LLMError (E9 retries)
            raise LLMError(f"anthropic request failed: {exc}") from exc

        text = resp.content[0].text if getattr(resp, "content", None) else ""
        usage = getattr(resp, "usage", None)
        parsed = extract_json(text) if (response_format == "json" and text) else None
        return Completion(
            text=text,
            usage=Usage(input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                        output_tokens=int(getattr(usage, "output_tokens", 0) or 0)),
            model=m, parsed=parsed,
        )


__all__ = ["AnthropicClient", "DEFAULT_MODEL"]


def __getattr__(name: str):  # PEP 562
    if name == "DEFAULT_MODEL":
        return _default_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
