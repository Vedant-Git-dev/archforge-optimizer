"""The unified real-LLM provider — an `LLMClient` over LiteLLM.

This is ArchForge's ONE real provider adapter (issue #1). Instead of a bespoke
adapter per vendor, every real provider (openai / anthropic / groq / gemini, and
any LiteLLM-recognized one) is reached through a single `litellm.completion(...)`
call by prefixing the bare model id with the provider (`openai/gpt-4o`,
`anthropic/claude-sonnet-5`, ...). Adding a provider = a new model prefix, no new
adapter.

Keeps the `LLMClient` seam / `Completion` / `Usage` / `LLMError` contract: the
Judge, Architect, runner and CLI depend only on that protocol. LiteLLM is imported
LAZILY inside `__init__`, so importing `archforge` / `archforge.llm` never pulls
LiteLLM (or any provider SDK) — only a real `complete()` does.

LiteLLM specifics this relies on (verified against the installed version):
  * response accessors: `resp.choices[0].message.content` (raw text string, may be
    None) and `resp.usage.prompt_tokens` / `.completion_tokens` (OpenAI-style;
    there is no input/output alias).
  * explicit provider prefixes are REQUIRED — an unprefixed `gemini-*` resolves to
    `vertex_ai` and unprefixed anthropic/groq raise; so bare config model ids are
    prefixed here.
  * JSON mode passes `response_format={"type": "json_object"}`; LiteLLM shims it
    per-provider (native where available, tool-call / output_format elsewhere) and
    `content` still returns the RAW JSON string — `extract_json` is the parse
    backstop (we do not assume valid JSON).
  * `api_key=` / `base_url=` work per call; standard env vars fall back
    automatically when not passed.
  * `litellm.exceptions.APIError` is the uniform base for every provider's
    RateLimit / Authentication / BadRequest errors — catch it once → `LLMError`.
"""

from __future__ import annotations

from typing import Any, Sequence

from archforge import userconfig as ucfg
from archforge.llm.base import Completion, LLMError, Message, Usage, extract_json

# Provider → the LiteLLM model-string prefix that routes to that vendor's API.
# Current four match the roster in archforge.config (REAL_PROVIDERS); add any
# LiteLLM-recognized provider here (e.g. "azure": "azure") to enable it with a
# model-string prefix and no new adapter.
_PROVIDER_PREFIX: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "groq": "groq",
    "gemini": "gemini",
}


def prefix_model(provider: str, model: str) -> str:
    """Prefix a model id with the provider (``openai/gpt-4o``).

    A model that ALREADY carries THIS provider's prefix (``openai/gpt-4o``
    under provider=openai) is passed through untouched, so fully-qualified
    overrides still work. But a ``/`` alone does NOT mean "already routed" —
    e.g. Groq's model ``openai/gpt-oss-120b`` has a slash in its native id,
    so it must still get the provider prefix (``groq/openai/gpt-oss-120b``)
    or LiteLLM would route it to OpenAI. Prefix unless it already matches.
    """
    prefix = _PROVIDER_PREFIX.get(provider, provider)
    if model.startswith(f"{prefix}/"):
        return model
    return f"{prefix}/{model}"


class LiteLLMClient:
    """An `LLMClient` backed by `litellm.completion` for one provider.

    Resolves the provider's bare default model from the centralized config
    (`DEFAULT_ARCHITECT_MODELS`) and prefixes it with the provider so LiteLLM
    routes to the right API.
    """

    def __init__(self, *, provider: str, api_key: str | None = None,
                 base_url: str | None = None) -> None:
        if provider not in _PROVIDER_PREFIX:
            raise LLMError(f"unsupported LiteLLM provider {provider!r}")
        self._provider = provider
        self._api_key = api_key
        self._base_url = base_url

    def _prefixed(self, model: str) -> str:
        """Prefix a model id with this client's provider (see `prefix_model`)."""
        return prefix_model(self._provider, model)

    def _default_model(self) -> str:
        """The provider's default model id for a bare complete() call, resolved
        lazily from the active config — the Architect (proposer) dict (the
        role-specific defaults are resolved by the runner/CLI from their own
        role dicts)."""
        return ucfg.get("DEFAULT_ARCHITECT_MODELS")[self._provider]

    def complete(
        self, messages: Sequence[Message], *, model: str | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
        response_format: str = "text",
    ) -> Completion:
        try:
            import litellm  # lazy: importing archforge must stay LiteLLM-free
        except ImportError as e:
            raise LLMError(
                "real providers need LiteLLM: pip install archforge-optimizer[litellm]"
            ) from e

        resolved = model or self._default_model()
        prefixed = self._prefixed(resolved)
        kwargs: dict[str, Any] = {
            "model": prefixed,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url

        try:
            resp = litellm.completion(**kwargs)
        except Exception as exc:  # noqa: BLE001 — surface any LiteLLM error as LLMError (E9)
            # litellm.exceptions.APIError is the uniform base for all provider
            # errors (RateLimitError, AuthenticationError, BadRequestError, ...);
            # a bare Exception here is the safe catch-all so a provider outage
            # still surfaces as one clear LLMError.
            raise LLMError(f"{self._provider} request failed: {exc}") from exc

        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        message = getattr(choice, "message", None) if choice is not None else None
        text = (getattr(message, "content", None) or "") if message is not None else ""
        usage_obj = getattr(resp, "usage", None)
        in_tok = getattr(usage_obj, "prompt_tokens", 0) or 0
        out_tok = getattr(usage_obj, "completion_tokens", 0) or 0
        parsed = extract_json(text) if (response_format == "json" and text) else None
        return Completion(
            text=text,
            usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
            model=prefixed, parsed=parsed,
        )


__all__ = ["LiteLLMClient", "prefix_model"]
