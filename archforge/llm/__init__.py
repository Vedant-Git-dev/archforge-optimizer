"""Provider-agnostic LLM abstraction (spec §3 Judge/Architect).

Both thinking components — the Judge and the Architect — talk
to a model only through the `LLMClient` protocol. This is the **single seam**
where "real vs fake" lives: tests plug in `ScriptedLLM` (deterministic, free,
crashable); a run plugs in a real provider via `make_client(provider)`.

The real provider adapters (`anthropic`, `openai`, `groq`, `gemini`) import their
SDKs lazily inside `__init__`, so importing THIS package never requires any
provider SDK to be installed — only an actual run with that provider does, at
which point a missing SDK surfaces as a clear `LLMError` rather than a bare
`ImportError`. The adapter classes are NOT re-exported here to keep the import
graph SDK-free; `make_client(provider)` resolves them by name on demand.
"""

from __future__ import annotations

from typing import Any

from archforge.config import ALL_PROVIDERS, REAL_PROVIDERS
from archforge.llm.base import (
    Completion,
    LLMClient,
    LLMError,
    Message,
    Role,
    Usage,
)
from archforge.llm.scripted import ScriptedLLM

# `REAL_PROVIDERS` / `ALL_PROVIDERS` are re-exported (below in __all__) straight
# from archforge.config — the single source of truth the CLI also imports.


def make_client(provider: str, **kwargs: Any) -> LLMClient:
    """Build a real `LLMClient` for `provider`.

    Resolves the adapter lazily (importing the SDK is deferred to here). Raises
    `LLMError` for an unknown provider so the CLI surfaces one clear message
    across the provider seam. `kwargs` (api_key/base_url) pass straight through.
    """

    if provider == "scripted":
        raise LLMError("'scripted' has no real client; use ScriptedLLM directly")
    table = {
        "anthropic": "archforge.llm.anthropic:AnthropicClient",
        "openai": "archforge.llm.openai:OpenAIClient",
        "groq": "archforge.llm.groq:GroqClient",
        "gemini": "archforge.llm.gemini:GeminiClient",
    }
    ref = table.get(provider)
    if ref is None:
        raise LLMError(f"unknown LLM provider {provider!r}; expected one of {REAL_PROVIDERS}")
    mod_name, cls_name = ref.split(":")
    import importlib

    module = importlib.import_module(mod_name)
    cls = getattr(module, cls_name)
    return cls(**kwargs)  # type: ignore[no-any-return]


__all__ = [
    "Completion", "LLMClient", "LLMError", "Message", "Role", "Usage", "ScriptedLLM",
    "make_client", "REAL_PROVIDERS", "ALL_PROVIDERS",
]
