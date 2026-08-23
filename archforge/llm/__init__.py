"""Provider-agnostic LLM abstraction (spec §3 Judge/Architect).

Both thinking components — the Judge and the Architect — talk
to a model only through the `LLMClient` protocol. This is the **single seam**
where "real vs fake" lives: tests plug in `ScriptedLLM` (deterministic, free,
crashable); a run plugs in a real provider via `make_client(provider)`.

Every real provider is reached through ONE adapter — `LiteLLMClient` (issue #1) —
which routes to the vendor's API by prefixing the model id (`openai/`, `anthropic/`,
`groq/`, `gemini/`). LiteLLM is imported lazily inside the client's `complete()`, so
importing THIS package never requires LiteLLM or any provider SDK to be installed —
only an actual run with a real provider does, at which point a missing LiteLLM
surfaces as a clear `LLMError` rather than a bare `ImportError`. `make_client`
resolves the client lazily; the class is not re-exported here to keep the import
graph SDK-free.
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

    Resolves `LiteLLMClient` lazily (so importing LiteLLM is deferred to here /
    to the first `complete()`). Raises `LLMError` for an unknown provider so the
    CLI surfaces one clear message across the provider seam. `kwargs`
    (api_key/base_url) pass straight through.
    """

    if provider == "scripted":
        raise LLMError("'scripted' has no real client; use ScriptedLLM directly")
    if provider not in REAL_PROVIDERS:
        raise LLMError(f"unknown LLM provider {provider!r}; expected one of {REAL_PROVIDERS}")
    from archforge.llm.litellm import LiteLLMClient

    return LiteLLMClient(provider=provider, **kwargs)  # type: ignore[no-any-return]


__all__ = [
    "Completion", "LLMClient", "LLMError", "Message", "Role", "Usage", "ScriptedLLM",
    "make_client", "REAL_PROVIDERS", "ALL_PROVIDERS",
]
