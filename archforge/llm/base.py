"""The LLM client contract.

A provider (∈ {ScriptedLLM, Anthropic, OpenAI}) implements `LLMClient`. The
Judge and Architect depend on this protocol only, never on a concrete provider.

Design choices:
  * `Message` is minimal (role + content) — providers translate to their native
    schema. Wrapped-agent prompts/system_prompts flow here as messages.
  * `response_format="json"` is the structured-output hint: the Judge and
    Architect use JSON for rubric scores and proposed changes. The client is
    expected to populate `Completion.parsed` when JSON is requested (and to raise
    `LLMError` on a parse failure rather than silently returning text).
  * `Usage` carries token counts so cost tracking (E3) and adaptive R work.
  * `LLMError` is the single wrapped exception type for any provider failure
    (rate limit, outage, malformed response) — the Judge/Architect retry on it
    (E9) instead of each provider's bespoke errors.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """One conversational turn. Content is plain text (no multimodal in v1)."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str


class Usage(BaseModel):
    """Token accounting for cost tracking (spec E3)."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class Completion(BaseModel):
    """A model completion: raw text + optional parsed JSON + token usage."""

    model_config = ConfigDict(extra="allow")

    text: str
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    # Populated when `response_format="json"` was requested and parsing succeeded.
    parsed: dict | None = None


class LLMError(RuntimeError):
    """Any provider failure (rate limit, outage, malformed JSON).

    The Judge's grader calls raise this (E9); downstream callers retry with
    backoff rather than handling each provider's exception types.
    """


@runtime_checkable
class LLMClient(Protocol):
    """The contract a provider (scripted or real) must satisfy."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str = "text",
    ) -> Completion: ...


__all__ = ["Role", "Message", "Usage", "Completion", "LLMError", "LLMClient"]
