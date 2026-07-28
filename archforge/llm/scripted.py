"""ScriptedLLM — a programmable, deterministic fake `LLMClient`.

This is what makes every "thinking" component testable without a real model.

Usage pattern (one queue, FIFO):
    llm = ScriptedLLM()
    llm.respond("plan A")                # next complete() returns this text
    llm.respond_json({"score": 0.7})     # next complete() returns parsed JSON
    llm.raise_on_next(LLMError("boom"))   # next complete() raises (E9 grader outage)

Each call to `complete()` pops the queue front; an empty queue is a test bug
(AssertionError) so silent under-scripting fails loudly rather than returning
garbage. All requests are recorded on `.calls` for assertions about prompts,
model, temperature, and response_format.

Determinism is paramount: identical scripts → identical behaviour, which is the
foundation for the R-repeat aggregation tests (E1) and every scenario test.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from archforge.llm.base import Completion, LLMError, Message, Usage


class ScriptedCall(BaseModel):
    """A recorded `complete()` request — for asserting what prompt was sent."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: str = "text"


class ScriptedLLM:
    """A fake `LLMClient` with a FIFO queue of canned responses or exceptions."""

    def __init__(self) -> None:
        self._queue: list[Completion | Exception] = []
        self.calls: list[ScriptedCall] = []

    # --------------------------------------------------------------- builders
    def respond(
        self, text: str, *, model: str | None = None,
        input_tokens: int = 0, output_tokens: int = 0,
    ) -> "ScriptedLLM":
        """Queue a plain-text completion (next complete() returns it)."""

        self._queue.append(
            Completion(text=text, model=model,
                       usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens))
        )
        return self

    def respond_json(
        self, obj: dict[str, Any], *, model: str | None = None,
        input_tokens: int = 0, output_tokens: int = 0,
    ) -> "ScriptedLLM":
        """Queue a JSON completion with `.parsed` populated (text is json.dumps)."""

        self._queue.append(
            Completion(
                text=json.dumps(obj, sort_keys=True),
                parsed=dict(obj),
                model=model,
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            )
        )
        return self

    def raise_on_next(self, exc: Exception | type[Exception]) -> "ScriptedLLM":
        """Make the next complete() raise `exc` (E9 grader outage, etc.)."""

        err = exc if isinstance(exc, Exception) else exc()
        self._queue.append(err)
        return self

    def reset(self) -> None:
        self._queue.clear()
        self.calls = []

    # --------------------------------------------------------------- contract
    @property
    def pending(self) -> int:
        """Responses still queued — handy for asserting all were consumed."""

        return len(self._queue)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str = "text",
    ) -> Completion:
        record = ScriptedCall(
            messages=list(messages), model=model, temperature=temperature,
            max_tokens=max_tokens, response_format=response_format,
        )
        self.calls.append(record)

        if not self._queue:
            raise AssertionError(
                "ScriptedLLM ran out of queued responses — under-scripted test "
                f"(call #{len(self.calls)} asked for {response_format!r} model={model!r})"
            )
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item

        # Enforce the JSON contract: JSON requested -> `parsed` must resolve.
        if response_format == "json" and item.parsed is None:
            try:
                item = item.model_copy(update={"parsed": json.loads(item.text)})
            except (json.JSONDecodeError, TypeError) as exc:
                raise LLMError(
                    f"ScriptedLLM: caller asked for JSON but got non-JSON text: {exc}"
                ) from exc
        # Caller asked for text: clear any parsed payload for contract consistency.
        if response_format != "json" and item.parsed is not None:
            item = item.model_copy(update={"parsed": None})
        return item


__all__ = ["ScriptedLLM", "ScriptedCall"]
