"""Unit tests for the LLM abstraction + ScriptedLLM (Phase 4).

Covers the contract seam: plain-text + JSON completions pop FIFO, the JSON
contract (text request strips parsed; json request parses or raises), the
under-scripting safety net (empty queue -> AssertionError, never silent), and
raise_on_next for grader-outage simulation (E9). Also records requests for
prompt assertions later phases need.
"""

from __future__ import annotations

import pytest

from archforge.llm import Completion, LLMClient, LLMError, Message, Role, ScriptedLLM
from archforge.llm.scripted import ScriptedCall


def _msgs() -> list[Message]:
    return [Message(role=Role.SYSTEM, content="sys"), Message(role=Role.USER, content="hi")]


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #


def test_scripted_llm_satisfies_llmclient_protocol() -> None:
    llm = ScriptedLLM()
    assert isinstance(llm, LLMClient)  # runtime_checkable protocol


# --------------------------------------------------------------------------- #
# FIFO + plain text
# --------------------------------------------------------------------------- #


def test_respond_text_pops_fifo() -> None:
    llm = ScriptedLLM().respond("first").respond("second")
    assert llm.complete(_msgs()).text == "first"
    assert llm.complete(_msgs()).text == "second"


def test_text_request_strips_parsed_payload() -> None:
    # If someone queued JSON but the caller asked for text, parsed is cleared.
    llm = ScriptedLLM().respond_json({"x": 1})
    c = llm.complete(_msgs(), response_format="text")
    assert c.parsed is None


def test_usage_passes_through() -> None:
    llm = ScriptedLLM().respond("ok", input_tokens=5, output_tokens=3)
    c = llm.complete(_msgs())
    assert (c.usage.input_tokens, c.usage.output_tokens, c.usage.total) == (5, 3, 8)


# --------------------------------------------------------------------------- #
# JSON contract
# --------------------------------------------------------------------------- #


def test_respond_json_returns_parsed_dict() -> None:
    llm = ScriptedLLM().respond_json({"score": 0.7, "ok": True})
    c = llm.complete(_msgs(), response_format="json")
    assert c.parsed == {"score": 0.7, "ok": True}
    assert c.text  # still serialised text too


def test_json_request_parses_queued_text() -> None:
    # A plain-text response that is valid JSON satisfies a json request.
    llm = ScriptedLLM().respond('{"k": 9}')
    c = llm.complete(_msgs(), response_format="json")
    assert c.parsed == {"k": 9}


def test_json_request_raises_on_non_json() -> None:
    llm = ScriptedLLM().respond("not json at all")
    with pytest.raises(LLMError):
        llm.complete(_msgs(), response_format="json")


# --------------------------------------------------------------------------- #
# Under-scripting safety net
# --------------------------------------------------------------------------- #


def test_empty_queue_raises_assertion_not_silent() -> None:
    llm = ScriptedLLM()
    with pytest.raises(AssertionError):
        llm.complete(_msgs())


def test_pending_counts_queued() -> None:
    llm = ScriptedLLM().respond("a").respond("b")
    assert llm.pending == 2
    llm.complete(_msgs())
    assert llm.pending == 1
    llm.reset()
    assert llm.pending == 0


# --------------------------------------------------------------------------- #
# Raise-on-next (E9 grader outage)
# --------------------------------------------------------------------------- #


def test_raise_on_next_propagates() -> None:
    llm = ScriptedLLM().raise_on_next(LLMError("rate limited"))
    with pytest.raises(LLMError, match="rate limited"):
        llm.complete(_msgs())
    # the exception is consumed; next call needs a new response
    assert llm.pending == 0


def test_raise_on_next_accepts_exception_class() -> None:
    llm = ScriptedLLM().raise_on_next(LLMError)
    with pytest.raises(LLMError):
        llm.complete(_msgs())


# --------------------------------------------------------------------------- #
# Request recording
# --------------------------------------------------------------------------- #


def test_calls_record_requests() -> None:
    llm = ScriptedLLM().respond("a").respond_json({"b": 1})
    llm.complete(_msgs(), model="gpt-4o", temperature=0.1, response_format="text")
    llm.complete(_msgs(), model="claude", response_format="json")
    assert len(llm.calls) == 2
    assert llm.calls[0].model == "gpt-4o" and llm.calls[0].temperature == 0.1
    assert llm.calls[0].response_format == "text"
    assert llm.calls[1].response_format == "json"
    # the messages were captured
    assert isinstance(llm.calls[0], ScriptedCall)
    assert [m.role for m in llm.calls[0].messages] == [Role.SYSTEM, Role.USER]


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_identical_scripts_identical_behaviour() -> None:
    def run() -> tuple[str, dict]:
        llm = ScriptedLLM().respond("text").respond_json({"s": 0.5})
        t = llm.complete(_msgs(), response_format="text").text
        j = llm.complete(_msgs(), response_format="json").parsed
        return t, dict(j)  # type: ignore[arg-type]

    assert run() == run()
