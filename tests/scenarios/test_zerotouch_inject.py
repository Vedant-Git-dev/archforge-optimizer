"""Zero-touch injector E2E — a real LangGraph run, a real vote, a fake SDK.

The missing link between ``tests/unit/test_inject.py`` (mechanics against fake
SDKs) and a real host: a ``LangGraphApp`` that does NOT override
``apply_llm_config`` (the zero-touch shape) drives a REAL compiled LangGraph
whose single LLM node calls the SDK boundary; a candidate Spec carrying a
``model_swap`` must land in the outbound request WITHOUT any host-side config
seam. The SDK is a fake ``groq`` installed in ``sys.modules`` for the test's
duration only (``monkeypatch`` restores it, so a real groq installed in the
environment is untouched elsewhere).

Also asserted: the incumbent run (seeded ``model=""``) leaves the request's
model alone (the vote's model is falsy → passthrough), the seeded temperature
from the live Spec IS applied (cfg_decay returns named knobs always — seeded
values equal the host's defaults on a base run), and the observed system
prompt is CAPTURED into the app's ``captured_prompts`` (the base a later
``prompt_edit`` diffs against).
"""
from __future__ import annotations

import sys
import types as pytypes

import pytest

import archforge.models as m
from archforge.host.adapters import LangGraphApp, LangGraphHostAdapter, Nd
from archforge.host.base import Task
from archforge.middleware import TracingMiddleware
from archforge.stores import TraceStore

pytest.importorskip("langgraph")
from langgraph.graph import END, StateGraph  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402


class _State(TypedDict, total=False):
    query: str
    answer: str


class _FakeCompletions:
    """Stands in for groq's Completions; records create() kwargs on the CLASS
    so tests can see what a node-constructed client sent."""

    calls: list[dict] = []

    def create(self, **kwargs):
        type(self).calls.append(dict(kwargs))
        return "the answer"


# The node function lives in its OWN synthetic module: NodeLocator attributes
# by the call frame's module, so the one-module-per-node layout is
# reproduced here — a fn defined in this test module would attribute every
# node to the same id.
_NODE_MOD_NAME = "zerotouch_demo.nodes.answer"
_mod = pytypes.ModuleType(_NODE_MOD_NAME)
_mod.__dict__["__name__"] = _NODE_MOD_NAME
sys.modules[_NODE_MOD_NAME] = _mod
exec(
    "def answer_node(state):\n"
    "    from groq.resources.chat.completions import Completions\n"
    "    client = Completions()\n"
    "    text = client.create(\n"
    "        model='seeded-model',\n"
    "        messages=[{'role': 'system', 'content': 'SEEDED PROMPT'},\n"
    "                  {'role': 'user', 'content': state['query']}],\n"
    "        temperature=0.9)\n"
    "    return {'answer': text}\n",
    _mod.__dict__,
)


def _build_graph():
    g = StateGraph(_State)
    g.add_node("answer", _mod.answer_node)  # type: ignore[attr-defined]
    g.set_entry_point("answer")
    g.add_edge("answer", END)
    return g.compile()


class ZeroTouchApp(LangGraphApp):
    """No apply_llm_config override — the default records votes and the
    injector does the rest."""

    graph_factory = staticmethod(_build_graph)
    final_output_key = "answer"
    nodes = [Nd("answer", "final answerer", m.NodeKind.LLM,
                knobs=m.Knobs(temperature=0.3, max_tokens=1024))]

    def initialize_state(self, task_input: str) -> dict:
        return {"query": task_input, "answer": ""}


@pytest.fixture
def fake_groq(monkeypatch):
    _FakeCompletions.calls = []
    chat_mod = pytypes.ModuleType("groq.resources.chat.completions")
    chat_mod.Completions = _FakeCompletions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "groq.resources.chat.completions", chat_mod)
    return _FakeCompletions.calls


def test_incumbent_run_passthrough_model_and_captures_prompt(
        fake_groq, tmp_path) -> None:
    app = ZeroTouchApp()
    adapter = LangGraphHostAdapter(app)
    mw = TracingMiddleware(TraceStore(tmp_path / ".archforge"))
    trace = adapter.instantiate(adapter.app_spec(), mw).run(
        Task(task_id="t1", input="q"))

    assert trace.ok
    assert len(fake_groq) == 1
    sent = fake_groq[0]
    assert sent["model"] == "seeded-model"     # vote.model falsy → untouched
    assert sent["temperature"] == 0.3          # seeded knob applied (not 0.9)
    # prompt capture: the observed system prompt becomes the cfg_decay base
    assert app.captured_prompts.get("answer") == "SEEDED PROMPT"
    assert adapter.base_prompts.get("answer") == "SEEDED PROMPT"


def test_step_tokens_come_from_real_usage(fake_groq, tmp_path, monkeypatch) -> None:
    """With a usage-carrying response, the step's cost is the metered total,
    not the len//4 estimate."""
    import types as _t

    def create_with_usage(self, **kwargs):
        type(self).calls.append(dict(kwargs))
        return _t.SimpleNamespace(
            usage=_t.SimpleNamespace(prompt_tokens=30, completion_tokens=12,
                                     total_tokens=42))

    monkeypatch.setattr(_FakeCompletions, "create", create_with_usage)
    app = ZeroTouchApp()
    adapter = LangGraphHostAdapter(app)
    mw = TracingMiddleware(TraceStore(tmp_path / ".archforge"))
    trace = adapter.instantiate(adapter.app_spec(), mw).run(
        Task(task_id="t1", input="q"))

    assert trace.ok
    step = next(s for s in trace.steps if s.node_id == "answer")
    assert step.perf.tokens == 42  # metered, not len("...")//4


def test_model_swap_lands_on_the_wire(fake_groq, tmp_path) -> None:
    app = ZeroTouchApp()
    adapter = LangGraphHostAdapter(app)
    spec = adapter.app_spec().model_copy(deep=True)
    for n in spec.nodes:
        if n.node_id == "answer":
            n.model = "swapped-model"          # the Architect's model_swap

    mw = TracingMiddleware(TraceStore(tmp_path / ".archforge"))
    trace = adapter.instantiate(spec, mw).run(Task(task_id="t1", input="q"))

    assert trace.ok
    assert fake_groq[0]["model"] == "swapped-model"
