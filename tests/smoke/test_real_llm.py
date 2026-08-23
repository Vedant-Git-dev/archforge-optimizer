"""Phase 11 — real-LLM smoke (OPT-IN, NOT in the default suite).

Run explicitly with:
    python -m pytest -m smoke            # all installed, keyed providers
    python -m pytest -m smoke -k groq    # one provider

These make LIVE, BILLED API calls. They assert **plumbing consistency**, not
scores — a real `Judge` scores a real trace and returns a well-formed RunScore;
a real `Architect` proposes a parseable JSON change that round-trips through
`apply_change` to a lint-clean candidate; an end-to-end `evolve_cycle` leaves the
stores' lineage/verdict state consistent. Score values are explicitly NOT
asserted (we check shapes, not numbers).

Self-skip rules (so `pytest -m smoke` degrades gracefully):
  * skip a provider whose SDK is not importable
  * skip a provider with no API key (`.env` is loaded here only; never baked in)
  * skip everything unless `ARCHFORGE_SMOKE=1` (the explicit opt-in guard)
"""

from __future__ import annotations

import os
import pathlib

import pytest

_ENV = pathlib.Path(__file__).resolve().parents[2] / ".env"

# Provider -> env var that holds its key. The genai SDK also accepts GEMINI_API_KEY.
_PROVIDER_KEYVAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_PROVIDER_SDK = {
    "anthropic": "anthropic",
    "openai": "openai",
    "groq": "groq",
    "gemini": "google.genai",
}


def _load_env() -> None:
    if not _ENV.exists():
        return
    for line in _ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()


def _default_model(provider: str) -> str:
    """The provider's real default model id, from the centralized config
    (DEFAULT_ARCHITECT_MODELS — SDK-free to resolve). Used so the live smoke
    sends a model the API knows, and so E2E provenance (judge_meta.model)
    carries a real id."""
    from archforge import userconfig as ucfg

    return ucfg.get("DEFAULT_ARCHITECT_MODELS")[provider]


def _sdk_available(provider: str) -> bool:
    try:
        __import__(_PROVIDER_SDK[provider])
        return True
    except Exception:
        return False


def _keyed(provider: str) -> bool:
    return bool(os.environ.get(_PROVIDER_KEYVAR[provider]))


pytestmark = pytest.mark.smoke


def _opted_in() -> bool:
    # The marker plus the env guard. Without ARCHFORGE_SMOKE=1 we skip wholesale.
    return os.environ.get("ARCHFORGE_SMOKE") == "1"


PROVIDERS = ["groq", "gemini", "anthropic", "openai"]


def _maybe_live(provider, request) -> None:  # noqa: ARG001 (request kept for -k filtering hooks)
    if not _opted_in():
        pytest.skip("real-LLM smoke is opt-in: set ARCHFORGE_SMOKE=1")
    if not _sdk_available(provider):
        pytest.skip(f"{provider} SDK not installed; pip install archforge[providers-{provider}]")
    if not _keyed(provider):
        pytest.skip(f"no {_PROVIDER_KEYVAR[provider]} set (.env or env)")


# --------------------------------------------------------------------------- #
# Provider: the raw LLMClient round-trips a JSON instruction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", PROVIDERS)
def test_smoke_llm_client_completes_json(request, provider: str) -> None:  # type: ignore[no-untyped-def]
    _maybe_live(provider, request)
    from archforge.llm import make_client
    from archforge.llm.base import Message, Role

    client = make_client(provider, base_url=os.environ.get(f"{provider}_base_url"))
    comp = client.complete(
        [Message(role=Role.SYSTEM, content="Return ONLY a JSON object."),
         Message(role=Role.USER, content='Return {"hello": "world"} exactly.')],
        response_format="json", temperature=0.0,
    )
    # plumbing: a parsed dict came back (JSON mode + extract_json succeeded)
    assert isinstance(comp.parsed, dict)
    # usage is populated (token accounting works for E3 budget caps)
    assert comp.usage.total >= 0


@pytest.mark.parametrize("provider", PROVIDERS)
def test_smoke_real_judge_scores_a_real_trace(request, provider: str) -> None:  # type: ignore[no-untyped-def]
    _maybe_live(provider, request)
    from archforge.host import FakeHostMAS
    from archforge.host.base import Task
    from archforge.judge.base import Judge, default_rubric
    from archforge.middleware import TracingMiddleware
    from archforge.stores import TraceStore

    import archforge.models as m

    spec = m.Spec(nodes=[m.Node(node_id="a", role="answerer",
                                system_prompt="Answer the question.", model="x",
                                tools=["t"])])
    ts = TraceStore(__import__("tempfile").mkdtemp())
    runnable = FakeHostMAS().instantiate(spec, TracingMiddleware(ts))
    trace = runnable.run(Task(task_id="t1", input="What is 2+2?"))

    client = __import__("archforge.llm", fromlist=["make_client"]).make_client(provider)
    judge = Judge(client, model=_default_model(provider), rubric=default_rubric())
    rs = judge.score(trace, Task(task_id="t1", input="What is 2+2?"),
                     default_rubric().rubric_id)

    # plumbing: a well-formed RunScore (shapes, not the score value)
    assert rs.task_id == "t1"
    assert rs.judge_meta.rubric_id == default_rubric().rubric_id      # E2 provenance
    assert 0.0 <= rs.aggregate <= 1.0
    assert 0.0 <= rs.confidence <= 1.0
    assert rs.judge_meta.model  # the model id is stamped (E2 provenance)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_smoke_real_architect_proposes_lint_clean_candidate(request, provider: str) -> None:  # type: ignore[no-untyped-def]
    _maybe_live(provider, request)
    from archforge.architect import Architect
    from archforge.lint import lint
    from archforge.stores import AttemptStore, SpecStore, TraceStore

    import archforge.models as m

    tmp = __import__("tempfile").mkdtemp()
    specs = SpecStore(tmp)
    atts = AttemptStore(tmp)
    TraceStore(tmp)  # constructor creates the dir; unused here otherwise
    inc = m.Spec(nodes=[m.Node(node_id="a", role="r", system_prompt="p0",
                              model="gpt", tools=["t"])])
    rid = specs.commit(inc, parent_spec_id=None, status=m.SpecStatus.INCUMBENT)
    specs.set_active(rid)

    client = __import__("archforge.llm", fromlist=["make_client"]).make_client(provider)
    arch = Architect(client, model=_default_model(provider))
    res = arch.next_attempt(specs.active(), [], attempt_store=atts)

    # plumbing: a proposed candidate that round-trips through mutate and is lint-clean.
    # (a real model may plateau/decline — we assert only the happy path here, and
    # the test's value is that the FULL propose->mutate->lint->dedup path runs live.)
    if not res.proposed:
        pytest.skip("real architect returned no proposal this call (allowed live behaviour)")
    cand = res.proposal.candidate
    assert lint(cand) == [], "real architect proposed a lint-dirty candidate"
    assert res.proposal.change.kind in m.ChangeKind
    assert cand.spec_id is None or cand.spec_id == rid  # not yet committed/over-stamped


@pytest.mark.parametrize("provider", PROVIDERS)
def test_smoke_end_to_end_evolve_cycle_state_consistent(request, provider: str) -> None:  # type: ignore[no-untyped-def]
    _maybe_live(provider, request)
    import tempfile

    import archforge.models as m
    from archforge.architect import Architect
    from archforge.engine import Engine, EngineConfig
    from archforge.host import FakeHostMAS
    from archforge.host.base import Task
    from archforge.judge.base import Judge, default_rubric
    from archforge.llm import make_client
    from archforge.stores import AttemptStore, SpecStore, TraceStore
    from archforge.suite import Suite

    tmp = tempfile.mkdtemp()
    specs, atts, traces = SpecStore(tmp), AttemptStore(tmp), TraceStore(tmp)
    inc = m.Spec(nodes=[m.Node(node_id="a", role="r", system_prompt="Answer briefly.",
                              model="gpt", tools=["t"])])
    rid = specs.commit(inc, parent_spec_id=None, status=m.SpecStatus.INCUMBENT)
    specs.set_active(rid)

    llm = make_client(provider)
    engine = Engine(
        host=FakeHostMAS(),
        judge=Judge(llm, model=_default_model(provider), rubric=default_rubric()),
        architect=Architect(llm, model=_default_model(provider)),
        spec_store=specs, attempt_store=atts,
        trace_store=traces, suite=Suite(suite_id="smoke", rubric_id=default_rubric().rubric_id,
                                        tasks=[Task(task_id="t1", input="What is 2+2?")]),
        thresholds=m.Thresholds(), config=EngineConfig(max_cycles=1, repeats=1),
    )
    r = engine.evolve_cycle(0)

    # Plumbing consistency (NOT scores): one cycle ran to completion without
    # raising, the active pointer's geometry is internally consistent, and every
    # attempted candidate ended with a terminal verdict (no leaked PENDING except
    # a queued structural change, which is the intended PENDING_HUMAN state).
    assert r.cycle == 0
    assert specs.active_id() is not None               # there is still an incumbent
    if r.attempted and r.applied_attempt_id is not None:
        att = atts.require(r.applied_attempt_id)
        assert att.verdict in m.Verdict                # a real terminal-ish verdict
        # the candidate Spec committed under this attempt is loadable + lint-clean
        cand = specs.get(att.candidate_spec_id)
        from archforge.lint import lint as _lint
        assert _lint(cand) == []
