"""Scaffolding text for `archforge-optimizer init` — the user-tunable config template.

Owns the ACTIVE defaults that become `.archforge/archforge.py` (the project's sole
source of ArchForge tunables, made by `init`) and the `.env.example` key template.

Two roles, one source:
  * `init` writes ``archforge_config_text()`` verbatim to ``.archforge/archforge.py`` →
    the user's tunables. It ships with **active sane values** so the CLI works
    immediately after `init`; the user edits a value to change behaviour.
  * Under the test runner, ``archforge.userconfig`` execs the SAME ``TEMPLATE``
    in-memory (no disk file) so the suite sees the sane defaults
    (e.g. ``Thresholds().tau == 0.05``) with zero per-test files.

The values here are ArchForge's sane defaults — keep them in sync with what the
package historically shipped. This module is pure data (string constants); it imports
nothing from `archforge` and never reads the live `.env`.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# the sane tunable defaults — the ONE place the values live
# --------------------------------------------------------------------------- #
# (name, active-value-as-assignment, one-line purpose). The assignment text is
# emitted verbatim into the generated file AND exec'd by the resolver, so it must be
# valid Python and carry the real sane value (note DEFAULT_ARCHITECT_MODELS /
# DEFAULT_JUDGE_MODELS / DEFAULT_SUB_RUBRICS are the full dicts, not `{}` — the
# resolver must resolve them without KeyError).

_DEFAULT_ARCHITECT_MODELS = (
    '{"anthropic": "claude-sonnet-5", "openai": "gpt-4o", '
    '"groq": "openai/gpt-oss-120b", "gemini": "gemini-3.6-flash"}'
)
_DEFAULT_JUDGE_MODELS = (
    '{"anthropic": "claude-sonnet-5", "openai": "gpt-4o", '
    '"groq": "openai/gpt-oss-120b", "gemini": "gemini-3.6-flash"}'
)
_DEFAULT_SUB_RUBRICS = (
    '{"correctness": "Is the final answer factually correct and aligned with the task?", '
    '"completeness": "Does the answer address every part of the task?", '
    '"grounding": "Are the claims supported by the inputs/context, not invented?"}'
)

# The starter suite `init` writes to .archforge/suite.json — byte-identical to the
# one-task CLI fallback fixture (cli-default / t1 / hello), so the generated default
# round-trips to the same Suite the CLI builds when the file is absent. A per-task
# rubric_id is optional in the file; omitting it scores against the active rubric.
_DEFAULT_SUITE_JSON = (
    '{\n'
    '  "suite_id": "cli-default",\n'
    '  "tasks": [\n'
    '    {"task_id": "t1", "input": "hello"}\n'
    '  ]\n'
    '}\n'
)

_FIELDS: tuple[tuple[str, str, str], ...] = (
    # --- LLM provider
    ("PROVIDER", '"gemini"',
     "which LLM to use (scripted|anthropic|openai|groq|gemini); scripted needs no API key"),
    ("DEFAULT_ARCHITECT_MODELS", _DEFAULT_ARCHITECT_MODELS,
     "default Architect (proposer) model per provider; a bare LLMClient call falls "
     "back here too — edit the dict to change it"),
    ("DEFAULT_JUDGE_MODELS", _DEFAULT_JUDGE_MODELS,
     "default Judge (scorer) model per provider — edit the dict to change it"),
    # --- optimization policy
    ("DEFAULT_TAU", "0.05", "how much better a candidate must score to be promoted (τ)"),
    ("DEFAULT_DELTA", "0.07", "how far a promoted run can drop before it's rolled back (δ, >= τ)"),
    ("DEFAULT_REPEATS", "1", "how many times each eval task is run; more = steadier scores, more cost (R)"),
    ("MAX_REPEATS", "3", "upper bound on R"),
    ("DEFAULT_UNRUNNABLE_FRAC", "0.25", "drop a candidate if more than this fraction of its tasks crash (ε)"),
    ("DEFAULT_PLATEAU_CYCLES", "5", "stop after this many cycles in a row with no improvement (K)"),
    ("DEFAULT_MAX_CYCLES", "20", "max optimization cycles per run"),
    # --- grader resilience
    ("DEFAULT_JUDGE_RETRIES", "2", "how many times to retry a failed judge call"),
    ("BACKOFF_CAP_SECONDS", "30.0", "max seconds to wait between judge retries"),
    # --- judge scoring
    ("DEFAULT_RUBRIC_ID", '"default-v1"', "name of the scoring rubric (keep it stable so runs compare)"),
    ("DEFAULT_SUB_RUBRICS", _DEFAULT_SUB_RUBRICS,
     "the rubric's dimensions: what a high score looks like, per dimension"),
    # --- environment / budget / storage
    ("DEFAULT_ROOT_DIR", '".archforge"', "where run state is written (relative to where you run the CLI)"),
    ("DEFAULT_ENV_FILE", '".env"', "the .env file loaded for API keys (a real env var always wins)"),
    ("DEFAULT_SUITE_FILE", '".archforge/suite.json"', "the suite file defining your eval tasks (absent → the one-task default)"),
    ("DEFAULT_MAX_TOKENS_TOTAL", "None", "whole-run token budget cap; None = no limit"),
    ("DEFAULT_MAX_TOKENS_PER_CYCLE", "None", "per-cycle token cap (aborts mid-cycle if exceeded); None = no limit"),
    ("DEFAULT_MAX_WALL_MS_PER_CYCLE", "None", "per-cycle wall-clock cap (ms); aborts if exceeded — "
     "covers non-LLM nodes (retriever/tool/rule) that cost time, not tokens; None = no limit"),
    ("DEFAULT_TRACE_TOTAL_BUDGET_TOK", "None", "total Judge-prompt token budget for OTel trace "
     "projection of per-step LLM prompt/completion; None = lossy summarize() path (parity, "
     "no tracing); an int turns on rich per-step Steps, shedding largest-evidence chunks first"),
)

# section break points in _FIELDS (for grouping the emitted file)
_BREAKS: dict[int, str] = {
    3: "# --- optimization policy ------------------------------------------------",
    10: "# --- grader resilience ---------------------------------------------------",
    12: "# --- judge scoring -------------------------------------------------------",
    14: "# --- environment / budget / storage --------------------------------------",
}

_HEADER = """\
# archforge.py — your ArchForge config (made by `archforge-optimizer init`).
#
# The values below already work — the CLI runs as-is after `init`. Edit any value
# to change that default. Nothing here is required to make ArchForge import.
#

# --- LLM provider ------------------------------------------------------------
"""

_FOOTER = """
# Changes here take effect on the next `archforge-optimizer` run.
"""


def archforge_config_text() -> str:
    """The full body of the generated `.archforge/archforge.py` (active sane defaults)."""
    lines = [_HEADER.rstrip("\n")]
    for i, (name, value, purpose) in enumerate(_FIELDS):
        if i in _BREAKS:
            lines.append("")
            lines.append(_BREAKS[i])
        lines.append(f"# {purpose}")
        lines.append(f"{name} = {value}")
    lines.append(_FOOTER.rstrip("\n"))
    return "\n".join(lines) + "\n"


# The active sane-default assignments, exec'd by archforge.userconfig under the test
# runner (so the suite sees sane defaults) — same string `init` writes to disk.
TEMPLATE: str = archforge_config_text()


def env_example_text() -> str:
    """The body of the generated `.env.example` (empty provider key var names)."""
    return (
        "# fill in your API keys for provider you are going to use for archforge-optimizer.\n\n"
        "# ANTHROPIC_API_KEY=\n"
        "# OPENAI_API_KEY=\n"
        "# GROQ_API_KEY=\n"
        "# GEMINI_API_KEY=\n"
        "\n# Other service keys you need for your tasks\n"
    )


# The tunable names — exported so callers/tests enumerate the editable surface.
EDITABLE_NAMES: tuple[str, ...] = tuple(name for name, _, _ in _FIELDS)

# --------------------------------------------------------------------------- #
# the generic LangGraph adapter scaffold — `init` ALSO writes these into a fresh
# `archforge_optimizer/` package in the user's project root, so they EDIT their MAS
# details instead of coding the wiring from scratch. A name-neutralized, generalized
# copy of the in-repo AEDE glue (`AEDE/backend/archforge_glue/`), which is NOT shipped
# (excluded from the sdist/wheel). Pure data: string constants, no archforge import,
# no real MAS, no secrets here. The placeholder roster (retrieve -> answer) forms a
# VALID Spec (it lints), so `--seed archforge_optimizer/spec.json` bootstraps BEFORE
# the user fills in the real graph; `app.py` self-writes `spec.json` at import
# (mirrors AEDE's `write_aede_spec_json()`). Keys are paths relative to the package
# dir (`archforge_optimizer/`); the folder name itself is fixed.
# --------------------------------------------------------------------------- #
_ADAPTER_PKG_FILES: dict[str, str] = {
    "__init__.py": '''"""ArchForge <-> your MAS adapter package.

Holds the instance of the generic ``LangGraphHostAdapter`` (``app.py`` +
``host.py``), the Tier-2 deploy consumer (``sidecar.py``), and the offline
zero-billing smoke test (``test_smoke_offline.py``).

Point ArchForge at it with::

    archforge-optimizer evolve --adapter archforge_optimizer.host:AppAdapter --seed archforge_optimizer/spec.json

``app.py`` describes your MAS's static agent DAG as a ``LangGraphApp``;
``host.py`` binds it to the generic adapter that drives the real
``graph.stream``. Edit the ``# EDIT:`` markers in ``app.py`` (node roster,
edges, knobs, hooks) to match your MAS -- the placeholder below builds a valid
Spec, so ``--seed`` bootstraps before you fill in the real graph.
"""
''',
    "host.py": '''"""Binds the generic ``LangGraphHostAdapter`` to your MAS's ``App`` description.

Zero-arg class (so the CLI's ``--adapter archforge_optimizer.host:AppAdapter``
imports + instantiates it). The generic adapter owns the run loop; this just binds
it to the ``App`` description and exposes the bootstrap incumbent Spec.
"""
from __future__ import annotations

from archforge.host.adapters import LangGraphHostAdapter

from .app import App


class AppAdapter(LangGraphHostAdapter):
    """A ``HostMAS`` driving your MAS's real LangGraph via the generic adapter.

    ``app_spec()`` is the bootstrap incumbent -- seed it into a ``SpecStore``
    as the active Spec (``archforge-optimizer evolve --seed
    archforge_optimizer/spec.json``), then ``archforge-optimizer evolve`` runs
    the Propose-Evaluate-Commit loop.
    """

    def __init__(self) -> None:
        super().__init__(App())


__all__ = ["AppAdapter"]
''',
    "app.py": '''"""Your MAS as a ``LangGraphApp`` -- the generic adapter's per-MAS description.

A small declarative description of your MAS's LangGraph graph: the static agent
DAG (a ``Nd`` roster), the seeded incumbent knobs, the knob->state map, and the
behavior hook (``initialize_state``) + summary hook (``summarize``) + the
LLM-injection hooks (``apply_llm_config``/``reset_llm_config``) that bridge to
your MAS's call-time config.

The adapter in ``host.py`` wraps this; the generic ``LangGraphHostAdapter`` owns
the run loop (drives the real ``graph.stream``, records one Step per emitted node).

The PLACEHOLDER below builds a VALID Spec (it lints), so ``--seed`` works before
you fill in the real graph. Replace every ``# EDIT:`` marker with your MAS's
details; until then ``build_graph`` raises only at RUN time (after a mutation
makes the Forge try to execute) -- ``build_spec()`` never needs a real graph.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import archforge.models as m
from archforge.host.adapters import EdgeSpec, LangGraphApp, Nd, node_ids

# -- your MAS import (EDIT: wire your real graph builder here) ------------------
# Replace the placeholder ``build_graph`` below with an import of your MAS's
# compiled graph builder, e.g.:
#     from mymas.graph import build_graph          # your MAS's real StateGraph builder
#     from mymas.state import create_initial_state # its initial-state factory (if any)
# ``langgraph`` is NOT an ArchForge dependency -- install it where your MAS runs.
# Until you wire ``build_graph``, the placeholder raises only at RUN time (after a
# mutation makes the Forge execute); ``build_spec()`` + the bootstrap ``--seed``
# work WITHOUT it -- so you are one command from a real run.
def build_graph() -> Any:
    raise NotImplementedError(
        "EDIT app.py: import your MAS's `build_graph` (the real langgraph.StateGraph "
        "builder) and point `graph_factory` at it. The placeholder Spec still lints "
        "+ seeds without it."
    )


# Keyed id source -- the SINGLE authority for each node id string. The reference
# adapter authors the id ONCE in ``build_graph``'s ``add(name, fn)`` line and
# sources it everywhere via ``_GID["retrieve"]``, so a rename there raises
# ``KeyError`` HERE at import (fail-fast per node). That needs a REAL graph; until
# you wire ``build_graph``, author the ids by hand:
_GID: dict[str, str] = {
    "retrieve": "retrieve",     # EDIT: the str your build_graph passes to add(name, fn)
    "answer":   "answer",       # EDIT: the str your build_graph passes to add(name, fn)
}
# Once build_graph() exists, replace the literal above with the fail-fast form:
#     _GID = node_ids(build_graph())   # KeyError at import on a build_graph rename


# --------------------------------------------------------------------------- #
# The static roster (EDIT: replace with your MAS's real nodes). kinds + seeded
# incumbent knobs + tunable. The placeholder below forms a VALID Spec (it lints),
# so ``--seed`` bootstraps before you fill in the real graph.
#
#   model=""         => the LLM injector falls back to your MAS's configured default
#                       on a bare run; a live `model_swap` rides through.
#   system_prompt="" => base_prompts={nid:""} => cfg_decay returns None on a base
#                       run (no override); a `prompt_edit` rides through.
#
# The id string is authored ONCE in your ``build_graph`` (the ``add(name, fn)``
# line); source it from ``_GID[...]`` so a rename raises ``KeyError`` at import,
# never a silently-mislabeled run. role/kind/knobs stay literals (they are the
# seeded VALUES, not the id).
# --------------------------------------------------------------------------- #
_NODES: list[Nd] = [
    Nd(_GID["retrieve"], "context retriever", m.NodeKind.RETRIEVER,
       knobs=m.Knobs(top_k=4, tunable=("top_k",))),           # EDIT: your retriever's knobs
    Nd(_GID["answer"], "final responder", m.NodeKind.LLM,
       knobs=m.Knobs(temperature=0.3, max_tokens=4096)),     # EDIT: your responder's knobs
]

# knob name -> state key the node reads via ``state.get(key, <default>)``.
# Named LLM knobs (model/temperature/max_tokens/...) are NOT here -- they flow
# through ``apply_llm_config`` (your call-time injector), not state.
_KNOB_TO_STATE: dict[str, str] = {
    "top_k": "current_top_k",       # EDIT: map each EXTRA knob to the state key your node reads
}

_EDGES: list[EdgeSpec] = [
    EdgeSpec(_GID["retrieve"], _GID["answer"], m.EdgeType.SEQUENCE),
]
# Runtime loops inside the real LangGraph (the adapter observes each iteration as
# an emitted node): document them below but do NOT add them as static ``edges`` --
# a back-edge would trip the linter's ``cycle`` rule, and the real graph drives the
# loop anyway. Example: a retrieve_more -> extract retry loop.
_RUNTIME_LOOPS: list[tuple[str, str]] = [
    # ("retrieve_more", "extract"),   # EDIT: any runtime back-edge in your real graph
]

NODE_IDS = tuple(_GID)                                   # build-order names, once a real graph exists


class App(LangGraphApp):
    """Your MAS's static description for the ArchForge generic LangGraph adapter."""

    graph_factory = staticmethod(build_graph)           # EDIT: your real build_graph (raises until wired)
    nodes = _NODES
    edges = _EDGES
    knob_to_state = _KNOB_TO_STATE
    runtime_loops = _RUNTIME_LOOPS
    final_output_key = "answer"                         # EDIT: your graph's terminal state key
    base_prompts = {nid: "" for nid in NODE_IDS}

    # ---- behavior hooks (override me) ---------------------------------- #
    # initialize_state: return the dict your graph's nodes read on the first step.
    # Two ways (set EITHER):
    #   * override this method (below), OR
    #   * delete this override and set the class attribute
    #     ``state_factory = <callable>`` below; the base
    #     ``LangGraphApp.initialize_state`` then calls it.
    # The placeholder returns a minimal dict so a bare import + run of the
    # placeholder does not crash before you fill in the real state shape.
    def initialize_state(self, task_input: str) -> dict[str, Any]:
        # EDIT: e.g. ``return dict(create_initial_state(task_input))`` from your MAS.
        return {"task_input": task_input, "current_top_k": 4}

    def summarize(self, node_id: str, partial: dict, merged: dict) -> str:
        """A deterministic per-node summary the Judge scores. Override to emit what
        the Judge should score per node.

        Reads ``merged`` (the adapter's own accumulation across the whole run) so a
        key LangGraph dropped from ``partial`` (an undeclared-key gotcha) is still
        observable here via the seeded/injected state. Kept short + bounded.

        EDIT: replace the placeholder with per-node summaries of YOUR state keys, e.g.::

            if node_id == "retrieve":
                return f"top_k={merged.get(\'current_top_k\')} chunks={len(merged.get(\'documents\') or [])}"

        The base default (if you delete this override) is ``json.dumps(partial)[:500]``.
        """
        # placeholder: a length-only snapshot valid for any node.
        return f"{node_id}: {len(str(merged))}b"

    # ---- LLM-injection (call-time, populated up-front per run) ----------- #
    def apply_llm_config(self, node_id: str, vote) -> None:
        # ``vote`` is a KnobVote (model/temperature/max_tokens/system_prompt); each
        # None => "no override" (base run = your MAS's default). The generic adapter
        # calls this up-front for every LLM node before graph.stream runs.
        # EDIT: route the vote into your MAS's call-time config (the reference
        # adapter points at its MAS's call-time node-config module), e.g.::
        #     _my_nodecfg.set_node_config(node_id, model=vote.model,
        #         temperature=vote.temperature, max_tokens=vote.max_tokens,
        #         system_prompt=vote.system_prompt)
        pass

    def reset_llm_config(self) -> None:
        # EDIT: clear the call-time injector before a run (the reference adapter
        # calls its node-config module's ``reset()``).
        pass


def write_spec_json(path: str | os.PathLike[str] | None = None) -> Path:
    """Dump the bootstrap Spec to JSON (for ``archforge-optimizer lint`` + ``--seed``).

    The builder is the single source of truth; the JSON is a derived artifact. This is a
    manual helper (call it yourself, or just run ``archforge-optimizer make-spec``) — it
    is NOT invoked at import time, so merely importing the app has no filesystem side
    effect. ``make-spec`` is the supported path: it builds the Spec via the adapter's
    ``app_spec()``, lints it, and writes ``archforge_optimizer/spec.json`` only if valid.
    """
    spec = App().build_spec()
    out = Path(path) if path else Path(__file__).resolve().parent / "spec.json"
    out.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return out


__all__ = ["App", "write_spec_json", "NODE_IDS"]
''',
    "sidecar.py": '''"""Tier-2 deploy consumer -- overlays a promoted Spec's knobs onto your MAS.

ArchForge promotes a candidate -> the Forge's ``on_promote`` hook calls
``export_spec_sidecar(spec, path)`` (in ``archforge.host.adapters.langgraph``) ->
writes ``optimized.json`` as ``{node_id: {knob: value}}``. At your MAS startup,
``load_at_startup(path)`` reads that file and applies it.

Why mutate-in-place (not return-and-rebind): if your MAS's node modules do
``from mymas.config import settings`` at import, the name is bound once; reassigning
``mymas.config.settings = new_settings`` leaves every already-imported node holding the
STALE reference. Mutating the singleton's fields (e.g.
``settings.pipeline.coverage_target = 0.4``) is seen by every holder of the same
object. (Pydantic v2 BaseModels are mutable by default.)

Keep your ``config`` module Forge-free (no ``archforge`` import there) -- this module
owns the ``node.knob -> config field`` map. Rollback = delete the sidecar file /
unset ``ARCHFORGE_SIDECAR_PATH``.

This is a name-neutral skeleton. Fill ``_KNOB_TO_CONFIG`` with your node -> config-field
maps and ``_LLM_NODES`` with your LLM node ids; point ``apply_node_config`` at your
MAS's call-time config store (the reference adapter points at its MAS's call-time
node-config module).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# EDIT: import your MAS's settings singleton + call-time node config. The reference
# adapter imports its MAS's ``Settings``/``settings`` singleton and call-time
# node-config module. Left as placeholders so this module imports before you wire
# them (the functions below no-op until then).
Settings = Any                # EDIT: from mymas.config import Settings
_settings_singleton = None    # EDIT: from mymas.config import settings as _settings_singleton
node_config = None            # EDIT: from mymas.utils import node_config

# node.knob -> (config attribute path) for routing/retrieval knobs. A single config
# field may be the target of two nodes; iterate sidecar nodes in Spec order (the
# exporter writes them in Spec order; dict insertion order preserved) so the LAST
# write wins -- identical to the live adapter's overlay order, so the deployed
# retune matches what was judged.
# EDIT: fill with your node -> {knob -> (section, field)} tuples.
_KNOB_TO_CONFIG: dict[str, dict[str, tuple[str, ...]]] = {
    # "retrieve": {"top_k": ("retrieval", "initial_k")},
    # "retrieve_more": {"max_k": ("retrieval", "max_k")},
}

# The LLM nodes whose model/temperature/max_tokens the call sites read. EDIT: your
# node ids (the reference adapter uses extract/analyze/compress/reason/small_reasoner).
_LLM_NODES: set[str] = set()          # EDIT: {"extract", "analyze", "reason", ...}
_LLM_KNOBS = ("model", "temperature", "max_tokens", "system_prompt")


def _coerce(value: Any, current: Any) -> Any:
    """Coerce a sidecar value to the current field's type (sidecar stores JSON
    ints/floats/strs; pydantic fields are typed). Minimal -- handles the knob types."""
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def apply_settings(settings: Any, sidecar: dict[str, dict[str, Any]]) -> None:
    """Overlay routing/retrieval knobs onto the ``settings`` singleton IN PLACE.

    Skips nodes/knobs not in ``_KNOB_TO_CONFIG`` (LLM knobs are handled by
    ``apply_node_config``; unmapped extras are silently dropped -- the scope seam).
    """
    for node_id, knobs in sidecar.items():
        cfg_map = _KNOB_TO_CONFIG.get(node_id, {})
        for kname, kval in knobs.items():
            if kval is None or kname not in cfg_map:
                continue
            sect, field = cfg_map[kname]
            section_obj = getattr(settings, sect, None)
            if section_obj is None:
                continue
            current = getattr(section_obj, field, None)
            setattr(section_obj, field, _coerce(kval, current))


def apply_node_config(sidecar: dict[str, dict[str, Any]]) -> None:
    """Overlay LLM knobs into your MAS's call-time node config (deploy target for
    model swaps). Silently skips unmapped nodes/knobs.

    EDIT: point ``node_config.set_node_config(...)`` at your MAS's config store.
    """
    if node_config is None:            # no-op until you import the real node_config above (EDIT)
        return
    for node_id, knobs in sidecar.items():
        if node_id not in _LLM_NODES:
            continue
        for kname in _LLM_KNOBS:
            kval = knobs.get(kname)
            if kval is None:
                continue
            node_config.set_node_config(node_id, **{kname: kval})  # type: ignore[arg-type]


def load_sidecar(path: str | os.PathLike[str] | None) -> dict[str, dict[str, Any]]:
    """Read a sidecar JSON; {} if absent or malformed (a missing rollout is a no-op)."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    import json
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                  # noqa: BLE001 -- a bad file never breaks startup
        return {}
    return data if isinstance(data, dict) else {}


def load_at_startup(path: str | os.PathLike[str] | None = None) -> None:
    """Apply a sidecar to the global ``settings`` singleton + node config.

    Read ``ARCHFORGE_SIDECAR_PATH`` if ``path`` is None. Noop (today's behaviour) if
    the env var is unset / the file missing. Call ONCE at your MAS startup (this
    mutates the shared singleton).
    """
    if path is None:
        path = os.getenv("ARCHFORGE_SIDECAR_PATH")
    sidecar = load_sidecar(path)
    if not sidecar:
        return
    if _settings_singleton is not None:
        apply_settings(_settings_singleton, sidecar)
    apply_node_config(sidecar)


__all__ = [
    "apply_settings", "apply_node_config", "load_sidecar", "load_at_startup",
]
''',
    "test_smoke_offline.py": '''"""Offline (zero-billing) smoke test for the adapter scaffold.

Builds ``App()``, calls ``build_spec()``, and asserts the placeholder Spec lints +
has the placeholder nodes -- proving the scaffold is one command from a real run,
with NO LLM/API keys and NO real graph required. The full routing-knob suite the
reference adapter ships (a span-emitting offline suite) needs a real ``build_graph``;
it is NOT part of the generic scaffold.

Run it either way:
    pytest archforge_optimizer/test_smoke_offline.py -q
    python archforge_optimizer/test_smoke_offline.py
"""
import pytest
pytest.importorskip("archforge")   # first executable line -- skip if archforge is absent

import os   # noqa: E402
import sys   # noqa: E402

import archforge.models as m   # noqa: E402
from archforge.lint import lint   # noqa: E402

# Resolve the adapter package whether run under pytest (parent on sys.path) or as a
# plain script (only the package dir is on sys.path): put the parent dir on sys.path
# so `archforge_optimizer` resolves.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from archforge_optimizer.app import App, NODE_IDS   # noqa: E402


def test_placeholder_spec_lints() -> None:
    """The placeholder roster + edges form a valid Spec (the load-bearing guarantee):
    `init` produces something one command from `evolve --seed`."""
    spec = App().build_spec()
    assert not lint(spec)
    assert {n.node_id for n in spec.nodes} == set(NODE_IDS)


def test_placeholder_spec_has_required_nodes() -> None:
    """The placeholder ships a 2-node DAG (retrieve -> answer) the user edits."""
    spec = App().build_spec()
    assert {n.node_id for n in spec.nodes} == {"retrieve", "answer"}
    kinds = {n.node_id: n.kind for n in spec.nodes}
    assert kinds["retrieve"] is m.NodeKind.RETRIEVER
    assert kinds["answer"] is m.NodeKind.LLM


if __name__ == "__main__":
    # `python archforge_optimizer/test_smoke_offline.py` (no pytest needed).
    test_placeholder_spec_lints()
    test_placeholder_spec_has_required_nodes()
    print("OK: adapter scaffold Spec lints + has the placeholder nodes.")
''',
}


def adapter_package_files() -> dict[str, str]:
    """The relative-path -> content entries ``init`` scaffolds into the user's project
    as a fresh ``archforge_optimizer/`` (LangGraph adapter skeleton). The folder name
    is fixed, so no substitution is needed (the accessor is kept as the single access
    point so the CLI does not reach into the private dict)."""
    return dict(_ADAPTER_PKG_FILES)


# The scaffolded file paths (relative to the package dir) -- exported so the CLI + the
# init tests enumerate the scaffolded surface without re-deriving the dict keys.
_EDITABLE_ADAPTER_FILES: tuple[str, ...] = tuple(_ADAPTER_PKG_FILES)

__all__ = ["archforge_config_text", "env_example_text", "TEMPLATE", "EDITABLE_NAMES",
           "_DEFAULT_SUITE_JSON", "adapter_package_files", "_ADAPTER_PKG_FILES",
           "_EDITABLE_ADAPTER_FILES"]
