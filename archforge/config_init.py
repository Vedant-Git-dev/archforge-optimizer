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
    '"groq": "openai/gpt-oss-120b", "gemini": "gemini-3.1-flash-lite"}'
)
_DEFAULT_JUDGE_MODELS = (
    '{"anthropic": "claude-sonnet-5", "openai": "gpt-4o", '
    '"groq": "openai/gpt-oss-120b", "gemini": "gemini-3.1-flash-lite"}'
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

__all__ = ["archforge_config_text", "env_example_text", "TEMPLATE", "EDITABLE_NAMES",
           "_DEFAULT_SUITE_JSON"]
