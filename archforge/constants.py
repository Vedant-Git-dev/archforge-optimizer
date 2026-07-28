"""ArchForge's single source of truth for project-wide raw constants.

This module is a **pure leaf**: it imports nothing from `archforge` (stdlib only),
and holds plain values only (str / int / float / tuple / dict) — never the enums,
Pydantic models, or enum-typed sets that live in `models.py` / the stores / the
judge. That keeps it cycle-free: every other module may `from archforge.constants
import …` with no risk of a back-edge.

What stays OUT of this file (and why):
  * Enums (`EdgeType`, `ChangeKind`, `Scope`, `Verdict`, `SpecStatus`, `Role`,
    `Action`) — domain identity types; their consumers are everywhere.
  * `STRUCTURAL_KINDS` (models.py) and `_BLOCKING_VERDICTS` (attempt_store) —
    frozensets over those enums, consumed *inside* the same module; centralizing
    them would either create an import cycle (constants → models → constants) or
    force string-typed sets. Their *rule* lives with the model that owns them.
  * `Rubric` model + the `default_rubric` instance (judge/base) — a Pydantic
    model; its *content* (id + sub-rubrics) IS centralized below, but the
    instance is built where `Rubric` is defined so `from … import default_rubric`
    keeps working.
  * `mutate.APPLY` dispatch — runtime wiring, not a constant.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #

VERSION: str = "0.1.0"

# --------------------------------------------------------------------------- #
# On-disk storage layout (consumed by the stores + the CLI's --root default)
# --------------------------------------------------------------------------- #

DEFAULT_ROOT_DIR: str = ".archforge"
SPECS_DIRNAME: str = "specs"
ATTEMPTS_DIRNAME: str = "attempts"
TRACES_DIRNAME: str = "traces"
ACTIVE_POINTER_FILE: str = "active.pointer"
ARCHIVED_FILE: str = "archived.jsonl"

# --------------------------------------------------------------------------- #
# Content-addressing hash truncation lengths (sha256 hex prefixes)
#   SPEC_ID_HASH_LEN  — full content+lineage spec id (models.py, attempt_store)
#   SHORT_HASH_LEN    — short deterministic suffixes (host/fake responder+run_id)
# --------------------------------------------------------------------------- #

SPEC_ID_HASH_LEN: int = 16
SHORT_HASH_LEN: int = 8

# --------------------------------------------------------------------------- #
# LLM providers + their default model ids (consumed by llm/__init__ + adapters)
# --------------------------------------------------------------------------- #

SCRIPTED_PROVIDER: str = "scripted"
REAL_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "groq", "gemini")
ALL_PROVIDERS: tuple[str, ...] = (SCRIPTED_PROVIDER,) + REAL_PROVIDERS

# Per-provider default model id. The four adapters each re-export their entry as
# the module-level `DEFAULT_MODEL` attribute (smoke tests + the CLI read that),
# so the attribute is preserved while its single source of truth lives here.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
}

# --------------------------------------------------------------------------- #
# Optimization thresholds / knobs (defaults for Thresholds + EngineConfig +
# SuiteRunner). These deduplicate values that were spelled in two places:
#   DEFAULT_UNRUNNABLE_FRAC — both Thresholds.unrunnable_frac and SuiteRunner.epsilon
#   DEFAULT_PLATEAU_CYCLES  — both Thresholds.plateau_cycles and EngineConfig.plateau_cycles
# --------------------------------------------------------------------------- #

DEFAULT_TAU: float = 0.05           # promotion margin τ
DEFAULT_DELTA: float = 0.07         # regression floor δ (rollback trigger, E6)
DEFAULT_REPEATS: int = 1           # R: repeats per task
MAX_REPEATS: int = 3               # cap on adaptive R
DEFAULT_UNRUNNABLE_FRAC: float = 0.25   # ε: > ε crashed tasks → unrunnable (E4)
DEFAULT_PLATEAU_CYCLES: int = 5    # K consecutive no-promotion → plateau (E8)
DEFAULT_MAX_CYCLES: int = 20       # loop cycle ceiling

DEFAULT_JUDGE_RETRIES: int = 2    # bounded grader-outage retries (E9)
BACKOFF_CAP_SECONDS: float = 30.0  # exponential grader-retry backoff ceiling

# --------------------------------------------------------------------------- #
# LLM runtime — Anthropic mandates a max_tokens; default when the caller omits
# --------------------------------------------------------------------------- #

ANTHROPIC_DEFAULT_MAX_TOKENS: int = 1024

# --------------------------------------------------------------------------- #
# Judge defaults — the content of default_rubric (instance built in judge/base)
# --------------------------------------------------------------------------- #

DEFAULT_RUBRIC_ID: str = "default-v1"
DEFAULT_SUB_RUBRICS: dict[str, str] = {
    "correctness": "Is the final answer factually correct and aligned with the task?",
    "completeness": "Does the answer address every part of the task?",
    "grounding": "Are the claims supported by the inputs/context, not invented?",
}

# --------------------------------------------------------------------------- #
# ScriptedJudge deterministic noise band (E1 jitter pattern)
# --------------------------------------------------------------------------- #

SCRIPTED_NOISE_PATTERN: tuple[float, ...] = (1.0, 0.0, -1.0, 0.5, -0.5, 0.25, -0.25)

# --------------------------------------------------------------------------- #
# CLI defaults (the free-run scripted suite/fixture + program name)
# --------------------------------------------------------------------------- #

PROG: str = "archforge"
DEFAULT_SUITE_ID: str = "cli-default"
DEFAULT_TASK_ID: str = "t1"
DEFAULT_TASK_INPUT: str = "hello"


__all__ = [
    # identity
    "VERSION",
    # storage
    "DEFAULT_ROOT_DIR", "SPECS_DIRNAME", "ATTEMPTS_DIRNAME", "TRACES_DIRNAME",
    "ACTIVE_POINTER_FILE", "ARCHIVED_FILE",
    # hashing
    "SPEC_ID_HASH_LEN", "SHORT_HASH_LEN",
    # providers
    "SCRIPTED_PROVIDER", "REAL_PROVIDERS", "ALL_PROVIDERS", "DEFAULT_MODELS",
    # thresholds / knobs
    "DEFAULT_TAU", "DEFAULT_DELTA", "DEFAULT_REPEATS", "MAX_REPEATS",
    "DEFAULT_UNRUNNABLE_FRAC", "DEFAULT_PLATEAU_CYCLES", "DEFAULT_MAX_CYCLES",
    "DEFAULT_JUDGE_RETRIES", "BACKOFF_CAP_SECONDS",
    # llm runtime
    "ANTHROPIC_DEFAULT_MAX_TOKENS",
    # judge
    "DEFAULT_RUBRIC_ID", "DEFAULT_SUB_RUBRICS",
    # scripted
    "SCRIPTED_NOISE_PATTERN",
    # cli
    "PROG", "DEFAULT_SUITE_ID", "DEFAULT_TASK_ID", "DEFAULT_TASK_INPUT",
]
