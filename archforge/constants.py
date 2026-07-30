"""ArchForge's single source of truth for project-wide raw constants.

This module is a **pure leaf**: it imports nothing from `archforge` (stdlib only),
and holds plain values only (str / int / float / tuple / dict) — never the enums,
Pydantic models, or enum-typed sets that live in `models.py` / the stores / the
judge. That keeps it cycle-free: every other module may `from archforge.constants
import …` with no risk of a back-edge.

== How to configure ArchForge ==
Tune the framework by editing the groups below, top to bottom (most-tuned first).
Every value is plain data imported *by name* throughout the package, so changing
a value here takes effect everywhere it's used; renaming a symbol requires a
find/replace across its consumers. Values the package treats as user preferences
(optimization policy, providers, rubric) sit at the top; wiring internals sit at
the bottom under "Internals — not for tuning".
"""

from __future__ import annotations

# =========================================================================== #
# Identity
# =========================================================================== #
# The package version. Read by hatchling for the built distribution and
# re-exported as `archforge.__version__` (archforge/__init__.py). One place.
VERSION: str = "0.1.0"


# =========================================================================== #
# LLM providers
# =========================================================================== #
# The providers the CLI's `--provider` flag accepts. `scripted` is the zero-cost
# default (deterministic fakes, no SDK, no API key); the real providers each need
# their SDK installed + an API key. `REAL_PROVIDERS` / `ALL_PROVIDERS` are also
# re-exported from archforge.llm; `make_client(provider)` dispatches on these.
SCRIPTED_PROVIDER: str = "scripted"
REAL_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "groq", "gemini")
ALL_PROVIDERS: tuple[str, ...] = (SCRIPTED_PROVIDER,) + REAL_PROVIDERS


# Set your default provider here; the CLI's `--provider` default comes from this value.
PROVIDER="gemini"

# The model id each real provider selects by default. Each adapter re-exports its
# entry as the module-level `DEFAULT_MODEL` (smoke tests + the CLI read it); the
# single source of truth lives HERE, so to switch a provider's default model you
# edit this dict rather than the adapter.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o",
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.5-flash",
}

# =========================================================================== #
# Optimization policy — the P-E-C knobs a user tunes most
# =========================================================================== #
# Defaults for `Thresholds` (models.py), `EngineConfig` (engine.py), and
# `SuiteRunner` (suite.py). These are the levers that shape the optimizer's
# explore/exploit + safety behaviour; tweak here and every default constructor
# adopts the new value. Spec-rule references (E-/I-) are in the design spec.

DEFAULT_TAU: float = 0.05
# τ — promotion margin. A candidate is kept only if its eval mean beats the
# incumbent's by at least τ. Higher τ = stricter, fewer (but safer) promotions.

DEFAULT_DELTA: float = 0.07
# δ — regression floor (>= τ). A promoted candidate that later drops by >= δ
# from its pre-promotion mean is rolled back (E6/I3). Higher δ = more tolerance
# for post-promote noise before a rollback fires.

DEFAULT_REPEATS: int = 1
# R — repeats per eval-suite task (averaging reduces run variance). The engine
# may raise it adaptively under E3; `MAX_REPEATS` caps that.

MAX_REPEATS: int = 3
# Cap on adaptive R; never exceeded even when the engine tries to raise repeats
# to harden a borderline measurement near ±τ.

DEFAULT_UNRUNNABLE_FRAC: float = 0.25
# ε — a candidate whose fraction of crashed tasks exceeds ε is discarded as
# "unrunnable" before any margin math (E4). Shared by `Thresholds.unrunnable_frac`
# and `SuiteRunner.epsilon` (one value, two consumers). Lower ε = less crash
# tolerance.

DEFAULT_PLATEAU_CYCLES: int = 5
# K — the loop plateaus (stops) after K consecutive cycles with no promotion
# (E8). Shared by `Thresholds.plateau_cycles` and `EngineConfig.plateau_cycles`.
# Higher K = the optimizer keeps trying longer before giving up as "stuck".

DEFAULT_MAX_CYCLES: int = 20
# Hard ceiling on P-E-C cycles per `evolve-loop` run (`EngineConfig.max_cycles`).


# =========================================================================== #
# Grader resilience
# =========================================================================== #
# How the SuiteRunner rides out a transient judge (LLM-as-judge) outage (E9):
# a failed judge call is retried up to `DEFAULT_JUDGE_RETRIES` times with
# exponential backoff capped at `BACKOFF_CAP_SECONDS` seconds; if all retries
# fail the run folds to "unscored" rather than fabricating a score.
DEFAULT_JUDGE_RETRIES: int = 2
BACKOFF_CAP_SECONDS: float = 30.0


# =========================================================================== #
# Judge scoring defaults
# =========================================================================== #
# The default rubric the Judge scores runs against (E2/I5 — comparisons stay
# within a rubric). The `Rubric` Pydantic model lives in judge/base.py; that
# module builds the `default_rubric` instance from these two values, so editing
# the id or the sub-rubric text here changes it everywhere. To use a different
# rubric, build a `Rubric(rubric_id=..., sub_rubrics=...)` and pass it to the
# Judge / suite explicitly (anything else risks cross-rubric comparisons, blocked
# by the Gatekeeper).
DEFAULT_RUBRIC_ID: str = "default-v1"
DEFAULT_SUB_RUBRICS: dict[str, str] = {
    "correctness": "Is the final answer factually correct and aligned with the task?",
    "completeness": "Does the answer address every part of the task?",
    "grounding": "Are the claims supported by the inputs/context, not invented?",
}


# =========================================================================== #
# Storage layout
# =========================================================================== #
# On-disk layout under the CLI's `--root` (default DEFAULT_ROOT_DIR). The stores
# (archforge/stores/) name their subdirectories + pointer files from these; the
# CLI's --root default comes from DEFAULT_ROOT_DIR. Rarely changed.
DEFAULT_ROOT_DIR: str = ".archforge"
SPECS_DIRNAME: str = "specs"
ATTEMPTS_DIRNAME: str = "attempts"
TRACES_DIRNAME: str = "traces"
ACTIVE_POINTER_FILE: str = "active.pointer"   # names the active incumbent spec_id
ARCHIVED_FILE: str = "archived.jsonl"          # rolled-back spec_ids (I3 reachability)


# =========================================================================== #
# Internals — not for tuning
# =========================================================================== #
# Bookkeeping values wired into specific consumers. Change only if you know the
# consumer; these are plumbing, not user preferences.

# Content-addressing hash truncation (sha256 hex prefix lengths). SPEC_ID_HASH_LEN
# is the Spec/Attempt content id (models.py, attempt_store); SHORT_HASH_LEN is the
# short deterministic suffixes used by the fake host (responder + run_id).
SPEC_ID_HASH_LEN: int = 16
SHORT_HASH_LEN: int = 8

# Anthropic's messages API requires a max_tokens; used when a caller omits it
# (only the Anthropic adapter). Not a user preference — a provider requirement.
ANTHROPIC_DEFAULT_MAX_TOKENS: int = 1024

# The ScriptedJudge's deterministic noise band (E1 jitter pattern): the jitter
# applied to a scripted aggregate is `pattern[run_index % len] * noise_width`,
# so identical runs yield identical jitter (reproducible E1 margin-boundary tests).
SCRIPTED_NOISE_PATTERN: tuple[float, ...] = (1.0, 0.0, -1.0, 0.5, -0.5, 0.25, -0.25)

# The CLI's free-run scripted fixtures: when `--provider scripted` is invoked
# with no injected `components`, the evolve family runs against this one-task suite
# so `archforge evolve` is runnable end-to-end with zero configuration.
PROG: str = "archforge"
DEFAULT_SUITE_ID: str = "cli-default"
DEFAULT_TASK_ID: str = "t1"
DEFAULT_TASK_INPUT: str = "hello"


__all__ = [
    # identity
    "VERSION",
    # llm providers
    "SCRIPTED_PROVIDER", "REAL_PROVIDERS", "ALL_PROVIDERS", "DEFAULT_MODELS",
    # optimization policy
    "DEFAULT_TAU", "DEFAULT_DELTA", "DEFAULT_REPEATS", "MAX_REPEATS",
    "DEFAULT_UNRUNNABLE_FRAC", "DEFAULT_PLATEAU_CYCLES", "DEFAULT_MAX_CYCLES",
    # grader resilience
    "DEFAULT_JUDGE_RETRIES", "BACKOFF_CAP_SECONDS",
    # judge scoring defaults
    "DEFAULT_RUBRIC_ID", "DEFAULT_SUB_RUBRICS",
    # storage layout
    "DEFAULT_ROOT_DIR", "SPECS_DIRNAME", "ATTEMPTS_DIRNAME", "TRACES_DIRNAME",
    "ACTIVE_POINTER_FILE", "ARCHIVED_FILE",
    # internals
    "SPEC_ID_HASH_LEN", "SHORT_HASH_LEN", "ANTHROPIC_DEFAULT_MAX_TOKENS",
    "SCRIPTED_NOISE_PATTERN", "PROG", "DEFAULT_SUITE_ID", "DEFAULT_TASK_ID",
    "DEFAULT_TASK_INPUT",
]
