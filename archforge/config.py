"""ArchForge's single source of truth for project-wide configuration.

This is **the** config file: one place to register a provider, set the default
model, tune the optimizer's policy knobs, and name the storage layout — every
default an ArchForge flag or constructor reads is sourced here, grouped top to
bottom (most-tuned first) under clear section comments. Editing a value takes
effect everywhere it's imported by name.

This module is a **pure leaf**: it imports nothing from `archforge` (stdlib only)
and holds plain values (str / int / float / tuple / dict). The one function in
it, ``load_env()`` (the project ``.env`` loader), is stdlib-only too — so the
file stays cycle-free and every other module may `from archforge.config import …`
with no risk of a back-edge.

== How to configure ArchForge ==
Values the package treats as user preferences (optimization policy, providers,
rubric, the ``.env`` loader) sit at the top; wiring internals sit at the bottom
under "Internals — not for tuning". Every value is plain data imported *by name*
throughout the package, so changing a value here takes effect everywhere it's
used; renaming a symbol requires a find/replace across its consumers.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

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
    "gemini": "gemini-3.1-flash-lite",
}


# =========================================================================== #
# Project environment (`.env`) — how secrets reach the provider SDKs
# =========================================================================== #
# A pip-installed `archforge` reads the project folder's `.env` so a checked-out
# repo "just runs" once API keys are added (the `.env` is gitignored — never
# commit secrets). `load_env()` populates `os.environ` with `setdefault`, so the
# precedence is: an explicit `--api-key` (passed straight to the SDK by the CLI)
# > a real process env var > the `.env` file. The loader is stdlib-only (this
# module is a pure leaf — no core deps); if `python-dotenv` happens to be
# installed it is deferred to for robust quote/multiline handling. Provider env
# var names are NOT centralized here — the provider SDKs read their own
# (`ANTHROPIC_API_KEY` … `GEMINI_API_KEY`); this loader only has to populate env.

DEFAULT_ENV_FILE: str = ".env"
# The project-folder env file the CLI / runner load by default (resolved against
# cwd = the user's repo root when they invoke `archforge` or run `run_loop`).

DEFAULT_MAX_TOKENS_TOTAL: int | None = None
# Default for `--max-tokens-total` / `RunnerConfig.max_tokens_total`: the whole-
# run budget cap (E3) — `None` = uncapped.

DEFAULT_MAX_TOKENS_PER_CYCLE: int | None = None
# Default for `--max-tokens-per-cycle` / `RunnerConfig.max_tokens_per_cycle`: the
# per-cycle cap that aborts mid-cycle (E3) — `None` = uncapped.


def load_env(path: str | os.PathLike[str] | None = DEFAULT_ENV_FILE) -> None:
    """Populate ``os.environ`` from a flat ``KEY=VALUE`` file via ``setdefault``.

    A real process env var therefore always wins (``setdefault`` never overwrites);
    the CLI's ``--api-key`` (handed straight to the SDK) wins above both. Skips
    blank and ``#``-comment lines and lines without ``=``; strips optional
    surrounding ``"``/``'`` on values. A no-op when the file is absent or ``path``
    is falsy — so callers (CLI, runner) can invoke it unconditionally.

    If ``python-dotenv`` is importable it is deferred to (handles ``export ``
    prefixes, multiline values, and ``#/`` edge cases more robustly); otherwise
    the stdlib parser below is used. Either way the precedence above holds.
    """
    if not path:
        return
    if importlib.util.find_spec("dotenv") is not None:
        from dotenv import load_dotenv   # type: ignore[import-not-found]
        load_dotenv(path, override=False)   # override=False ⇒ real env wins
        return
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):      # tolerate `export KEY=…`
            key = key[len("export "):].strip()
        val = val.strip().strip("'").strip('"')
        os.environ.setdefault(key, val)


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
PROG: str = "archforge-optimizer"
DEFAULT_SUITE_ID: str = "cli-default"
DEFAULT_TASK_ID: str = "t1"
DEFAULT_TASK_INPUT: str = "hello"


# =========================================================================== #
# User-config override — the project-local ``archforge.py`` loader
# =========================================================================== #
# A checked-out repo drops a plain ``archforge.py`` INSIDE its run state dir
# (``<DEFAULT_ROOT_DIR>/archforge.py``, i.e. ``.archforge/archforge.py``) holding
# only the user-editable defaults. ``load_user_config()`` execs it and propagates
# the whitelisted overrides onto this module's globals BEFORE any consumer
# from-imports a value — the single chokepoint that makes a project override reach
# the frozen dataclass field defaults (the linchpin; see load_user_config's docstring).

_EDITABLE: tuple[str, ...] = (
    # The shipped names a project ``archforge.py`` may override — the project's
    # user-preference categories (this file's own section comments: providers, policy,
    # grader resilience, judge scoring, storage root, the .env knobs). Framework
    # internals (provider roster tuples, storage dirnames, hash lengths,
    # ANTHROPIC_DEFAULT_MAX_TOKENS, SCRIPTED_NOISE_PATTERN, PROG, the CLI fixtures,
    # VERSION) are deliberately ABSENT: the user file may set one, but it is NOT copied
    # (whitelist-copy), so the shipped/internal split is enforced even if the file
    # assigns an internal name. Keep this in sync with the names below.
    "PROVIDER", "DEFAULT_MODELS",
    "DEFAULT_TAU", "DEFAULT_DELTA", "DEFAULT_REPEATS", "MAX_REPEATS",
    "DEFAULT_UNRUNNABLE_FRAC", "DEFAULT_PLATEAU_CYCLES", "DEFAULT_MAX_CYCLES",
    "DEFAULT_JUDGE_RETRIES", "BACKOFF_CAP_SECONDS",
    "DEFAULT_RUBRIC_ID", "DEFAULT_SUB_RUBRICS",
    "DEFAULT_ROOT_DIR", "DEFAULT_ENV_FILE",
    "DEFAULT_MAX_TOKENS_TOTAL", "DEFAULT_MAX_TOKENS_PER_CYCLE",
)


def load_user_config(
    path: str | os.PathLike[str] | None = None, *, target: dict | None = None,
) -> bool:
    """Exec a project-local ``archforge.py`` and override this module's editable names.

    The file lives at ``<DEFAULT_ROOT_DIR>/archforge.py`` (``.archforge/archforge.py``
    in the cwd by default) — *inside* the run state dir so it can never shadow the
    installed ``archforge`` package (a top-level ``archforge.py`` would: ``import
    archforge`` would resolve to the file before site-packages, breaking embedders).
    It is exec'd into a FRESH namespace (NOT this module's globals) so it can't touch
    internals (``os``, ``VERSION``, ``load_env`` …); only the names in ``_EDITABLE``
    present in the file are copied onto ``target`` (default = this module's globals).

    Returns ``True`` if a file was applied, ``False`` if it is absent (a no-op — the
    common case). There is NO guard inside: the caller decides whether to call. The
    module-body auto-load (at the very end of this file) wraps this in
    ``_config_disabled()`` so the test suite always sees the shipped defaults.

    Why this is the override chokepoint: ``archforge.config`` is the FIRST archforge
    module imported in the whole graph — pulled transitively by ``archforge/__init__``
    before any consumer (``models``/``engine``/``suite``/``runner``/``cli``) from-imports
    a ``DEFAULT_*`` value. Defaults are captured by those consumers AT CLASS-DEFINITION
    time as frozen ``from``-import bindings (``Thresholds.tau = DEFAULT_TAU`` etc.), so a
    LATER patch of ``archforge.config.DEFAULT_TAU`` cannot reach them. Executing the user
    file onto THIS module's globals at import time, before those from-imports fire, makes
    every captured binding + dataclass field default + argparse ``default=`` read the
    user's value. Edit a name in ``.archforge/archforge.py`` and it takes effect
    everywhere — provided the consumer in question reads the config name (see
    "drift literals"; the few that hardcoded literals are wired to the config names).
    """
    p = Path(path) if path is not None else Path(DEFAULT_ROOT_DIR) / "archforge.py"
    if not p.exists():
        return False
    src = p.read_text(encoding="utf-8")
    code = compile(src, str(p), "exec")
    ns: dict[str, object] = {}
    exec(code, ns)  # noqa: S102  — the file is the user's own config; fresh namespace
    tgt = target if target is not None else globals()
    for name in _EDITABLE:
        if name in ns:
            tgt[name] = ns[name]
    return True


def _config_disabled() -> bool:
    """True when user-config auto-load should NOT run.

    Two gates: an explicit ``ARCHFORGE_CONFIG_DISABLE`` env var (manual opt-out; also
    lets a test force shipped defaults), and ``"pytest" in sys.modules``. pytest is
    imported long before any archforge module is first imported during collection, so
    the latter reliably disables auto-load during the suite — protecting tests that
    assert EXACT shipped defaults (e.g. ``Thresholds().tau == 0.05``) from being
    re-painted by a real ``.archforge/archforge.py`` checked into a dev workspace.
    """
    return bool(os.environ.get("ARCHFORGE_CONFIG_DISABLE")) or "pytest" in sys.modules


__all__ = [
    # identity
    "VERSION",
    # llm providers
    "SCRIPTED_PROVIDER", "REAL_PROVIDERS", "ALL_PROVIDERS", "DEFAULT_MODELS",
    # project environment (api keys from a `.env` in the working folder)
    "DEFAULT_ENV_FILE", "load_env", "load_user_config",
    "DEFAULT_MAX_TOKENS_TOTAL", "DEFAULT_MAX_TOKENS_PER_CYCLE",
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


# Auto-load the project's ``.archforge/archforge.py`` (if present) so the user's
# editable overrides are live on this module's globals BEFORE any consumer from-imports
# them (see load_user_config's docstring for the import-order chokepoint). Guarded off
# under pytest / ARCHFORGE_CONFIG_DISABLE so the test suite asserts shipped defaults.
if not _config_disabled():  # pragma: no cover  (auto-disabled under the test runner)
    load_user_config()
