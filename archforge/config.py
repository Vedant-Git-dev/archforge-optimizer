"""ArchForge system config — internals only (NOT user-tunable).

This holds the NON-tunable framework plumbing: identity (``VERSION``), the LLM
provider ROSTER (the tuples the CLI's ``--provider`` choices come from), the
on-disk storage DIRNAMES + pointer files, the content-addressing hash lengths,
Anthropic's required ``max_tokens``, the ScriptedJudge noise pattern, the CLI
program name + its scripted fixtures, and ``load_env`` (the ``.env`` loader for
provider API keys). **Nothing here is a user preference** — every value a user
might tune (τ, δ, R, the default provider + models, the rubric, the storage root,
token budgets, grader resilience …) lives instead in the project's
``.archforge/archforge.py`` (made by ``archforge-optimizer init``) and is read
lazily by ``archforge.userconfig`` at use time. So ``config.py`` ↔ ``archforge.py``
share NO variable (the two-file split), and this module never depends on the
user's config — it imports cleanly before ``init`` has run.

Pure leaf: stdlib + ``python-dotenv`` (a core dependency) only — imports nothing
else from ``archforge``, so every other module may ``from archforge.config import …``
with no risk of a back-edge.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# =========================================================================== #
# Identity
# =========================================================================== #
# The package version. Read by hatchling for the built distribution and
# re-exported as `archforge.__version__` (archforge/__init__.py). One place.
VERSION: str = "0.3.0"


# =========================================================================== #
# LLM providers — the ROSTER only (NOT the default choice; that's a tunable)
# =========================================================================== #
# The providers the CLI's `--provider` flag ACCEPTS (its `choices`). `scripted` is
# the zero-cost option (deterministic fakes, no SDK, no API key); the real
# providers each need their SDK + an API key. WHICH provider is the default, and
# each provider's default MODEL id, are tunables → they live in .archforge/archforge.py
# (resolved by archforge.userconfig), NOT here. `REAL_PROVIDERS`/`ALL_PROVIDERS` are
# re-exported from archforge.llm; `make_client(provider)` dispatches on these.
SCRIPTED_PROVIDER: str = "scripted"
REAL_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "groq", "gemini")
ALL_PROVIDERS: tuple[str, ...] = (SCRIPTED_PROVIDER,) + REAL_PROVIDERS


# =========================================================================== #
# Project environment (`.env`) — how secrets reach the provider SDKs
# =========================================================================== #
# A pip-installed `archforge` reads the project folder's `.env` so a checked-out
# repo "just runs" once API keys are added (the `.env` is gitignored — never commit
# secrets). `load_env()` populates `os.environ` with `override=False`, so the
# precedence is: an explicit `--api-key` (passed straight to the SDK by the CLI)
# > a real process env var > the `.env` file. `python-dotenv` is a CORE dependency
# here (handles `export ` prefixes, quoting, multiline values robustly); an absent
# file is a no-op. Provider env-var names are NOT centralized here — the provider
# SDKs read their own (`ANTHROPIC_API_KEY` … `GEMINI_API_KEY`); this loader only
# populates env.
#
# The DEFAULT `.env` path is a tunable (DEFAULT_ENV_FILE in archforge.py); this
# loader's own default arg is the literal ".env" so the function stays a pure leaf
# that never touches the resolver (callable pre-init, e.g. to load keys).

def load_env(path: str | os.PathLike[str] | None = ".env") -> None:
    """Populate ``os.environ`` from a ``.env`` file (via python-dotenv, no override).

    A real process env var therefore always wins (``override=False``); the CLI's
    ``--api-key`` (handed straight to the SDK) wins above both. No-op when the file
    is absent or ``path`` is falsy — so callers (CLI, runner) invoke it
    unconditionally. ``python-dotenv`` (core dep) does the parsing.
    """
    if not path:
        return
    if importlib.util.find_spec("dotenv") is not None:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
        load_dotenv(path, override=False)   # override=False ⇒ real env wins


# =========================================================================== #
# Storage layout
# =========================================================================== #
# On-disk layout under the CLI's `--root` (a path the user passes, or ".archforge" —
# a tunable, but the dirnames themselves are plumbing). The stores name their
# subdirectories + pointer files from these; rare to change.
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

# The CLI program name (the installed console command + the prefix on result lines).
PROG: str = "archforge-optimizer"

# The CLI's free-run scripted fixtures: when `--provider scripted` is invoked
# with no injected `components`, the evolve family runs against this one-task suite
# so `archforge-optimizer evolve` is runnable end-to-end with zero configuration.
DEFAULT_SUITE_ID: str = "cli-default"
DEFAULT_TASK_ID: str = "t1"
DEFAULT_TASK_INPUT: str = "hello"


__all__ = [
    # identity
    "VERSION",
    # llm provider roster
    "SCRIPTED_PROVIDER", "REAL_PROVIDERS", "ALL_PROVIDERS",
    # project environment (.env loader for provider API keys)
    "load_env",
    # storage layout
    "SPECS_DIRNAME", "ATTEMPTS_DIRNAME", "TRACES_DIRNAME",
    "ACTIVE_POINTER_FILE", "ARCHIVED_FILE",
    # internals
    "SPEC_ID_HASH_LEN", "SHORT_HASH_LEN", "ANTHROPIC_DEFAULT_MAX_TOKENS",
    "SCRIPTED_NOISE_PATTERN", "PROG",
    "DEFAULT_SUITE_ID", "DEFAULT_TASK_ID", "DEFAULT_TASK_INPUT",
]
