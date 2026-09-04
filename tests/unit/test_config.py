"""Unit tests for `archforge.config.load_env` + `archforge.userconfig` (the resolver).

Pins two contracts:

  * ``load_env`` — the ``.env`` loader precedence that makes a pip-installed
    ``archforge`` "just work" once the user drops keys in a project ``.env``:
    ``--api-key`` (passed to make_client) > a real process env var > ``.env``.
    Real env always wins (``override=False``); an absent file is a no-op.
    ``python-dotenv`` (core dep) does the parsing — quotes/spacing/export prefixes
    stripped, comments/blanks/malformed lines skipped.
  * ``archforge.userconfig`` — the lazy tunables resolver. Under the test runner
    (``"pytest" in sys.modules``) it execs the SAME sane-default ``TEMPLATE`` that
    ``init`` writes, in-memory, so the suite never depends on a per-test disk file
    (e.g. ``Thresholds().tau == 0.05``). The disk branch is exercised here by
    forcing the gate off (``monkeypatch``-ing ``_disabled``) + ``chdir``-ing to a
    tmp project that has a ``.archforge/archforge.py``; the init-required contract
    is pinned with a tmp project that has NONE.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from archforge import userconfig as ucfg, config
from archforge.config import load_env
from archforge.config_init import TEMPLATE, EDITABLE_NAMES, archforge_config_text


# --------------------------------------------------------------------------- #
# resolver: every test starts with a fresh cache (no cross-test resolve leak)
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_cfg():
    ucfg._reset_cache()
    yield
    ucfg._reset_cache()


# --------------------------------------------------------------------------- #
# load_env fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    """Write a representative `.env`: values, blanks, comments, quoted values."""
    p = tmp_path / ".env"
    p.write_text(
        "# a comment line (skipped)\n"
        "GROQ_API_KEY=gsk_plain\n"
        '\n'                              # blank line (skipped)
        'GEMINI_API_KEY="quoted double"\n'
        "OPENAI_API_KEY='quoted single'\n"
        "   ANTHROPIC_API_KEY = spaced   \n"   # surrounding whitespace stripped
        "#no=equals-here is also skipped\n"
        "MALFORMED_NO_EQUALS\n"               # no '=' → skipped, not a raise
        "\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """No cross-test leakage of the loader's own keys."""
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
              "ANTHROPIC_API_KEY", "MALFORMED_NO_EQUALS"):
        monkeypatch.delenv(k, raising=False)
    yield


# --------------------------------------------------------------------------- #
# load_env tests (config.py still owns the .env loader)
# --------------------------------------------------------------------------- #


def test_load_env_populates_from_file(env_file: Path):
    load_env(env_file)
    assert os.environ["GROQ_API_KEY"] == "gsk_plain"
    assert os.environ["OPENAI_API_KEY"] == "quoted single"
    assert os.environ["ANTHROPIC_API_KEY"] == "spaced"


def test_load_env_strips_quotes(env_file: Path):
    load_env(env_file)
    assert os.environ["GEMINI_API_KEY"] == "quoted double"   # surrounding " gone
    assert os.environ["OPENAI_API_KEY"] == "quoted single"   # surrounding ' gone


def test_load_env_skips_comments_blanks_and_malformed(env_file: Path):
    load_env(env_file)
    assert "MALFORMED_NO_EQUALS" not in os.environ            # no '=' → skipped


def test_load_env_precedence_real_env_wins(env_file: Path, monkeypatch):
    """override=False: a pre-existing real env var is NOT overwritten."""
    monkeypatch.setenv("GROQ_API_KEY", "from_real_env")
    load_env(env_file)
    assert os.environ["GROQ_API_KEY"] == "from_real_env"     # .env did NOT clobber


def test_load_env_missing_file_is_noop(tmp_path, monkeypatch):
    """No raise, no mutation when the file is absent (the common case)."""
    monkeypatch.setenv("ZZ_TEST_KEY", "present_before")
    load_env(tmp_path / "does_not_exist.env")
    assert os.environ["ZZ_TEST_KEY"] == "present_before"     # untouched


def test_load_env_empty_or_none_path_is_noop(monkeypatch):
    """Passing '' or None must not blow up (the CLI/runner pass args.env_file)."""
    monkeypatch.setenv("ZZ_TEST_KEY", "present_before")
    load_env(None)
    load_env("")
    assert os.environ["ZZ_TEST_KEY"] == "present_before"


def test_default_env_file_tunable_is_dotenv_under_gate():
    # DEFAULT_ENV_FILE is now a tunable resolved by ucfg; under the pytest gate its
    # template value is ".env" (matches the historic shipped default).
    assert ucfg.get("DEFAULT_ENV_FILE") == ".env"


# --------------------------------------------------------------------------- #
# archforge.userconfig — the lazy resolver (the new tunables bridge)
# --------------------------------------------------------------------------- #


def test_disabled_gate_is_on_under_pytest():
    # pytest is imported long before archforge at collection → the gate the suite
    # relies on (sane template defaults, no per-test disk file) is literally true.
    assert "pytest" in sys.modules
    assert ucfg._disabled() is True


def test_get_returns_template_sane_defaults_under_gate():
    # Thresholds()/EngineConfig()/RunnerConfig() defaults all read through ucfg;
    # under the gate they resolve to the SAME template `init` ships.
    assert ucfg.get("DEFAULT_TAU") == 0.05
    assert ucfg.get("DEFAULT_DELTA") == 0.07
    assert ucfg.get("DEFAULT_REPEATS") == 1
    assert ucfg.get("MAX_REPEATS") == 3
    assert ucfg.get("DEFAULT_MAX_CYCLES") == 20
    assert ucfg.get("PROVIDER") == "gemini"
    assert ucfg.get("DEFAULT_RUBRIC_ID") == "default-v1"
    assert ucfg.get("BACKOFF_CAP_SECONDS") == 30.0
    assert ucfg.get("DEFAULT_ARCHITECT_MODELS")["gemini"] == "gemini-3.6-flash"
    assert ucfg.get("DEFAULT_JUDGE_MODELS")["gemini"] == "gemini-3.6-flash"
    assert ucfg.get("DEFAULT_SUB_RUBRICS")["correctness"].startswith("Is the final")
    assert ucfg.get("DEFAULT_SUITE_FILE") == ".archforge/suite.json"


def test_template_execs_to_every_editable_name():
    # The TEMPLATE the resolver execs (in-memory, under the gate) must define every
    # editable name — the same string `init` writes to disk.
    ns: dict = {}
    exec(compile(TEMPLATE, "<template>", "exec"), ns)   # noqa: S102
    assert set(EDITABLE_NAMES) <= set(ns)
    assert ns["DEFAULT_TAU"] == 0.05
    assert len(EDITABLE_NAMES) == 23


def test_editable_names_match_locked_set():
    assert set(EDITABLE_NAMES) == {
        "PROVIDER", "DEFAULT_ARCHITECT_MODELS", "DEFAULT_JUDGE_MODELS",
        "DEFAULT_TAU", "DEFAULT_DELTA",
        "DEFAULT_REPEATS", "MAX_REPEATS", "DEFAULT_UNRUNNABLE_FRAC",
        "DEFAULT_PLATEAU_CYCLES", "DEFAULT_MAX_CYCLES", "DEFAULT_JUDGE_RETRIES",
        "BACKOFF_CAP_SECONDS", "DEFAULT_RUBRIC_ID", "DEFAULT_SUB_RUBRICS",
        "DEFAULT_ROOT_DIR", "DEFAULT_ENV_FILE", "DEFAULT_SUITE_FILE",
        "DEFAULT_MAX_TOKENS_TOTAL", "DEFAULT_MAX_TOKENS_PER_CYCLE",
        "DEFAULT_MAX_WALL_MS_PER_CYCLE",
        "DEFAULT_TRACE_TOTAL_BUDGET_TOK",
        "DEFAULT_EVALUATOR", "DEFAULT_DEEPEVAL_METRICS",
    }


def test_disk_file_overrides_when_gate_forced_off(tmp_path, monkeypatch):
    """A project's .archforge/archforge.py is the SOLE tunable source in production.
    The pytest gate is forced off here (monkeypatch _disabled) so the disk branch
    runs against a tmp project that DOES have the file."""
    (tmp_path / ".archforge").mkdir()
    (tmp_path / ".archforge" / "archforge.py").write_text(
        "DEFAULT_TAU = 0.42\nDEFAULT_SUB_RUBRICS = {'a': 'x'}\n", encoding="utf-8")
    monkeypatch.setattr(ucfg, "_disabled", lambda: False)
    monkeypatch.chdir(tmp_path)
    ucfg._reset_cache()
    assert ucfg.get("DEFAULT_TAU") == 0.42
    assert ucfg.get("DEFAULT_SUB_RUBRICS") == {"a": "x"}
    # a name the file omits, with a default fallback, returns the default (not err)
    assert ucfg.get("DEFAULT_MAX_CYCLES", default=99) == 99


def test_missing_config_raises_when_gate_forced_off(tmp_path, monkeypatch):
    """The init-required contract: with no disk file and the gate OFF, a tuned use
    raises ConfigNotInitialized carrying the init hint (no bogus fallback)."""
    monkeypatch.setattr(ucfg, "_disabled", lambda: False)
    monkeypatch.chdir(tmp_path)        # empty tmp — no .archforge/archforge.py
    ucfg._reset_cache()
    with pytest.raises(ucfg.ConfigNotInitialized) as exc:
        ucfg.get("DEFAULT_TAU")
    assert "archforge-optimizer init" in str(exc.value)


def test_ensure_initialized_raises_pre_init_when_gate_off(tmp_path, monkeypatch):
    """The evolve gate: a real run in an un-initialized project fails cleanly
    (rather than building a real provider with bogus defaults)."""
    monkeypatch.setattr(ucfg, "_disabled", lambda: False)
    monkeypatch.chdir(tmp_path)
    ucfg._reset_cache()
    with pytest.raises(ucfg.ConfigNotInitialized):
        ucfg.ensure_initialized()


def test_ensure_initialized_noop_under_gate_even_with_no_file(tmp_path, monkeypatch):
    """Under pytest the gate is on → ensure_initialized is a no-op (tests resolve
    template defaults; the disk file's absence is irrelevant)."""
    monkeypatch.chdir(tmp_path)
    ucfg._reset_cache()
    # must not raise (disabled branch short-circuits)
    ucfg.ensure_initialized()


def test_active_default_file_is_sane_out_of_box(tmp_path):
    # archforge_config_text() — exactly what `init` writes — ships ACTIVE sane
    # defaults (not commented no-ops), so the CLI works immediately after init.
    ns: dict = {}
    exec(compile(archforge_config_text(), "archforge.py", "exec"), ns)   # noqa: S102
    assert ns["DEFAULT_TAU"] == 0.05
    assert ns["PROVIDER"] == "gemini"
    assert ns["DEFAULT_ARCHITECT_MODELS"]["gemini"] == "gemini-3.6-flash"
    assert ns["DEFAULT_JUDGE_MODELS"]["gemini"] == "gemini-3.6-flash"
    assert ns["DEFAULT_SUITE_FILE"] == ".archforge/suite.json"
    # the ACTIVE assignment is literal in the text (a user can find + edit it)
    assert 'DEFAULT_SUITE_FILE = ".archforge/suite.json"' in archforge_config_text()


def test_config_module_has_no_tunables():
    # config.py is SYSTEM-ONLY: none of the tunable names leaked into it.
    leaks = [n for n in (
        "DEFAULT_TAU", "PROVIDER", "DEFAULT_ARCHITECT_MODELS",
        "DEFAULT_JUDGE_MODELS", "DEFAULT_ENV_FILE",
        "DEFAULT_ROOT_DIR", "DEFAULT_MAX_CYCLES", "BACKOFF_CAP_SECONDS",
        "DEFAULT_RUBRIC_ID", "DEFAULT_SUB_RUBRICS", "DEFAULT_DELTA",
        "DEFAULT_REPEATS", "MAX_REPEATS", "DEFAULT_UNRUNNABLE_FRAC",
        "DEFAULT_PLATEAU_CYCLES", "DEFAULT_JUDGE_RETRIES",
        "DEFAULT_MAX_TOKENS_TOTAL", "DEFAULT_MAX_TOKENS_PER_CYCLE",
        "DEFAULT_MAX_WALL_MS_PER_CYCLE", "DEFAULT_TRACE_TOTAL_BUDGET_TOK",
        "DEFAULT_SUITE_FILE",
        "load_user_config", "_EDITABLE", "_config_disabled",
    ) if hasattr(config, n)]
    assert leaks == []
