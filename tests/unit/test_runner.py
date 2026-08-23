"""Unit tests for `archforge.runner._build_organs` — the Architect/Judge wiring seam.

Pins the bug where ``RunnerConfig.rubric`` (a string id) was passed straight to
``Judge`` whenever it was NOT the literal ``"default-v1"`` (the old
``rubric=default_rubric() if cfg.rubric == "default-v1" else cfg.rubric``
ternary). A string landed where a ``Rubric`` is expected, so the first
``judge.score(...)`` crashed on ``self._rubric.sub_rubrics.items()`` with
``AttributeError: 'str' object has no attribute 'sub_rubrics'``.

The trigger is editing ``DEFAULT_RUBRIC_ID`` in ``.archforge/archforge.py`` away
from ``"default-v1"`` (e.g. bumping to ``"default-v2"`` after a sub-rubric
change) and then running the embedder API — ``run_loop``/``run_cycle`` (and the
Lumina glue through them). The CLI was already robust (it always builds
``default_rubric()``); this brings the kit API to parity.

`make_client` is stubbed so no SDK/key is needed — the regression never makes an
LLM call (it would only at ``Judge.score``, which these tests don't invoke).
"""
from __future__ import annotations

import pytest

from archforge import runner, userconfig as ucfg
from archforge.config_init import archforge_config_text
from archforge.judge.base import Rubric
from archforge.runner import RunnerConfig, _build_organs


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_cfg():
    """Fresh resolver cache per test (no cross-test resolve leak)."""
    ucfg._reset_cache()
    yield
    ucfg._reset_cache()


@pytest.fixture(autouse=True)
def _noop_make_client(monkeypatch):
    """`_build_organs` would otherwise build a real SDK client (needs keys/SDK).
    The rubric regression never makes an LLM call, so a sentinel client isolates it."""
    monkeypatch.setattr(runner, "make_client", lambda *a, **k: object())


def _bumped_rubric_config_text() -> str:
    """The real generated archforge.py with DEFAULT_RUBRIC_ID bumped off the
    literal 'default-v1' (to 'regression-v9') — exactly the edit a user makes
    when they change the sub-rubrics and bump the version to keep scores
    comparable. A complete config (all 19 names) so RunnerConfig's default
    factories resolve without KeyError."""
    return archforge_config_text().replace(
        'DEFAULT_RUBRIC_ID = "default-v1"',
        'DEFAULT_RUBRIC_ID = "regression-v9"',
    )


# --------------------------------------------------------------------------- #
# the (unchanged) happy path: default-v1 under the pytest gate
# --------------------------------------------------------------------------- #


def test_build_organs_rubric_is_a_real_object_for_default_v1(tmp_path, monkeypatch):
    """Under the gate (template sane default, rubric_id 'default-v1'), the Judge
    is wired with a Rubric object — never a bare str. Pins the unchanged path so
    the fix can't quietly regress it."""
    monkeypatch.chdir(tmp_path)            # so load_env(".env") finds nothing
    cfg = RunnerConfig(provider="gemini", api_key="dummy")
    arch, judge = _build_organs(cfg)
    assert arch is not None
    assert isinstance(judge._rubric, Rubric)
    assert judge._rubric.rubric_id == "default-v1"
    assert isinstance(judge._rubric.sub_rubrics, dict)        # the old crash attribute


# --------------------------------------------------------------------------- #
# the regression: a non-"default-v1" rubric id no longer hands a str to Judge
# --------------------------------------------------------------------------- #


def test_build_organs_survives_non_default_v1_rubric_id(tmp_path, monkeypatch):
    """Regression: an active disk config with DEFAULT_RUBRIC_ID edited away from
    the literal 'default-v1' must still hand the Judge a Rubric — NOT the string
    id. The old `else cfg.rubric` branch crashed here (AttributeError at first
    score). Force the gate off (disk branch) against a tmp project."""
    (tmp_path / ".archforge").mkdir()
    (tmp_path / ".archforge" / "archforge.py").write_text(
        _bumped_rubric_config_text(), encoding="utf-8",
    )
    monkeypatch.setattr(ucfg, "_disabled", lambda: False)
    monkeypatch.chdir(tmp_path)
    ucfg._reset_cache()

    cfg = RunnerConfig(provider="gemini", api_key="dummy")
    arch, judge = _build_organs(cfg)          # the old code raised AttributeError here

    assert isinstance(judge._rubric, Rubric)       # a real Rubric — not the "regression-v9" str
    assert judge._rubric.rubric_id == "regression-v9"
    assert isinstance(judge._rubric.sub_rubrics, dict)            # the old crash attribute
    assert judge._rubric.sub_rubrics["correctness"].startswith("Is the final")


def test_runnerconfig_rubric_default_differs_when_disk_says_so(tmp_path, monkeypatch):
    """RunnerConfig.rubric (a string id) still resolves from the active config —
    it's kept for the embedder's metadata use; the fix just no longer SELECTS a
    grading object from it. Confirms the field wiring is intact (and that the
    bug was specifically the consumption at _build_organs, not the field itself)."""
    (tmp_path / ".archforge").mkdir()
    (tmp_path / ".archforge" / "archforge.py").write_text(
        _bumped_rubric_config_text(), encoding="utf-8",
    )
    monkeypatch.setattr(ucfg, "_disabled", lambda: False)
    monkeypatch.chdir(tmp_path)
    ucfg._reset_cache()
    cfg = RunnerConfig(provider="gemini", api_key="dummy")
    assert cfg.rubric == "regression-v9"        # the field reads the disk value
    assert _build_organs(cfg)                   # and _build_organs no longer cares
