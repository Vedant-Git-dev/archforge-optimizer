"""Unit tests for `archforge.suite.load_suite_file` — the JSON sidecar loader.

Pins the contract for `.archforge/suite.json` (made by `init`, overridable via
`--suite`):

  * absent / falsy path → ``None`` (so the CLI's ``load_suite_file(p) or <fallback>``
    drops straight back to the one-task fixture — no special-casing);
  * a valid file → a ``Suite``; an omitted ``rubric_id`` is filled from the active
    rubric (``default_rubric().rubric_id``), and a per-task ``rubric_id`` survives;
  * a present-but-malformed file → ``ValueError`` (NOT silent coerce) since the user
    wrote a file they expect to load.

Runs under the pytest gate, so ``default_rubric()`` resolves the sane template
(``default-v1``) — no per-test disk config. Uses ``tmp_path``; never touches the
real ``.env``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from archforge.config_init import _DEFAULT_SUITE_JSON
from archforge.judge.base import default_rubric
from archforge.suite import Suite, load_suite_file


def test_absent_file_returns_none(tmp_path: Path):
    # the CLI resolves `load_suite_file(p) or <fallback>`; absence drops to fallback.
    assert load_suite_file(tmp_path / "no_such_suite.json") is None


def test_falsy_path_returns_none():
    # CLI/runner pass args that may be None / empty; falsy must not be a
    # FileNotFoundError, just None.
    assert load_suite_file(None) is None
    assert load_suite_file("") is None


def test_starter_json_round_trips_to_default_fixture(tmp_path: Path):
    """`init`'s seeded suite.json == the CLI's one-task fallback fixture."""
    f = tmp_path / "suite.json"
    f.write_text(_DEFAULT_SUITE_JSON, encoding="utf-8")
    loaded = load_suite_file(f)
    assert loaded is not None
    assert isinstance(loaded, Suite)
    assert loaded.suite_id == "cli-default"
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].task_id == "t1"
    assert loaded.tasks[0].input == "hello"
    # the file omits rubric_id -> the loader fills the active default (default-v1
    # under the pytest gate, matching the hardcoded CLI fallback).
    assert loaded.rubric_id == default_rubric().rubric_id == "default-v1"


def test_omitted_rubric_id_defaults_to_active(tmp_path: Path):
    f = tmp_path / "suite.json"
    f.write_text(json.dumps({"suite_id": "s2", "tasks": [{"task_id": "a", "input": "x"}]}),
                 encoding="utf-8")
    loaded = load_suite_file(f)
    assert loaded is not None
    assert loaded.rubric_id == default_rubric().rubric_id   # filled, not None
    assert loaded.tasks[0].rubric_id is None                # per-task unset


def test_explicit_suite_rubric_id_is_used(tmp_path: Path):
    f = tmp_path / "suite.json"
    f.write_text(json.dumps({"suite_id": "s3", "rubric_id": "rubric-x",
                             "tasks": [{"task_id": "a", "input": "x"}]}),
                 encoding="utf-8")
    loaded = load_suite_file(f)
    assert loaded is not None
    assert loaded.rubric_id == "rubric-x"


def test_per_task_rubric_id_survives(tmp_path: Path):
    """A task may carry its own rubric_id (overrides the suite default at scoring
    time — SuiteRunner.run_suite does `task.rubric_id or suite.rubric_id`). The
    loader must preserve it on the Task."""
    f = tmp_path / "suite.json"
    f.write_text(json.dumps({"suite_id": "s4", "rubric_id": "rubric-x",
                             "tasks": [{"task_id": "a", "input": "x",
                                        "rubric_id": "rubric-y"}]}),
                 encoding="utf-8")
    loaded = load_suite_file(f)
    assert loaded is not None
    assert loaded.tasks[0].rubric_id == "rubric-y"   # per-task supersedes suite default


def test_malformed_json_raises_value_error(tmp_path: Path):
    """A present-but-broken file does NOT silently coerce/fallback — the user wrote
    a file they expect to use, so a clear ValueError beats a confusing wrong run."""
    f = tmp_path / "suite.json"
    f.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_suite_file(f)
    assert "not valid JSON" in str(exc.value)


def test_missing_required_keys_raise_value_error(tmp_path: Path):
    f = tmp_path / "suite.json"
    f.write_text(json.dumps({"tasks": []}), encoding="utf-8")   # no suite_id
    with pytest.raises(ValueError) as exc:
        load_suite_file(f)
    assert "suite_id" in str(exc.value)


def test_bad_task_shape_raises_value_error(tmp_path: Path):
    f = tmp_path / "suite.json"
    f.write_text(json.dumps({"suite_id": "s", "tasks": [{"task_id": "a"}]}),
                 encoding="utf-8")   # task missing required `input`
    with pytest.raises(ValueError) as exc:
        load_suite_file(f)
    assert "malformed task" in str(exc.value)


def test_extra_task_fields_are_allowed(tmp_path: Path):
    """Task has extra='allow' (the host may stamp routing/labels); a sensible extra
    like `expected` must survive — it is not a malformed task."""
    f = tmp_path / "suite.json"
    f.write_text(json.dumps({"suite_id": "s", "tasks": [
        {"task_id": "a", "input": "x", "expected": "answer"}]}), encoding="utf-8")
    loaded = load_suite_file(f)
    assert loaded is not None
    assert loaded.tasks[0].task_id == "a"
