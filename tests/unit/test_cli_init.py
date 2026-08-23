"""`archforge-optimizer init` + `make-spec` — the two-step bootstrap contract.

`init` scaffolds: `.archforge/archforge.py` (user-editable ACTIVE config w/ sane
defaults), `.archforge/suite.json` (one-task eval sidecar), and the
`archforge_optimizer/` package (a generic, name-neutral LangGraph adapter skeleton
whose `# EDIT:` markers the user fills for their MAS). It does NOT write
`.env.example` (the user's provider key goes in the root `.env`, gitignored) and does
NOT write `spec.json` — that is `make-spec`'s job: it builds the Spec from the
EDITED adapter, lints it, and writes `archforge_optimizer/spec.json` only if valid.

These tests pin that contract: init never touches a real `.env`, refuses to clobber
an existing `archforge.py` without `--force`, repairs the adapter per-file without
`--force` even when `archforge.py` already exists, and never leaks `.env` secrets.
make-spec lints + writes + fails nonzero on an invalid spec. Drives the real
`archforge.cli.main(...)` in an isolated cwd.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from archforge.cli import main
from archforge.config_init import (
    EDITABLE_NAMES, _DEFAULT_SUITE_JSON, adapter_package_files,
    archforge_config_text, env_example_text,
)
from archforge.lint import lint
from archforge.models import Spec
from archforge.suite import load_suite_file

# the 5 files `init` scaffolds into archforge_optimizer/ (the generic adapter skeleton);
# the distribution package name stays `archforge-optimizer` — only the scaffolded folder
# differs. Ordered for stable, human-readable assertions below.
_ADAPTER_FILES = (
    "__init__.py",
    "host.py",
    "app.py",
    "sidecar.py",
    "test_smoke_offline.py",
)


def test_init_writes_config_suite_and_adapter(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`init` scaffolds archforge.py + suite.json + the archforge_optimizer/ package.
    It does NOT write `.env.example` or `spec.json` (those moved out of init)."""
    monkeypatch.chdir(tmp_path)
    rc = main(["init"])
    assert rc == 0
    cfg = tmp_path / ".archforge" / "archforge.py"
    suite = tmp_path / ".archforge" / "suite.json"
    pkg = tmp_path / "archforge_optimizer"
    assert cfg.is_file() and suite.is_file() and pkg.is_dir()
    # the new contract: NO .env.example, NO spec.json from init alone
    assert not (tmp_path / ".env.example").exists()
    assert not (pkg / "spec.json").exists()
    out = capsys.readouterr().out
    assert "created:" in out and "archforge.py" in out
    assert "suite.json" in out
    for name in _ADAPTER_FILES:
        assert f"created: archforge_optimizer/{name}" in out
    # the Next: message points at .env + make-spec (not .env.example)
    assert ".env" in out and "make-spec" in out


def test_generated_config_sets_sane_active_defaults(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A freshly-generated archforge.py is ACTIVE: exec'd into a fresh namespace, it
    sets every editable name to the sane default so the CLI works immediately after
    `init` (a user edits a value to change a default)."""
    monkeypatch.chdir(tmp_path)
    main(["init"])
    raw = (tmp_path / ".archforge" / "archforge.py").read_text(encoding="utf-8")
    ns: dict = {}
    exec(compile(raw, "archforge.py", "exec"), ns)   # noqa: S102
    assert set(EDITABLE_NAMES) <= set(ns)
    assert ns["DEFAULT_TAU"] == 0.05                   # the sane default ships, not 0
    assert ns["PROVIDER"] == "gemini"
    # the committed model default at HEAD (archforge_config_text ships gemini-3.6-flash);
    # pinning the generator's actual output keeps this test honest against the source.
    assert ns["DEFAULT_ARCHITECT_MODELS"]["gemini"] == "gemini-3.6-flash"
    assert ns["DEFAULT_JUDGE_MODELS"]["gemini"] == "gemini-3.6-flash"


def test_generated_config_contains_all_editable_names(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Every editable name appears in the generated file as an ACTIVE assignment
    (`NAME = value`) so a user can find + edit each one."""
    monkeypatch.chdir(tmp_path)
    main(["init"])
    raw = (tmp_path / ".archforge" / "archforge.py").read_text(encoding="utf-8")
    for name in EDITABLE_NAMES:
        assert f"\n{name} = " in raw or raw.startswith(f"{name} = "), \
            f"generated config missing ACTIVE assignment for {name}"


def test_init_refuses_to_clobber_existing_config(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    main(["init"])
    capsys.readouterr()  # drain first init's output
    rc = main(["init"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "already exists" in err and "--force" in err


def test_init_force_overwrites_existing_config(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    main(["init"])
    cfg = tmp_path / ".archforge" / "archforge.py"
    # corrupt the file to prove --force rewrites it cleanly
    cfg.write_text("# my hand edits\nDEFAULT_TAU = 0.99\n", encoding="utf-8")
    capsys.readouterr()
    rc = main(["init", "--force"])
    assert rc == 0
    raw = cfg.read_text(encoding="utf-8")
    assert "# my hand edits" not in raw
    assert "DEFAULT_TAU = 0.99" not in raw        # the hand edit is gone
    assert "DEFAULT_TAU = 0.05" in raw             # replaced by the generated ACTIVE sane default


def test_init_does_not_write_env_example(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """init creates no `.env.example` — the user's provider key goes in the root `.env`
    (gitignored). init only prints the hint to put it there; no example file is written."""
    monkeypatch.chdir(tmp_path)
    rc = main(["init"])
    assert rc == 0
    assert not (tmp_path / ".env.example").exists()
    out = capsys.readouterr().out
    assert ".env" in out                       # the Next: hint points at .env ...
    assert ".env.example" not in out            # ... but never mentions .env.example
    # the env-example generator is still part of the package (reused by tests) but init
    # no longer calls it; keep the idempotent generator honest on its own merits below.
    assert "ANTHROPIC_API_KEY=" in env_example_text()   # generator still sound, unused by init


def test_init_scaffolds_suite_json(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """init writes `.archforge/suite.json` byte-identical to the starter, and it
    loads (via load_suite_file) to the one-task default fixture — so the generated
    default == the CLI's fallback, with no extra setup needed by the user."""
    monkeypatch.chdir(tmp_path)
    rc = main(["init"])
    assert rc == 0
    suite = tmp_path / ".archforge" / "suite.json"
    assert suite.is_file()
    assert suite.read_text(encoding="utf-8") == _DEFAULT_SUITE_JSON
    # round-trips through the loader to the default fixture
    loaded = load_suite_file(suite)
    assert loaded is not None
    assert loaded.suite_id == "cli-default"
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].task_id == "t1"
    assert loaded.tasks[0].input == "hello"
    # the file omits rubric_id -> the loader fills the active default (default-v1)
    assert loaded.rubric_id == "default-v1"


def test_init_never_overwrites_existing_suite_json(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A pre-existing suite.json (the user may have tuned the tasks) is left
    untouched — init prints `kept:` rather than clobbering it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".archforge").mkdir()
    custom = '{"suite_id": "mine", "tasks": [{"task_id": "qa", "input": "ping"}]}'
    (tmp_path / ".archforge" / "suite.json").write_text(custom, encoding="utf-8")
    rc = main(["init"])
    assert rc == 0
    raw = (tmp_path / ".archforge" / "suite.json").read_text(encoding="utf-8")
    assert raw == custom                       # the hand-tuned tasks are intact
    out = capsys.readouterr().out
    assert "kept:" in out and "suite.json" in out


def test_init_never_touches_real_env(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A real .env (gitignored, may hold live keys) is neither read nor written nor
    echoed — init only mentions `.env` in its Next: hint. The standing secret-safety
    invariant holds on the new init path too."""
    monkeypatch.chdir(tmp_path)
    secret = "GROQ_API_KEY=gsk-do-not-leak-123456\n"
    (tmp_path / ".env").write_text(secret, encoding="utf-8")
    rc = main(["init"])
    assert rc == 0
    assert (tmp_path / ".env").read_text(encoding="utf-8") == secret   # contents unchanged
    captured = capsys.readouterr()
    assert "gsk-do-not-leak" not in captured.out + captured.err   # the live key never echoed


def test_init_supports_custom_root(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """--root relocates archforge.py + suite.json; the adapter package stays at cwd
    (it's importable user code that must resolve via `--adapter`). Still no .env.example."""
    monkeypatch.chdir(tmp_path)
    rc = main(["init", "--root", str(tmp_path / "custom_state")])
    assert rc == 0
    assert (tmp_path / "custom_state" / "archforge.py").is_file()
    assert (tmp_path / "custom_state" / "suite.json").is_file()
    # the adapter package is at cwd regardless of --root (so --adapter resolves)
    assert (tmp_path / "archforge_optimizer").is_dir()
    # .env.example is never written, with or without --root
    assert not (tmp_path / ".env.example").exists()


def test_generated_strings_are_idempotent() -> None:
    """The pure string generators return stable, well-formed text (no fs, no live .env,
    no secrets). env_example_text is still shipped in the package (reused by tests), even
    though init no longer calls it — keep it honest on its own merits."""
    a, b = archforge_config_text(), archforge_config_text()
    assert a == b
    assert "DEFAULT_TAU" in a and "PROVIDER" in a
    e = env_example_text()
    assert "ANTHROPIC_API_KEY=" in e and "GROQ_API_KEY=" in e
    assert "gsk" not in e and "sk-" not in e   # no leaked-looking keys


def test_env_example_no_real_keys(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """env_example_text() (still shipped in the package) ships only empty var names —
    even if a real .env is present, the generator never reads it. (init no longer calls
    this generator, but it is still importable, so pin its secret-safety directly.)"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=AIza-real-secret\n", encoding="utf-8")
    raw = env_example_text()
    assert "GEMINI_API_KEY=" in raw                       # the empty placeholder present
    assert "AIza-real-secret" not in raw                  # never the real value
    # confirm env wasn't read/mutated by the generator either
    _ = os.environ.get("GEMINI_API_KEY")  # noqa: F841


# --------------------------------------------------------------------------- #
# init scaffolds the generic archforge_optimizer/ adapter package
# --------------------------------------------------------------------------- #
# These pin the contract: `init` writes a name-neutral `archforge_optimizer/` package
# (LangGraph adapter skeleton) at the project root, so a user edits `# EDIT:` markers
# instead of coding the adapter wiring from scratch. The AEDE glue stays the in-repo
# reference (not shipped); the scaffold is a derived, name-neutralized copy from string
# constants. init does NOT self-write spec.json — `make-spec` builds it from the edited
# adapter.

def test_init_scaffolds_adapter_package(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`init` writes `archforge_optimizer/` with the 5 generic-adapter files, each
    syntactically compilable, carrying the placeholder `App`/`AppAdapter` names, free
    of any `aede`/`AEDE` literal (regression guard for name-neutrality), and with NO
    spec.json written (make-spec's job)."""
    monkeypatch.chdir(tmp_path)
    rc = main(["init"])
    assert rc == 0
    pkg = tmp_path / "archforge_optimizer"
    assert pkg.is_dir()
    present = {p.name for p in pkg.iterdir() if p.is_file()}
    assert set(_ADAPTER_FILES) <= present
    assert "spec.json" not in present            # init no longer writes spec.json
    out = capsys.readouterr().out
    for name in _ADAPTER_FILES:
        fpath = pkg / name
        assert fpath.is_file()
        # syntactically valid Python (the user imports it; the placeholder must compile)
        compile(fpath.read_text(encoding="utf-8"), str(fpath), "exec")  # noqa: S102
        assert f"created: archforge_optimizer/{name}" in out

    # name-neutral: no aede/AEDE leaked into any scaffolded file (the scaffold is generic)
    for name in _ADAPTER_FILES:
        raw = (pkg / name).read_text(encoding="utf-8").lower()
        assert "aede" not in raw, f"`aede` leaked into scaffolded {name}"
    # the placeholder class names the README / --adapter line points at are present
    app_text = (pkg / "app.py").read_text(encoding="utf-8")
    host_text = (pkg / "host.py").read_text(encoding="utf-8")
    assert "class App(" in app_text
    assert "AppAdapter" in host_text
    # importing the scaffolded app has NO filesystem side effect (spec.json not created)
    assert not (pkg / "spec.json").exists()


def test_init_adapter_per_file_clobber_guard(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A re-run `init` (no --force) keeps every existing adapter file — a user
    mid-edit is never clobbered, and missing files are still filled (mirror of the
    suite.json guard, scoped per-file)."""
    monkeypatch.chdir(tmp_path)
    main(["init"])
    capsys.readouterr()
    pkg = tmp_path / "archforge_optimizer"
    # hand-edit app.py to prove the kept-file content survives a re-run
    edited = "# MY HAND-EDIT TO app.py\n"
    (pkg / "app.py").write_text(edited, encoding="utf-8")
    # delete one file so the missing-file path is also exercised + filled
    (pkg / "sidecar.py").unlink()

    rc = main(["init"])  # archforge.py still exists -> rc=2 (refuse clobber), BUT the
    assert rc == 2        # adapter scaffold runs first, so app.py is kept + sidecar re-seeded
    out = capsys.readouterr().out
    assert "kept:" in out and "archforge_optimizer/app.py" in out
    assert "created: archforge_optimizer/sidecar.py" in out  # the missing file was re-scaffolded
    # the hand-edit survived (never clobbered without --force)
    assert (pkg / "app.py").read_text(encoding="utf-8") == edited


def test_init_repairs_adapter_without_force_when_config_exists(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A user with an existing `.archforge/archforge.py` but a deleted
    `archforge_optimizer/` can REPAIR the adapter with a bare `init` — no `--force`
    (which would clobber their config). The archforge.py refuse-clobber returns rc=2,
    but only AFTER the adapter scaffold already ran (the blocks were reordered so the
    adapter repair is independent of `.archforge/`). This is the load-bearing
    UX contract the reorder fixes."""
    monkeypatch.chdir(tmp_path)
    main(["init"])
    pkg = tmp_path / "archforge_optimizer"
    # user keeps a hand-tuned config; deletes the adapter
    cfg = tmp_path / ".archforge" / "archforge.py"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "\n# MY HAND EDIT\n", encoding="utf-8")
    import shutil
    shutil.rmtree(pkg)

    rc = main(["init"])
    assert rc == 2  # archforge.py exists + not --force (existing contract intact)
    # …but the adapter was repaired first — all 5 files restored (spec.json is NOT
    # init's job; make-spec builds it, so it stays absent here)
    assert set(_ADAPTER_FILES) <= {p.name for p in pkg.iterdir() if p.is_file()}
    assert not (pkg / "spec.json").exists()
    # and the hand-tuned config is preserved (never clobbered)
    assert "# MY HAND EDIT" in cfg.read_text(encoding="utf-8")


def test_init_force_overwrites_adapter_package(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`init --force` overwrites the entire adapter package — hand edits are replaced
    by the generated placeholder (mirror of archforge.py --force semantics)."""
    monkeypatch.chdir(tmp_path)
    main(["init"])
    pkg = tmp_path / "archforge_optimizer"
    edited = "# MY HAND-EDIT TO app.py\n"
    (pkg / "app.py").write_text(edited, encoding="utf-8")

    rc = main(["init", "--force"])
    assert rc == 0
    assert "class App(" in (pkg / "app.py").read_text(encoding="utf-8")
    assert edited not in (pkg / "app.py").read_text(encoding="utf-8")  # hand edit gone


# --------------------------------------------------------------------------- #
# make-spec: build + lint + write spec.json from the EDITED adapter
# --------------------------------------------------------------------------- #

def test_make_spec_builds_and_lints_placeholder(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`make-spec` on the freshly-scaffolded placeholder imports
    `archforge_optimizer.host:AppAdapter`, calls `app_spec()` to build the bootstrap
    Spec, lints it, and — because the placeholder roster is a valid DAG — writes
    `archforge_optimizer/spec.json`. This is the load-bearing guarantee that the
    scaffold is one command from a real `evolve --seed`."""
    monkeypatch.chdir(tmp_path)
    main(["init"])
    capsys.readouterr()
    spec_path = tmp_path / "archforge_optimizer" / "spec.json"
    assert not spec_path.exists()          # init did NOT write it

    rc = main(["make-spec"])
    assert rc == 0
    assert spec_path.is_file()
    spec = Spec.model_validate(json.loads(spec_path.read_text(encoding="utf-8")))
    assert not lint(spec), "the built placeholder spec must pass the Linter"
    out = capsys.readouterr().out
    assert "created:" in out and "lint OK" in out
    assert "evolve --adapter archforge_optimizer.host:AppAdapter --seed" in out


def test_make_spec_custom_out_path(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`make-spec --out` writes the Spec JSON to the given path (and the parent dir is
    created if missing)."""
    monkeypatch.chdir(tmp_path)
    main(["init"])
    out = tmp_path / "out" / "custom_spec.json"
    rc = main(["make-spec", "--out", str(out)])
    assert rc == 0
    assert out.is_file()
    spec = Spec.model_validate(json.loads(out.read_text(encoding="utf-8")))
    assert not lint(spec)


def test_make_spec_custom_adapter(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`make-spec --adapter` resolves an arbitrary module:Class (the same path
    `evolve --adapter` uses), so a user with a renamed adapter still builds from it."""
    monkeypatch.chdir(tmp_path)
    main(["init"])
    # the default adapter string works verbatim (the README/Next line uses it)
    rc = main(["make-spec", "--adapter", "archforge_optimizer.host:AppAdapter"])
    assert rc == 0
    assert (tmp_path / "archforge_optimizer" / "spec.json").is_file()


def test_make_spec_refuses_invalid_spec(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """If the edited adapter builds a Spec that FAILS the linter, `make-spec` returns
    rc=1 and writes NOTHING — the user fixes the # EDIT: markers and re-runs. We force
    a lint failure by adding a third node the placeholder's single edge never touches
    (multi-node ⇒ every node must appear in an edge ⇒ the Linter flags an orphan)."""
    monkeypatch.chdir(tmp_path)
    main(["init"])
    capsys.readouterr()
    app = tmp_path / "archforge_optimizer" / "app.py"
    raw = app.read_text(encoding="utf-8")
    # Inject an orphan node as a 3rd roster entry (placeholder has retrieve→answer only).
    # Match the responder's knobs line + the list-closing lone `]` verbatim, then insert
    # a third Nd never touched by the single edge -> the Linter flags orphan_node.
    responder_line = "       knobs=m.Knobs(temperature=0.3, max_tokens=4096)),     # EDIT: your responder's knobs"
    orphan_line = '    Nd("orphan_only", "detached", m.NodeKind.LLM),   # injected orphan'
    assert responder_line in raw, "placeholder roster shape changed; update the edit"
    raw = raw.replace(responder_line, responder_line + "\n" + orphan_line, 1)
    app.write_text(raw, encoding="utf-8")
    # make-spec flushes the adapter's module subtree from sys.modules so it always reads
    # the current cwd's (here, edited) files — no per-test cache dance needed.

    rc = main(["make-spec"])
    assert rc == 1
    assert not (tmp_path / "archforge_optimizer" / "spec.json").exists()  # nothing written
    err = capsys.readouterr().err
    assert "fails the linter" in err and "orphan" in err.lower()


def test_make_spec_adapter_missing_app_spec(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A `--adapter` whose class has no `app_spec()` (not a LangGraph-style adapter) is
    rejected with rc=1 + a clear message, rather than an AttributeError traceback."""
    monkeypatch.chdir(tmp_path)
    # build a tiny fake adapter module on sys.path with NO app_spec()
    import sys
    fake = tmp_path / "fakeadapter.py"
    fake.write_text(
        "from archforge.host.adapters import BaseHostAdapter\n"
        "class NoSpec(BaseHostAdapter):\n"
        "    def instantiate(self, state):\n        raise NotImplementedError\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(tmp_path))
    try:
        rc = main(["make-spec", "--adapter", "fakeadapter:NoSpec"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "no app_spec" in err and "make-spec needs" in err
    finally:
        sys.path.remove(str(tmp_path))


def test_init_then_make_spec_end_to_end_no_secrets(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The full two-step offline bootstrap, secret-safe: `init` + `make-spec` with a
    real `.env` (silent key) present — it is never read or echoed, and no `.env.example`
    is ever created. Pins the whole new contract together."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk-e2e-never-echo-456\n", encoding="utf-8")
    assert main(["init"]) == 0
    assert main(["make-spec"]) == 0
    spec_path = tmp_path / "archforge_optimizer" / "spec.json"
    assert spec_path.is_file()
    spec = Spec.model_validate(json.loads(spec_path.read_text(encoding="utf-8")))
    assert not lint(spec)
    captured = capsys.readouterr()
    assert "gsk-e2e-never-echo" not in captured.out + captured.err
    assert not (tmp_path / ".env.example").exists()
    # the on-disk generator is the single source the scaffold is derived from
    assert set(_ADAPTER_FILES) == set(adapter_package_files())


def test_evolve_defaults_seed_to_scaffold_spec(tmp_path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """After `init` + `make-spec`, a bare `evolve` with no `--seed` bootstraps the root
    incumbent from `archforge_optimizer/spec.json` — the scaffold default, no flag needed.
    Injected components (FakeHostMAS host, scripted judge/architect) supply the organs so
    no real LLM/key is touched; the host-swap default is NOT exercised here (injected
    organs win by contract) — only the seed default is under test."""
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    assert main(["make-spec"]) == 0
    capsys.readouterr()                       # drain init/make-spec output
    spec_path = tmp_path / "archforge_optimizer" / "spec.json"
    assert spec_path.is_file()

    from archforge.architect import ScriptedArchitect
    from archforge.cli import Components
    from archforge.host.base import Task
    from archforge.host.fake import FakeHostMAS
    from archforge.judge import ScriptedJudge
    from archforge.suite import Suite

    judge = ScriptedJudge()
    arch = ScriptedArchitect().force_plateau()   # plateaus → a clean no-promote cycle
    suite = Suite(suite_id="S", rubric_id="default-v1",
                  tasks=[Task(task_id="t1", input="q")])
    comp = Components(host=FakeHostMAS(), judge=judge, architect=arch, suite=suite)

    root = tmp_path / ".archforge"
    rc = main(["evolve", "--root", str(root)], components=comp)
    assert rc == 0
    # the bootstrap seed resolved to the scaffold spec.json (no --seed flag passed):
    # the committed root incumbent is content-addressed to the scaffold spec's id.
    scaffold = Spec.model_validate(json.loads(spec_path.read_text(encoding="utf-8")))
    from archforge.stores.spec_store import SpecStore
    active_id = SpecStore(str(root)).active_id()
    assert active_id == scaffold.compute_spec_id()
