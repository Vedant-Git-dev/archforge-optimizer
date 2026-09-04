"""Phase 0 scaffold smoke tests.

Keeps `pytest` exit code green (0) by ensuring at least one test is collected,
and verifies the package is importable and the CLI entrypoint is callable.
"""

from __future__ import annotations

import archforge
from archforge.cli import main


def test_package_imports() -> None:
    assert archforge.__version__ == "0.4.0"


def test_cli_no_arg_returns_zero() -> None:
    rc = main([])
    assert rc == 0


def test_cli_evolve_entrypoint_is_callable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Phase 9 wired `evolve` for real: with no incumbent and no --seed it fails
    # cleanly with rc 1 (and a message) rather than crashing — confirming the
    # entrypoint is callable and the subcommand is no longer a stub. Uses an
    # isolated --root (a fresh empty dir) so the assertion holds regardless of
    # any stale incumbent left in the repo-root .archforge/ by out-of-band runs.
    rc = main(["evolve", "--root", str(tmp_path)])
    assert rc == 1
