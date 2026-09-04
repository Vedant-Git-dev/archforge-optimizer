"""ArchForge tunables resolver — the lazy bridge from ``.archforge/archforge.py``.

System config (``archforge.config``) holds only non-tunable *internals* (version,
provider roster, storage dirnames, hash lengths, the ``.env`` loader, …). The
*tunable* defaults (τ, δ, R, providers, rubric, storage root, …) live in the
project's ``.archforge/archforge.py`` — the file ``archforge-optimizer init``
writes — and every consumer reads them lazily through THIS module at **use** time
(via ``ucfg.get(name)``), NOT at import time. Because nothing binds a tunable at
class-definition time, ``import archforge`` (and ``init`` itself) succeed *before*
``init`` has run, no import-timing chokepoint.

Precedence (one resolve per process, cached until ``_reset_cache()``):

  1. **disabled** — ``ARCHFORGE_CONFIG_DISABLE`` set OR ``"pytest" in sys.modules`` →
     exec the sane-default ``archforge.config_init.TEMPLATE`` **in-memory** (no disk
     file) so the test suite sees shipped values (e.g. ``Thresholds().tau == 0.05``)
     with zero per-test files.
  2. **disk** — ``.archforge/archforge.py`` (relative to cwd) exists → exec it → the
     project's sole tunable source.
  3. **raise** — neither, and we are NOT disabled → a tuned use raises
     ``ConfigNotInitialized`` (message: run ``archforge-optimizer init``). ``import``
     still succeeds; only *constructing* a tuned object / running ``evolve`` without
     ``init`` raises — the "pip install → init → CLI works" contract.

Discovery is the FIXED ``.archforge/archforge.py`` — a system path, deliberately
independent of the tunable ``DEFAULT_ROOT_DIR`` (which is one of the values DISCOVERED
here): a user who moves their run-state dir still finds config at the default spot.
Stdlib-only — a pure leaf that imports nothing from the archforge core.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Fixed config-discovery location — system plumbing, independent of the tunable
# DEFAULT_ROOT_DIR (that value is itself read from the file discovered here).
_DISCOVERY_DIR: Path = Path(".archforge")
_DISCOVERY_FILE: Path = _DISCOVERY_DIR / "archforge.py"

_MAIN_MSG = (
    "ArchForge is not initialized. Run `archforge-optimizer init`."
)


class ConfigNotInitialized(RuntimeError):
    """A tunable was needed but no config is initialized and we are not disabled."""


def _disabled() -> bool:
    """True when the in-memory template (sane defaults) should win over the disk file.

    Two gates: an explicit ``ARCHFORGE_CONFIG_DISABLE`` env var (manual opt-out / force
    sane defaults — e.g. to ignore a checked-in ``.archforge/archforge.py``) and
    ``"pytest" in sys.modules`` (auto-disabled during the test suite so tests assert
    sane shipped values like ``Thresholds().tau == 0.05`` without per-test files).
    """
    return bool(os.environ.get("ARCHFORGE_CONFIG_DISABLE")) or "pytest" in sys.modules


def _exec(src: str, label: str) -> dict[str, object]:
    """Exec config source into a FRESH namespace and return it (the user's own code)."""
    code = compile(src, label, "exec")
    ns: dict[str, object] = {}
    exec(code, ns)  # noqa: S102  — the config file is the user's own; fresh namespace
    return ns


def _template_namespace() -> dict[str, object]:
    """The sane defaults, exec'd in-memory from the SAME template ``init`` writes."""
    from archforge.config_init import TEMPLATE  # local import: keep this a pure leaf
    return _exec(TEMPLATE, "<archforge config template>")


def _disk_namespace() -> dict[str, object] | None:
    """The project's ``.archforge/archforge.py`` exec'd, or None if absent."""
    if not _DISCOVERY_FILE.exists():
        return None
    return _exec(_DISCOVERY_FILE.read_text(encoding="utf-8"), str(_DISCOVERY_FILE))


_NS: dict[str, object] | None = None


def _resolve() -> dict[str, object]:
    """Resolve the active tunable namespace once per process (until ``_reset_cache()``).

    See the module docstring for precedence. Raises ``ConfigNotInitialized`` when a
    real (non-disabled) use has no disk file — the "init required" contract.
    """
    global _NS
    if _NS is not None:
        return _NS
    if _disabled():
        _NS = _template_namespace()
        return _NS
    disk = _disk_namespace()
    if disk is None:
        raise ConfigNotInitialized(_MAIN_MSG)
    _NS = disk
    return _NS


_MISSING = object()


def get(name: str, *, default: object = _MISSING) -> object:
    """Look up a tunable by name from the resolved config.

    Raises ``ConfigNotInitialized`` if no config is initialized (and not disabled) —
    the init gate fires here too. Raises ``KeyError`` if the initialized config omits
    ``name`` and no ``default`` was supplied; with ``default`` a missing name returns
    it instead (no error). Internal names belong to ``archforge.config``, not here.
    """
    ns = _resolve()
    if name in ns:
        return ns[name]
    if default is not _MISSING:
        return default
    raise KeyError(
        f"{name!r} is not defined in the active ArchForge config "
        f"({'template' if _disabled() else str(_DISCOVERY_FILE)})."
    )


def ensure_initialized() -> None:
    """The evolve gate: raise ``ConfigNotInitialized`` if a real run lacks ``init``.

    A pure precondition (does NOT populate the cache). The CLI calls this before
    building any real provider so an uninitialized project prints the init hint and
    exits with rc 1 instead of making a bogus API call. No-op when disabled (tests).
    """
    if not _disabled() and not _DISCOVERY_FILE.exists():
        raise ConfigNotInitialized(_MAIN_MSG)


def _reset_cache() -> None:
    """Forget the cached namespace (test hook — forces the next ``get`` to re-resolve)."""
    global _NS
    _NS = None


__all__ = ["ConfigNotInitialized", "get", "ensure_initialized"]
