"""Pins the ResourceWarning-silence contract for ``archforge.cli``.

The evolve-loop prints ``ResourceWarning: unclosed <ssl.SSLSocket>`` noise from
the per-call LLM clients' (groq/google-genai) sockets closing in langgraph's
async executor. The CLI silences them by overriding ``warnings.showwarning``
(the terminal sink) — NOT via a ``simplefilter("ignore", ResourceWarning)``,
which a library importing after us (httpx/httpcore/chromadb/langgraph) re-arms by
prepending its own entry above ours (first matching filter wins → our "ignore" is
shadowed). The ``showwarning`` sink runs AFTER the filters decide to *show*, so
it can't be shadowed by filter re-arming — the load-bearing property this test
reproduces. These are exactly the socket-warning emissions the real
``evolve-loop`` was spamming; archforge opens no raw sockets itself, so this is
third-party async-pool noise, not our bug.
"""
from __future__ import annotations

import io
import sys
import warnings

import pytest

from archforge.cli import _install_resource_warning_quieteners


@pytest.fixture
def _restored_sinks():
    """Save + restore the process-global warning sinks + filter list so a test
    failure can't poison the rest of the suite."""
    saved_show = warnings.showwarning
    saved_hook = sys.unraisablehook
    saved_filters = warnings.filters[:]
    # Clear any once-per-location registry so a prior test's warn() doesn't
    # suppress this test's (``__warningregistry__`` is per-module, lazily created
    # on first ``warn``; just clearing it is enough — we restore the sinks).
    globals().setdefault("__warningregistry__", {})
    globals()["__warningregistry__"] = {}
    yield
    warnings.showwarning = saved_show
    sys.unraisablehook = saved_hook
    warnings.filters[:] = saved_filters


class _SpySink:
    """A stand-in for the terminal ``warnings.showwarning``: records every call.
    Installed as ``warnings.showwarning`` BEFORE ``_install_resource_warning_quieteners``
    so the quietener captures the spy as its "default" and forwards non-ResourceWarning
    warnings to it. This tests the forwarding contract directly (drop RW, forward
    others) instead of via ``sys.stderr`` capture — which is unreliable under pytest
    (pytest replaces ``showwarning`` with its recorder that writes to an internal
    list, not stderr, so ``redirect_stderr`` can't see the forwarded warning even
    though it's emitted)."""
    def __init__(self):
        self.calls = []

    def __call__(self, message, category, filename, lineno, file=None, line=None):
        self.calls.append((str(message), category.__name__))


def test_quietener_silences_resource_warning_when_filter_would_show_it(_restored_sinks):
    """With ResourceWarning's filter armed to ``default`` (the httpx/langgraph
    re-arm scenario — by default Python ignores ResourceWarning, so the libs flip
    it to "default" to surface it), our sink still drops it: the captured default
    (the spy) is NOT called."""
    spy = _SpySink()
    warnings.showwarning = spy
    warnings.simplefilter("default", ResourceWarning)   # it WOULD show without us
    _install_resource_warning_quieteners()

    warnings.warn("unclosed <ssl.SSLSocket fd=17>",
                  ResourceWarning, stacklevel=2)
    assert spy.calls == [], (
        f"ResourceWarning should be dropped (not forwarded); got {spy.calls}")


def test_quietener_is_shadow_proof_against_filter_re_arming(_restored_sinks):
    """THE load-bearing property: a library importing AFTER our install re-arms
    ResourceWarning's filter (PREPENDS its "default" above where ours sat). A
    plain ``simplefilter("ignore")`` would be shadowed here and the warning would
    print; the ``showwarning`` sink can't be, because it runs AFTER the filters
    decide to show — the spy default is still NOT called."""
    spy = _SpySink()
    warnings.showwarning = spy
    warnings.simplefilter("default", ResourceWarning)
    _install_resource_warning_quieteners()
    # A late library import re-arms the filter ABOVE ours (the real-world defeat
    # of the simplefilter approach, observed in the first CLI run).
    warnings.simplefilter("default", ResourceWarning)

    warnings.warn("unclosed <ssl.SSLSocket fd=42>",
                  ResourceWarning, stacklevel=2)
    assert spy.calls == [], (
        f"shadowed re-armed filter should still route through our sink (dropped); "
        f"got {spy.calls}")


def test_quietener_keeps_other_warnings_audible(_restored_sinks):
    """Only ResourceWarning is silenced — a DeprecationWarning (a real archforge
    signal worth debugging) is FORWARDED to the captured default (the spy)."""
    spy = _SpySink()
    warnings.showwarning = spy
    warnings.simplefilter("default", DeprecationWarning)
    _install_resource_warning_quieteners()

    warnings.warn("a real archforge deprecation",
                  DeprecationWarning, stacklevel=2)
    assert len(spy.calls) == 1, f"DeprecationWarning should be forwarded: {spy.calls}"
    assert spy.calls[0] == ("a real archforge deprecation", "DeprecationWarning")


def test_quietener_unraisable_hook_silences_resource_warning(_restored_sinks):
    """Belt-and-suspenders: the ``__del__``-RAISES path routes through
    ``sys.unraisablehook`` (a DIFFERENT path than ``warnings.warn`` — sockets call
    warn, but a custom object could raise). Our hook drops a ResourceWarning
    there and forwards everything else to the default."""
    called = {"n": 0}

    def _spy(unr_args, /):
        called["n"] += 1

    # Install the spy as the default BEFORE the quietener captures it, so the
    # quietener's captured "default" IS the spy (same pattern as the showwarning
    # tests). Then install the quietener over the spy.
    sys.unraisablehook = _spy
    _install_resource_warning_quieteners()

    # A fake unraisable that raised a ResourceWarning -> dropped (not forwarded).
    class _Unr:
        exc_value = ResourceWarning("unclosed thing")

    sys.unraisablehook(_Unr())
    assert called["n"] == 0, (
        "a ResourceWarning unraisable should be dropped (not forwarded)")

    # A non-ResourceWarning unraisable IS forwarded to the captured default.
    class _UnrOther:
        exc_value = ValueError("a genuine bug")

    sys.unraisablehook(_UnrOther())
    assert called["n"] == 1, (
        "a non-ResourceWarning unraisable should be forwarded to the default")
