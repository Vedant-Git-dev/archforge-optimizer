"""Runner — the embedder's single entrypoint into the P-E-C loop.

Wraps ``Engine`` so a concrete adapter author never touches the four organs or
the stores directly: hand in a ``HostAdapter`` (your MAS) + a ``Spec`` (your
bootstrap) + a ``Suite`` (your eval set) + a ``RunnerConfig``, and get a
``CycleResult`` (one cycle) or ``LoopResult`` (a full evolve-loop) back.

The construction mirrors the proven ``LuminaAI/archforge_glue/live_loop.py``
path — the same organs, stores, seeding, and budget caps — factored once so
every external adapter reuses it instead of re-deriving ~150 lines of wiring.

Env priority (documented): an explicit ``config.api_key`` > a real process env
var > a ``.env`` file (loaded with ``setdefault`` so a real env var always
wins). The kit keeps no hard dependency on ``python-dotenv`` — a flat stdlib
parser covers ``.env``; absent file is a no-op.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import archforge.models as m
from archforge.architect import Architect
from archforge.engine import CycleResult, Engine, EngineConfig, LoopResult
from archforge.judge.base import Judge, default_rubric
from archforge.llm import make_client
from archforge.stores import AttemptStore, SpecStore, TraceStore
from archforge.suite import Suite


# --------------------------------------------------------------------------- #
# .env loader (stdlib-only; no python-dotenv dependency on core)
# --------------------------------------------------------------------------- #


def _load_env(path: str | os.PathLike[str] | None = ".env") -> None:
    """Populate ``os.environ`` from a flat ``KEY=VALUE`` file via ``setdefault``.

    A real process env var therefore always wins; a passed ``api_key`` (handed
    straight to the SDK in ``make_client``) wins above both. Skips blank and
    ``#``-comment lines; strips optional surrounding ``"``/``'`` on values.
    No-op if the file is missing.
    """
    if not path:
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
        val = val.strip().strip("'").strip('"')
        os.environ.setdefault(key, val)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


@dataclass
class RunnerConfig:
    """Everything an embedder tunes; every field has a sensible default."""

    # LLM provider for the Architect + Judge (NOT for the host's own agents —
    # those keep whatever provider the adapter wires, per the design's split).
    provider: str = "gemini"
    architect_model: str | None = None    # None → provider default
    judge_model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    rubric: str = "default-v1"
    env_file: str | None = ".env"

    # loop / budget
    tau: float = m.DEFAULT_TAU
    max_cycles: int = EngineConfig().max_cycles
    repeats: int = EngineConfig().repeats
    max_tokens_per_cycle: int | None = None
    max_tokens_total: int | None = None
    plateau_cycles: int = EngineConfig().plateau_cycles

    # persistence (a directory, NOT a tmp path — reruns continue evolving)
    storage_root: str | os.PathLike[str] = ".archforge"


# --------------------------------------------------------------------------- #
# internal: build organs + scaffold an Engine exactly like live_loop.py
# --------------------------------------------------------------------------- #


def _build_organs(cfg: RunnerConfig):
    """Build the real Architect + Judge over a configured provider client."""
    if cfg.provider == "scripted":
        # The kit doesn't import scripted organs by default (they live in the
        # CLI/scenarios); an embedder wanting the zero-cost path supplies its
        # own via the explicit-Components entrypoint in the CLI. Here we only
        # support real providers — matches live_loop's billed intent.
        raise ValueError(
            "RunnerConfig(provider='scripted') is not supported by run_loop; "
            "use the CLI's --provider scripted path or supply scripted organs "
            "directly to Engine."
        )
    _load_env(cfg.env_file)
    llm = make_client(cfg.provider, api_key=cfg.api_key, base_url=cfg.base_url)
    from archforge.constants import DEFAULT_MODELS
    arch = Architect(llm, model=cfg.architect_model or DEFAULT_MODELS[cfg.provider])
    judge = Judge(llm, model=cfg.judge_model or DEFAULT_MODELS[cfg.provider],
                  rubric=default_rubric if cfg.rubric == "default-v1" else cfg.rubric)
    return arch, judge


def _scaffold(cfg: RunnerConfig, adapter, spec: m.Spec, suite: Suite):
    """Open the persistent stores, seed an active incumbent from ``spec`` if
    the store is empty, and build the wired ``Engine``. Mirrors live_loop.py."""
    root = Path(cfg.storage_root)
    specs = SpecStore(root)
    atts = AttemptStore(root)
    ts = TraceStore(root)

    if specs.active_id() is None:
        # fresh store: commit the bootstrap spec as the initial incumbent
        sid = specs.commit(spec, parent_spec_id=None, status=m.SpecStatus.INCUMBENT)
        specs.set_active(sid)

    arch, judge = _build_organs(cfg)
    engine = Engine(
        host=adapter, judge=judge, architect=arch,
        spec_store=specs, attempt_store=atts, trace_store=ts, suite=suite,
        thresholds=m.Thresholds(tau=cfg.tau),
        config=EngineConfig(
            max_cycles=cfg.max_cycles, repeats=cfg.repeats,
            max_tokens_per_cycle=cfg.max_tokens_per_cycle,
            max_tokens_total=cfg.max_tokens_total,
            plateau_cycles=cfg.plateau_cycles,
        ),
    )
    return engine, specs, atts, ts


def run_loop(
    adapter, spec: m.Spec, suite: Suite, *, config: RunnerConfig | None = None,
) -> LoopResult:
    """Run a full evolve-loop over ``adapter`` (a ``HostMAS``) seeded from ``spec``.

    The ``storage_root`` is PERSISTENT: a second call with the same root
    continues evolving the prior active incumbent (mirrors Lumina's live loop).
    Delete the directory to start fresh.
    """
    cfg = config or RunnerConfig()
    engine, specs, atts, ts = _scaffold(cfg, adapter, spec, suite)
    return engine.evolve_loop()


def run_cycle(
    adapter, spec: m.Spec, suite: Suite, *, config: RunnerConfig | None = None,
) -> CycleResult:
    """Run exactly ONE evolve_cycle over ``adapter`` seeded from ``spec``."""
    cfg = config or RunnerConfig()
    engine, specs, atts, ts = _scaffold(cfg, adapter, spec, suite)
    return engine.evolve_cycle(cycle=0)


__all__ = ["RunnerConfig", "run_cycle", "run_loop", "_load_env"]
