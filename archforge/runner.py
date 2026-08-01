"""Runner — the embedder's single entrypoint into the P-E-C loop.

Wraps ``Engine`` so a concrete adapter author never touches the four organs or
the stores directly: hand in a ``HostAdapter`` (your MAS) + a ``Spec`` (your
bootstrap) + a ``Suite`` (your eval set) + a ``RunnerConfig``, and get a
``CycleResult`` (one cycle) or ``LoopResult`` (a full evolve-loop) back.

The construction mirrors the proven ``LuminaAI/archforge_glue/live_loop.py``
path — the same organs, stores, seeding, and budget caps — factored once so
every external adapter reuses it instead of re-deriving ~150 lines of wiring.

Env priority (documented): an explicit ``api_key`` > a real process env var >
a ``.env`` file (loaded via ``archforge.config.load_env`` so a real env var always
wins). ``python-dotenv`` is a core dependency; an absent file is a no-op.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import archforge.models as m
from archforge import userconfig as ucfg
from archforge.architect import Architect
from archforge.config import load_env
from archforge.engine import CycleResult, Engine, EngineConfig, LoopResult
from archforge.judge.base import Judge, default_rubric
from archforge.llm import make_client
from archforge.stores import AttemptStore, SpecStore, TraceStore
from archforge.suite import Suite


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


@dataclass
class RunnerConfig:
    """Everything an embedder tunes; every field has a sensible default."""

    # LLM provider for the Architect + Judge (NOT for the host's own agents —
    # those keep whatever provider the adapter wires, per the design's split).
    # Every tunable default resolves lazily from archforge.userconfig (the active
    # config), so building a RunnerConfig() default works BEFORE `init` has run
    # under the pytest gate (template sane values); a real run reads the disk file.
    provider: str = field(default_factory=lambda: ucfg.get("PROVIDER"))
    architect_model: str | None = None    # None → provider default
    judge_model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    rubric: str = field(default_factory=lambda: ucfg.get("DEFAULT_RUBRIC_ID"))
    env_file: str | None = field(default_factory=lambda: ucfg.get("DEFAULT_ENV_FILE"))

    # loop / budget
    tau: float = field(default_factory=lambda: ucfg.get("DEFAULT_TAU"))
    max_cycles: int = field(default_factory=lambda: ucfg.get("DEFAULT_MAX_CYCLES"))
    repeats: int = field(default_factory=lambda: ucfg.get("DEFAULT_REPEATS"))
    max_tokens_per_cycle: int | None = field(default_factory=lambda: ucfg.get("DEFAULT_MAX_TOKENS_PER_CYCLE"))
    max_tokens_total: int | None = field(default_factory=lambda: ucfg.get("DEFAULT_MAX_TOKENS_TOTAL"))
    plateau_cycles: int = field(default_factory=lambda: ucfg.get("DEFAULT_PLATEAU_CYCLES"))

    # persistence (a directory, NOT a tmp path — reruns continue evolving)
    storage_root: str | os.PathLike[str] = field(default_factory=lambda: ucfg.get("DEFAULT_ROOT_DIR"))


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
    load_env(cfg.env_file)
    llm = make_client(cfg.provider, api_key=cfg.api_key, base_url=cfg.base_url)
    models = ucfg.get("DEFAULT_MODELS")
    arch = Architect(llm, model=cfg.architect_model or models[cfg.provider])
    judge = Judge(llm, model=cfg.judge_model or models[cfg.provider],
                  rubric=default_rubric() if cfg.rubric == "default-v1" else cfg.rubric)
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


__all__ = ["RunnerConfig", "run_cycle", "run_loop"]
