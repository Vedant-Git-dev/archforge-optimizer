"""The evolve engine — orchestrates one P-E-C cycle and the loop.

This is the "nervous system" that wires the four organs (Architect, SuiteRunner,
Gatekeeper) plus the stores into a coherent run-over-run loop:

    incumbent = SpecStore.active()
    result = architect.next_attempt(incumbent, worst_task_scores, attempt_store)
      if not result.proposed -> record, maybe plateau, continue
    candidate_spec_id = spec_store.commit(result.proposal.candidate, parent=incumbent)
    attempt = Attempt{candidate_spec_id, parent=incumbent, change, verdict=PROMOTED}
        (PROMOTED is the *initial* verdict; the Gatekeeper flips it)
    attempt_id = attempt_store.append(attempt)
    inc_run  = suite_runner.run_suite(incumbent_suite, R)      # baseline, cacheable
    cand_run = suite_runner.run_suite(candidate_spec, R)
    decision = gatekeeper.decide(attempt_id, cand_run, inc_run)
    applied  = gatekeeper.apply_decision(decision)
    record CycleResult

The engine is constructed with explicit components (host/judge/architect) +
stores, so tests inject fakes directly and a CLI shim wires the real or
scripted variants per `--provider` config. Departmental rules honored:
  E3  budget caps abort cleanly (budget_guard raises CycleAborted; incumbent untouched)
  E8  K consecutive no-promotion cycles -> plateau (loop stops, status surfaced)
  I1  only the Gatekeeper moves `active`; the engine never touches it
  I5  every run is stamped with (suite_id, rubric_id); the Gatekeeper checks it
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from archforge import userconfig as ucfg
import archforge.models as m
from archforge.architect import ArchitectProtocol, ArchitectResult
from archforge.gatekeeper import Action, Decision, Gatekeeper
from archforge.judge.base import JudgeProtocol
from archforge.stores import AttemptStore, SpecStore, TraceStore
from archforge.suite import Suite, SuiteRunner, SuiteRun


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #


class CycleAborted(Exception):
    """Raised when a budget cap is hit mid-cycle (E3). The incumbent is untouched."""

    def __init__(self, reason: str, partial: "CycleResult | None" = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.partial = partial


@dataclass
class CycleResult:
    """One cycle's outcome, for the report + plateau accounting."""

    cycle: int
    attempted: bool                       # did the Architect propose a candidate?
    architect_status: ArchitectResult | None = None
    decision: Decision | None = None
    applied_attempt_id: str | None = None
    incumbent_mean: float | None = None
    candidate_mean: float | None = None
    tokens: int = 0
    latency_ms: float = 0.0               # summed wall-clock this cycle (inc + cand)
    note: str = ""

    @property
    def promoted(self) -> bool:
        return self.decision is not None and self.decision.action is Action.AUTO_PROMOTE

    @property
    def queued(self) -> bool:
        return self.decision is not None and self.decision.action is Action.QUEUE_HUMAN


@dataclass
class LoopResult:
    """The aggregate outcome of `evolve_loop`."""

    cycles_run: int = 0
    promotions: int = 0
    queued: int = 0
    plateaued: bool = False
    aborted: bool = False
    abort_reason: str = ""
    results: list[CycleResult] = field(default_factory=list)
    final_incumbent_id: str | None = None
    final_incumbent_mean: float | None = None


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


@dataclass
class EngineConfig:
    """Tunables for the loop (Beyond thresholds, which live in m.Thresholds)."""

    # Tunable defaults resolve lazily from archforge.userconfig (the active config),
    # so importing the engine — and even materializing an EngineConfig() default —
    # works BEFORE `init` has run (no from-import at module load).
    max_cycles: int = field(default_factory=lambda: ucfg.get("DEFAULT_MAX_CYCLES"))
    max_tokens_per_cycle: int | None = field(default_factory=lambda: ucfg.get("DEFAULT_MAX_TOKENS_PER_CYCLE"))  # E3 (None=∞)
    max_tokens_total: int | None = field(default_factory=lambda: ucfg.get("DEFAULT_MAX_TOKENS_TOTAL"))
    repeats: int = field(default_factory=lambda: ucfg.get("DEFAULT_REPEATS"))    # R (adaptive raises it)
    plateau_cycles: int = field(default_factory=lambda: ucfg.get("DEFAULT_PLATEAU_CYCLES"))  # K (E8)
    # per-cycle wall-clock cap (ms). Defaults to None (opt-in) — closing the budget
    # hole for non-LLM-heavy pipelines whose cost is *time* not tokens (a retriever/
    # tool/rule node costs ~0 tokens). Mirrors max_tokens_per_cycle's per-cycle
    # shape + none-means-∞ contract; timed post-eval (same as the token cap), not
    # mid-step. No total-wall cap (YAGNI; the loop is already bounded by max_cycles
    # + this per-cycle cap).
    max_wall_ms_per_cycle: float | None = field(default_factory=lambda: ucfg.get("DEFAULT_MAX_WALL_MS_PER_CYCLE"))


class Engine:
    """Runs Propose-Evaluate-Commit cycles and loops over them.

    Holds the long-lived stores + judge + runner + gatekeeper; the host and
    architect are supplied per construction (real or scripted).
    """

    def __init__(
        self,
        *,
        host: object,                  # HostMAS (typed loosely to avoid import cycle)
        judge: JudgeProtocol,
        architect: ArchitectProtocol,
        spec_store: SpecStore,
        attempt_store: AttemptStore,
        trace_store: TraceStore,
        suite: Suite,
        thresholds: m.Thresholds | None = None,
        config: EngineConfig | None = None,
        on_promote: Callable[[m.Spec], None] | None = None,
    ) -> None:
        self._host = host
        self._judge = judge
        self._architect = architect
        self._specs = spec_store
        self._attempts = attempt_store
        self._traces = trace_store
        self._suite = suite
        self._th = thresholds or m.Thresholds()
        self._cfg = config or EngineConfig()
        self._on_promote = on_promote
        self._runner = SuiteRunner(host, judge, trace_store,
                                    epsilon=self._th.unrunnable_frac,
                                    judge_retries=ucfg.get("DEFAULT_JUDGE_RETRIES"))
        self._gatekeeper = Gatekeeper(spec_store, attempt_store, thresholds=self._th)
        # baseline cache: incumbent_suite_run keyed by spec_id (stable until rubric changes)
        self._baseline_cache: dict[str, SuiteRun] = {}
        self._tokens_total = 0
        # worst-task scores cache for the Architect (last incumbent's run scores)
        self._last_incumbent_scores: list[m.RunScore] = []

    # ----------------------------------------------------------------- one cycle
    def evolve_cycle(self, cycle: int = 0) -> CycleResult:
        """Run exactly one P-E-C cycle from the active incumbent."""
        incumbent = self._specs.active()           # raises NoActiveSpecError if none
        arch = self._architect.next_attempt(
            incumbent, self._last_incumbent_scores, attempt_store=self._attempts,
        )

        if not arch.proposed:
            return CycleResult(cycle=cycle, attempted=False, architect_status=arch,
                                note=arch.note or _status_note(arch))

        proposal = arch.proposal
        candidate = proposal.candidate
        candidate_spec_id = self._specs.commit(candidate, parent_spec_id=incumbent.spec_id,
                                               status=m.SpecStatus.CANDIDATE)
        attempt = m.Attempt(
            candidate_spec_id=candidate_spec_id, parent_spec_id=incumbent.spec_id,
            change=proposal.change, verdict=m.Verdict.PROMOTED,
        )
        attempt_id = self._attempts.append(attempt)

        # Evaluate both, same suite + rubric (I5). Baseline is cacheable.
        inc_run = self._baseline_for(incumbent)
        self._last_incumbent_scores = list(inc_run.scores)
        candidate_spec = self._specs.get(candidate_spec_id)
        cand_run = self._bounded_run(candidate_spec, tag="candidate")

        self._check_budget_cycle(
            cand_run.tokens + inc_run.tokens,
            cand_run.latency_ms + inc_run.latency_ms,
        )
        decision = self._gatekeeper.decide(attempt_id, cand_run, inc_run)
        applied = self._gatekeeper.apply_decision(decision)

        # Deploy hook (Tier-2 sidecar auto-sync): an opt-in callback fired ONLY on
        # an AUTO_PROMOTE — i.e. the candidate just became the active incumbent.
        # The callback owns any side effect (e.g. export_spec_sidecar → a JSON file
        # the MAS overlays onto its config consts so the win reaches production
        # without the Forge on the hot path). Not fired for QUEUE_HUMAN (a human
        # gate) or a discard. The engine does no I/O; the callback does.
        if self._on_promote is not None and decision.action is Action.AUTO_PROMOTE:
            self._on_promote(self._specs.get(candidate_spec_id))

        # Persist the scored result so the human-facing surfaces (status, report,
        # Approval Queue) show the real delta + cost without re-running the suite.
        # `tokens` carries only the candidate-side marginal cost — the incumbent
        # baseline is a shared, cacheable cost accounted for at the loop level
        # (CycleResult.tokens), never double-counted onto a single attempt.
        self._attempts.set_result(
            attempt_id,
            m.SuiteResult(
                mean=cand_run.mean,
                margin_vs_incumbent=decision.margin,
                repeats=cand_run.repeats,
                unrunnable=cand_run.unrunnable,
                rubric_id=cand_run.rubric_id,
                suite_id=cand_run.suite_id,
                tokens=cand_run.tokens,
            ),
        )

        return CycleResult(
            cycle=cycle, attempted=True, architect_status=arch, decision=decision,
            applied_attempt_id=applied.attempt_id,
            incumbent_mean=inc_run.mean,
            candidate_mean=cand_run.mean,
            tokens=cand_run.tokens + inc_run.tokens,
            latency_ms=cand_run.latency_ms + inc_run.latency_ms,
            note=decision.reason,
        )

    # ----------------------------------------------------------------- the loop
    def evolve_loop(self) -> LoopResult:
        """Repeat `evolve_cycle` until budget cap or plateau (E3/E8)."""
        out = LoopResult()
        plateau_streak = 0
        for i in range(self._cfg.max_cycles):
            if self._over_total_budget():
                out.aborted = True
                out.abort_reason = "total token budget reached"
                break
            try:
                r = self.evolve_cycle(cycle=i)
            except CycleAborted as exc:
                out.aborted = True
                out.abort_reason = exc.reason
                if exc.partial is not None:
                    out.results.append(exc.partial)
                break
            out.results.append(r)
            out.cycles_run += 1
            if r.promoted:
                out.promotions += 1
                plateau_streak = 0
            elif r.queued:
                out.queued += 1
                plateau_streak = 0        # a queued change is forward progress
            elif not r.attempted:
                plateau_streak += 1
            else:
                plateau_streak += 1       # a discard also counts toward plateau

            self._tokens_total += r.tokens
            if plateau_streak >= self._cfg.plateau_cycles:
                out.plateaued = True
                break
        out.final_incumbent_id = self._specs.active_id()
        return out

    # ----------------------------------------------------------------- helpers
    def _baseline_for(self, incumbent: m.Spec) -> SuiteRun:
        """The incumbent's suite run, cached until the rubric/suite changes."""
        key = incumbent.spec_id or incumbent.compute_spec_id()
        if key in self._baseline_cache:
            cached = self._baseline_cache[key]
            # invalidate if rubric/suite drifted since cache (I5 — same geometry)
            if (cached.rubric_id, cached.suite_id) == (self._suite.rubric_id, self._suite.suite_id):
                return cached
        run = self._bounded_run(incumbent, tag="incumbent")
        self._baseline_cache[key] = run
        return run

    def _bounded_run(self, spec: m.Spec, *, tag: str) -> SuiteRun:
        return self._runner.run_suite(spec, self._suite, R=self._cfg.repeats)

    def _check_budget_cycle(self, cycle_tokens: int, cycle_latency_ms: float = 0.0) -> None:
        cap = self._cfg.max_tokens_per_cycle
        if cap is not None and cycle_tokens > cap:
            raise CycleAborted(
                f"per-cycle token cap exceeded ({cycle_tokens} > {cap}; tag budget)",
            )
        # Wall-clock cap: the cost fix for non-LLM-heavy pipelines (a retriever/
        # tool/rule node costs ~0 tokens, so it sails past the token cap). Same
        # post-eval timing + > contract as the token cap — incumbent untouched.
        wall_cap = self._cfg.max_wall_ms_per_cycle
        if wall_cap is not None and cycle_latency_ms > wall_cap:
            raise CycleAborted(
                f"per-cycle wall-clock cap exceeded "
                f"({cycle_latency_ms:.1f}ms > {wall_cap}ms)",
            )

    def _over_total_budget(self) -> bool:
        # A total token budget is a *ceiling*: stop at or before reaching it, not
        # only after crossing it. `>=` makes a 0 budget mean "spend nothing" (the
        # loop aborts before the first cycle) and keeps a real budget from being
        # overrun by a single cycle (> would let one cycle blow straight past it).
        cap = self._cfg.max_tokens_total
        return cap is not None and self._tokens_total >= cap


def _status_note(arch: ArchitectResult) -> str:
    if arch.status == "lint_rejected":
        reasons = "; ".join(e.message for e in arch.rejected_reasons) or "malformed"
        return f"candidate rejected by linter: {reasons}"
    if arch.status == "plateau":
        return arch.note or "architect plateaued (no fresh proposal)"
    return arch.note or f"architect status={arch.status}"


__all__ = [
    "Engine", "EngineConfig", "CycleResult", "LoopResult", "CycleAborted",
]
