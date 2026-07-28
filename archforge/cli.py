"""ArchForge command-line interface — the Forge.

Wires the four organs (Architect, SuiteRunner, Judge, Gatekeeper) plus the
filesystem stores into the runnable surface the user actually touches:

    archforge lint <spec.json>                 validate a Spec
    archforge evolve  [--root R] [--seed S]    one Propose-Evaluate-Commit cycle
    archforge evolve-loop [...]                repeat until budget cap or plateau
    archforge approve [<id>...|--all]          drain the structural-change queue
    archforge reject <id>                      reject a queued change (active kept)
    archforge status                           incumbent Spec + lineage + counts
    archforge report                           aggregate deltas across attempts

Provider seam (the single place "real vs fake" lives at the CLI):
  * `--provider scripted` (default) builds the zero-cost fakes — `FakeHostMAS`,
    `ScriptedJudge`, `ScriptedArchitect` — so `evolve` is runnable end-to-end
    with no LLM. Left unconfigured, the scripted architect simply plateaus
    (it has no proposal to make); a deterministic *promotion* needs the organs
    pre-configured, which is exactly what an embedding test supplies via
    `components=...` (see `tests/integration/test_cli.py`).
  * `--provider anthropic|openai|groq|gemini` wires a real `LLMClient` and the
    real `Architect` + `Judge` over it — one provider swap, same organs.

`_ensure_incumbent` bootstraps the root incumbent from `--seed <spec.json>`
zero-LLM (commit as INCUMBENT + set_active) so the very first `evolve` has an
active spec to mutate. Approval/rollback keep `active` moving only through the
Gatekeeper (invariant I1); the CLI never mutates the pointer itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import archforge.models as m
from archforge.architect import Architect, ArchitectProtocol, ScriptedArchitect
from archforge.engine import CycleResult, Engine, EngineConfig, LoopResult
from archforge.gatekeeper import Gatekeeper
from archforge.host.base import HostMAS, Task
from archforge.host.fake import FakeHostMAS
from archforge.judge import ScriptedJudge
from archforge.judge.base import Judge, JudgeProtocol, default_rubric
from archforge.lint import lint
from archforge.stores import AttemptStore, SpecStore, TraceStore
from archforge.suite import Suite

# Project-wide constants — single source of truth in archforge.constants.
from archforge.constants import (
    ALL_PROVIDERS as _PROVIDERS,
    DEFAULT_MODELS, DEFAULT_ROOT_DIR as _DEFAULT_ROOT, DEFAULT_SUITE_ID,
    DEFAULT_TASK_ID, DEFAULT_TASK_INPUT, PROG, PROVIDER
)


# --------------------------------------------------------------------------- #
# Injectable runtime organs — the test/embedding seam
# --------------------------------------------------------------------------- #


@dataclass
class Components:
    """The four organs the Engine runs, injectable so a test/embedding can
    supply pre-configured fakes (a scripted architect with a queued proposal +
    a scripted judge with per-spec aggregates → a deterministic promotion).

    When `main(..., components=None)` the CLI builds defaults per `--provider`:
    `scripted` builds the inert fakes; a real provider builds the real
    `Architect` + `Judge` over a real `LLMClient`.
    """

    host: HostMAS
    judge: JudgeProtocol
    architect: ArchitectProtocol
    suite: Suite


# --------------------------------------------------------------------------- #
# arg parsing
# --------------------------------------------------------------------------- #


def _add_store_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=_DEFAULT_ROOT,
                   help="archforge state directory (default: .archforge)")


def _add_evolve_args(p: argparse.ArgumentParser, *, loop: bool) -> None:
    p.add_argument("--seed", metavar="PATH",
                   help="bootstrap the root incumbent from this Spec JSON (no active yet)")
    p.add_argument("--provider", choices=_PROVIDERS, default=PROVIDER,
                   help="LLM provider (default: scripted; anthropic/openai/groq/gemini are real)")
    # real-provider configuration (ignored for 'scripted'). API key/base-url
    # default to the provider SDK's env vars when omitted (ANTHROPIC_API_KEY,
    # OPENAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY/GEMINI_API_KEY).
    p.add_argument("--api-key", default=None, help="provider API key (else read from env)")
    p.add_argument("--base-url", default=None, help="provider base URL override")
    p.add_argument("--architect-model", default=None,
                   help="model id for the Architect (else the provider default)")
    p.add_argument("--judge-model", default=None,
                   help="model id for the Judge (else the provider default)")
    # thresholds
    p.add_argument("--tau", type=float, default=m.Thresholds().tau,
                   help="promotion margin τ (default: %(default)s)")
    p.add_argument("--delta", type=float, default=m.Thresholds().delta,
                   help="regression floor δ (default: %(default)s)")
    p.add_argument("--repeats", type=int, default=1,
                   help="R: repeats per task (default: 1)")
    if loop:
        p.add_argument("--max-cycles", type=int, default=EngineConfig().max_cycles,
                       help="cap on P-E-C cycles (default: %(default)s)")
        p.add_argument("--plateau-cycles", type=int, default=EngineConfig().plateau_cycles,
                       help="K consecutive no-promotion cycles → plateau (default: %(default)s)")
        p.add_argument("--max-tokens-total", type=int, default=None,
                       help="total token budget cap; stop at or before reaching it (E3)")
    p.add_argument("--max-tokens-per-cycle", type=int, default=None,
                   help="per-cycle token cap; abort mid-cycle if exceeded (E3)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="ArchForge — a self-improving meta-layer over multi-agent systems.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- evolve (one cycle) --------------------------------------------------
    ev = sub.add_parser("evolve",
                        help="run one Propose-Evaluate-Commit cycle from the active incumbent")
    _add_store_args(ev)
    _add_evolve_args(ev, loop=False)

    # --- evolve-loop ---------------------------------------------------------
    evl = sub.add_parser("evolve-loop",
                         help="repeat evolve until the budget cap or a plateau (E3/E8)")
    _add_store_args(evl)
    _add_evolve_args(evl, loop=True)

    # --- approve (human gate, I4) --------------------------------------------
    ap = sub.add_parser("approve",
                        help="approve queued (PENDING_HUMAN) structural changes → active")
    _add_store_args(ap)
    ap.add_argument("attempts", nargs="*",
                    help="attempt ids to approve (default: every pending change, in order)")
    ap.add_argument("--all", action="store_true",
                    help="approve every pending change (default when none are named)")

    # --- reject --------------------------------------------------------------
    rj = sub.add_parser("reject",
                        help="reject a queued structural change — active is left alone")
    _add_store_args(rj)
    rj.add_argument("attempt_id", help="attempt id to reject")

    # --- status / report -----------------------------------------------------
    st = sub.add_parser("status", help="print the incumbent Spec + lineage + counts")
    _add_store_args(st)

    rp = sub.add_parser("report", help="print aggregate deltas across attempts")
    _add_store_args(rp)

    # --- lint ----------------------------------------------------------------
    lint_p = sub.add_parser("lint", help="run the Spec Linter on a JSON Spec file")
    _add_store_args(lint_p)
    lint_p.add_argument("path", help="path to a Spec JSON file")

    return parser


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _stores(root: str) -> tuple[SpecStore, AttemptStore, TraceStore]:
    return SpecStore(root), AttemptStore(root), TraceStore(root)


def _thresholds(args: argparse.Namespace) -> m.Thresholds:
    return m.Thresholds(tau=args.tau, delta=args.delta, repeats=args.repeats,
                        plateau_cycles=getattr(args, "plateau_cycles", m.Thresholds().plateau_cycles))


def _config(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        max_cycles=getattr(args, "max_cycles", EngineConfig().max_cycles),
        max_tokens_per_cycle=getattr(args, "max_tokens_per_cycle", None),
        max_tokens_total=getattr(args, "max_tokens_total", None),
        repeats=args.repeats,
        plateau_cycles=getattr(args, "plateau_cycles", EngineConfig().plateau_cycles),
    )


def _default_components(args: argparse.Namespace) -> Components:
    """Build the runtime organs for `--provider` (no injected `components`).

    `--provider scripted` (default) builds the zero-cost fakes — well-defined but
    inert: the scripted architect has no queued proposal so it plateaus, and the
    scripted judge returns a neutral 0.5 base. A deterministic *promotion* needs
    the organs pre-configured, which is the test path through
    `main(..., components=...)`.

    A real provider (`anthropic`/`openai`/`groq`/`gemini`)
    builds ONE `LLMClient` via `make_client` and the REAL `Architect` + `Judge`
    over it — the provider abstraction is the single seam, so neither organ
    changes when the provider changes. The host stays `FakeHostMAS` for now
    (a real MAS host is its own integration; the seam already accepts it).
    """

    suite = Suite(suite_id=DEFAULT_SUITE_ID, rubric_id=default_rubric.rubric_id,
                  tasks=[Task(task_id=DEFAULT_TASK_ID, input=DEFAULT_TASK_INPUT)])
    if args.provider == "scripted":
        return Components(host=FakeHostMAS(), judge=ScriptedJudge(),
                          architect=ScriptedArchitect(), suite=suite)
    from archforge.llm import make_client, LLMError

    try:
        llm = make_client(args.provider, api_key=args.api_key, base_url=args.base_url)
    except LLMError as exc:
        print(f"[provider] {exc}", file=sys.stderr)
        raise
    arch = Architect(llm, model=args.architect_model or DEFAULT_MODELS[args.provider])
    judge = Judge(llm, model=args.judge_model or DEFAULT_MODELS[args.provider],
                  rubric=default_rubric)
    return Components(host=FakeHostMAS(), judge=judge, architect=arch, suite=suite)


def _ensure_incumbent(args: argparse.Namespace, specs: SpecStore) -> str | None:
    """Make sure an active incumbent exists before `evolve` runs.

    If one already exists, return its id (no LLM, no overwrite). If none and a
    `--seed` is supplied, lint + commit it as the root INCUMBENT and set active
    (zero-LLM bootstrap). Returns the active id, or None if it could not.
    """

    current = specs.active_id()
    if current is not None:
        return current
    seed_path = getattr(args, "seed", None)
    if not seed_path:
        return None
    spec = m.Spec.model_validate(json.loads(Path(seed_path).read_text(encoding="utf-8")))
    faults = lint(spec)
    if faults:
        raise SystemExit(
            "refusing to bootstrap from --seed: spec fails the linter: "
            + "; ".join(f"{f.code}({f.location or ''}): {f.message}" for f in faults)
        )
    spec_id = specs.commit(spec, parent_spec_id=None, status=m.SpecStatus.INCUMBENT)
    specs.set_active(spec_id)
    return spec_id


def _pending(atts: AttemptStore) -> list[m.Attempt]:
    return [a for a in atts.all() if a.verdict is m.Verdict.PENDING_HUMAN]


def _fmt_spec(spec: m.Spec) -> str:
    parent = spec.parent_spec_id or "(root)"
    return (f"  spec_id:  {spec.spec_id}\n"
            f"  parent:   {parent}\n"
            f"  status:   {spec.status.value}\n"
            f"  created:  {spec.created_at}\n"
            f"  nodes:    {len(spec.nodes)}\n"
            f"  edges:    {len(spec.edges)}")


# --------------------------------------------------------------------------- #
# subcommand handlers
# --------------------------------------------------------------------------- #


def _cmd_status(args: argparse.Namespace) -> int:
    specs, atts, _ = _stores(args.root)
    active_id = specs.active_id()
    if active_id is None:
        print("No incumbent yet. Bootstrap with `archforge evolve --seed <spec.json>`.")
        return 0
    spec = specs.get(active_id)
    print("active incumbent:")
    print(_fmt_spec(spec))
    chain = specs.lineage(active_id)
    print("lineage: " + " <- ".join(chain))
    print(f"specs known: {len(specs.known_ids())}    "
          f"archived: {len(specs.archived_ids())}    "
          f"attempts: {len(atts.all())}    "
          f"pending: {len(_pending(atts))}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    specs, atts, _ = _stores(args.root)
    rows = atts.all()
    if not rows:
        print("No attempts recorded yet.")
        return 0
    print(f"{'attempt_id':<18}{'verdict':<14}{'kind':<13}{'target':<8}"
          f"{'mean':>7}{'margin':>9}{'tokens':>8}  change")
    print("-" * 90)
    for a in rows:
        r = a.suite_result
        mean = f"{r.mean:.3f}" if r is not None else "-"
        margin = f"{r.margin_vs_incumbent:+.3f}" if r is not None else "-"
        toks = str(r.tokens) if r is not None else "-"
        print(f"{(a.attempt_id or '-'):<18}{a.verdict.value:<14}"
              f"{a.change.kind.value:<13}{a.change.target:<8}"
              f"{mean:>7}{margin:>9}{toks:>8}  {a.change.diff}")
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    specs, atts, _ = _stores(args.root)
    gk = Gatekeeper(specs, atts, thresholds=_thresholds_for_approval())
    if args.all or not args.attempts:
        pending = _pending(atts)
        if not pending:
            print("No queued (PENDING_HUMAN) changes to approve.")
            return 0
        ids = [a.attempt_id for a in pending]
    else:
        ids = list(args.attempts)

    approved: list[str] = []
    for aid in ids:
        try:
            att = gk.approve(aid)            # only path w/ Gatekeeper that moves active
        except ValueError as exc:
            print(f"! {aid}: {exc}")
            continue
        approved.append(aid)
        print(f"approved {aid}: active = {att.candidate_spec_id}  (verdict={att.verdict.value})")
    if not approved:
        print("Nothing approved.")
        return 1
    print(f"approved {len(approved)} change(s). active incumbent: {specs.active_id()}")
    return 0


def _cmd_reject(args: argparse.Namespace) -> int:
    specs, atts, _ = _stores(args.root)
    gk = Gatekeeper(specs, atts, thresholds=_thresholds_for_approval())
    try:
        att = gk.reject(args.attempt_id)
    except ValueError as exc:
        print(f"! {args.attempt_id}: {exc}")
        return 1
    print(f"rejected {args.attempt_id}: verdict={att.verdict.value}; "
          f"active unchanged at {specs.active_id()}")
    return 0


def _thresholds_for_approval() -> m.Thresholds:
    # approve/reject never consult τ/δ; defaults are fine (the verdicts already
    # carry the cycle's margin on their suite_result).
    return m.Thresholds()


def _cmd_evolve(args: argparse.Namespace, *, components: Components | None,
                loop: bool) -> int:
    # Injected `components` (the test/embedding path) always win — they ARE the
    # scripted organs, by contract. Otherwise build organs per `--provider`:
    # a real provider now wires the REAL Architect + Judge over a real LLMClient.
    if components is None and args.provider != "scripted":
        try:
            components = _default_components(args)           # builds a real LLMClient
        except Exception:
            return 2                                          # message already printed

    specs, atts, traces = _stores(args.root)

    # zero-LLM bootstrap of the root incumbent from --seed (if none active)
    active = _ensure_incumbent(args, specs)
    if active is None:
        print("No active incumbent and no --seed given; bootstrap with "
              "`archforge evolve --seed <spec.json>` first.", file=sys.stderr)
        return 1

    organs = components or _default_components(args)
    engine = Engine(
        host=organs.host, judge=organs.judge, architect=organs.architect,
        spec_store=specs, attempt_store=atts, trace_store=traces,
        suite=organs.suite, thresholds=_thresholds(args), config=_config(args),
    )

    if not loop:
        return _print_cycle(engine.evolve_cycle(cycle=0))
    return _print_loop(engine.evolve_loop())


def _print_cycle(r: CycleResult) -> int:
    if not r.attempted:
        print(f"archforge evolve: cycle={r.cycle} attempted=false action=none "
              f"note={r.note}")
        return 0
    action = r.decision.action.value if r.decision else "none"
    inc = f"{r.incumbent_mean:.3f}" if r.incumbent_mean is not None else "-"
    cand = f"{r.candidate_mean:.3f}" if r.candidate_mean is not None else "-"
    margin = f"{r.decision.margin:+.3f}" if r.decision else "-"
    print(f"archforge evolve: cycle={r.cycle} attempted=true action={action} "
          f"margin={margin} incumbent_mean={inc} candidate_mean={cand} "
          f"promoted={'true' if r.promoted else 'false'} "
          f"queued={'true' if r.queued else 'false'} "
          f"attempt_id={r.applied_attempt_id} tokens={r.tokens}")
    if r.note:
        print(f"  note: {r.note}")
    return 0


def _print_loop(r: LoopResult) -> int:
    print(f"archforge evolve-loop: cycles_run={r.cycles_run} promotions={r.promotions} "
          f"queued={r.queued} plateaued={'true' if r.plateaued else 'false'} "
          f"aborted={'true' if r.aborted else 'false'} "
          f"final_incumbent={r.final_incumbent_id} "
          f"final_mean={r.final_incumbent_mean if r.final_incumbent_mean is not None else '-'!s}")
    if r.aborted and r.abort_reason:
        print(f"  abort_reason: {r.abort_reason}")
    return 0


def _cmd_lint(path: str) -> int:
    from archforge.models import Spec

    spec = Spec.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    errors = lint(spec)
    if not errors:
        print("OK: spec is structurally valid.")
        return 0
    for e in errors:
        loc = f" [{e.location}]" if e.location else ""
        print(f"{e.code}{loc}: {e.message}")
    return 1


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None, *,
        components: Components | None = None) -> int:
    """Run the Forge CLI. `components` injects pre-configured organs (tests/embedding).

    When `components` is None the CLI builds them per `--provider`: the default
    `scripted` uses inert fakes; `--provider anthropic|openai|groq|gemini` wires
    real LLMs. Only the evolve-family consults `components`; `status`/`report`/
    `approve`/`reject`/`lint` read the stores directly.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "lint":
        return _cmd_lint(args.path)
    if args.command == "status":
        return _cmd_status(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "approve":
        return _cmd_approve(args)
    if args.command == "reject":
        return _cmd_reject(args)
    if args.command == "evolve":
        return _cmd_evolve(args, components=components, loop=False)
    if args.command == "evolve-loop":
        return _cmd_evolve(args, components=components, loop=True)
    # argparse rejects unknown subcommands before dispatch, so this is unreachable.
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
