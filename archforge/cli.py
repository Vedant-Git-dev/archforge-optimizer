"""ArchForge command-line interface — the Forge.

Wires the four organs (Architect, SuiteRunner, Judge, Gatekeeper) plus the
filesystem stores into the runnable surface the user actually touches:

    archforge-optimizer lint <spec.json>       validate a Spec
    archforge-optimizer evolve  [--root R] [--seed S]   one Propose-Evaluate-Commit cycle
    archforge-optimizer evolve-loop [...]      repeat until budget cap or plateau
    archforge-optimizer approve [<id>...|--all]   drain the structural-change queue
    archforge-optimizer reject <id>            reject a queued change (active kept)
    archforge-optimizer status                 incumbent Spec + lineage + counts
    archforge-optimizer report                 aggregate deltas across attempts

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
import importlib
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
from archforge.suite import Suite, load_suite_file
from archforge.config_init import archforge_config_text, env_example_text, _DEFAULT_SUITE_JSON

# System config (provider roster, CLI fixtures, PROG, .env loader) — single source
# in archforge.config. The TUNABLE defaults (tau/delta/repeats/provider/models/…)
# are NOT imported here; they resolve lazily from archforge.userconfig (the active
# project config made by `init`) at use time, so importing+running the CLI works
# before `init` has created .archforge/archforge.py and so an edit takes effect on
# the next run. See archforge/userconfig.py.
from archforge import userconfig as ucfg
from archforge.config import (
    ALL_PROVIDERS as _PROVIDERS,
    DEFAULT_SUITE_ID, DEFAULT_TASK_ID, DEFAULT_TASK_INPUT,
    PROG, load_env,
)
from archforge.userconfig import ConfigNotInitialized

# Fixed run-state / config-discovery dir (a system path, independent of the
# tunable DEFAULT_ROOT_DIR which the embedder API reads via archforge.userconfig).
_DEFAULT_ROOT = ".archforge"


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


def _import_adapter(dotted: str) -> HostMAS:
    """Import an external MAS adapter from a ``module:Class`` (or ``module``)
    dotted path and instantiate it. The class implements ``HostMAS`` (the kit's
    ``BaseHostAdapter`` does), so it drops straight in as ``components.host`` —
    its own ``__init__`` carries whatever its MAS needs (Lumina loads its base
    prompts; a framework adapter wraps its graph). No PR into core to adapt a
    new MAS: ``--adapter mypkg:MyAdapter`` wires it; ``--provider`` keeps the
    Architect/Judge organs, ``--seed`` the bootstrap Spec."""
    if ":" in dotted:
        modpath, cls = dotted.split(":", 1)
    else:
        modpath, cls = dotted, ""
    module = importlib.import_module(modpath)
    if not cls:
        # Bare module: expect it to expose a ``HostMAS``-protocol attr named
        # ``HostMAS`` or the last path segment; else error loudly.
        cls = "HostMAS"
    try:
        obj = getattr(module, cls)
    except AttributeError as exc:
        raise SystemExit(
            f"--adapter: module {modpath!r} has no attribute {cls!r}. "
            f"Pass it as `module:ClassName`."
        ) from exc
    if isinstance(obj, type):
        return obj()                 # a HostMAS/BaseHostAdapter subclass → instance
    if isinstance(obj, HostMAS):
        return obj                   # already an instance
    raise SystemExit(f"--adapter: {dotted!r} resolved to a {type(obj).__name__}, "
                     "not a HostMAS subclass or instance.")


# --------------------------------------------------------------------------- #
# arg parsing
# --------------------------------------------------------------------------- #


def _add_store_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=_DEFAULT_ROOT,
                   help="archforge state directory (default: .archforge)")


def _add_evolve_args(p: argparse.ArgumentParser, *, loop: bool) -> None:
    p.add_argument("--seed", metavar="PATH",
                   help="bootstrap the root incumbent from this Spec JSON (no active yet)")
    p.add_argument("--adapter", metavar="DOTTED.PATH[:Class]",
                   help="import an external MAS adapter (a HostMAS/BaseHostAdapter "
                        "subclass) as the runtime host; pair with --provider for the "
                        "Architect/Judge organs. e.g. --adapter archforge_glue:LuminaAdapter")
    p.add_argument("--provider", choices=_PROVIDERS, default=None,
                   help="LLM provider (default from archforge.py: anthropic/openai/groq/gemini "
                        "are real; scripted is the zero-cost fake)")
    # API keys: a `.env` in the cwd is loaded first (load_env, ∴ real env wins), then
    # the provider SDK reads its key var; --api-key overrides both. Tunable flags
    # default to None here so the active config (.archforge/archforge.py) supplies
    # the real default at resolve time — an explicit flag overrides the file.
    p.add_argument("--env-file", default=None,
                   help="load provider API keys from this file before --provider "
                        "builds the organs (default from archforge.py; no-op if absent)")
    p.add_argument("--api-key", default=None, help="provider API key (overrides .env/env)")
    p.add_argument("--base-url", default=None, help="provider base URL override")
    p.add_argument("--architect-model", default=None,
                   help="model id for the Architect (else the provider default)")
    p.add_argument("--judge-model", default=None,
                   help="model id for the Judge (else the provider default)")
    p.add_argument("--suite", metavar="PATH", default=None,
                   help="path to a suite.json (overrides the DEFAULT_SUITE_FILE tunable; "
                        "the file's tasks define what you optimize against)")
    # thresholds — every default resolves from the active config (.archforge/archforge.py);
    # a flag is None at the parser and filled from ucfg unless the user set it.
    p.add_argument("--tau", type=float, default=None, help="promotion margin τ")
    p.add_argument("--delta", type=float, default=None, help="regression floor δ (>= τ)")
    p.add_argument("--repeats", type=int, default=None, help="R: repeats per eval-suite task")
    if loop:
        p.add_argument("--max-cycles", type=int, default=None,
                       help="cap on P-E-C cycles")
        p.add_argument("--plateau-cycles", type=int, default=None,
                       help="K consecutive no-promotion cycles → plateau (E8)")
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

    # --- init (scaffold user config) -----------------------------------------
    init_p = sub.add_parser("init",
                            help="scaffold .archforge/archforge.py + .env.example for this project")
    _add_store_args(init_p)   # --root selects where archforge.py is written
    init_p.add_argument("--force", action="store_true",
                        help="overwrite an existing .archforge/archforge.py")

    return parser


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _stores(root: str) -> tuple[SpecStore, AttemptStore, TraceStore]:
    return SpecStore(root), AttemptStore(root), TraceStore(root)


def _arg(args: argparse.Namespace, attr: str, cfg_name: str):
    """Resolve a CLI tunable: the explicit flag value, else the active-config default.

    Every tunable flag defaults to ``None`` at the parser (so `--help` works pre-init
    and a user's `.archforge/archforge.py` override isn't frozen into argparse). The
    real default is read here from ``ucfg`` (disk post-init, or the in-memory sane
    template under the pytest gate) only when the flag is omitted.
    """
    v = getattr(args, attr, None)
    return v if v is not None else ucfg.get(cfg_name)


def _thresholds(args: argparse.Namespace) -> m.Thresholds:
    return m.Thresholds(
        tau=_arg(args, "tau", "DEFAULT_TAU"),
        delta=_arg(args, "delta", "DEFAULT_DELTA"),
        repeats=_arg(args, "repeats", "DEFAULT_REPEATS"),
        plateau_cycles=_arg(args, "plateau_cycles", "DEFAULT_PLATEAU_CYCLES"),
    )


def _config(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        max_cycles=_arg(args, "max_cycles", "DEFAULT_MAX_CYCLES"),
        max_tokens_per_cycle=_arg(args, "max_tokens_per_cycle", "DEFAULT_MAX_TOKENS_PER_CYCLE"),
        max_tokens_total=_arg(args, "max_tokens_total", "DEFAULT_MAX_TOKENS_TOTAL"),
        repeats=_arg(args, "repeats", "DEFAULT_REPEATS"),
        plateau_cycles=_arg(args, "plateau_cycles", "DEFAULT_PLATEAU_CYCLES"),
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

    # The eval suite: a JSON sidecar (.archforge/suite.json by default) if present,
    # else the one-task fallback fixture (byte-identical to `init`'s seeded
    # suite.json, so out-of-box == generated-default). --suite overrides the tunable.
    suite_path = getattr(args, "suite", None) or ucfg.get("DEFAULT_SUITE_FILE")
    suite = load_suite_file(suite_path) or Suite(
        suite_id=DEFAULT_SUITE_ID, rubric_id=default_rubric().rubric_id,
        tasks=[Task(task_id=DEFAULT_TASK_ID, input=DEFAULT_TASK_INPUT)])
    provider = args.provider or ucfg.get("PROVIDER")
    if provider == "scripted":
        return Components(host=FakeHostMAS(), judge=ScriptedJudge(),
                          architect=ScriptedArchitect(), suite=suite)
    from archforge.llm import make_client, LLMError

    try:
        llm = make_client(provider, api_key=args.api_key, base_url=args.base_url)
    except LLMError as exc:
        print(f"[provider] {exc}", file=sys.stderr)
        raise
    arch_models = ucfg.get("DEFAULT_ARCHITECT_MODELS")
    judge_models = ucfg.get("DEFAULT_JUDGE_MODELS")
    arch = Architect(llm, model=args.architect_model or arch_models[provider])
    judge = Judge(llm, model=args.judge_model or judge_models[provider],
                  rubric=default_rubric())
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
        print("No incumbent yet. Bootstrap with `archforge-optimizer evolve --seed <spec.json>`.")
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
    # organs, by contract. Otherwise build organs per `--provider`. While a real
    # tunable is needed we require `init` to have run (the "pip install → init → CLI
    # works" contract): a project with no .archforge/archforge.py gets the init hint
    # and rc 1 instead of a bogus run. No-op under the pytest gate (tests use the
    # in-memory sane template).
    if components is None:
        try:
            ucfg.ensure_initialized()
        except ConfigNotInitialized as exc:
            print(str(exc), file=sys.stderr)
            return 1
        # Load API keys from the cwd's .env (real env wins; --api-key wins above
        # both). No-op if the file is missing or for --provider scripted.
        load_env(getattr(args, "env_file", None) or ucfg.get("DEFAULT_ENV_FILE"))
        if (args.provider or ucfg.get("PROVIDER")) != "scripted":
            try:
                components = _default_components(args)           # builds a real LLMClient
            except Exception:
                return 2                                          # message already printed

    specs, atts, traces = _stores(args.root)

    # zero-LLM bootstrap of the root incumbent from --seed (if none active)
    active = _ensure_incumbent(args, specs)
    if active is None:
        print("No active incumbent and no --seed given; bootstrap with "
              "`archforge-optimizer evolve --seed <spec.json>` first.", file=sys.stderr)
        return 1

    organs = components or _default_components(args)

    # --adapter: swap the runtime host for an external MAS adapter (a HostMAS /
    # BaseHostAdapter) while keeping the --provider organs (Architect/Judge).
    # This is the "adapt any MAS" seam: point at an adapter class, no core edit.
    adapter_path = getattr(args, "adapter", None)
    if adapter_path:
        organs = Components(host=_import_adapter(adapter_path), judge=organs.judge,
                            architect=organs.architect, suite=organs.suite)

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
        print(f"{PROG} evolve: cycle={r.cycle} attempted=false action=none "
              f"note={r.note}")
        return 0
    action = r.decision.action.value if r.decision else "none"
    inc = f"{r.incumbent_mean:.3f}" if r.incumbent_mean is not None else "-"
    cand = f"{r.candidate_mean:.3f}" if r.candidate_mean is not None else "-"
    margin = f"{r.decision.margin:+.3f}" if r.decision else "-"
    print(f"{PROG} evolve: cycle={r.cycle} attempted=true action={action} "
          f"margin={margin} incumbent_mean={inc} candidate_mean={cand} "
          f"promoted={'true' if r.promoted else 'false'} "
          f"queued={'true' if r.queued else 'false'} "
          f"attempt_id={r.applied_attempt_id} tokens={r.tokens}")
    if r.note:
        print(f"  note: {r.note}")
    return 0


def _print_loop(r: LoopResult) -> int:
    print(f"{PROG} evolve-loop: cycles_run={r.cycles_run} promotions={r.promotions} "
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


def _cmd_init(args: argparse.Namespace) -> int:
    """Scaffold `.archforge/archforge.py` (user-editable config) + `.env.example`
    (repo root). Never touches a real `.env`. Refuses to clobber an existing
    `archforge.py` unless `--force`; never overwrites an existing `.env.example`."""
    root = Path(args.root or _DEFAULT_ROOT)
    cfg_path = root / "archforge.py"

    # 1. archforge.py — refuse-clobber unless --force.
    if cfg_path.exists() and not args.force:
        print(f"! {cfg_path} already exists. Re-run with --force to overwrite "
              "(your edits would be lost).", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)        # .archforge/ (also the run state dir)
    cfg_path.write_text(archforge_config_text(), encoding="utf-8")
    print(f"created: {cfg_path}  (edit a value to change a default; the file is ACTIVE as-is)")

    # 2. .env.example — create once at the repo root (cwd), never overwrite.
    env_example = Path(".env.example")
    if env_example.exists():
        print(f"kept:    {env_example}  (already present — left untouched)")
    else:
        env_example.write_text(env_example_text(), encoding="utf-8")
        print(f"created: {env_example}  (copy to .env and fill in your API keys)")

    # 3. suite.json — seed the eval-task sidecar next to archforge.py (so --root
    # relocations also move the seeded suite); never overwrite — the user may have
    # tuned the tasks. Byte-identical to the CLI's one-task fallback fixture.
    suite_path = root / "suite.json"
    if suite_path.exists():
        print(f"kept:    {suite_path}  (already present — left untouched)")
    else:
        suite_path.write_text(_DEFAULT_SUITE_JSON, encoding="utf-8")
        print(f"created: {suite_path}  (edit the tasks to change what you optimize against)")
    print(f"\nNext: edit {cfg_path}, then run `{PROG} evolve --seed <spec.json>`.")
    return 0


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
    if args.command == "init":
        return _cmd_init(args)
    if args.command == "evolve":
        return _cmd_evolve(args, components=components, loop=False)
    if args.command == "evolve-loop":
        return _cmd_evolve(args, components=components, loop=True)
    # argparse rejects unknown subcommands before dispatch, so this is unreachable.
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
