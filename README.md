# ArchForge

> A self-improving meta-layer over multi-agent systems.
> Point it at your graph, give it a rubric, and it evolves your pipeline — one proven change per cycle.

**Python ≥ 3.11** · **zero hard deps beyond pydantic** · **no live LLMs in the test suite** · exercised end-to-end on a real LangGraph MAS (groq + google-genai + chroma, OpenTelemetry-traced) · *not yet released under an open-source license — see [License](#license).*

ArchForge sits **on top** of an existing multi-agent system (MAS) and improves it run-over-run. Each cycle it inspects where the judge docked points, proposes **one** targeted change — rewriting an agent's prompt, tuning a knob, adding a verifier, re-wiring a node, swapping a model — and keeps it only if it measurably beats the incumbent on a held-out suite. The host MAS keeps running tasks as normal; ArchForge observes the runs and feeds back an improved pipeline.

The design is deliberately minimal and verifiable:

- **One protected incumbent.** A candidate never touches production config; it's promoted only on strict improvement.
- **Immutable, versioned Specs are the single source of truth.** Evolving the pipeline = swapping which Spec the host instantiates, never patching live state.
- **Hybrid autonomy.** Safe small edits (prompt/knob) auto-apply; structural edits (roster/graph/model) queue for human approval.
- **Observation/control asymmetry.** The wrapper records traces and reports the active Spec, but never rewrites prompts mid-run. All mutation happens *between* runs, on the Spec.

No ground truth is required — an LLM-as-judge scores each run against a rubric.

### At a glance

| Component | Purpose |
| ---- | --- |
| [**Engine**](archforge/engine.py) | Orchestrates one P-E-C cycle and the loop (budget caps → clean abort; plateau → stop) |
| [**Architect**](archforge/architect.py) | Proposes **one** change per cycle from the last trace + judge scores + history (credit assignment, dedup of dead-ends) |
| [**SuiteRunner**](archforge/suite.py) | Runs the candidate against the held-out eval suite `R` repeats (noise absorption). The only component that invokes the host MAS |
| [**Judge**](archforge/judge/) | LLM-as-judge: scores each run per a versioned rubric, with a per-step breakdown for credit assignment |
| [**Gatekeeper**](archforge/gatekeeper.py) | Decides promote / queue-for-human / discard / rollback by margin `τ` + scope |
| [**Stores**](archforge/stores/) | `SpecStore` (versioned, content-addressed) + `TraceStore` + `AttemptStore` (all append-only) |
| [**TracingMiddleware**](archforge/middleware.py) | The host seam — wraps every agent, records each `Step`, reports the active Spec |

---

## A Simple Example

One Propose-Evaluate-Commit cycle, **zero cost** — no LLM, no network, no API keys. It wires the scripted organs (a fake host, a scripted Architect that proposes one prompt edit, a scripted Judge that scores it a win) into the real `Engine`, and you watch an auto-promotion end-to-end. A real run swaps the scripted organs for `--adapter <your_mas>` + `--provider <llm>` (see [Quickstart](#quickstart)).

```python
# save this as evolve_demo.py  —  run from an `init`-ed project dir
import tempfile
from pathlib import Path

import archforge.models as m
from archforge.architect import ScriptedArchitect
from archforge.engine import Engine, EngineConfig
from archforge.gatekeeper import Action
from archforge.host import FakeHostMAS
from archforge.host.base import Task
from archforge.judge import ScriptedJudge
from archforge.judge.base import Rubric
from archforge.stores import AttemptStore, SpecStore, TraceStore
from archforge.suite import Suite

rubric = Rubric(rubric_id="demo-v1",
                sub_rubrics={"correctness": "the answer is right",
                             "grounding": "the answer cites its sources"})

root = m.Node(node_id="a", role="responder", system_prompt="p0",
             model="gpt", tools=["t0"])
seed = m.Spec(nodes=[root], edges=[])

# the candidate the Architect will propose: same graph, a tighter prompt
cand = m.Spec(nodes=[m.Node(node_id="a", role="responder", system_prompt="p1",
                            model="gpt", tools=["t0"])], edges=[])
change = m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a",
                           "tighten the prompt", "reduce hallucination")

with tempfile.TemporaryDirectory() as d:
    root_dir = Path(d)
    specs, atts, ts = SpecStore(root_dir), AttemptStore(root_dir), TraceStore(root_dir)
    rid = specs.commit(seed, parent_spec_id=None, status=m.SpecStatus.INCUMBENT)
    specs.set_active(rid)
    # the candidate's spec_id is content-hashed OVER its parent — mirror the
    # engine's commit (parent = rid) when scripting the judge's score for it
    cand.parent_spec_id = rid
    judge = (ScriptedJudge(rubric=rubric)
             .set_aggregate(rid, "t1", 0.55)            # incumbent baseline
             .set_aggregate(cand.compute_spec_id(), "t1", 0.70))  # +0.15 >= τ

    engine = Engine(
        host=FakeHostMAS(), judge=judge, architect=ScriptedArchitect().propose(change, {"prompt": "p1"}),
        spec_store=specs, attempt_store=atts, trace_store=ts,
        suite=Suite(suite_id="S", rubric_id="default-v1", tasks=[Task(task_id="t1", input="q")]),
        config=EngineConfig(max_cycles=1, repeats=1, plateau_cycles=5),
    )
    r = engine.evolve_cycle(0)
    print(f"action={r.decision.action.name}  margin={r.decision.margin:+.2f}")
    print(f"incumbent_mean={r.incumbent_mean:.2f}  candidate_mean={r.candidate_mean:.2f}")
    print(f"active_spec_id={specs.active_id()[:8]}  promoted={r.decision.action is Action.AUTO_PROMOTE}")
```

```
$ archforge-optimizer init       # once — scaffolds the project config the Engine reads
$ python evolve_demo.py
action=AUTO_PROMOTE  margin=+0.15
incumbent_mean=0.55  candidate_mean=0.70
active_spec_id=57396a49  promoted=True
```

The candidate's tighter prompt beat the incumbent by `+0.15 ≥ τ`, so the Gatekeeper **auto-promoted** it to the new active Spec — all within immutable, versioned storage. Nothing was patched in place; the host will now instantiate the new Spec on its next run.

---

## Table of contents

- [How it works](#how-it-works)
- [Quickstart](#quickstart)
- [The optimization loop (P-E-C)](#the-optimization-loop-p-e-c)
- [Adapters: connecting your MAS](#adapters-connecting-your-mas)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Observability (OpenTelemetry GenAI tracing)](#observability-opentelemetry-genai-tracing)
- [Deployment: shipping optimizations to production](#deployment-shipping-optimizations-to-production)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Extending ArchForge](#extending-archforge)
- [Requirements](#requirements)
- [Roadmap](#roadmap)
- [Design references](#design-references)

---

## How it works

```
  HOST MAS  ──runs tasks──>  each agent wrapped by TracingMiddleware
   │                          records Step {prompt_in, response_out, tools, timing}
   │                          instantiates each node from the active Spec (prompt/model/knobs)
   ▼
  ┌──────────────────────────── FORGE ─────────────────────────────┐
  │  TraceStore ──> Judge (LLM-as-judge, rubric) ──> scored run     │
  │       │ + per-agent rubric breakdown                            │
  │       ▼                                                         │
  │  Architect (P-E-C): trace + judge + history → credit assign      │
  │       → propose ONE Spec change (diff + rationale + scope)       │
  │       ▼                                                         │
  │  SuiteRunner: run candidate on eval suite (R repeats)            │
  │       ▼                                                         │
  │  Gatekeeper: vs incumbent by margin τ + scope flag              │
  │     small + win  → auto-promote │ structural + win → human gate │
  │     lose         → discard      │ regress → rollback (lineage)  │
  │       ▼                                                         │
  │  SpecStore (versioned, immutable) — "active incumbent" pointer  │
  └─────────────────────────────┬──────────────────────────────────┘
                                 └── next host runs use the new incumbent Spec
```

Four organs, one loop:

| Organ | Role |
|---|---|
| **Architect** | Reads the last trace + judge scores + history, credit-assigns the rubric loss to a node/route, proposes **one** change. Forgets nothing — skips mutations already tried-and-rejected. |
| **SuiteRunner** | Runs each candidate against the held-out suite `R` times (repeats absorb judge noise). The only component that invokes the host MAS. |
| **Judge** | LLM-as-judge: scores each run per a versioned rubric (`grounding`, `correctness`, `completeness`, …), with a per-step breakdown for credit assignment. |
| **Gatekeeper** | Decides by margin `τ` + scope: auto-promote small wins, queue structural wins for a human, discard regressions, rollback if a later measurement regresses. |

---

## Quickstart

ArchForge imports and tests with **zero LLM** installed — adapters are import-lazy and self-skip when a provider SDK is absent. A real run needs one provider SDK.

```bash
# 1. Install (editable, in your normal env — no venv needed)
pip install -e . --break-system-packages

# Optional: install the provider(s) you actually run
pip install "archforge[providers-groq,providers-gemini]" --break-system-packages

# 2. Scaffold per-project config (.archforge/archforge.py, .env.example, suite.json)
archforge-optimizer init

# 3. Put your API keys in .env (gitignored)  ── e.g. GROQ_API_KEY=..., GEMINI_API_KEY=...

# 4. Lint a Spec before running
archforge-optimizer lint path/to/spec.json

# 5. Run one Propose-Evaluate-Commit cycle against your MAS, wired by an adapter
archforge-optimizer evolve \
    --adapter archforge_glue.aede_host:AEDEAdapter \
    --seed aede_spec.json

# 6. Run the full loop until it plateaus or hits a budget cap
archforge-optimizer evolve-loop \
    --adapter archforge_glue.aede_host:AEDEAdapter \
    --seed aede_spec.json --max-cycles 50

# 7. Inspect
archforge-optimizer status     # incumbent Spec + lineage + counts
archforge-optimizer report     # aggregate deltas across attempts
archforge-optimizer approve --all   # apply queued structural changes
```

> You can also invoke as `python -m archforge ...` — identical surface.

The `--provider` flag selects the LLM backing the Architect + Judge (`scripted` by default for zero-cost runs; `anthropic` / `openai` / `groq` / `gemini` for real runs). The host MAS is wired via `--adapter my_pkg.my_host:MyAdapter`.

---

## The optimization loop (P-E-C)

One cycle, end-to-end:

1. **Baseline.** Run the incumbent on the suite (`R` repeats) → judge → scored baseline, cached until the rubric/suite changes.
2. **Propose.** The Architect reads the incumbent's worst task scores + last trace, credit-assigns the loss, proposes **one** `Change` (a candidate = incumbent + that mutation).
3. **Evaluate.** The SuiteRunner runs the candidate on the *same* suite + *same* rubric (`R` repeats) → judge → candidate score.
4. **Commit (or don't).** The Gatekeeper decides by margin `τ` + scope:
   - `small + win` → **auto-promote** (`SpecStore.active ← candidate`)
   - `structural + win` → **queue for human** (the Approval Queue; the active pointer does *not* move)
   - `lose` → **discard**
   - a promoted incumbent that later regresses ≥ `δ` → **rollback** (a pointer swap to `parent_spec_id`)
5. **Persist.** The Attempt is appended; the promoted Spec is committed iff promoted/approved. `evolve-loop` repeats until a budget cap or `K` consecutive non-promotions (a plateau).

### The action space

`ChangeKind` ∈ `prompt_edit | knob | add_node | remove_node | rewire | model_swap`. Scope is mechanical: `small` (prompt/knob) auto-promotes; `structural` (roster/graph/model) requires a human. A Spec Linter validates every candidate *before* it reaches the SuiteRunner — orphans, dangling refs, self-loops, type rules.

### Governing invariants (asserted every test tier)

- **I1** — `SpecStore.active()` is the only Spec any host run can instantiate.
- **I2** — no committed Spec ever changes after `commit`.
- **I3** — every non-root Spec has a reachable `parent_spec_id` chain; rollback preserves it.
- **I4** — every structural win goes to `queue_for_human`; auto-promote never bypasses.
- **I5** — no `Attempt.suite_result` ever compares scores across a different `rubric_id` or task set.

### Error handling, by design

Every failure that touches the lineage **fails closed** — the incumbent is untouched, the candidate discarded or held, traces retained. Noise is absorbed by `R` repeats + margin `τ` + regression floor `δ ≥ τ` (so a noisy measurement never yo-yos the pointer). Host/agent errors mid-run are caught per-task (`Trace.ok=false`, partial trace retained); a candidate that fails > ε of tasks is auto-rejected *before* margin math. See [the design doc](docs/superpowers/specs/2026-07-22-archforge-design.md) §6 for the full E1–E10 catalog.

---

## Adapters: connecting your MAS

ArchForge couples to a host through one protocol — `HostMAS`:

```python
class HostMAS(Protocol):
    def instantiate(self, spec: m.Spec, middleware: "TracingMiddleware") -> Runnable: ...
```

Your adapter builds a runnable pipeline from `spec` (the active incumbent's nodes/edges/prompts/knobs) and threads `TracingMiddleware` through it so every step is recorded. Everything below the seam is your pipeline; everything above it is the Forge.

A **generic LangGraph adapter** ships in `archforge/host/adapters/langgraph.py` and drives a real `graph.stream(...)` — "describe, don't introspect" (it reads node *names*, the stable surface; it never climbs your graph's internals). It powers AEDE (the bundled example) and is the easiest path for any LangGraph-based MAS. For other frameworks (CrewAI, AutoGen, raw call loops), subclass `BaseHostAdapter` (`archforge/host/adapters/base.py`) — the kit is factored so adapting *any* MAS is cheap, not bespoke-per-framework.

Run it via the dotted-path seam:

```bash
archforge-optimizer evolve-loop --adapter your_pkg.your_host:YourAdapter --seed your_spec.json
```

### Bundled example: AEDE

`AEDE/backend/` ships a real retrieval/reasoning graph adapted to ArchForge via `archforge_glue/aede_host.py:AEDEAdapter`. It's the reference for the LangGraph adapter, the OTel tracing seam, and the deploy envelope. See `AEDE/backend/archforge_glue/` for the glue.

---

## CLI reference

```
archforge-optimizer <command> [flags]

  init          scaffold .archforge/archforge.py + .env.example + suite.json for this project
  lint <path>   run the Spec Linter on a JSON Spec file
  evolve        run one Propose-Evaluate-Commit cycle from the active incumbent
  evolve-loop   repeat evolve until the budget cap or a plateau
  approve       approve queued (PENDING_HUMAN) structural changes → active
  reject <id>   reject a queued structural change (active left alone)
  status        print the incumbent Spec + lineage + counts
  report        print aggregate deltas across attempts
```

### `evolve` / `evolve-loop` flags

| Flag | Purpose |
|---|---|
| `--root <dir>` | project root holding `.archforge/` (default `.`) |
| `--seed <path>` | bootstrap the root incumbent from a Spec JSON (first run) |
| `--adapter <dotted.path[:Class]>` | your `HostMAS` adapter |
| `--provider {scripted\|anthropic\|openai\|groq\|gemini}` | LLM backing the Architect + Judge |
| `--suite <path>` | evaluation suite JSON (default: `.archforge/suite.json`) |
| `--tau <float>` | promotion margin τ |
| `--delta <float>` | regression floor δ (≥ τ) |
| `--repeats <int>` | R: repeats per eval-suite task (noise absorption) |
| `--env-file <path>` | `.env` to load API keys from |
| `--api-key`, `--base-url`, `--architect-model`, `--judge-model` | per-call overrides |
| `--max-cycles <int>` | (*loop only*) cycle ceiling |
| `--plateau-cycles <int>` | (*loop only*) K: consecutive non-promotions → stop |
| `--max-tokens-total <int>` | (*loop only*) total token budget → abort |
| `--max-tokens-per-cycle <int>` | per-cycle token cap → clean abort (incumbent untouched) |
| `--max-wall-ms-per-cycle <float>` | per-cycle wall-clock cap (bounds non-LLM nodes that cost time, not tokens) |

`approve` takes attempt ids (or `--all`); `init` takes `--force` to overwrite.

---

## Configuration

Per-project config lives in **`.archforge/archforge.py`** — a plain Python file, **active as-is** (no registration), so `init → CLI works` immediately. Edit a value to change a default. `init` scaffolds it with sane defaults. Tunables include `PROVIDER`, `DEFAULT_TAU`, `DEFAULT_DELTA`, `DEFAULT_REPEATS`, `DEFAULT_MAX_CYCLES`, `DEFAULT_PLATEAU_CYCLES`, the budget caps, the architect model roster, and `DEFAULT_TRACE_TOTAL_BUDGET_TOK` (the tracing toggle — see below).

API keys live in **`.env`** (gitignored — your own keys, never logged or committed). `evolve` loads them from `.env` for `--provider != scripted`; the environment always wins, and `--api-key` wins above both.

The evaluation suite is **`.archforge/suite.json`** — the representative tasks the Judge scores. Optimization targets the *suite*, never a single repeated task (the primary defense against overfitting structural mutations).

---

## Observability (OpenTelemetry GenAI tracing)

By default, each `Step` the Judge reads carries a host-authored one-liner summary (e.g. `answer_len=1189`) — lossy on the **host-streaming path**. ArchForge can instead auto-instrument your SDK calls as **OpenTelemetry GenAI spans** and project a **bounded slice** of the real prompt/completion into each `Step` — so the Judge compares real content against the task, not length stubs.

- **Cooperative attribution.** A forge-owned `wrapped(name, fn)` opens an `archforge.node` parent span; auto-instrumented LLM/retriever spans nest as children by parent-link (not temporal order) — robust to retries, multi-call, and fan-out.
- **Bounded.** Per-kind caps keep the total judge-prompt token budget bounded; a post-loop shed trims the largest remaining steps while **protecting the final-answer step**.
- **Gated, not forked.** `DEFAULT_TRACE_TOTAL_BUDGET_TOK = None` reproduces the lossy `summarize()` path **byte-identically** → the existing suite stays green. Set a number to turn on rich steps. Toggle, not fork.
- **Zero-dep by default.** `archforge.otel` is import-lazy — `import archforge` and `import archforge.otel` pull **zero** OpenTelemetry. Per-SDK instrumentors (`opentelemetry-instrumentation-<sdk>`) are the MAS owner's install.
- **Secrets stay in-process.** The in-memory span buffer has no exporter — nothing leaves the process. Never wire an OTLP exporter without a redaction processor.

See [`docs/superpowers/specs/2026-08-08-tracing-otel-design.md`](docs/superpowers/specs/2026-08-08-tracing-otel-design.md) for the full spec.

---

## Deployment: shipping optimizations to production

When a candidate auto-promotes, ArchForge can emit a **deploy envelope** — a self-contained JSON with the promoted Spec, the knobs to overlay, the scores, and the decision (margin + rule). Your MAS reads it at startup (e.g. `aede_sidecar.load_at_startup`) and applies the knobs without the Forge on the hot path. Opt-in via the engine's `on_deploy` hook (the CLI wires it to write `.archforge/optimized.json`); `on_cycle` is the richer per-cycle surface (specs, runs, change) for custom rendering/telemetry.

---

## Project layout

```
archforge/
  cli.py            the Forge — argparse entrypoint + per-command wiring
  engine.py         the P-E-C orchestrator + loop (E3/E8 budget/plateau)
  architect.py       proposes one change per cycle (credit assignment, dedup)
  suite.py           SuiteRunner — runs the eval suite R repeats
  judge/             LLM-as-judge (base.py + scripted.py)
  gatekeeper.py       decides promote / queue / discard / rollback
  stores/            SpecStore (versioned) + TraceStore + AttemptStore (append-only)
  middleware.py       TracingMiddleware — the host seam
  host/              HostMAS protocol + adapter kit (base.py, langgraph.py, ...)
  llm/               provider clients (anthropic/openai/groq/gemini, lazy + self-skip)
  otel.py            OpenTelemetry GenAI tracing (import-lazy, bounded projection)
  lint.py            Spec Linter (validate-DAG, refs, type rules)
  mutate.py          apply a Change to a Spec
  diff.py             spec_diff / format_diff (human-readable mutation deltas)
  runlog.py           per-cycle run log (cards)
  models.py          Spec/Node/Edge/Step/Trace/RunScore/Attempt/Change/...
  config.py .py / userconfig.py / config_init.py   versioning + tunable resolver + `init`
AEDE/backend/        bundled example MAS + its archforge_glue adapter
docs/superpowers/specs/  design docs
tests/               unit / integration / scenarios / adapters / llm / smoke
```

---

## Testing

No real LLMs in CI. The suite uses `ScriptedJudge`, `ScriptedArchitect`, and a `FakeHostMAS` for deterministic, zero-cost runs.

```bash
python -m pytest -q          # default: everything except the opt-in real-LLM smoke tier
python -m pytest -m smoke    # opt-in real-LLM smoke (make live, billed API calls)
```

Tiers — **unit** (each component behind its interface), **integration** (real store wiring on a tmp dir), **scenarios** (drives AEDE's *real* LangGraph graph offline with span-emitting fakes), **adapters**, **llm** (provider clients), and **smoke** (opt-in real-LLM, gated by the `smoke` marker).

The invariants I1–I5 and the error scenarios E1–E10 from the design doc are pinned as tests — see [`tests/`](tests/) and the design doc §7.

---

## Extending ArchForge

- **A new MAS.** Subclass `BaseHostAdapter` (or use the LangGraph adapter if you're on LangGraph), implement `instantiate(spec, middleware) -> Runnable`, and pass it via `--adapter`.
- **A new provider.** Add a client under `archforge/llm/` (subclass `LLMClient`); register it in the CLI's `_PROVIDERS`.
- **A new mutation kind.** Add it to `ChangeKind` + `scope_for_kind`, implement it in `mutate.apply_change`, and teach the Architect to propose it.
- **A richer rubric.** Write a `suite.json` + rubric; the Judge scores each run against it. Comparisons are only valid within `(rubric_id, suite_id)` — bumping either starts a fresh baseline (I5).
- **Custom cycle/deploy surfaces.** Wire `on_cycle` / `on_deploy` callbacks into the `Engine` for telemetry, painting, or sidecar sync.

The public model surface (`archforge.models`) is the stable contract: `Spec`, `Node`, `Edge`, `Knobs`, `Step`, `Trace`, `RunScore`, `Attempt`, `Change`, `Thresholds`, and the `ChangeKind`/`Scope`/`Verdict`/`SpecStatus` enums. `archforge.host.base` defines `Task`, `AgentResponse`, `Agent`, `Runnable`, `HostMAS`.

---

## Requirements

- Python ≥ 3.11 (developed on 3.14)
- `pydantic >= 2.7`, `python-dotenv >= 1.0` (only hard deps — ArchForge imports cleanly with nothing else)
- Provider SDKs (optional, install only what you run): `anthropic`, `openai`, `groq`, `google-genai`
- For rich tracing (optional): `opentelemetry-sdk` + the per-SDK instrumentors you call
- Dev: `pytest`, `hypothesis`, `ruff`, `mypy` (`pip install -e ".[dev]"`)

> **Install discipline.** Use `pip install -e .` (editable). A flat `pip install .` makes a *non-editable* copy in site-packages — any later source edit won't take effect at the CLI. If a repo edit ever seems to no-op, check `python -c "import archforge; print(archforge.__file__)"` resolves to the repo, not site-packages.

---

## Roadmap

ArchForge is real and exercised end-to-end on the bundled AEDE example (real groq + google-genai + chroma, OTel-traced). Active directions:

- **Delegation specs** (replace hand-rolled subsystems with vetted libraries):
  - ✅ #1 — Tracing → OpenTelemetry GenAI (built + verified; bounded real I/O to the Judge)
  - 🚧 #2 — LLM clients → LiteLLM (unify the per-provider clients behind one library)
  - 🚧 #3 — Judge → DeepEval / Ragas (rubric scoring via a mature eval framework)
- **Adapter kit.** Generalize so adapting *any* MAS is cheap (LangGraph done; CrewAI/AutoGen/raw-loops next).
- **Hierarchical search (v2).** A Strategist layer that emits scoped optimization goals, layered over the P-E-C loop once the cheap one-change loop is reliable.

No part of the roadmap requires breaking the model surface — additions are additive and gated behind tunables.

---

## Design references

- [`docs/superpowers/specs/2026-07-22-archforge-design.md`](docs/superpowers/specs/2026-07-22-archforge-design.md) — core architecture, data model, P-E-C loop, error handling (E1–E10), invariants (I1–I5), alternatives.
- [`docs/superpowers/specs/2026-07-30-adapter-kit-design.md`](docs/superpowers/specs/2026-07-30-adapter-kit-design.md) — the general adapter kit.
- [`docs/superpowers/specs/2026-08-03-mixed-non-llm-nodes-design.md`](docs/superpowers/specs/2026-08-03-mixed-non-llm-nodes-design.md) — non-LLM (retriever/tool/rule) nodes.
- [`docs/superpowers/specs/2026-08-08-tracing-otel-design.md`](docs/superpowers/specs/2026-08-08-tracing-otel-design.md) — OTel GenAI tracing delegation.

---

## License

ArchForge is not yet under a released open-source license — no `LICENSE` file ships in this repository. **Treat it as source-available, all rights reserved, until one is added.** A permissive license will be attached before the first public release; in the meantime, contact the maintainer before redistributing or basing derivative work on this code.

---

*ArchForge never patches live state — it swaps which versioned pipeline the host uses.*
