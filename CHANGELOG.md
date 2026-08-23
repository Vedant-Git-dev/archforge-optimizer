# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-23

### Changed
- **LiteLLM unifies the provider layer (issue #1).** The four bespoke provider adapters (`anthropic`/`openai`/`groq`/`gemini`) collapse into one `LiteLLMClient` (`archforge/llm/litellm.py`) behind the same `LLMClient` seam: a provider is now just a model-id prefix (`openai/gpt-4o`, `gemini/gemini-3.6-flash`, …), so adding a provider needs no new adapter. `make_client` builds the LiteLLM client for any real provider; the `LLMClient`/`Completion`/`Usage`/`LLMError` contract, the centralized `DEFAULT_ARCHITECT_MODELS` config, and `ScriptedLLM` are unchanged. LiteLLM is import-lazy (a core dep) and shells out to the per-provider SDKs at call time (`providers-*` extras). The 5 adapter/helper files were deleted; the stubbed-SDK provider tests became `tests/llm/test_litellm.py` (stubbing `litellm.completion`).

## [0.2.0] - 2026-08-22

### Added
- **`init` scaffolds the adapter package.** `archforge-optimizer init` now writes a generic, name-neutral `archforge_optimizer/` package (5 files: `__init__.py`, `host.py`, `app.py`, `sidecar.py`, `test_smoke_offline.py`) into the project root, a LangGraph adapter skeleton the user fills in via `# EDIT:` markers instead of coding the wiring from scratch. Derived from the AEDE glue (which stays the in-repo, non-shipped reference); the scaffold is runtime-generated from string constants, so the sdist/wheel still ship only `archforge/`.
- **`make-spec` command.** New subcommand that builds the bootstrap `archforge_optimizer/spec.json` from the *edited* adapter: it imports `--adapter`, builds the Spec via `app_spec()`, lints it, and writes only if it passes (rc=1 + the lint faults if not). So the spec `evolve --seed` consumes is the user's real roster, not a placeholder template.
- **`init` no longer writes `.env.example` or `spec.json`.** The provider API key goes in the root `.env` (gitignored); `init` only prints the hint. `spec.json` is `make-spec`'s job.
- **Scaffold defaulting for `evolve`.** After `init` (+ `make-spec`), `evolve`/`evolve-loop` auto-default `--adapter archforge_optimizer.host:AppAdapter` and `--seed archforge_optimizer/spec.json` when the scaffold is present, no flags needed. The flags remain for custom adapters/seeds; the `--provider scripted` fake path still keeps `FakeHostMAS`.

### Changed
- **Class names in the scaffold.** The placeholder adapter classes are now `App` (`app.py`, a `LangGraphApp`) and `AppAdapter` (`host.py`, a `LangGraphHostAdapter`). The redundant `MyHostMAS = MyAdapter` alias was removed.
- **`init` adapter-scaffold ordering.** The adapter scaffold runs before the `archforge.py` refuse-clobber, so a user with existing config but a deleted/half-edited adapter can repair it with a bare `init` (per-file keep/`--force` guards; no `--force` clobbering their config).

### Security
- `init` never reads or echoes a real `.env`; `make-spec` and the scaffold are secret-safe.

## [0.1.0] - 2026-08-13

### Added
- **The P-E-C engine**: `evolve` runs one Propose-Evaluate-Commit cycle; `evolve-loop` repeats until a budget cap or a plateau.
- **Reusable adapter kit** (`archforge.host.adapters`): `BaseHostAdapter`, `BaseAgent`, `BasePipeline`, plus a generic **LangGraph adapter** that drives a real `graph.stream(...)` ("describe, don't introspect").
- **Architect** with credit assignment + dead-end dedup; **Judge** (LLM-as-judge, versioned rubric, per-step breakdown); **Gatekeeper** deciding promote / queue-for-human / discard / rollback by margin τ + scope.
- **Immutable, versioned Specs**: content-addressed `SpecStore` (active-incumbent pointer), append-only `TraceStore` + `AttemptStore`.
- **`init`** scaffolds `.archforge/archforge.py` (user-editable tunables with sane defaults) + `.archforge/suite.json` (eval tasks), plus OpenTelemetry-based tracing and a deploy sidecar (Tier-2 `optimized.json` rollout).