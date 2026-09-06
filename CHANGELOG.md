# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Real provider token usage in traces (zero-touch).** The `SdkInjector` now also reads the metered usage off each SDK response (`usage.total_tokens` for groq/openai, `usage_metadata.total_token_count` for google.genai) and the LangGraph adapter drains it per step: a node's step consumes exactly the calls made since its previous step, so loops and multi-call nodes account correctly. The `len//4` estimate remains the fallback for explicit-seam apps, missing usage fields, and non-LLM kinds. Any SDK/response shape deviation records nothing; accounting never breaks the host's call.
- **Zero-touch host integration for the LangGraph adapter** (`archforge/host/adapters/inject.py`). A host MAS now integrates by filling the scaffolded `archforge_optimizer/app.py` only; its own source stays untouched. Three mechanisms: (1) the default `LangGraphApp.apply_llm_config` records each node's `KnobVote` and a run-scoped `SdkInjector` patches the `groq` / `openai` / `google.genai` SDK boundaries, attributing each call to its node via the call stack (`NodeLocator`, derived from the compiled graph's node callables) and rewriting model/temperature/max_tokens/system_prompt per the vote; (2) new `settings_getter` / `knob_to_settings` app attrs make settings-backed knobs tunable by snapshot-mutate-restore on the host's config singleton; (3) the injector captures the first observed system prompt per node so a later `prompt_edit` has a real base even with `base_prompts` empty. Apps that override `apply_llm_config` (an explicit call-time seam) are detected and the injector stays off, so existing adapters are unaffected.

## [0.4.0] - 2026-09-04

### Added
- **Pluggable evaluation backends (issue #2).** The Judge is now selectable behind the existing `JudgeProtocol` seam: a new `EVALUATORS` roster (`native` / `deepeval`), a `make_evaluator` factory (mirroring `make_client`), a `--evaluator` CLI flag (with `--deepeval-metric`, repeatable), and `DEFAULT_EVALUATOR` / `DEFAULT_DEEPEVAL_METRICS` tunables in `.archforge/archforge.py`. The native LLM-as-judge stays the default and is unchanged.
- **DeepEval backend** (`archforge/judge/deepeval.py`, optional `deepeval` extra). `DeepEvalEvaluator` projects a trace into a DeepEval `LLMTestCase` and scores it with standalone metrics (`answer_relevancy`, `faithfulness`); metric scores map to `RunScore.rubric_scores` with the aggregate as their mean, comparable to the native judge's [0,1] aggregate. It is run-level (empty `step_scores`; credit assignment degrades gracefully), imports DeepEval lazily (a missing install surfaces as a clear `LLMError`, never an `ImportError` at package import), and converts DeepEval's own errors to `LLMError` so the SuiteRunner treats its failures exactly like a native judge failure (E9). The judge model is configurable: `--judge-model` / `DEFAULT_JUDGE_MODELS` are reused, prefixed with the provider (new public `archforge.llm.litellm.prefix_model` helper) and wrapped in DeepEval's `LiteLLMModel`, so scoring routes through LiteLLM to your provider instead of DeepEval's OpenAI default.

## [0.3.0] - 2026-08-23

### Changed
- **LiteLLM unifies the provider layer (issue #1).** The four bespoke provider adapters (`anthropic`/`openai`/`groq`/`gemini`) collapse into one `LiteLLMClient` (`archforge/llm/litellm.py`) behind the same `LLMClient` seam: a provider is now just a model-id prefix (`openai/gpt-4o`, `gemini/gemini-3.6-flash`, …), so adding a provider needs no new adapter. `make_client` builds the LiteLLM client for any real provider; the `LLMClient`/`Completion`/`Usage`/`LLMError` contract, the centralized `DEFAULT_ARCHITECT_MODELS` config, and `ScriptedLLM` are unchanged. LiteLLM is import-lazy (a core dep) and shells out to the per-provider SDKs at call time (`providers-*` extras). The 5 adapter/helper files were deleted; the stubbed-SDK provider tests became `tests/llm/test_litellm.py` (stubbing `litellm.completion`).

## [0.2.0] - 2026-08-22

### Added
- **`init` scaffolds the adapter package.** `archforge-optimizer init` now writes a generic, name-neutral `archforge_optimizer/` package (5 files: `__init__.py`, `host.py`, `app.py`, `sidecar.py`, `test_smoke_offline.py`) into the project root, a LangGraph adapter skeleton the user fills in via `# EDIT:` markers instead of coding the wiring from scratch. The scaffold is runtime-generated from string constants, so the sdist/wheel ship only `archforge/`.
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