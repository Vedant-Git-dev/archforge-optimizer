"""Shared helpers for host adapters — the single source for the recurring
wiring every MAS adapter (and `FakeHostMAS`) needs.

These were lifted from two as-built implementations so the kit has ONE copy:

  * `topo_order` / `run_id`  — from `archforge.host.fake` (graph walk + run-id).
  * `cfg_decay`             — from the Lumina adapter's `_cfg` (the phase-2
    "thread model/temp/max_tokens always, system_prompt only when mutated" rule).

This module is a pure leaf: it imports only `archforge.models` /
`archforge.constants`, so importing it from `host/fake.py` (or any adapter)
cannot form a cycle.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import archforge.models as m
from archforge.constants import SHORT_HASH_LEN


# --------------------------------------------------------------------------- #
# Run identity
# --------------------------------------------------------------------------- #


def run_id(spec_id: str, task_id: str, counter: int) -> str:
    """A unique, human-readable run_id: ``<spec hash>-<task>-<seq>``.

    Uniqueness comes from the per-pipeline ``counter`` (one increment per
    ``run()`` call), so two repeats of the same (spec, task) — and two candidates
    that share a spec_id — never collide. Determinism within a run comes from the
    spec/task hash. (Semantics preserved from ``host/fake._run_id``.)
    """
    h = hashlib.sha256(f"{spec_id}|{task_id}".encode()).hexdigest()[:SHORT_HASH_LEN]
    return f"{h}-{task_id}-{counter:04d}"


# --------------------------------------------------------------------------- #
# Execution order
# --------------------------------------------------------------------------- #


def topo_order(spec: m.Spec) -> list[str]:
    """Kahn's algorithm over the (lint-clean, acyclic) Spec graph.

    Roots (no in-edges) come first; frontier ties are broken by node insertion
    order so the result is stable across runs — important for deterministic
    traces. (Semantics preserved from ``host/fake._topo_order``.)
    """
    ids = [n.node_id for n in spec.nodes]
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {nid: 0 for nid in ids}
    for e in spec.edges:
        if e.from_ == e.to:
            continue
        adjacency[e.from_].append(e.to)
        indegree[e.to] += 1
    order_index = {nid: i for i, nid in enumerate(ids)}
    ready = sorted((nid for nid in ids if indegree[nid] == 0), key=lambda n: order_index[n])
    out: list[str] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for nxt in adjacency.get(n, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=lambda n: order_index[n])
    return out


# --------------------------------------------------------------------------- #
# Live-config decay — the phase-2 rule, centralized
# --------------------------------------------------------------------------- #


@dataclass
class KnobVote:
    """What each live-knob becomes when the adapter hands it to its agent call.

    ``None`` on a field means "let the agent use its own hardcoded default" —
    the agents' ``or``/``if X is None`` fallbacks honor that. So a base run
    (the seeded Spec's values match the agents' consts) is behavior-identical to
    the MAS running untouched, and a mutation (model_swap / knob / prompt_edit)
    takes effect because the seeded value no longer equals the const.
    """

    model: str | None
    temperature: float | None
    max_tokens: int | None
    retries: int | None
    system_prompt: str | None


def cfg_decay(
    node_id: str,
    system_prompt: str | None,
    model: str | None,
    knobs: m.Knobs | None,
    tools: list[str] | None,        # noqa: ARG001  (kept on the seam for parity; unused here)
    *,
    base_prompts: dict[str, str] | None = None,
) -> KnobVote:
    """Live-Node-config → the per-call override an agent accepts.

    Returns ``model`` / ``temperature`` / ``max_tokens`` / ``retries`` ALWAYS
    (they equal the agents' hardcoded consts on a base run → identical
    behaviour; they take effect when mutated), and ``system_prompt`` ONLY when
    it differs from ``base_prompts[node_id]`` (an Architect ``prompt_edit``).

    Passing a default ``system_prompt`` verbatim would corrupt MASes whose
    prompts are runtime f-strings (Lumina's ``task``/``retrieval`` hold literal
    ``{placeholders}`` the static Spec can't resolve); a *mutated* prompt is a
    complete replacement, so it rides through untouched. Base run unchanged;
    ``prompt_edit`` mutations honored.
    """
    base = base_prompts or {}
    sp_out = system_prompt if system_prompt != base.get(node_id) else None
    return KnobVote(
        model=model,
        temperature=knobs.temperature if knobs is not None else None,
        max_tokens=knobs.max_tokens if knobs is not None else None,
        retries=knobs.retries if knobs is not None else None,
        system_prompt=sp_out,
    )


def cfg_as_kwargs(
    vote: KnobVote,
    *,
    keys: tuple[str, ...] = ("model", "temperature", "max_tokens", "retries", "system_prompt"),
) -> dict[str, Any]:
    """Flatten a ``KnobVote`` into the ``**kwargs`` shape an agent accepts.

    The adapter declares ``keys`` so it binds EXACTLY the overrides its agent's
    signature carries (the Lumina agents take model/temperature/max_tokens/
    system_prompt but NOT retries); the kit owns the decay logic, not the
    agent's signature. Default = the full set.
    """
    out: dict[str, Any] = {}
    if "model" in keys:           out["model"] = vote.model
    if "temperature" in keys:      out["temperature"] = vote.temperature
    if "max_tokens" in keys:      out["max_tokens"] = vote.max_tokens
    if "retries" in keys:          out["retries"] = vote.retries
    if "system_prompt" in keys:   out["system_prompt"] = vote.system_prompt
    return out


def estimate_tokens(text: str | None) -> int:
    """A rough, dependency-free token estimate for cost tracking.

    The same ``len//4`` heuristic ``host/fake.FakeAgent`` uses inline; good enough
    for budget caps and dedup. Real provider usage accounting is a later
    fidelity item (out of scope for the kit), not a blocker here.
    """
    return max(1, len(text or "") // 4)


__all__ = ["run_id", "topo_order", "cfg_decay", "cfg_as_kwargs", "KnobVote", "estimate_tokens"]
