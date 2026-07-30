"""SpecBuilder — a typed, fluent DSL for constructing an ArchForge `Spec`.

The honest universal floor for bootstrapping a MAS into ArchForge: instead of
the framework trying to *inspect* an existing MAS, the author *describes* it in
typed Python. Works even for a MAS you can't introspect; a MAS you can (Lumina)
may emit a SpecBuilder from an extractor as an *adapter convenience* — that
stays optional and out of core.

`build()` validates through the existing `archforge.lint` (the same linter the
Architect runs and the `archforge lint` subcommand uses), raising the model's
`LintError` on the first defect — so a malformed builder never reaches the
engine. The builder owns no new validation; it is sugar over `m.Spec` / `m.Node`
/ `m.Edge` / `m.Knobs`, with exactly the fields those models accept (Node has no
`kind` field — the DSL does not invent one).
"""
from __future__ import annotations

import archforge.models as m
from archforge.lint import LintError, lint

# Re-export the edge-kind enum so an author can write `.edge("a","b", kind=JOIN)`
# without reaching into `archforge.models` separately.
EdgeType = m.EdgeType


class SpecBuildError(ValueError):
    """Raised by `SpecBuilder.build()` on the first structural defect.

    `LintError` is a pydantic model (not an Exception), so the builder wraps it
    here; the original defect is attached as `.defect` for inspection/logging.
    """

    def __init__(self, defect: LintError) -> None:
        self.defect = defect
        where = f" [{defect.location}]" if defect.location else ""
        super().__init__(f"{defect.code}: {defect.message}{where}")

# Friendly aliases for the DSL (the names a spec author thinks in).
JOIN = EdgeType.JOIN
FANOUT = EdgeType.FANOUT
SEQUENCE = EdgeType.SEQUENCE
CONDITIONAL = EdgeType.CONDITIONAL


class SpecBuilder:
    """Fluent builder over `m.Spec`.

        spec = (SpecBuilder()
                .node("task", role="planner", model=MODEL_8B, prompt=T, temperature=0.3, max_tokens=1500)
                .node("retrieval", role="retriever", model=MODEL_8B, prompt=R, tools=["tavily"])
                .edge("task", "retrieval", kind=SEQUENCE)
                .edge("refine", "cross", kind=FANOUT)
                .build())

    A `Spec` is *content-addressed*: it carries only nodes + edges (+ lineage
    metadata). It has no `name`/`rubric_id` of its own — a rubric is a property
    of the `Suite` (associated at the runner level, not on the Spec). So this
    builder owns structure only; names/rubrics are the caller's concern.

    Each call mutates the builder in place (fluent) and returns ``self``; the
    final `.build()` returns a content-addressed `Spec` (id computed lazily by
    `Spec.compute_spec_id`, assigned at commit).
    """

    def __init__(self) -> None:
        self._nodes: list[m.Node] = []
        self._edges: list[m.Edge] = []

    # -- construction ------------------------------------------------------- #

    def node(
        self,
        node_id: str,
        *,
        role: str,
        model: str,
        prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int | None = None,
        tools: list[str] | None = None,
    ) -> "SpecBuilder":
        """Add a node. `prompt` (optional) may be a plain constant or a template
        holding literal ``{placeholders}`` — both are just strings to the Spec;
        the host's `call()` resolves placeholders at runtime. A text-in/text-out
        node need not declare one (the agent has no system-prompt override)."""
        n = m.Node(
            node_id=node_id,
            role=role,
            system_prompt=prompt,
            model=model,
            knobs=m.Knobs(temperature=temperature, max_tokens=max_tokens, retries=retries),
            tools=list(tools) if tools else [],
        )
        self._nodes.append(n)
        return self

    def edge(
        self,
        from_: str,
        to: str,
        *,
        kind: EdgeType = SEQUENCE,
        gate: str | None = None,
    ) -> "SpecBuilder":
        """Add a directed edge. `from_` is the Python attribute (``from`` is a
        keyword); the persisted JSON alias is ``from``. `gate` only used for
        CONDITIONAL edges."""
        self._edges.append(m.Edge(from_=from_, to=to, type=kind, gate=gate))
        return self

    # -- finalize ----------------------------------------------------------- #

    def build(self) -> m.Spec:
        """Return a lint-clean `Spec`, raising `SpecBuildError` on the first defect."""
        spec = m.Spec(nodes=list(self._nodes), edges=list(self._edges))
        errors = lint(spec)
        if errors:
            raise SpecBuildError(errors[0])
        return spec


__all__ = [
    "SpecBuilder", "SpecBuildError", "EdgeType",
    "JOIN", "FANOUT", "SEQUENCE", "CONDITIONAL",
]
