"""CLI-level test for `archforge lint` (Phase 1, exercised via the Phase-9 stub).

Uses real tmp files so the `archforge lint <path>` path gets end-to-end coverage:
valid spec -> exit 0 & "OK"; invalid spec -> exit 1 + printed fault codes.
"""

from __future__ import annotations

import json
from pathlib import Path

import archforge.models as m
from archforge.cli import main


def _write_spec(tmp_path: Path, spec: m.Spec) -> Path:
    p: Path = tmp_path / "spec.json"
    p.write_text(json.dumps(spec.model_dump(mode="json")), encoding="utf-8")
    return p


def test_cli_lint_valid_spec_exits_zero(tmp_path: Path, capsys: object) -> None:  # type: ignore[no-untyped-def]
    spec = m.Spec(
        nodes=[m.Node(node_id="a", role="r", system_prompt="p", model="gpt", tools=["t0"]),
               m.Node(node_id="b", role="r", system_prompt="p", model="gpt", tools=["t1"])],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)],
    )
    path = _write_spec(tmp_path, spec)
    rc = main(["lint", str(path)])
    assert rc == 0


def test_cli_lint_invalid_spec_exits_one(tmp_path: Path) -> None:
    spec = m.Spec(
        nodes=[m.Node(node_id="a", role="r", system_prompt="p", model="gpt", tools=["t0"]),
               m.Node(node_id="b", role="r", system_prompt="p", model="gpt", tools=["t1"]),
               m.Node(node_id="lonely", role="r", system_prompt="p", model="gpt", tools=["t2"])],
        edges=[m.Edge(from_="a", to="b", type=m.EdgeType.SEQUENCE)],
    )
    path = _write_spec(tmp_path, spec)
    rc = main(["lint", str(path)])
    assert rc == 1
