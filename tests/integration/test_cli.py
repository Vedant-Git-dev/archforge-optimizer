"""Phase 9 CLI integration — the vertical-slice click.

Drives the REAL `archforge.cli.main(...)` against `FakeHostMAS` + `ScriptedJudge`
+ `ScriptedArchitect` on a tmp `.archforge` root, proving the wired subcommands
move the active pointer (P-E-C → promote) and stop the loop cleanly (plateau,
budget ceiling). No real LLM is touched — the organs are injected via the
`Components` seam.

Spec ids are content-addressed (path-independent), so this mirrors exactly what
the Engine commits: the candidate's id is computed with `parent_spec_id` set to
the seed's id, the same way `SpecStore.commit` stamps it. That lets us script
`ScriptedJudge.set_aggregate(<spec_id>, "t1", score)` ahead of time.
"""

from __future__ import annotations

from pathlib import Path

import archforge.models as m
from archforge.architect import ScriptedArchitect
from archforge.cli import Components, main
from archforge.host.base import Task
from archforge.host.fake import FakeHostMAS
from archforge.judge import ScriptedJudge
from archforge.mutate import apply_change
from archforge.suite import Suite

SUITE_ID = "S"
RUBRIC = "default-v1"


# --------------------------------------------------------------------------- #
# fixtures built deterministically (content-addressed, path-independent)
# --------------------------------------------------------------------------- #


def _seed() -> m.Spec:
    """A lint-clean root incumbent: one node, no edges (single-node → no orphan)."""
    return m.Spec(
        nodes=[m.Node(node_id="a", role="r", system_prompt="p0", model="gpt", tools=["t0"])]
    )


def _small_change() -> m.Change:
    return m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a",
                             "rewrite system_prompt of 'a'", "tighten")


def _struct_change() -> m.Change:
    return m.Change.for_kind(m.ChangeKind.ADD_NODE, "v", "add node 'v'", "verification")


def _seed_id() -> str:
    return _seed().compute_spec_id()           # parent_spec_id=None


def _cand_id_small() -> str:
    """The candidate the Engine will commit for the small prompt_edit."""
    cand = apply_change(_seed(), _small_change(), {"prompt": "p1"})
    cand.parent_spec_id = _seed_id()           # what SpecStore.commit stamps
    return cand.compute_spec_id()


def _cand_id_struct() -> str:
    """The candidate the Engine will commit for the add_node change."""
    vnode = m.Node(node_id="v", role="verifier", system_prompt="pv",
                   model="gpt", tools=["t0"])
    payload = {"node": vnode, "wiring": {"in_edges": [["a", "sequence"]]}}
    cand = apply_change(_seed(), _struct_change(), payload)
    cand.parent_spec_id = _seed_id()
    return cand.compute_spec_id()


def _judge(*pairs: tuple[str, str, float]) -> ScriptedJudge:
    j = ScriptedJudge()
    for spec_id, task_id, score in pairs:
        j.set_aggregate(spec_id, task_id, score)
    return j


def _organs(judge: ScriptedJudge, arch: ScriptedArchitect) -> Components:
    suite = Suite(suite_id=SUITE_ID, rubric_id=RUBRIC,
                  tasks=[Task(task_id="t1", input="q")])
    return Components(host=FakeHostMAS(), judge=judge, architect=arch, suite=suite)


def _write_seed(tmp_path: Path) -> Path:
    p = tmp_path / "seed.json"
    p.write_text(_seed().model_dump_json(), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# the click
# --------------------------------------------------------------------------- #


def test_evolve_promotes_and_status_reflects_click(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    # small win clears τ=0.05 (0.66 - 0.55 = 0.11) → AUTO_PROMOTE, active moves.
    seed = _write_seed(tmp_path)
    root = tmp_path / "af"
    comp = _organs(
        _judge((_seed_id(), "t1", 0.55), (_cand_id_small(), "t1", 0.66)),
        ScriptedArchitect().propose(_small_change(), {"prompt": "p1"}),
    )

    rc = main(["evolve", "--root", str(root), "--seed", str(seed)], components=comp)
    assert rc == 0
    out = capsys.readouterr().out
    assert "attempted=true" in out
    assert "action=auto_promote" in out
    assert "promoted=true" in out

    # the click lands in the store: status now shows the candidate as incumbent.
    rc = main(["status", "--root", str(root)])
    assert rc == 0
    stat = capsys.readouterr().out
    assert _cand_id_small() in stat
    assert _seed_id() in stat                      # lineage walks back to root
    assert "lineage:" in stat


def test_evolve_loop_plateaus_after_one_promotion(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    seed = _write_seed(tmp_path)
    root = tmp_path / "af"
    comp = _organs(
        _judge((_seed_id(), "t1", 0.55), (_cand_id_small(), "t1", 0.66)),
        ScriptedArchitect().propose(_small_change(), {"prompt": "p1"}),
    )

    rc = main(["evolve-loop", "--root", str(root), "--seed", str(seed),
               "--max-cycles", "50", "--plateau-cycles", "3"], components=comp)
    assert rc == 0
    out = capsys.readouterr().out
    # cycle 0 promotes (streak reset); cycles 1-3 plateau (empty architect queue)
    # → K=3 consecutive no-promotion trips the plateau break (E8).
    assert "promotions=1" in out
    assert "plateaued=true" in out
    assert "cycles_run=4" in out
    assert _cand_id_small() in out          # final_incumbent is the promoted candidate


def test_evolve_loop_budget_cap_aborts_before_first_cycle(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    # E3 ceiling: a total budget of 0 means "spend nothing" → stop before cycle 1.
    seed = _write_seed(tmp_path)
    root = tmp_path / "af"
    comp = _organs(
        _judge((_seed_id(), "t1", 0.55), (_cand_id_small(), "t1", 0.66)),
        ScriptedArchitect().propose(_small_change(), {"prompt": "p1"}),
    )

    rc = main(["evolve-loop", "--root", str(root), "--seed", str(seed),
               "--max-tokens-total", "0"], components=comp)
    assert rc == 0
    out = capsys.readouterr().out
    assert "aborted=true" in out
    assert "cycles_run=0" in out
    assert "total token budget reached" in out
    # incumbent untouched by the abort — pointer still on the seed.
    rc = main(["status", "--root", str(root)])
    assert rc == 0
    assert _seed_id() in capsys.readouterr().out


def test_approve_drains_structural_queue_and_promotes(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    # clear structural win (0.80 - 0.55 = 0.25 >= τ) → QUEUE_HUMAN (I4, never
    # auto-promote); then `approve --all` flips it to PROMOTED via the Gatekeeper.
    seed = _write_seed(tmp_path)
    root = tmp_path / "af"
    vnode = m.Node(node_id="v", role="verifier", system_prompt="pv",
                   model="gpt", tools=["t0"])
    payload = {"node": vnode, "wiring": {"in_edges": [["a", "sequence"]]}}
    comp = _organs(
        _judge((_seed_id(), "t1", 0.55), (_cand_id_struct(), "t1", 0.80)),
        ScriptedArchitect().propose(_struct_change(), payload),
    )

    rc = main(["evolve", "--root", str(root), "--seed", str(seed)], components=comp)
    assert rc == 0
    out = capsys.readouterr().out
    assert "queued=true" in out
    assert "action=queue_human" in out

    # active is still the seed — the structural change only gates (I1/I4).
    rc = main(["status", "--root", str(root)])
    assert rc == 0
    assert _seed_id() in capsys.readouterr().out

    # the human approves → active moves to the candidate (the only other path
    # besides AUTO_PROMOTE/ROLLBACK that may touch the active pointer).
    rc = main(["approve", "--all", "--root", str(root)])
    assert rc == 0
    ap = capsys.readouterr().out
    assert _cand_id_struct() in ap
    assert "approved" in ap

    rc = main(["status", "--root", str(root)])
    assert rc == 0
    stat = capsys.readouterr().out
    assert _cand_id_struct() in stat
    assert _seed_id() in stat                       # lineage still reaches the root


def test_evolve_without_seed_or_incumbent_errors(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "af"
    comp = _organs(_judge(), ScriptedArchitect())
    rc = main(["evolve", "--root", str(root)], components=comp)
    assert rc == 1
    err = capsys.readouterr().err
    assert "no active incumbent" in err.lower() or "--seed" in err


def test_report_lists_attempts_with_deltas(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    seed = _write_seed(tmp_path)
    root = tmp_path / "af"
    comp = _organs(
        _judge((_seed_id(), "t1", 0.55), (_cand_id_small(), "t1", 0.66)),
        ScriptedArchitect().propose(_small_change(), {"prompt": "p1"}),
    )
    main(["evolve", "--root", str(root), "--seed", str(seed)], components=comp)
    capsys.readouterr()                              # drain evolve output

    rc = main(["report", "--root", str(root)])
    assert rc == 0
    rep = capsys.readouterr().out
    assert "attempt_id" in rep                       # header
    assert "promoted" in rep                          # the verdict column
    assert "+0.110" in rep                            # the real margin surfaced
