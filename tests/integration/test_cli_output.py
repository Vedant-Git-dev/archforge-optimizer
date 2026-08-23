"""CLI integration for the expressive surfaces — improvements #2/#3/#4/#5.

Drives the REAL ``archforge.cli.main(["evolve-loop", …])`` against
``FakeHostMAS`` + ``ScriptedJudge`` + ``ScriptedArchitect`` on a tmp
``.archforge`` root, asserting the additive output works WITHOUT breaking the
existing integration-test substrings (kept verbatim on their lines). Mirrors the
content-addressed pattern of ``tests/integration/test_cli.py`` exactly, so the
suite the Engine commits is what we script the Judge against ahead of time.

Asserts:
  (a) legacy substrings still present (the ``test_cli`` contract survives);
  (b) new card lines ``change:``/``scores:``/``cost:``/``why:`` present
      (#2/#3/#5) — the per-cycle visibility the loop previously lacked;
  (c) ``<root>/runs/last.json`` exists with a ``cycles`` array matching
      cycles_run (#5 file sink);
  (d) ``<root>/optimized.json`` exists after the promote and its ``knobs``
      matches ``load_spec_sidecar`` shape — the consumer-compat seam (#4);
  (e) ``final_mean=`` is a number (not ``-``) when there's an active baseline
      (the trivial-bug fix: it was never populated).
"""
from __future__ import annotations

import json
from pathlib import Path

import archforge.models as m
from archforge.architect import ScriptedArchitect
from archforge.cli import Components, main
from archforge.host.adapters import load_optimized, load_spec_sidecar
from archforge.host.base import Task
from archforge.host.fake import FakeHostMAS
from archforge.judge import ScriptedJudge
from archforge.mutate import apply_change
from archforge.suite import Suite

SUITE_ID = "S"
RUBRIC = "default-v1"


# --------------------------------------------------------------------------- #
# content-addressed fixtures (path-independent, matching test_cli.py)
# --------------------------------------------------------------------------- #


def _seed() -> m.Spec:
    """A lint-clean root incumbent: one node, no edges (single-node → no orphan)."""
    return m.Spec(
        nodes=[m.Node(node_id="a", role="r", system_prompt="p0", model="gpt", tools=["t0"])]
    )


def _small_change() -> m.Change:
    return m.Change.for_kind(m.ChangeKind.PROMPT_EDIT, "a",
                             "rewrite system_prompt of 'a'", "tighten")


def _seed_id() -> str:
    return _seed().compute_spec_id()                   # parent_spec_id=None


def _cand_id_small() -> str:
    """The candidate the Engine will commit for the small prompt_edit."""
    cand = apply_change(_seed(), _small_change(), {"prompt": "p1"})
    cand.parent_spec_id = _seed_id()                   # what SpecStore.commit stamps
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
# (a) + (b) legacy substrings survive AND the new card lines appear
# --------------------------------------------------------------------------- #


def test_evolve_loop_emits_legacy_lines_and_card(tmp_path: Path, capsys) -> None:
    """``evolve-loop`` with a wired promote prints BOTH the legacy cycle line
    (kept verbatim — the ``test_cli`` substring contract survives) AND the new
    per-cycle card (``change:``/``scores:``/``cost:``/``why:`` — #2/#3/#5)."""
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

    # (a) the legacy contract survives (kept verbatim on their lines) —
    # cycle 0 is the promote; cycles 1-3 plateau (architect runs dry).
    assert "attempted=true" in out and "action=auto_promote" in out
    assert "promoted=true" in out and "promotions=1" in out
    assert "plateaued=true" in out and "cycles_run=4" in out
    assert _cand_id_small() in out            # final_incumbent is the promoted candidate

    # (b) the new per-cycle card lines appear (the loop now shows each cycle,
    # not just the summary). The change line carries the prompt diff (p0→p1).
    assert "  change: prompt_edit" in out
    assert "system_prompt: p0 -> p1" in out
    assert "  scores: cand 0.660" in out and "vs inc 0.550" in out
    assert "  cost:" in out and "tokens" in out and "latency" in out
    assert "  why:" in out and "rule=" in out and "action=auto_promote" in out


# --------------------------------------------------------------------------- #
# (c) the run-log file sink — runs/last.json with a cycles array
# --------------------------------------------------------------------------- #


def test_runlog_file_records_cycles_matching_cycles_run(tmp_path: Path, capsys) -> None:
    """``<root>/runs/last.json`` exists after the run, carries the
    ``archforge.runlog/v1`` schema, and its ``cycles`` array length matches the
    printed ``cycles_run`` (#5 file sink — the narrative survives the terminal)."""
    seed = _write_seed(tmp_path)
    root = tmp_path / "af"
    comp = _organs(
        _judge((_seed_id(), "t1", 0.55), (_cand_id_small(), "t1", 0.66)),
        ScriptedArchitect().propose(_small_change(), {"prompt": "p1"}),
    )

    main(["evolve-loop", "--root", str(root), "--seed", str(seed),
          "--max-cycles", "50", "--plateau-cycles", "3"], components=comp)
    capsys.readouterr()                         # drain stdout

    log_path = root / "runs" / "last.json"
    assert log_path.exists(), "run log not written to <root>/runs/last.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["schema"] == "archforge.runlog/v1"
    # cycles_run=4 (cycle 0 promote + cycles 1-3 plateau), but ``on_cycle`` fires
    # ONLY on ATTEMPTED cycles (engine.py returns early on the not-proposed path,
    # before the fire site) → exactly ONE logged cycle (the promote). The runlog
    # records the cycles that did work, not the no-op plateau ticks.
    cycles = log["cycles"]
    assert isinstance(cycles, list) and len(cycles) == 1
    c0 = cycles[0]
    assert c0["cycle"] == 0 and c0["attempted"] is True
    assert c0["action"] == "auto_promote"
    assert c0["change_kind"] == "prompt_edit" and c0["target"] == "a"
    assert c0["candidate_mean"] == 0.66 and c0["incumbent_mean"] == 0.55
    # the summary block was written at the end and reflects the FULL run (4 cycles).
    assert isinstance(log.get("summary"), dict)
    assert log["summary"]["cycles_run"] == 4 and log["summary"]["promotions"] == 1


# --------------------------------------------------------------------------- #
# (d) optimized.json exists after a promote and its knobs match the sidecar shape
# --------------------------------------------------------------------------- #


def test_optimized_json_written_on_promote_and_knobs_match_sidecar(
        tmp_path: Path, capsys) -> None:
    """On the AUTO_PROMOTE the ``on_deploy`` hook writes
    ``<root>/optimized.json`` — the unified envelope (knobs + lineage + decision
    + scores). Its ``knobs`` block matches ``export_spec_sidecar``'s body (the
    consumer-compat seam: a sidecar consumer switches to ``env["knobs"]``)."""
    seed = _write_seed(tmp_path)
    root = tmp_path / "af"
    comp = _organs(
        _judge((_seed_id(), "t1", 0.55), (_cand_id_small(), "t1", 0.66)),
        ScriptedArchitect().propose(_small_change(), {"prompt": "p1"}),
    )

    rc = main(["evolve-loop", "--root", str(root), "--seed", str(seed),
               "--max-cycles", "50", "--plateau-cycles", "3"], components=comp)
    assert rc == 0
    capsys.readouterr()

    opt_path = root / "optimized.json"
    assert opt_path.exists(), "optimized.json not written on the promote"
    # the CLI prints a deploy notice on the promote.
    # (drained above; existence + shape is the artifact assertion.)

    env = load_optimized(opt_path)
    assert env["schema"] == "archforge.optimized/v1"
    assert env["spec_id"] == _cand_id_small()                # the promoted candidate
    assert env["parent_spec_id"] == _seed_id()               # lineage to the root
    assert env["promoted_at_cycle"] == 0
    assert env["decision"]["action"] == "auto_promote"
    assert env["scores"]["mean"] == 0.66
    assert env["scores"]["incumbent_mean"] == 0.55

    # (d) the compat seam: envelope["knobs"] == the bare sidecar body for the same
    # Spec. A sidecar consumer could switch ``load_spec_sidecar`` → ``load_optimized
    # (...)["knobs"]`` with no projection change.
    from archforge.host.adapters import export_spec_sidecar
    cand = apply_change(_seed(), _small_change(), {"prompt": "p1"})
    sidecar_body = export_spec_sidecar(cand, "/dev/null")    # path unused → the body

    assert env["knobs"] == sidecar_body
    # the winner's prompt (p1) and model (gpt) shipped into the envelope's knobs.
    assert env["knobs"]["a"]["system_prompt"] == "p1"
    assert env["knobs"]["a"]["model"] == "gpt"


def test_optimized_json_not_written_without_a_promote(tmp_path: Path, capsys) -> None:
    """No optimized.json is written when nothing promotes (the deploy hook is
    promote-gated) — a losing run ships no deploy artifact."""
    seed = _write_seed(tmp_path)
    root = tmp_path / "af"
    # candidate scores BELOW incumbent → discard (margin < τ), no promote.
    comp = _organs(
        _judge((_seed_id(), "t1", 0.66), (_cand_id_small(), "t1", 0.55)),
        ScriptedArchitect().propose(_small_change(), {"prompt": "p1"}),
    )

    rc = main(["evolve-loop", "--root", str(root), "--seed", str(seed),
               "--max-cycles", "50", "--plateau-cycles", "3"], components=comp)
    assert rc == 0
    capsys.readouterr()

    assert not (root / "optimized.json").exists(), \
        "optimized.json written without a promote — the deploy hook isn't promote-gated"


# --------------------------------------------------------------------------- #
# (e) final_mean= is a number (not "-") when there's an active baseline
# --------------------------------------------------------------------------- #


def test_final_mean_populated_after_a_promote(tmp_path: Path, capsys) -> None:
    """``final_mean=`` carries a number after a promote (the trivial-bug fix —
    it was never populated so the summary always read ``-``). The promote's
    candidate mean (0.66) becomes the final incumbent mean (zero re-scoring)."""
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

    # the legacy loop one-liner carries final_mean=<number>, NOT "-".
    assert "final_mean=0.660" in out
    assert "final_mean=-" not in out
    # and the summary block (appended after) repeats the populated mean.
    assert "final_mean=0.660" in out


def test_final_mean_dash_when_no_promote_no_baseline(tmp_path: Path, capsys) -> None:
    """Honest ``-`` when the loop never promoted AND never scored a baseline —
    a plateau-only loop (architect never proposes) has no scored mean, so ``-``
    is the truth (never force a fresh suite run to fill a summary field)."""
    seed = _write_seed(tmp_path)
    root = tmp_path / "af"
    # never proposes → every cycle attempted=false → no score recorded.
    comp = _organs(
        _judge((_seed_id(), "t1", 0.55), (_cand_id_small(), "t1", 0.66)),
        ScriptedArchitect(),                                    # empty queue → plateau
    )

    main(["evolve-loop", "--root", str(root), "--seed", str(seed),
          "--max-cycles", "5", "--plateau-cycles", "3"], components=comp)
    out = capsys.readouterr().out

    assert "final_mean=-" in out
