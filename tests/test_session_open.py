"""scripts/session_open.py — the first line every session starts with (DECISIONS.md D32).

Pins: the pen's mark carries main, the docs drift from the hash PROJECT_STATE.md
records, the other-branch count, the data age from the manifest frontier, and the
nearest open deadline, an overdue row first; a slot whose source fails reads
``unknown``; the Executor's and Codex's marks; the Executor needs a run mode; a
data root that runs past the recorded frontier is flagged.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "session_open", _REPO / "scripts" / "session_open.py"
)
so = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(so)

TODAY = dt.date(2026, 9, 23)
DEADLINES = """# Deadlines

| Due | What | Owner | Status | Source |
| --- | ---- | ----- | ------ | ------ |
| **2026-10-26** | Domain renewal. Carries the mail. | Operator | **OPEN** | a doc |
| 2026-08-12 (past) | Shut the old host down. | Operator | **OVERDUE, UNVERIFIED** | a doc |
| 2026-07-01 | Something already done. | Operator | CLOSED 2026-07-02 | a doc |
| Conditional: when X starts | Approve within 7 days. | Operator | Not started | a doc |

## Data dates

| Dataset | Last date | Where it lives | Source |
| ------- | --------- | -------------- | ------ |
| Prices | 2026-07-02 | the root | the manifest |
"""


def test_recorded_main_hash():
    assert so.recorded_main_hash("**Branches:** `main` is at `f1c0066` (2026-09-23).") == "f1c0066"
    assert so.recorded_main_hash("no hash here") is None


def test_deadlines_overdue_first_then_nearest_then_none():
    rows = so.parse_deadlines(DEADLINES)
    assert len(rows) == 4  # the data-dates table below is not read as deadlines
    assert so.nearest_deadline(rows, TODAY) == "Shut the old host down, overdue by 42 days"
    future_only = [r for r in rows if not r[0].startswith("2026-08-12")]
    assert so.nearest_deadline(future_only, TODAY) == "Domain renewal, 33 days"
    assert so.nearest_deadline([r for r in rows if "Conditional" in r[0]], TODAY) == "none open"
    assert so.nearest_deadline([], TODAY) == "none open"


def test_data_slot_is_the_oldest_frontier_with_its_age(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "frontier": {
                    "iv": {"path": "iv.csv", "last_date": "2026-07-10"},
                    "prices": {"path": "p.csv", "last_date": "2026-07-02"},
                }
            }
        )
    )
    slot, detail = so.data_slot(manifest, TODAY)
    assert slot == "2026-07-02 (83 days old)"
    assert detail.startswith("data frontier: prices 2026-07-02 (83 days), iv 2026-07-10 (75 days)")
    manifest.write_text(json.dumps({"files": []}))
    assert so.data_slot(manifest, TODAY)[0] == "unknown"
    assert so.data_slot(tmp_path / "absent.json", TODAY)[0] == "unknown"


def _fake_repo(tmp_path: Path, state: str = "`main` is at `aaaaaaa` (2026-09-20)") -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "PROJECT_STATE.md").write_text(f"**Branches:** {state}.\n")
    (tmp_path / "docs" / "deadlines.md").write_text(DEADLINES)
    (tmp_path / "data" / "DATA_MANIFEST.json").write_text(
        json.dumps({"frontier": {"prices": {"path": "p.csv", "last_date": "2026-07-02"}}})
    )
    return tmp_path


def _git_stub(answers):
    def git(*args, repo=None):
        for key, value in answers.items():
            if key in args:
                return value
        return None

    return git


def test_pen_mark_fills_every_slot(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    monkeypatch.delenv("SWE_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        so,
        "git",
        _git_stub(
            {
                "log": "f1c0066 2026-09-23",
                "rev-list": "2",
                "ls-remote": "a\trefs/heads/main\nb\trefs/heads/x\nc\trefs/heads/y",
            }
        ),
    )
    mark, details = so.pen_mark(repo, TODAY)
    assert mark == (
        "Report from the Strategist, Sir — main `f1c0066` (2026-09-23) · docs 2 commits behind"
        " · 2 other branches · data 2026-07-02 (83 days old)"
        " · nearest deadline: Shut the old host down, overdue by 42 days"
    )
    assert details[0].startswith("data frontier: prices 2026-07-02")


def test_pen_mark_says_unknown_when_sources_fail(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path, state="no recorded hash")
    monkeypatch.delenv("SWE_DATA_ROOT", raising=False)
    monkeypatch.setattr(so, "git", _git_stub({}))
    mark, details = so.pen_mark(repo, TODAY)
    assert "main `unknown` (unknown)" in mark
    assert "docs unknown commits behind" in mark
    assert "unknown other branches" in mark
    assert any("records no main hash" in d for d in details)


def test_executor_and_codex_marks(tmp_path, monkeypatch):
    def git(*args, repo=None):
        return {
            ("rev-parse", "--abbrev-ref", "HEAD"): "claude/x",
            ("rev-parse", "--short", "HEAD"): "abc1234",
            ("rev-list", "--left-right", "--count", "origin/main...HEAD"): "3\t1",
            ("log", "-1", "--format=%h %ad", "--date=short", "origin/main"): "f1c0066 2026-09-23",
        }.get(args)

    monkeypatch.setattr(so, "git", git)
    assert so.executor_mark(tmp_path, "change") == (
        "Report from the Executor, Sir — branch `claude/x` · HEAD `abc1234` · pushed nothing yet"
        " · behind/ahead of main 3/1 · run mode change"
    )
    assert so.codex_mark(tmp_path, "PR #1", "the diff") == (
        "Second opinion from Codex, Sir — reviewing PR #1 · main `f1c0066` (2026-09-23)"
        " · verified myself: the diff"
    )


def test_executor_needs_a_run_mode():
    with pytest.raises(SystemExit):
        so.main(["--executor", "--no-fetch"])


def test_a_root_past_the_recorded_frontier_is_flagged(tmp_path, monkeypatch):
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps({"frontier": {"prices": {"path": "data/p.csv", "last_date": "2026-07-02"}}})
    )
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    (root / "data" / "p.csv").write_bytes(b"date,ticker\n2026-07-02,A\n2026-09-18,A\n")
    monkeypatch.setenv("SWE_DATA_ROOT", str(root))
    note = so.live_frontier_note(manifest)
    assert note and "the root runs to 2026-09-18" in note and "rebuild the manifest" in note
    monkeypatch.delenv("SWE_DATA_ROOT")
    assert so.live_frontier_note(manifest) is None
