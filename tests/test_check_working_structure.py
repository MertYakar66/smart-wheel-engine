"""scripts/check_working_structure.py — the rule-books still match the repository (D32).

Pins: the committed repository passes; one changed word in the AGENTS.md Appendix
fails and names the fix; a PROJECT_STATE.md without the main hash fails; a
malformed deadline row fails while a second table is ignored; a manifest without
a frontier fails; a mark missing from CLAUDE.md fails; a cited path that does not
exist fails, while data paths, placeholders and fenced examples are skipped.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "check_working_structure", _REPO / "scripts" / "check_working_structure.py"
)
cws = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(cws)


def _copy(tmp_path: Path, *rels: str) -> Path:
    for rel in rels:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REPO / rel, dest)
    return tmp_path


def test_the_committed_repository_passes(capsys):
    assert cws.main([str(_REPO)]) == 0
    assert "6 of 6 checks pass" in capsys.readouterr().out


def test_one_word_changed_in_the_appendix_fails_and_names_the_fix(tmp_path):
    root = _copy(tmp_path, "CLAUDE.md", "AGENTS.md")
    assert cws.check_appendix(root) == []
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    head, appendix = agents.split("\n## Appendix", 1)
    (root / "AGENTS.md").write_text(
        head + "\n## Appendix" + appendix.replace("Four roles.", "Five roles.", 1), encoding="utf-8"
    )
    failures = cws.check_appendix(root)
    assert failures and "copy CLAUDE.md §1–§6 into the AGENTS.md Appendix" in failures[0]


def test_project_state_without_the_main_hash_fails(tmp_path):
    (tmp_path / "PROJECT_STATE.md").write_text("# Project State\n\nno branches line\n")
    failures = cws.check_main_hash(tmp_path)
    assert failures and "Branches line" in failures[0]
    (tmp_path / "PROJECT_STATE.md").write_text(
        "**Branches:** `main` is at `abc1234` (2026-09-23).\n"
    )
    assert cws.check_main_hash(tmp_path) == []  # not a git repository: format only


def test_deadlines_rows_are_checked_and_other_tables_ignored(tmp_path):
    (tmp_path / "docs").mkdir()
    good = (
        "| Due | What | Owner | Status | Source |\n| --- | --- | --- | --- | --- |\n"
        "| 2026-10-26 | x | Operator | OPEN | a doc |\n| Undated | y | Operator | OPEN | a doc |\n"
        "\n| Dataset | Last date | Where | Source |\n| --- | --- | --- | --- |\n| p | 2026-07-02 | root | m |\n"
    )
    (tmp_path / "docs" / "deadlines.md").write_text(good)
    assert cws.check_deadlines(tmp_path) == []
    (tmp_path / "docs" / "deadlines.md").write_text(good.replace("| 2026-10-26 |", "| soon |"))
    assert any("malformed row" in f for f in cws.check_deadlines(tmp_path))
    # the right shape but not a calendar date would crash session-open
    (tmp_path / "docs" / "deadlines.md").write_text(
        good.replace("| 2026-10-26 |", "| 2026-02-30 |")
    )
    assert any("malformed row" in f for f in cws.check_deadlines(tmp_path))
    (tmp_path / "docs" / "deadlines.md").unlink()
    assert "missing" in cws.check_deadlines(tmp_path)[0]


def test_a_manifest_without_a_frontier_fails(tmp_path):
    root = _copy(tmp_path, "scripts/data_manifest.py")
    (root / "data").mkdir()
    (root / "data" / "DATA_MANIFEST.json").write_text('{"schema": 2, "files": []}')
    failures = cws.check_frontier(root)
    assert failures and "rebuild the manifest" in failures[0]


def test_a_mark_missing_from_claude_md_fails(tmp_path):
    root = _copy(tmp_path, "CLAUDE.md", "scripts/session_open.py")
    assert cws.check_marks(root) == []
    text = (root / "CLAUDE.md").read_text(encoding="utf-8")
    (root / "CLAUDE.md").write_text(
        text.replace("Second opinion from Codex, Sir —", "Codex says —")
    )
    assert any("Second opinion from Codex" in f for f in cws.check_marks(root))


def test_cited_paths_must_exist_but_data_placeholders_and_examples_are_skipped(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "real.py").write_text("")
    (tmp_path / "BOOK.md").write_text(
        "Run `scripts/real.py` and `scripts/nope.py`; the data is `data/bloomberg/sp500_ohlcv.csv`;\n"
        "branches are `claude/<slug>`; a test is `scripts/real.py::test_x`.\n"
        "```\nwrite docs/worklog/example.md\n`scripts/fenced.py`\n```\n"
    )
    failures = cws.check_cited_paths(tmp_path, ("BOOK.md",))
    assert failures == ["BOOK.md: cites `scripts/nope.py`, which does not exist"]
