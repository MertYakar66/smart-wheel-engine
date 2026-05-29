"""Tests for ``scripts/check_manifest_coverage.py``'s conflict-marker guard.

The guard exists because the existing manifest-coverage parser only reads
``|``-prefixed table rows, so a ``FILE_MANIFEST.md`` committed with git
merge-conflict markers passed CI silently (caught on 2026-05-29 by grep,
not by the guard). These tests pin the marker-shape precision so that
visual ``=========`` separators, pytest ``================ 93 passed
================`` lines, indented occurrences, and substrings inside
prose are NOT flagged.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_manifest_coverage import (  # noqa: E402  — sys.path injection above
    _find_conflict_markers,
    _is_conflict_marker_line,
)


class TestIsConflictMarkerLine:
    def test_clean_lines_not_flagged(self) -> None:
        for line in ("", "# Title", "Some prose.", "| col | row |", "---"):
            assert _is_conflict_marker_line(line) is False, line

    def test_opening_marker_with_ref_flagged(self) -> None:
        assert _is_conflict_marker_line("<<<<<<< HEAD") is True
        assert _is_conflict_marker_line("<<<<<<< feature/branch-name") is True

    def test_opening_marker_alone_flagged(self) -> None:
        assert _is_conflict_marker_line("<<<<<<<") is True

    def test_separator_alone_flagged(self) -> None:
        assert _is_conflict_marker_line("=======") is True

    def test_closing_marker_with_ref_flagged(self) -> None:
        assert _is_conflict_marker_line(">>>>>>> branch-name") is True
        assert _is_conflict_marker_line(">>>>>>> 90d7e78 (fix(...))") is True

    def test_closing_marker_alone_flagged(self) -> None:
        assert _is_conflict_marker_line(">>>>>>>") is True

    def test_visual_separator_more_than_seven_equals_not_flagged(self) -> None:
        for line in (
            "========",
            "=========",
            "============================",
            "======================== 93 passed in 18.6s ========================",
        ):
            assert _is_conflict_marker_line(line) is False, line

    def test_long_angle_runs_not_flagged(self) -> None:
        # Prose / decorative content with >7 of the marker char.
        assert _is_conflict_marker_line("<<<<<<<<") is False
        assert _is_conflict_marker_line(">>>>>>>>") is False
        assert _is_conflict_marker_line("<<<<<<<<<< section <<<<<<<<<<") is False

    def test_indented_marker_not_flagged(self) -> None:
        # git writes markers at column 0; indented occurrences are intentional content.
        assert _is_conflict_marker_line("  <<<<<<< HEAD") is False
        assert _is_conflict_marker_line("\t=======") is False
        assert _is_conflict_marker_line("    >>>>>>> branch") is False

    def test_marker_substring_in_prose_not_flagged(self) -> None:
        assert _is_conflict_marker_line("We use <<<<<<< as a placeholder.") is False
        assert _is_conflict_marker_line("Code emits ======= as separator.") is False

    def test_no_space_after_marker_token_not_flagged(self) -> None:
        # Malformed git output would always include a space before the ref;
        # without one, this is more likely something else (e.g. a typo or
        # rare prose). Strict-precision rule: require space or end-of-line.
        assert _is_conflict_marker_line("<<<<<<<HEAD") is False
        assert _is_conflict_marker_line(">>>>>>>branch") is False


class TestFindConflictMarkers:
    def test_clean_content_returns_empty(self) -> None:
        content = "# Title\n\nSome prose.\n\n| File | Purpose |\n|---|---|\n"
        assert _find_conflict_markers("docs/foo.md", content) == []

    def test_full_three_way_conflict_returns_three_offenders(self) -> None:
        content = (
            "| `tradingview/launch-tradingview-cdp.sh` | Launches CDP. |\n"
            "<<<<<<< HEAD\n"
            "| `tradingview/research/*.md` | Glob version. |\n"
            "=======\n"
            "| `tradingview/research/specific.md` | Specific version. |\n"
            ">>>>>>> branch-90d7e78\n"
            "| `utils/__init__.py` | Re-exports. |\n"
        )
        offenders = _find_conflict_markers("FILE_MANIFEST.md", content)
        assert [(path, line_num) for path, line_num, _ in offenders] == [
            ("FILE_MANIFEST.md", 2),
            ("FILE_MANIFEST.md", 4),
            ("FILE_MANIFEST.md", 6),
        ]

    def test_line_numbers_are_one_indexed(self) -> None:
        # First line of file containing a marker → line_num 1, not 0.
        offenders = _find_conflict_markers("a.md", "<<<<<<< HEAD\nbody\n")
        assert offenders == [("a.md", 1, "<<<<<<< HEAD")]

    def test_path_passes_through_verbatim(self) -> None:
        offenders = _find_conflict_markers("nested/dir/file.md", "=======\n")
        assert offenders[0][0] == "nested/dir/file.md"

    def test_visual_separators_not_flagged_in_real_content(self) -> None:
        # The pattern that caused the false positive on archive/2026-05/*.md
        # (visual separators in pytest output captures, archived data reports).
        content = (
            "# Section\n"
            "========================\n"
            "Some content.\n"
            "======================= 93 passed in 18.64s =======================\n"
            "More content.\n"
        )
        assert _find_conflict_markers("archive/some.md", content) == []
