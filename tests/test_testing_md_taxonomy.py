"""TESTING.md taxonomy completeness gate.

Every ``tests/test_*.py`` file must be named (literal filename) somewhere in
``TESTING.md`` so the taxonomy a fresh agent navigates by can never silently
drift from the suite again — the 2026-06 audit found 89 of 144 files had
become invisible to it. Mirrors the FILE_MANIFEST coverage gate
(``scripts/check_manifest_coverage.py``) and the worklog-index ``--check``:
adding a test file costs one taxonomy row.

If this test fails: add a one-line row for your new test file to the matching
taxonomy table in ``TESTING.md`` (or the most specific section that fits).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTING_MD = REPO_ROOT / "TESTING.md"
TESTS_DIR = REPO_ROOT / "tests"


def test_every_test_file_is_named_in_testing_md():
    taxonomy = TESTING_MD.read_text(encoding="utf-8")
    missing = sorted(p.name for p in TESTS_DIR.glob("test_*.py") if p.name not in taxonomy)
    assert not missing, (
        f"{len(missing)} test file(s) are missing from the TESTING.md "
        f"taxonomy — add a one-line row for each: {missing}"
    )


def test_taxonomy_names_no_phantom_test_files():
    """Literal ``tests/test_*.py`` paths in TESTING.md must exist on disk.

    Wildcard mentions (``test_audit_viii_*``) and bare function names are out
    of scope — only explicit ``tests/<name>.py`` paths are checked, so prose
    examples stay unconstrained.
    """
    import re

    taxonomy = TESTING_MD.read_text(encoding="utf-8")
    referenced = set(re.findall(r"tests/(test_[a-z0-9_]+\.py)", taxonomy))
    phantoms = sorted(name for name in referenced if not (TESTS_DIR / name).exists())
    assert not phantoms, (
        f"TESTING.md names {len(phantoms)} test file(s) that do not exist: {phantoms}"
    )
