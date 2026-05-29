#!/usr/bin/env python3
"""Sync helper: keep `FILE_MANIFEST.md` in step with the tracked tree.

Same scan as `scripts/check_manifest_coverage.py` (the CI guard), with an
optional ``--fix`` that appends rows for the missing files into a marked
"Untriaged additions" section at the tail of the manifest. The placeholder
purpose text is obviously human-follow-up so the next reviewer moves the
rows under the right ``## <directory>`` section with a real description.

Orphans (manifest tokens that match zero tracked files) are NEVER
auto-deleted by ``--fix`` — they often signal a planned-but-not-shipped
file or a recently moved/renamed entry, both of which need a human call.
They are reported, and the exit code reflects them so CI or the operator
still sees the work that remains.

Run:
  python scripts/sync_manifest.py          # report only (same exit code as check_manifest_coverage)
  python scripts/sync_manifest.py --fix    # append missing rows; flag orphans
"""

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from check_manifest_coverage import (  # noqa: E402  — sys.path injection above
    MANIFEST,
    _glob_matches,
    _is_glob,
    _manifest_tokens,
    _repo_root,
    _tracked_files,
)

_MARKER_HEADER = "## Untriaged additions (auto-appended by `scripts/sync_manifest.py`)"
_MARKER_BLURB = (
    "Rows below were added automatically because the file was tracked but "
    "absent from the manifest. Move each entry under the correct "
    "`## <directory>` section with a real purpose description, then delete "
    "it from here. Re-running `--fix` rebuilds this section from scratch."
)
_PURPOSE_PLACEHOLDER = "_TODO: describe (auto-added by `scripts/sync_manifest.py --fix`)._"


def _classify(manifest_text: str, tracked: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (uncovered, invented_exact, dead_globs) using the CI guard's rules.

    Takes the manifest text explicitly so callers can pass either the live file
    or a copy with the auto-marker section stripped (for idempotent ``--fix``).
    """
    tracked_set = set(tracked)
    tokens = _manifest_tokens(manifest_text)

    exact = [t for t in tokens if not _is_glob(t)]
    globs = [t for t in tokens if _is_glob(t)]

    invented = sorted(t for t in exact if t not in tracked_set)
    dead_globs = sorted(g for g in globs if not _glob_matches(g, tracked))

    covered: set[str] = {t for t in exact if t in tracked_set}
    for glob in globs:
        covered.update(_glob_matches(glob, tracked))
    uncovered = sorted(f for f in tracked if f not in covered)

    return uncovered, invented, dead_globs


def _strip_marker_section(text: str) -> str:
    """Remove the existing marker section if present so ``--fix`` is idempotent."""
    lines = text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == _MARKER_HEADER)
    except StopIteration:
        return text
    end = start + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    return "".join(lines[:start] + lines[end:]).rstrip() + "\n"


def _build_marker_section(missing: list[str]) -> str:
    rows = "\n".join(f"| `{path}` | {_PURPOSE_PLACEHOLDER} |" for path in sorted(set(missing)))
    return f"\n{_MARKER_HEADER}\n\n{_MARKER_BLURB}\n\n| File | Purpose |\n|---|---|\n{rows}\n"


def _print_offenders(uncovered: list[str], invented: list[str], dead_globs: list[str]) -> None:
    for path in uncovered:
        print(f"  tracked but absent from the manifest  : {path}")
    for token in invented:
        print(f"  manifest names a non-existent path    : {token}")
    for glob in dead_globs:
        print(f"  manifest glob matches no tracked file : {glob}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Append rows for missing files under a marker section at the tail of FILE_MANIFEST.md.",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    tracked = _tracked_files(root)
    manifest_path = root / MANIFEST
    original = manifest_path.read_text(encoding="utf-8")

    if not args.fix:
        uncovered, invented, dead_globs = _classify(original, tracked)
        if not (uncovered or invented or dead_globs):
            print("OK: FILE_MANIFEST.md accounts for every tracked file.")
            return 0
        print("FILE_MANIFEST.md is out of sync with the tree:")
        _print_offenders(uncovered, invented, dead_globs)
        print()
        print(
            "Fix: re-run with --fix to append rows for the missing files, or edit the manifest manually."
        )
        return 1

    # --fix: strip the auto-marker first so the same input always yields the
    # same output. Computing uncovered against the *stripped* manifest is what
    # makes repeated --fix calls converge: if the user has not yet moved a TODO
    # row out of the marker section, it stays; if they have, it drops out.
    stripped = _strip_marker_section(original)
    uncovered, invented, dead_globs = _classify(stripped, tracked)
    new_text = (
        stripped.rstrip() + "\n" + _build_marker_section(uncovered) if uncovered else stripped
    )

    if new_text != original:
        manifest_path.write_text(new_text, encoding="utf-8")
        if uncovered:
            print(f"--fix: appended {len(uncovered)} row(s) under {_MARKER_HEADER}.")
            print(
                "Move each row under its proper `## <directory>` section and replace the placeholder description."
            )
        else:
            print("--fix: removed empty marker section (no missing files).")
    else:
        print("--fix: no changes needed; manifest already covers every tracked file.")

    if invented or dead_globs:
        print()
        print("Orphans (NOT auto-removed — needs human review):")
        for token in invented:
            print(f"  manifest names a non-existent path    : {token}")
        for glob in dead_globs:
            print(f"  manifest glob matches no tracked file : {glob}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
