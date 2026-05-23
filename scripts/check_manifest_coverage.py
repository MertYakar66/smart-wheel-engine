#!/usr/bin/env python3
"""Guard: FILE_MANIFEST.md must account for every tracked file.

FILE_MANIFEST.md is the repo's exhaustive per-file index (see DECISIONS.md
D14); AGENTS.md, README.md and MODULE_INDEX.md point agents at it as the
canonical map. If a file is added, moved or removed without updating the
manifest, that index silently rots. This script is the automatic check that
prevents it.

It cross-checks the manifest against `git ls-files`:

  * (a) every tracked file must be covered by some manifest entry — an exact
        path, or a glob / directory token (e.g. `data/bloomberg/*.csv`,
        `data_raw/ohlcv/*.csv`, `data/features/<group>/ticker=AAPL/...`);
  * (b) every manifest path/glob must match at least one tracked file.

Exit code 0 when the manifest accounts for every tracked file; 1 (with the
offenders printed) otherwise. Stdlib + git only — no third-party deps.

The matching semantics are lifted verbatim from the one-off cross-check in
commit 4de0cca (the PR #133 review): backtick-quoted tokens in the left
column of each Markdown table row, brace expansion of `{a,b,c}` groups,
`<group>` placeholders treated as `*`, and a glob match of `fnmatch` OR a
directory-prefix `startswith`. They already correctly handle the
directory-level data entries (`data_raw/`, feature shards, …) that the
manifest intentionally does not enumerate per file.

Run:  python scripts/check_manifest_coverage.py
"""

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

MANIFEST = "FILE_MANIFEST.md"


def _git(args: list[str], cwd: Path | None = None) -> str:
    """Run a git command and return its stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _repo_root() -> Path:
    """Locate the repository root so the check works from any CWD."""
    return Path(_git(["rev-parse", "--show-toplevel"]).strip())


def _tracked_files(root: Path) -> list[str]:
    """Every file git tracks, as repo-relative paths."""
    return [line for line in _git(["ls-files"], cwd=root).splitlines() if line]


def _expand_braces(token: str) -> list[str]:
    """Expand one `{a,b,c}` group: `d/{x,y}.json` -> [`d/x.json`, `d/y.json`]."""
    match = re.search(r"\{([^}]+)\}", token)
    if not match:
        return [token]
    return [
        token[: match.start()] + option + token[match.end() :]
        for option in match.group(1).split(",")
    ]


def _manifest_tokens(manifest_text: str) -> list[str]:
    """Backtick-quoted tokens from the left column of every Markdown table row."""
    tokens: list[str] = []
    for line in manifest_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = stripped.split("|")
        if len(cells) < 3:
            continue
        tokens += re.findall(r"`([^`]+)`", cells[1])
    expanded: list[str] = []
    for token in tokens:
        expanded += _expand_braces(token)
    return expanded


def _is_glob(token: str) -> bool:
    """A token is a glob if it carries `*` / `<group>` or names a directory."""
    return any(ch in token for ch in "*<") or token.endswith("/")


def _glob_matches(pattern: str, tracked: list[str]) -> list[str]:
    """Tracked files matched by a glob token (fnmatch or directory prefix)."""
    resolved = pattern.replace("<group>", "*")
    return [f for f in tracked if fnmatch.fnmatch(f, resolved) or f.startswith(resolved)]


def main() -> int:
    root = _repo_root()
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        print(f"ERROR: {MANIFEST} not found at {manifest_path}", file=sys.stderr)
        return 1

    tracked = _tracked_files(root)
    tracked_set = set(tracked)
    tokens = _manifest_tokens(manifest_path.read_text(encoding="utf-8"))

    exact = [t for t in tokens if not _is_glob(t)]
    globs = [t for t in tokens if _is_glob(t)]

    # (b) manifest path/glob that matches zero tracked files
    invented = sorted(t for t in exact if t not in tracked_set)
    dead_globs = sorted(g for g in globs if not _glob_matches(g, tracked))

    # (a) tracked file covered by no manifest entry
    covered: set[str] = {t for t in exact if t in tracked_set}
    for glob in globs:
        covered.update(_glob_matches(glob, tracked))
    uncovered = sorted(f for f in tracked if f not in covered)

    print(f"tracked files                   : {len(tracked)}")
    print(
        f"manifest entries                : {len(tokens)} ({len(exact)} exact, {len(globs)} glob)"
    )
    print(f"uncovered tracked files         : {len(uncovered)}")
    print(f"manifest paths matching nothing : {len(invented) + len(dead_globs)}")

    if not (uncovered or invented or dead_globs):
        print("OK: FILE_MANIFEST.md accounts for every tracked file.")
        return 0

    print()
    print("FILE_MANIFEST.md is out of sync with the tree:")
    for path in uncovered:
        print(f"  tracked but absent from the manifest  : {path}")
    for token in invented:
        print(f"  manifest names a non-existent path    : {token}")
    for glob in dead_globs:
        print(f"  manifest glob matches no tracked file : {glob}")
    print()
    print("Fix: add or correct the row(s) in FILE_MANIFEST.md, then re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
