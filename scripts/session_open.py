#!/usr/bin/env python3
"""Session-open: print the first line every session starts with, its mark (DECISIONS.md D32).

Usage (from anywhere inside the checkout)::

    python scripts/session_open.py                                  # the Strategist (the pen)
    python scripts/session_open.py --executor --run-mode change     # the Executor
    python scripts/session_open.py --codex --reviewing "PR #531" --verified "the diff"

The pen's mark::

    Report from the Strategist, Sir — main `<hash>` (<date>) · docs <n> commits behind ·
    <n> other branches · data <date> (<n> days old) · nearest deadline: <what>, <n> days

Every slot comes from a command or a file in the repository, never from memory,
and a slot whose source fails reads ``unknown``:

* **main** is ``origin/main`` after ``git fetch origin``. It never uses ``--prune``.
* **docs behind** counts the content commits on ``main`` after the hash that
  ``PROJECT_STATE.md`` records (a "`main` is at `<hash>`" line). Merge commits and
  ``docs(close)`` commits are excluded.
* **other branches** is ``git ls-remote --heads origin``, less ``main``.
* **data** is the oldest date in the frontier that ``data/DATA_MANIFEST.json``
  records (``scripts/data_manifest.py build`` writes it), with its age in days.
* **nearest deadline** is the nearest open row of ``docs/deadlines.md``. An
  overdue row comes first, as ``overdue by <n> days``. With no open dated row,
  the slot reads ``none open``.

Stdlib and git only, so it runs the same on Linux and Windows. It always exits 0:
the mark is information, and a failed source shows as ``unknown``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UNKNOWN = "unknown"
PEN = "Report from the Strategist, Sir —"
EXECUTOR = "Report from the Executor, Sir —"
CODEX = "Second opinion from Codex, Sir —"
MAIN_HASH = re.compile(r"`?main`? is at `([0-9a-f]{7,40})`")
ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


def git(*args: str, repo: Path = REPO) -> str | None:
    """Output of a git command, or None when it fails."""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def recorded_main_hash(project_state: str) -> str | None:
    """The main commit PROJECT_STATE.md records, or None."""
    m = MAIN_HASH.search(project_state)
    return m.group(1) if m else None


def main_tip(repo: Path = REPO) -> tuple[str, str]:
    out = git("log", "-1", "--format=%h %ad", "--date=short", "origin/main", repo=repo)
    if not out or " " not in out:
        return UNKNOWN, UNKNOWN
    h, d = out.split(" ", 1)
    return h, d


def docs_drift(recorded: str | None, repo: Path = REPO) -> str:
    if not recorded:
        return UNKNOWN
    out = git(
        "rev-list",
        "--count",
        "--no-merges",
        "--invert-grep",
        "--grep=^docs(close)",
        f"{recorded}..origin/main",
        repo=repo,
    )
    return out if out is not None and out.isdigit() else UNKNOWN


def other_branches(repo: Path = REPO) -> str:
    out = git("ls-remote", "--heads", "origin", repo=repo)
    if out is None:
        return UNKNOWN
    heads = [line.split("\t", 1)[1] for line in out.splitlines() if "\t" in line]
    return str(sum(1 for h in heads if h != "refs/heads/main"))


def frontier_dates(manifest: dict) -> list[tuple[str, str]]:
    """(dataset, last_date) pairs from the manifest's frontier, oldest first."""
    pairs = [
        (name, f.get("last_date", ""))
        for name, f in manifest.get("frontier", {}).items()
        if isinstance(f, dict) and ISO.fullmatch(f.get("last_date", ""))
    ]
    return sorted(pairs, key=lambda p: p[1])


def data_slot(manifest_path: Path, today: dt.date) -> tuple[str, str]:
    """The mark's data slot and a detail line naming every dataset."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return UNKNOWN, f"data: {manifest_path.name} unreadable"
    pairs = frontier_dates(manifest)
    if not pairs:
        return UNKNOWN, "data: the manifest records no frontier"

    def age(d: str) -> int:
        return (today - dt.date.fromisoformat(d)).days

    oldest = pairs[0][1]
    detail = ", ".join(f"{name} {d} ({age(d)} days)" for name, d in pairs)
    return (
        f"{oldest} ({age(oldest)} days old)",
        f"data frontier: {detail} (data/DATA_MANIFEST.json)",
    )


def live_frontier_note(manifest_path: Path) -> str | None:
    """On a machine holding the data, say so when the root runs past the recorded frontier."""
    raw = os.environ.get("SWE_DATA_ROOT", "").strip()
    if not raw:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    notes = []
    for name, f in manifest.get("frontier", {}).items():
        p = Path(raw).expanduser() / f.get("path", "")
        if not p.is_file():
            continue
        best = ""
        with open(p, encoding="utf-8", errors="replace") as fh:
            next(fh, None)
            for line in fh:
                d = line.split(",", 1)[0].strip()
                if ISO.fullmatch(d) and d > best:
                    best = d
        if best and best != f.get("last_date"):
            notes.append(
                f"{name}: the root runs to {best}, the manifest records {f.get('last_date')}"
            )
    if not notes:
        return None
    return "; ".join(notes) + " (rebuild the manifest: python scripts/data_manifest.py build)"


def parse_deadlines(text: str) -> list[list[str]]:
    """Data rows of the deadlines table: the one headed | Due | What | Owner | Status | Source |."""
    rows: list[list[str]] = []
    in_table = False
    for line in text.splitlines():
        if not line.startswith("|"):
            if in_table:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not in_table:
            in_table = [c.lower() for c in cells[:5]] == [
                "due",
                "what",
                "owner",
                "status",
                "source",
            ]
            continue
        if set(cells[0]) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def _short(what: str, limit: int = 70) -> str:
    plain = re.sub(r"[*`]", "", what).strip()
    first = re.split(r"(?<=\w)\.\s", plain, maxsplit=1)[0].rstrip(".")
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def nearest_deadline(rows: list[list[str]], today: dt.date) -> str:
    open_rows = []
    for cells in rows:
        m = ISO.search(cells[0])
        status = re.sub(r"[*`_]", "", cells[3]).strip().lower()
        if not m or status.startswith(("closed", "done")):
            continue
        open_rows.append((dt.date.fromisoformat(m.group(0)), _short(cells[1])))
    if not open_rows:
        return "none open"
    overdue = sorted(r for r in open_rows if r[0] < today)
    if overdue:
        due, what = overdue[0]
        return f"{what}, overdue by {(today - due).days} days"
    due, what = min(open_rows)
    return f"{what}, {(due - today).days} days"


def pen_mark(repo: Path, today: dt.date) -> tuple[str, list[str]]:
    h, d = main_tip(repo)
    try:
        state = (repo / "PROJECT_STATE.md").read_text(encoding="utf-8")
    except OSError:
        state = ""
    recorded = recorded_main_hash(state)
    drift = docs_drift(recorded, repo)
    branches = other_branches(repo)
    data, data_detail = data_slot(repo / "data" / "DATA_MANIFEST.json", today)
    try:
        deadline = nearest_deadline(
            parse_deadlines((repo / "docs" / "deadlines.md").read_text(encoding="utf-8")), today
        )
    except OSError:
        deadline = f"{UNKNOWN} (docs/deadlines.md missing)"
    mark = (
        f"{PEN} main `{h}` ({d}) · docs {drift} commits behind · {branches} other branches"
        f" · data {data} · nearest deadline: {deadline}"
    )
    details = [data_detail]
    if recorded is None:
        details.append("docs: PROJECT_STATE.md records no main hash, so the drift is unknown")
    note = live_frontier_note(repo / "data" / "DATA_MANIFEST.json")
    if note:
        details.append(note)
    return mark, details


def executor_mark(repo: Path, run_mode: str) -> str:
    branch = git("rev-parse", "--abbrev-ref", "HEAD", repo=repo) or UNKNOWN
    head = git("rev-parse", "--short", "HEAD", repo=repo) or UNKNOWN
    pushed = git("rev-parse", "--short", "@{upstream}", repo=repo)
    if pushed is None and branch not in (UNKNOWN, "HEAD"):
        pushed = git("rev-parse", "--short", f"origin/{branch}", repo=repo)
    counts = git("rev-list", "--left-right", "--count", "origin/main...HEAD", repo=repo)
    behind_ahead = "/".join(counts.split()) if counts else UNKNOWN
    pushed_slot = f"`{pushed}`" if pushed else "nothing yet"
    return (
        f"{EXECUTOR} branch `{branch}` · HEAD `{head}` · pushed {pushed_slot}"
        f" · behind/ahead of main {behind_ahead} · run mode {run_mode}"
    )


def codex_mark(repo: Path, reviewing: str, verified: str) -> str:
    h, d = main_tip(repo)
    return f"{CODEX} reviewing {reviewing} · main `{h}` ({d}) · verified myself: {verified}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    role = ap.add_mutually_exclusive_group()
    role.add_argument("--executor", action="store_true", help="the Executor's mark")
    role.add_argument("--codex", action="store_true", help="the second opinion's mark")
    ap.add_argument("--run-mode", choices=("read-only", "change"), help="Executor: the run mode")
    ap.add_argument("--reviewing", default="<what>", help="Codex: what is under review")
    ap.add_argument("--verified", default="nothing", help="Codex: what it checked itself")
    ap.add_argument("--no-fetch", action="store_true", help="skip git fetch (offline)")
    ap.add_argument("--today", help="YYYY-MM-DD instead of the system date (scenario tests)")
    ap.add_argument("--repo", default=str(REPO), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    repo = Path(args.repo)
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()

    if args.executor and not args.run_mode:
        ap.error("--executor needs --run-mode read-only|change (line 1 of the Execution Prompt)")
    fetched = True if args.no_fetch else git("fetch", "origin", "--quiet", repo=repo) is not None
    if args.executor:
        print(executor_mark(repo, args.run_mode))
    elif args.codex:
        print(codex_mark(repo, args.reviewing, args.verified))
    else:
        mark, details = pen_mark(repo, today)
        print(mark)
        for line in details:
            print(f"  {line}")
    if not fetched:
        print("  (git fetch origin failed: main and the branch counts are as of the last fetch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
