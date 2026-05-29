#!/usr/bin/env python3
"""Generate ``docs/worklog/INDEX.md`` — the one index over every work record.

The documentation redesign (DECISIONS.md D14 extension, 2026-05) replaces two
treadmills with per-task **worklog fragments** + this generated index:

  * the 490 KB append-only ``docs/USAGE_TEST_LEDGER.md`` monolith (one giant
    file every task had to edit — a rebase magnet, and unreadable whole), and
  * the "write a dated report -> hand-maintain an INDEX over the dated reports
    -> archive the batch" cycle (``docs/VERIFICATION_INDEX_*.md``).

A **worklog fragment** is one file per task/scenario under ``docs/worklog/``
with YAML-ish front-matter and a fixed learning-record body (see
``docs/worklog/README.md``). This script reads the front-matter of every
fragment AND of the in-place dated reports (``docs/ENGINE_BACKTEST_*.md`` etc.,
which are NOT moved — they carry 243 inbound references across 43 files
including CLAUDE.md and decision-layer docstrings, so relocating them is pure
risk), and writes a single grouped, sorted ``INDEX.md``.

Sources scanned:
  1. ``docs/worklog/*.md`` (excluding README / _template / INDEX) — the
     canonical home for new records. Front-matter expected.
  2. The in-place legacy reports (``_LEGACY_GLOBS``) — front-matter used if
     present, else a minimal entry is derived from the filename + first
     ``# `` heading. ``VERIFICATION_INDEX*`` is skipped (this file replaces it).

Stdlib only (no PyYAML) — runs identically on Cowork / Windows / CI, same as
the other doc guards. The front-matter parser handles scalar ``key: value``
lines and a simple ``[a, b]`` list; that is all the index needs.

Modes:
  (default)   regenerate docs/worklog/INDEX.md in place.
  --check     regenerate to memory and diff against the committed INDEX.md;
              exit 1 if stale (the CI gate — keeps the generated file current).

Run:  python scripts/gen_worklog_index.py
      python scripts/gen_worklog_index.py --check
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOG_DIR = os.path.join(ROOT, "docs", "worklog")
INDEX_PATH = os.path.join(WORKLOG_DIR, "INDEX.md")

# Non-fragment files that live in docs/worklog/ but are not work records.
_FRAGMENT_SKIP = {"README.md", "_template.md", "INDEX.md"}

# In-place dated reports — indexed where they live, never moved (see module
# docstring for the 243-reference reason). Globs are relative to docs/.
_LEGACY_GLOBS = (
    "ENGINE_BACKTEST_*.md",
    "ENGINE_REVERIFY_*.md",
    "*REALISM_VERIFICATION*.md",
    "REAL_DATA_VERIFICATION*.md",
    "REVERIFICATION_REPORT*.md",
    "PROB_PROFIT_CALIBRATION*.md",
    "F4_TAIL_RISK_DIAGNOSTIC.md",
    "BACKTEST_REGRESSION_CAMPAIGN.md",
)

# Display order + headings for the kind buckets.
_KIND_ORDER = [
    "feature",
    "fix",
    "backtest",
    "verification",
    "usage",
    "refactor",
    "docs",
    "research",
]
_KIND_TITLE = {
    "feature": "Features",
    "fix": "Fixes",
    "backtest": "Backtests",
    "verification": "Verification & realism",
    "usage": "Usage-test scenarios",
    "refactor": "Refactors",
    "docs": "Docs / process",
    "research": "Research records",
    "other": "Other",
}


def _parse_front_matter(text: str) -> dict[str, object]:
    """Parse a leading ``---``-delimited front-matter block. ``{}`` if none."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end].strip("\n")
    fm: dict[str, object] = {}
    for line in block.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [v.strip() for v in val[1:-1].split(",") if v.strip()]
            fm[key] = items
        else:
            fm[key] = val.strip().strip("'\"")
    return fm


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _kind_from_filename(name: str) -> str:
    up = name.upper()
    if up.startswith("ENGINE_BACKTEST") or up.startswith("BACKTEST_REGRESSION"):
        return "backtest"
    if "VERIFICATION" in up or "REALISM" in up or up.startswith("ENGINE_REVERIFY"):
        return "verification"
    if "CALIBRATION" in up or "F4_TAIL_RISK" in up:
        return "research"
    return "docs"


def _entry(path: str, *, legacy: bool) -> dict[str, object] | None:
    """Build one index row from a file, or None to skip."""
    name = os.path.basename(path)
    with open(path, encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    fm = _parse_front_matter(text)

    if legacy:
        if "VERIFICATION_INDEX" in name.upper():
            return None  # this generator replaces that hand-maintained index
        rid = str(fm.get("id") or os.path.splitext(name)[0])
        title = str(fm.get("title") or _first_heading(text) or rid)
        kind = str(fm.get("kind") or _kind_from_filename(name))
        status = str(fm.get("status") or "legacy")
    else:
        if not fm:
            return None  # a worklog file with no front-matter is not a record
        rid = str(fm.get("id") or os.path.splitext(name)[0])
        title = str(fm.get("title") or rid)
        kind = str(fm.get("kind") or "other")
        status = str(fm.get("status") or "unknown")

    return {
        "id": rid,
        "title": title,
        "kind": kind if kind in _KIND_TITLE else "other",
        "status": status,
        "pr": str(fm.get("pr") or ""),
        "headline": str(fm.get("headline") or ""),
        "link": os.path.relpath(path, WORKLOG_DIR).replace(os.sep, "/"),
    }


def _natural_key(rid: str) -> tuple:
    """Sort S2 < S10; fall back to lexical for slugs."""
    parts = re.split(r"(\d+)", rid)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def collect() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(glob.glob(os.path.join(WORKLOG_DIR, "*.md"))):
        if os.path.basename(path) in _FRAGMENT_SKIP:
            continue
        e = _entry(path, legacy=False)
        if e:
            entries.append(e)
    seen = {os.path.realpath(p) for p in glob.glob(os.path.join(WORKLOG_DIR, "*.md"))}
    for pat in _LEGACY_GLOBS:
        for path in glob.glob(os.path.join(ROOT, "docs", pat)):
            if os.path.realpath(path) in seen:
                continue
            seen.add(os.path.realpath(path))
            e = _entry(path, legacy=True)
            if e:
                entries.append(e)
    return entries


def render(entries: list[dict[str, object]]) -> str:
    lines = [
        "<!-- GENERATED by scripts/gen_worklog_index.py — DO NOT EDIT BY HAND.",
        "     Add/redit a worklog fragment under docs/worklog/, then re-run the",
        "     generator (CI checks it is current via --check). -->",
        "# Worklog index",
        "",
        "Every work record — features, fixes, backtests, verification runs, usage",
        "scenarios — at a glance. Each row links to the full learning record",
        "(*what we tried / what worked / what didn't / how we fixed it*). New",
        "records are per-task fragments under `docs/worklog/`; the dated backtest /",
        "verification reports are indexed in place. See `docs/worklog/README.md`.",
        "",
        f"**{len(entries)} records.**",
        "",
    ]
    by_kind: dict[str, list[dict[str, object]]] = {}
    for e in entries:
        by_kind.setdefault(str(e["kind"]), []).append(e)

    ordered_kinds = [k for k in _KIND_ORDER if k in by_kind]
    ordered_kinds += [k for k in by_kind if k not in _KIND_ORDER]

    for kind in ordered_kinds:
        rows = sorted(by_kind[kind], key=lambda e: _natural_key(str(e["id"])))
        lines.append(f"## {_KIND_TITLE.get(kind, kind.title())} ({len(rows)})")
        lines.append("")
        lines.append("| ID | Status | PR | Headline | Record |")
        lines.append("|---|---|---|---|---|")
        for e in rows:
            pr = f"#{e['pr']}" if e["pr"] else ""
            headline = str(e["headline"]).replace("|", "\\|")
            title = str(e["title"]).replace("|", "\\|")
            link = f"[{e['id']}]({e['link']})"
            lines.append(f"| {link} | {e['status']} | {pr} | {headline or title} | `{e['link']}` |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate docs/worklog/INDEX.md from fragments.")
    ap.add_argument("--check", action="store_true", help="fail if INDEX.md is stale (CI mode)")
    args = ap.parse_args(argv)

    content = render(collect())

    if args.check:
        existing = ""
        if os.path.exists(INDEX_PATH):
            with open(INDEX_PATH, encoding="utf-8") as fh:
                existing = fh.read()
        if existing != content:
            print(
                "FAIL: docs/worklog/INDEX.md is stale — "
                "run: python scripts/gen_worklog_index.py"
            )
            return 1
        print("worklog-index: OK (INDEX.md is current).")
        return 0

    os.makedirs(WORKLOG_DIR, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"wrote {os.path.relpath(INDEX_PATH, ROOT)} ({content.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
