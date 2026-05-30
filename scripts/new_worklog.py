#!/usr/bin/env python3
"""Scaffold a new worklog fragment under ``docs/worklog/``.

A worklog fragment is one task's learning record (what we tried / worked /
didn't / how we fixed it). See ``docs/worklog/README.md`` for the format and
``scripts/gen_worklog_index.py`` for the generated index.

Usage:
    python scripts/new_worklog.py S31 --title "Sever verbal news from EV path" --kind feature
    python scripts/new_worklog.py fix-nat-eventgate --title "NaT in event gate" --kind fix

Writes ``docs/worklog/<id-lowered>-<title-slug>.md`` from ``_template.md`` with
the front-matter filled in. Refuses to overwrite an existing fragment. Stdlib
only.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOG_DIR = os.path.join(ROOT, "docs", "worklog")
TEMPLATE = os.path.join(WORKLOG_DIR, "_template.md")

_KINDS = ("feature", "fix", "backtest", "verification", "usage", "refactor", "docs", "research")


def _slug(text: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen].rstrip("-")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scaffold a docs/worklog/ fragment.")
    ap.add_argument("id", help="canonical id (e.g. S31, or a short slug)")
    ap.add_argument("--title", required=True)
    ap.add_argument("--kind", choices=_KINDS, default="feature")
    ap.add_argument("--terminal", default="")
    ap.add_argument("--pr", default="")
    args = ap.parse_args(argv)

    if not os.path.exists(TEMPLATE):
        print(f"ERROR: template not found at {TEMPLATE}", file=sys.stderr)
        return 1

    fname = f"{args.id.lower()}-{_slug(args.title)}.md"
    out = os.path.join(WORKLOG_DIR, fname)
    if os.path.exists(out):
        rel = os.path.relpath(out, ROOT)
        print(f"ERROR: {rel} already exists — not overwriting.", file=sys.stderr)
        return 1

    with open(TEMPLATE, encoding="utf-8") as fh:
        text = fh.read()

    text = text.replace("id: REPLACE_ID", f"id: {args.id}")
    text = text.replace("title: REPLACE_TITLE", f"title: {args.title}")
    text = text.replace("kind: feature", f"kind: {args.kind}")
    if args.terminal:
        text = text.replace("terminal:", f"terminal: {args.terminal}")
    if args.pr:
        text = text.replace("pr:", f"pr: {args.pr}")

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"created {os.path.relpath(out, ROOT)}")
    print("Fill the sections, then: python scripts/gen_worklog_index.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
