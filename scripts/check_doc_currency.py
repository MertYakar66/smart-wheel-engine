#!/usr/bin/env python3
"""Doc-currency guard — catch the temporal docs drifting behind ``main``.

Stdlib only, so it runs identically on Cowork/Linux, a Windows box
(``py -3.12 scripts/check_doc_currency.py``), and CI. Deliberately
low-friction: it WARNs on mild staleness and only FAILs on egregious
staleness or structural breakage, so normal PRs are never blocked by it.

The decaying temporal docs are ``PROJECT_STATE.md`` (its ``Last updated``
date) and ``CHANGELOG.md`` (its newest month section). This guard does NOT
re-pin a commit SHA or a test count — those are intentionally kept out of
the docs (see PROJECT_STATE's header); it only checks that *someone has
touched the temporal docs recently*.

Checks:
  1. ``PROJECT_STATE.md`` has a parseable ``**Last updated:** YYYY-MM-DD``
     line. FAIL if missing/unparseable, or older than ``--fail-days``
     (default 45). WARN if older than ``--warn-days`` (default 21).
  2. ``CHANGELOG.md``'s newest ``## YYYY-MM ...`` section is recent. Same
     thresholds, measured month-to-month.

Exit 0 = OK or warn-only. Exit 1 = a FAIL-level finding (or, with
``--strict``, any WARN). The SessionStart hook runs it non-strict
(informational); CI runs it as a soft gate that only trips on real
abandonment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _classify(age_days: int, warn_days: int, fail_days: int) -> str:
    if age_days > fail_days:
        return "FAIL"
    if age_days > warn_days:
        return "WARN"
    return "OK"


def _check_project_state(warn_days: int, fail_days: int) -> tuple[str, str]:
    path = ROOT / "PROJECT_STATE.md"
    if not path.exists():
        return ("FAIL", "PROJECT_STATE.md: file not found")
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"\*\*Last updated:\*\*\s*(\d{4}-\d{2}-\d{2})", text)
    if not m:
        return ("FAIL", "PROJECT_STATE.md: no parseable '**Last updated:** YYYY-MM-DD' line")
    try:
        last = dt.date.fromisoformat(m.group(1))
    except ValueError:
        return ("FAIL", f"PROJECT_STATE.md: unparseable date {m.group(1)!r}")
    age = (dt.date.today() - last).days
    return (
        _classify(age, warn_days, fail_days),
        f"PROJECT_STATE.md last updated {last} ({age}d ago)",
    )


def _check_changelog(warn_days: int, fail_days: int) -> tuple[str, str]:
    path = ROOT / "CHANGELOG.md"
    if not path.exists():
        return ("FAIL", "CHANGELOG.md: file not found")
    text = path.read_text(encoding="utf-8", errors="ignore")
    months = re.findall(r"^##\s+(\d{4})-(\d{2})", text, re.MULTILINE)
    if not months:
        return ("FAIL", "CHANGELOG.md: no '## YYYY-MM ...' section headers found")
    newest = max(dt.date(int(y), int(mo), 1) for y, mo in months)
    age = (dt.date.today().replace(day=1) - newest).days
    return (
        _classify(age, warn_days, fail_days),
        f"CHANGELOG.md newest section {newest:%Y-%m} ({age}d behind)",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Warn/fail when the temporal docs drift behind main.")
    ap.add_argument("--warn-days", type=int, default=21)
    ap.add_argument("--fail-days", type=int, default=45)
    ap.add_argument("--strict", action="store_true", help="treat WARN as failure")
    args = ap.parse_args(argv)

    rc = 0
    for level, msg in (
        _check_project_state(args.warn_days, args.fail_days),
        _check_changelog(args.warn_days, args.fail_days),
    ):
        print(f"{level}: {msg}")
        if level == "FAIL" or (args.strict and level == "WARN"):
            rc = 1
    if rc == 0:
        print("doc-currency: OK")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
