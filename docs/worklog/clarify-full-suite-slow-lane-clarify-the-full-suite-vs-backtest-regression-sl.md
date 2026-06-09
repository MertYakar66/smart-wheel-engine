---
id: clarify-full-suite-slow-lane
title: Clarify the full-suite vs backtest_regression slow-lane trap in TESTING.md
kind: docs
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-08
headline: TESTING.md called the full suite a bare `pytest tests/ -v`, but that does NOT auto-deselect the `backtest_regression` marker — with the S27/S32/S34/S35 snapshots committed locally a bare run pulls the ~4–5h slow lane inline. Added a callout pinning the per-PR gate to `-m "not backtest_regression"` (what CI runs) and naming the trap.
surface: [TESTING.md]
---

## Goal
<!-- What we set out to do, and why. -->

Surfaced during the post-merge integration sweep on `origin/main` @ `d0cdcde`.
`TESTING.md:3` defines "the full suite" as a bare `pytest tests/ -v`, and the
marker table (`:179`) says `backtest_regression` is "Excluded from per-PR CI".
But that exclusion is **only** via CI's explicit `-m "not backtest_regression"`
flag — there is no local default deselection. A reader following the doc
literally would run the ~4–5 h slow lane inline without warning.

## What tripped it

Three facts compound:

1. `conftest.py::pytest_configure` only **registers** the `backtest_regression`
   marker (`config.addinivalue_line(...)`) — it does not skip or deselect it.
2. `[tool.pytest.ini_options].addopts` in `pyproject.toml` is just `"-v"` — no
   `-m "not backtest_regression"` and no `--strict-markers` deselection.
3. The four reproducers `skip` **only when their snapshot JSON is absent**
   (`.claude/commands/backtest-regression.md`). All four snapshots
   (`s27_ivpit_24t_100k` / `s32_friction_24t_1m` / `s34_universe_100t_1m` /
   `s35_oos_24t_100k`) are committed under `backtests/regression/snapshots/`.

So a literal `pytest tests/` collects them, they do **not** skip, and the run
balloons to ~4–5 h (S34-dominated) instead of the ~11 min the doc implies.

## How we fixed it

Added a blockquote callout immediately under the `TESTING.md` full-suite line
that (a) pins the per-PR / launch-blocker gate to the exact CI command
`pytest tests/ -m "not backtest_regression"`, (b) names the no-auto-deselect
trap and why a bare run pulls the slow lane locally, and (c) notes that the
data-drift guards (`test_snapshot_*fingerprint*`) are *not* behind the marker,
so they run in the fast gate and pre-flag snapshot drift before any multi-hour
run. Docs-only; no code, no trio, no data-file edits.

## Evidence
<!-- Exact commands run, numbers, links to raw artifacts. -->

Confirmed empirically on `d0cdcde` during the sweep:

- Fast gate `pytest tests/ -m "not backtest_regression" -v` →
  `collected 3007 items / 4 deselected / 3003 selected` (the 4 deselected =
  exactly the `backtest_regression` reproducers) → `2970 passed, 14 skipped,
  4 deselected, 15 xfailed` in **695 s (11 m 35 s)**. (The 4 failures were
  live-Theta-Terminal connector tests — environmental, auto-skip without a
  Terminal, unrelated to this doc.)
- Slow lane `pytest tests/test_backtest_regression.py -m backtest_regression` →
  `collected 13 items / 9 deselected / 4 selected` — confirming the same four
  reproducers are the marked slow lane, and that the 9 unmarked items
  (fingerprint/required-keys/universe guards) run in the fast gate.

## Unresolved / handoff
<!-- What's still open; what the next agent should look at next. -->

`CLAUDE.md` §4.3 (`run the full test suite (pytest tests/ -v)`) carries the same
literal wording but was left untouched on purpose — it is the structural
contract and `TESTING.md` is the authoritative test-taxonomy home for this
distinction. If the operator wants the contract doc cross-referenced too, that's
a one-line parenthetical follow-up. Related: memory `canonical-doc-rcount-drift`.
