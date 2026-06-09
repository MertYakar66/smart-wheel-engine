---
id: data-tests-ev-sign-earnings
title: Data-test PR-2 EV sign controls + real earnings lockout (W15/W16)
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Phase-2 PR-2 of the data-layer test audit. Adds the two HIGH data→engine gaps — W15 the real-data EV SIGN controls (XOM +EV / UNH -EV through rank_candidates_by_ev at the FRONTIER, sign pinned not magnitude — catches a transform sign inversion that finite+banded tests miss), and W16 the real earnings→event-lockout wire (JPM dropped gate=='event' with the gate on, produces with it off — asserts the existing §2 first-gate on the real sp500_earnings.csv). Test-only; trio/data untouched.
surface: [tests/test_data_to_engine.py]
---

## Goal
<!-- What we set out to do, and why. -->

Second Phase-2 PR after PR-1 (#370) merged. PR-2 = the two HIGH data→engine gaps
the register surfaced: a real-data EV **sign** control (W15, the crown jewel) and a
real earnings→lockout wire-in (W16). Both route through `rank_candidates_by_ev`
(no §2 bypass). Held for review before PR-3.

## What we tried
<!-- Approaches, in the order we tried them. -->

Added 2 tests to `tests/test_data_to_engine.py`:

- **W15** `test_ev_dollars_sign_controls` — rank `[XOM, UNH]` at FRONTIER; assert
  `ev_dollars` SIGN only: XOM > 0 (clean +EV CSP, ~+113 at lock), UNH < 0
  (structurally -EV, fat left tail, ~-77). Magnitude NOT pinned → the pending
  ev_mean re-baseline moves the numbers, a sign INVERSION fails.
- **W16** `test_real_earnings_event_lockout_fires` — JPM has a real earnings date
  inside the 35-DTE window at FRONTIER. Gate ON → dropped `gate=='event'`; gate OFF
  (`use_event_gate=False`) → produces, no event drop. Asserts the EXISTING wire
  (real `sp500_earnings.csv` → `get_next_earnings` → `EventGate` → `evaluate`).

## What worked

Both pass on the bundled data; the existing data→engine suite stays green.

## What didn't
<!-- The dead ends + WHY. -->

First cut of W16 did `set(on["ticker"])` on the gate-ON frame, which is **empty**
when the only requested name is fully gated (no `ticker` column) → `KeyError`. Fixed
with the file's own guard pattern `set(frame["ticker"]...) if len(frame) else set()`
(mirrors `test_thin_names_degrade_gracefully`); the drop is read from
`frame.attrs["drops"]`, which is populated even on an empty frame.

## How we fixed it
<!-- The approach that shipped. -->

Test-only. W15 pins SIGN (per the reviewer: "pin the sign, never a stale
magnitude"), FRONTIER-tied so the pending re-baseline moves it in lockstep. W16
asserts the existing lockout, builds no new wiring (`use_event_gate` already exists
on both rankers). Names pre-verified at FRONTIER before writing the tests.

## Evidence
<!-- Exact commands run, numbers. -->

Worktree off `origin/main 2a1e1c5` (post PR-1 merge), provider `MarketDataConnector`.

- Pre-verify probe at as_of 2026-06-04: XOM ev_dollars **+112.86**, UNH **-77.35**;
  JPM gate ON → drop `event`, gate OFF → produces **+10.88** (no drops).
- `py -3.12 -m pytest tests/test_data_to_engine.py -m "not slow" -q` →
  **13 passed, 12 xfailed, 1 deselected** (12 xfails = pre-existing W2 PIT + 11 W6
  backfill; deselect = the slow full-universe split).
- `ruff check` clean.

## Unresolved / handoff
<!-- What's still open. -->

- **HOLD for review before PR-3.** PR-3 = fundamentals/sector (W19 yield-band +
  GICS-11, W20 dividend_yield→carry real, W17 DEFAULT_SECTOR_MAP coverage —
  characterize before any R9 rewire). Then PR-4 (OHLCV/dividends hygiene + rate_1m
  residual), PR-5 (credit, off-EV).
- Pending re-baseline session re-pins ev_mean on all 4 backtest snapshots (#363
  re-pricing) — W15's signs are robust to it by design.
