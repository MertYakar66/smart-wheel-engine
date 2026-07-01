---
id: d2-live-spot-staleness
title: D-2 fix: live (as_of=None) spot-staleness guard
kind: fix
status: shipped
terminal: X
pr:
decisions: []
date: 2026-06-15
headline: as_of=None runs no longer silently price off a stale spot — warn-once + spot_date provenance + opt-in refusal
surface: [engine/wheel_runner.py, tests/test_pit_leaks.py]
---

## Goal
Close adversarial-review finding **D-2** (`docs/ADVERSARIAL_WEAKNESS_REVIEW_2026-06-15.md`,
the #1 hardening priority). The S32-F3 staleness gate (`wheel_runner.py:976`)
refuses to silently substitute a back-dated close as "current spot" — but it is
wrapped in `if as_of is not None:`. On the **live path (`as_of=None`)** the spot
is set unconditionally from `ohlcv["close"].iloc[-1]` (`:1015`) with no staleness
check, so an operator running a scan "today" against 87-day-stale committed data
prices the BSM delta→strike solve, synthetic premium, and EV off a 2026-03-20
close with **no warning** — the exact D11 "no silent substitution" violation the
as_of gate exists to prevent, but live.

## What we tried
A hard refusal-by-default on the `as_of=None` path (mirror the as_of gate using
`date.today()` as the live reference).

## What didn't
Hard refusal-by-default is a non-starter: dozens of tests (and every backtest)
call `rank_candidates_by_ev(as_of=None)` against the committed CSVs, which end
2026-03-20 and are *necessarily* back-dated relative to the real CI clock. A
default refusal would drop every ticker and red the suite. The live staleness
reference can only be `date.today()` (there is no other "now"), so committed data
is always "stale" on that axis — the gate must not refuse by default.

## How we fixed it
Loud-but-non-breaking by default, with opt-in hard refusal — mirroring the
codebase's existing opt-in gate pattern (`enforce_history_gate`,
`require_ev_authority`):

1. **`spot_date` on every row** (core column) — the trading date the spot is
   actually priced from (`ohlcv.index[-1]`, after any as_of PIT trim). Provenance
   is now explicit; an operator never has to assume "spot == today".
2. **Warn once per call** when `as_of is None` and the latest close lags
   `date.today()` by more than `max_as_of_staleness_days` (default 30). Still
   ranks (backtests unaffected), but no longer silent.
3. **New `refuse_stale_live: bool = False`** param — when `True`, hard-drops
   stale-live candidates with a `gate="data"` reason (for real-money live
   operation where pricing off a back-dated spot is unacceptable).

The `as_of`-provided path is byte-identical (the new block is under
`if as_of is None:`), so no backtest baseline shifts and the regression snapshots
are untouched. `engine/wheel_runner.py` only; `ev_engine`/`candidate_dossier`
untouched. §2 invariant unaffected.

## Evidence
- 5-ticker smoke on the **real** committed data (`SWE_DATA_PROVIDER=bloomberg`,
  `as_of=None`): now emits `WARNING ... live spot is 87 days stale (latest OHLCV
  2026-03-20 vs today 2026-06-15)` and every row carries `spot_date='2026-03-20'`.
- New tests `tests/test_pit_leaks.py::TestLiveSpotStaleness` (4): refuse-when-opted-in
  (df empty + drop reason carries the stale gap and real spot date), warn-but-rank
  by default (caplog asserts the warning; `spot_date` exposed), fresh-data
  not-flagged (no over-refusal), and as_of-path-unchanged provenance. Use a
  synthetic connector ending at a FIXED old date (2020-01-01) so the test is
  robust to tomorrow's data refresh.
- Regression: `test_pit_leaks` (incl. the existing `TestRankerAsOfBeyondData`),
  `test_diagnostic_column_honesty`, `test_covered_call_ranker` (separate pinned
  column set — intact), `test_audit_improvements`, `test_audit_invariants`,
  `test_dossier_invariant` → 141 passed. ruff check + format clean. Full-suite
  (minus backtest_regression) re-run launched for final confirmation.

## Unresolved / handoff
- `analyze_ticker` (the advisory/display path, `wheel_runner.py:~513`) has the
  same `as_of=None` → latest-close substitution; this fix targeted the tradeable
  ranker only. Mirror the warning there as a small follow-up if the display
  surface needs the same provenance.
- Consider arming `refuse_stale_live=True` on the live operator/cockpit surface
  (engine_api) once fresh data is the norm — currently default-off to keep
  backtests green.
- Not committed (shared dirty tree); part of the `claude/weakness-review-fixes`
  working-tree batch awaiting review.
