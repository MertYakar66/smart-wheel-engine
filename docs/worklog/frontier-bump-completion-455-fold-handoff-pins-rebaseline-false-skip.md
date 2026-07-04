---
id: frontier-bump-completion
title: "Frontier-bump #472 completion: #455 fold, handoff pins, 4x re-baseline, false-skip catch"
kind: fix
status: shipped
terminal: X
pr: 472
decisions: []
date: 2026-07-04
headline: "The operator's 2026-07-02 Bloomberg pull taken to merge: held #455 splice re-applied to the fresh monolith (dividends leg self-refused — fresh pull already adjusted), all handoff pins landed, 4 snapshots re-baselined (S34 drift = the corrected BKNG scale), and a probe-window bug un-masked 16 silently-skipping frontier tests whose 'passed' claim was actually 'skipped'"
surface:
  - tests/test_data_to_engine.py
  - tests/test_data_integrity_bloomberg.py
  - backtests/regression/snapshots/s34_universe_100t_1m.json
---

## Goal

Complete PR #472 (the Terminal session's frontier bump 2026-06-04 →
2026-07-02 + earnings snapshot vintage 07-03) per the operator's three
decisions: re-baseline now; fold held #455 first; keep it data-only.

## What we tried / worked

- **#455 fold**: re-ran `scripts/fix_ohlcv_split_scale_439.py` against
  the NEW monolith (never text-merge the stale data blob). OHLCV leg
  applied (BKNG ÷25, CVNA ÷5 pre-2026-03-23); the dividends ÷25 leg
  **self-refused via its idempotency guard** — the fresh Bloomberg
  dividends pull already serves split-adjusted amounts (0.384/0.42),
  unlike #455's original environment. Scale-break set EMPTY (two-sided
  pin); both strict-xfails lifted. #455 closed as superseded.
- **Handoff pins**: 11 broad_pull byte-pins (verified against actual
  bytes by the loader tests themselves), HONA → KNOWN_THIN,
  earnings-backfill-pending seam exceptions {ECHO, MRVL, FLEX} +
  structural joiner-skip on the leavers half, TSLA pin upgraded to a
  two-vintage PIT supersession test, DATA_INVENTORY §6 re-pinned.
- **4× re-baseline** (data-only, parallel-3, ~2h): S27/S32 moved at
  the 4th-5th decimal; S35 byte-identical (2018-2020 untouched);
  **S34's real drift** (ev_mean 18.92→10.65, final_cash 380k→92k,
  trades 300→287, ρ 0.3174→0.3130 with rows/assignments/hit-rate
  stable) = the corrected BKNG scale ($4,300→$172 in-window; BKNG is
  in UNIVERSE_100 and historically dominated S34's realized P&L) —
  #455 was held precisely to own this re-baseline.

## What didn't

- **The pull session's "480/31 + W15/W16 PASSED at the new frontier"
  claim.** The pin bump left the coverage probe's hardcoded
  `end_date="2026-06-30"` in place — the probe could never reach
  FRONTIER=2026-07-02, so **all 16 frontier-gated tests in
  `test_data_to_engine.py` silently self-skipped** and read as "did
  not fail". Caught only by a branch-vs-main skip census (44 vs 28).
  The probe window is now DERIVED from FRONTIER. The
  xfail-false-green class strikes again: a gate that silently skips
  its subject is worse than no gate.
- Un-skipping exposed 7 real invalidations (all re-pinned to byte
  truth): the 5-ticker smoke needed the #464 oracle; three mechanics
  tests needed `use_event_gate=False` (they pin cascade/IV/sign
  wiring, not gating); the DIS ex-div pick → JPM (07-06 ex, $1.50;
  DIS's next declaration postdates 07-02); the full-universe split
  480/31 → **66/449**.

## Evidence

- **The seasonal-book headline**: at the 2026-07-02 frontier, 421 of
  515 names event-lock (July earnings season under the restored
  100%-coverage lockout) — the tradeable book is ~13% of the index at
  35 DTE. Designed behavior, not a defect; operationally significant
  for any live deployment this month.
- Independent verifier (pre-merge): seam ratios BKNG 1.0266 / CVNA
  1.0022; full-universe no-break scan clean; S35 value-identical;
  drift-guard hashes freshly recomputed and matching; both TSLA
  vintage directions; 54/54 loader pins. Its two mechanical blockers
  (ruff wrap, missing FILE_MANIFEST rows) fixed pre-merge; its record
  corrections (S34 hit_rate moved at the 4th decimal; bid_ask-leg
  assignments 81→83) disclosed on the PR.
- Final: full fast suite 3,436 passed / 1 known deep-read box
  artifact, skips back to 28; CI 9/9 at `46ccbe5`; merged as
  `14b21cb`.

## Unresolved / handoff

- `sp500_earnings.csv` historical backfill still deferred (bds hung on
  the Terminal box; #472 handoff item 5) — the {ECHO, MRVL, FLEX} seam
  exceptions clear when it lands.
- `iv_surface.csv.gz` at **90.8% of the 100 MiB cap** — the next
  ticking clock; split/restructure due within a few refresh cycles.
- The 45d earnings-snapshot alarm resets from vintage 2026-07-03 →
  next re-pull due ~mid-August 2026.
- MRVL/ECHO/FLEX carry only recent-window rows in iv_surface /
  vol_term_rv (staged panels, unconsumed) — full backfill can ride the
  next refresh.
