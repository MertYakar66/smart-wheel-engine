---
id: validation-v6-lockbox
title: V6 lockbox spend — the one pre-registered deep-history read (2008 refusal generalization)
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-07-13
headline: The phase's single reserved-data spend — spec frozen (plan §9, bc1446a) and the driver pinned with in-code H-verdicts + test-pinned verdict math (1473857) BEFORE the read; deep panels 2007-01-03 → 2009-06-30 with PIT/delisted membership through the survivorship harness, validating selection/refusal/assignment only (synthetic BSM premiums — NAV is not evidence); running on the operator's deep-data terminal.
surface: [scripts/run_v6_lockbox.py, tests/test_v6_lockbox.py, docs/VALIDATION_PHASE_PLAN.md, docs/VALIDATION_PHASE_FINDINGS_2026-07-13.md]
---

## Goal

Does the refusal mechanism generalize to a crisis regime the tuning window
never saw? One pre-registered read of the deep-history lockbox
(2007-01-03 → 2009-06-30: baseline → grind → Lehman cliff → trough),
reported whatever it says. 1998/LTCM stays locked. Frozen hypotheses:
H1 grind refusal (EV-positive rate ≤ 0.5× 2007 baseline over Oct–Dec
2008), H2 cliff lag (opens/day into Lehman eve ≥ 0.7× August = onset
blindness generalizes; the falsifier is GOOD news), H3 assignment wave
(≥ 50% ITM some Sep–Dec month; delisting census report-only), H4 rank
rho (report-only, synthetic-premium caveat).

## What we tried

- Reading assignment off the tracker's closed positions: rejected on
  schema recon — `_finalize_position` records carry no assignment flag
  and assigned puts wheel into STOCK_OWNED rather than closing at
  expiry; the first H3 draft would have returned INSUFFICIENT on the
  unrepeatable spend.
- CLI parameters on the driver: rejected by design — the spec IS the
  driver; a frozen dict with no overrides is what makes
  "never re-parameterized" mechanical rather than aspirational.

## What worked

`scripts/run_v6_lockbox.py` — frozen SPEC dict, precondition check
(refuses cleanly without `data/bloomberg/deep/`), the four H-verdict
functions computed in code, H3 rewritten onto the schema-safe
forward-replay identity (ITM ⟺ realized_pnl < premium×100 − 1e-9 on
EV-positive ranked rows), H2 onto entry dates across closed+open
positions, H3b delisted-participant census via the deep connector's
last-bar date. `tests/test_v6_lockbox.py` pins the verdict math on
synthetic rank logs BEFORE the spend (a bug found after the read could
not be fixed by re-reading). Sandbox smoke: precondition refusal path +
4/4 tests green.

## What didn't

- **Attempts 1+2 (2026-07-13) crashed identically in driver
  post-processing — read NOT spent.** The engine pass completed both
  times (~61 min, well under the 3-5 h estimate); the driver then died
  at its opens-collection line before any verdict was computed or
  written. The schema recon that saved H3 missed the OTHER half of the
  vehicle's return: `open_positions` is a `{ticker: state_string}` map,
  and the driver's shape guard handled only the DataFrame case —
  `list(dict)` yields ticker strings and `.get("entry_date")` raises.
  The pre-spend tests pinned the H-verdict MATH but not the TRANSPORT
  (the vehicle's actual return shape), which is exactly why 4/4 passed
  and the run died live. Deeper gap surfaced by the executor: the state
  map carries no entry dates at all — still-open positions' opens were
  unrecoverable from the return value as it stood.

## How we fixed it

Transport/reporting only; SPEC and all four H-verdict functions are
byte-identical to the pinned pre-spend commit (`1473857`):

- `backtests/survivorship.py` gains a non-breaking
  `open_position_records` key (ticker/state/entry_date from the
  tracker's Position objects); the legacy `open_positions` map is
  untouched (its only consumer was this driver).
- The driver's opens-collection became the shape-defensive
  `collect_entry_dates` (records/DataFrame/legacy-map/None all
  tolerated); raw artifacts (`v6_rank_log.csv.gz` + `v6_positions.json`)
  are written BEFORE verdict computation; each H-verdict is
  exception-captured (`verdict=ERROR` + traceback recorded in the
  report). The engine pass can no longer be lost to a reporting bug.
- Regression locks added: `collect_entry_dates` vs every vehicle shape
  including the exact crash shape, `_safe` capture, and `build_report`
  post-processing the vehicle's byte-exact return shape end-to-end (the
  test that would have caught this — it fails on the `1473857` driver).
- The crash + fix are recorded in plan §9.2's run-record BEFORE the
  restart; §2 semantics hold (no result surfaced → spend unspent;
  1998/LTCM unread).

## Result

*(PENDING — the executor's H-verdict block + `v6_report.json` land here
verbatim; plan §9.3 + the §2 lockbox ledger row + findings-doc §7
complete in the closing commit.)*
