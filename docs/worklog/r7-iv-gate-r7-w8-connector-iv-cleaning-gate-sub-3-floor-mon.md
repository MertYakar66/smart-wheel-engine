---
id: r7-iv-gate
title: R7/W8 connector IV cleaning gate (sub-3 floor + monolith sentinel)
kind: fix
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-08
headline: Non-trio data-layer fix for the IV unit bug — connector NULLs implausible vol_iv implied-vol (sub-3.0 low floor + ~134217.7 high sentinel) on the monolith read; obviates #356 (W1) + #360 with ZERO trio edit; snapshot-neutral by enumeration
surface: [engine/data_connector.py]
---

## Goal

Operator redirect: land the audit IV-unit fix (W1/#356) as a **data-layer
gate**, not the trio edit. The earlier W1 PR (#361, unconditional /100 in
wheel_runner) was correct but had a 72-test blast radius and touched the
decision trio. Instead, clean the bad IV at the connector so the existing
`if iv > 3.0: iv/100` heuristic (wheel_runner #356 + wheel_tracker #360) only
ever sees unambiguous-percent IV — making it always-correct with no trio touch.

## What we tried / what worked

`engine/data_connector.py`: a shared `_clean_vol_iv_inplace` helper that NULLs
`hist_put_imp_vol` / `hist_call_imp_vol` cells outside the PERCENT band
`(_IV_LOW_FLOOR=3.0, _DEEP_IV_SENTINEL_FLOOR=10_000]`, KEEPING the row (and its
realized-vol columns). Applied to BOTH `_load` (the default monolith read — new)
and `_load_assembled` (the deep read — was high-only; now also low). The low
floor is set exactly at the rankers'/tracker's percent->decimal threshold, so
every served IV is > 3.0 → the conditional conversion is always correct →
**#356 + #360 obviated with ZERO decision-trio edit**.

## What didn't / blast radius (much smaller than the trio route)

The gate touches only `MarketDataConnector` vol_iv reads. The wheel_runner
heuristic stays intact, so the 72 fundamentals-decimal-stub tests that broke
under the W1 trio route are unaffected here. Only **6 tests** broke — all in
`test_data_connector.py`, whose synthetic `data_dir` vol_iv fixture used DECIMAL
IV (0.10–0.50) that the gate (correctly) nulls. Fixed by ×100 the fixture to
PERCENT (behaviour-preserving: iv_rank/percentile are scale-invariant) +
updating the one IV-RV `vol_risk_premium` assertion to percentage-points
(0.20 -> 20.0). Reverted the W1 trio edit + 11-file migration (branch off main).

## How we fixed it

Connector gate + the 6 fixture migrations + 5 new `TestVolIvCleaningGate` tests
(sub-floor nulled & row/realized kept; high sentinel nulled on the monolith;
band boundaries 3.0/3.01/10000/10000.1; normal percent untouched). #361 closed.

## Evidence

- **Snapshot-neutral by complete enumeration:** the gate trims 17 implied-vol
  cells across 7 names (all low-tail <=3.0; 0 high-tail — the monolith has no
  134217 sentinel, so the high cut is defensive). Only BG/CEG are in a snapshot
  universe (UNIVERSE_100), both on 2025-11-14 — outside every snapshot window
  (S27/S32/S34 end 2024-12-31; S35 ends 2020-12-31). `backtest_regression`
  markers re-run to confirm byte-identical (see PR).
- Targeted: 128 passed / 13 xfailed (pre-existing #358 W2/W6). Full fast suite +
  markers in the PR report. Decision trio: untouched (`engine/` diff =
  data_connector.py only).

## Unresolved / handoff

Cutoffs: low floor 3.0 (percent), high sentinel floor 10,000 — documented on
the constants. Closes #356 + #360 at the data layer (the heuristic code remains
but can no longer mis-fire in production). Held for operator review + sign-off;
no auto-merge.
