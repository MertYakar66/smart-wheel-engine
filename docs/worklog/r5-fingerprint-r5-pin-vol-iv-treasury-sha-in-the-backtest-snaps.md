---
id: r5-fingerprint
title: "R5: pin vol_iv + treasury sha in the backtest snapshot fingerprint"
kind: fix
status: complete
terminal: desktop
pr:
decisions: []
date: 2026-06-06
headline: Closed the snapshot-fingerprint blind-spot — the regression fingerprint pinned OHLCV only, so a vol_iv or treasury refresh could silently move S27/S32/S34/S35 results. Now also captures vol_iv + treasury sha256; backfilled the 4 pinned snapshots with the current (main) provenance, claim numbers untouched.
surface: [backtests/regression/_common.py, tests/test_backtest_regression.py, backtests/regression/snapshots/]
---

## Goal
The 2026-06-05 data-layer QA flagged a fingerprint blind-spot: the regression
snapshot fingerprint pinned `data_csv_sha256` (OHLCV) **only**, but `vol_iv` and
`treasury` also feed the EV path / backtests. A refresh of either would move the
pinned S27/S32/S34/S35 claims without tripping any provenance guard. Land this
**before** R1's re-baseline so the guard is complete when the data changes.

## What we did
- `backtests/regression/_common.py`: factored a `_file_sha256(path)` helper;
  added `vol_iv_sha256()` + `treasury_sha256()` (mirroring `ohlcv_sha256()`); added
  `vol_iv_sha256` + `treasury_sha256` to the fingerprint dict in BOTH drivers
  (`run_backtest`, `run_backtest_multi_friction`). `ohlcv_sha256`'s public
  signature is unchanged.
- `tests/test_backtest_regression.py`: added the two keys to
  `_FINGERPRINT_REQUIRED` so every present snapshot must carry them.
- Backfilled the 4 pinned snapshots' `fingerprint` with the current (main)
  provenance via surgical text insertion after the `data_csv_sha256` line —
  **+2 lines each, nothing else touched**. A guard asserted the
  `aggregate`/`per_year`/`per_quartile`/`per_friction_level` claim sections are
  byte-identical (deep-equal) before/after.

## Why this is NOT a re-baseline
This EXTENDS the captured fingerprint (provenance metadata); it does not
regenerate any computed claim number. The slow `test_backtest_matches_snapshot`
compares only metric sections (it explicitly ignores the fingerprint), so adding
fingerprint keys cannot affect it. No trio file touched.

## Evidence
- Main data shas (== what produced the pinned claims): ohlcv `c3d5443158b12ec5`
  (matches all 4 pinned `data_csv_sha256` exactly), vol_iv `a64b747d81eb68da`,
  treasury `a76a3ef85fa60ef9`.
- `git diff --stat` snapshots = 4 files, **8 insertions(+)** (2 each), 0 deletions.
- `pytest tests/test_backtest_regression.py -m "not backtest_regression"` ->
  5 passed, 4 deselected (slow suite NOT run — confounded + ~5h, per the run rules).
- ruff clean.

## Unresolved / handoff
**R1 dependency:** when the refreshed data merges (R1) and S27/S32/S34/S35 are
re-baselined, the regenerated snapshots will capture the NEW vol_iv/treasury shas
automatically (the drivers now emit them). Until then these snapshots pin the
stale-main provenance, which is correct. The fingerprint is still not an *active*
CI assertion (the slow test ignores it) — it is provenance the re-baseline
workflow reads; making it an active guard would be a separate change.
