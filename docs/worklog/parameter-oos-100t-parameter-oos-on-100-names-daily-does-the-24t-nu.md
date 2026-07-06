---
id: parameter-oos-100t
title: Parameter-OOS on 100 names (daily) — does the 24t null generalize?
kind: verification
status: in-flight
terminal: builder
pr:
decisions: []
date: 2026-07-06
headline: Replicated the #484 parameter-OOS rank test on UNIVERSE_100 at daily cadence with date-clustered bootstrap significance, to decide whether the 24-name OOS null generalizes or an out-of-parameter edge appears on the S34 universe (in-sample rho 0.313). Reporting-only, off §2; reuses the shared harness verbatim.
surface:
  - backtests/parameter_oos.py
  - scripts/run_parameter_oos_100t.py
  - tests/test_parameter_oos_100t.py
  - tests/fixtures/param_oos/rank_table_100t.csv
  - backtests/regression/snapshots/param_oos_regime_100t.json
  - docs/PARAMETER_OOS.md
---

## Goal

PR #484 found NO out-of-sample rank skill on the 24-name universe (holdout
pooled ρ ≈ 0, optimism gap +0.096). But the strongest *in-sample* signal (S34,
ρ 0.313) lived on the **100-name** universe. Decisive question: does the 24-name
null generalize to 100 names, or does an out-of-parameter / out-of-window rank
edge appear where S34's in-sample edge was? Report honestly whichever way it
comes out.

Hard constraints (same as #484): read-only measurement, never mutate an engine
default, trio + all `engine/` untouched, per-row leakage certificate on every OOS
number, no real IBKR data.

## What we tried

1. **Reused the shared harness, did not fork it.** New functions were ADDED to
   `backtests/parameter_oos.py` (per-date cross-sectional ρ, date-clustered
   bootstrap, E3 drop-name/leave-one-out); the 24t and 100t runs call the SAME
   leakage/re-fit/scorecard code so they are directly comparable. A separate
   driver `scripts/run_parameter_oos_100t.py` supplies the 100t config.
2. **Daily capture over 2020-06 → 2025-06** (~1326 as_of dates), checkpointed per
   6-month batch so a killed run keeps every completed batch and is honestly
   labelled by its actual date coverage.
3. **Independence correction** for the daily overlap: a per-date cross-sectional
   rank-ρ + a date-clustered bootstrap that resamples whole as_of dates (never
   rows), so no significance figure rides an inflated pooled N.
4. **Same leakage-certified split as #484** (train ≤ 2023-06-30, holdout ≥
   2023-08-20) for a direct 24t↔100t comparison, plus an S34-window (2022-2024)
   slice for the in-sample reconciliation and an E3 BKNG drop / leave-one-out.

## What worked

<!-- FILL AFTER RUN -->

## What didn't

<!-- FILL AFTER RUN -->

## How we fixed it

<!-- FILL AFTER RUN -->

## Evidence

Provider + smoke: `MarketDataConnector` (bloomberg); 5-ticker smoke @ 2026-06-02
= 5 rows, 0 nulls. 100-name rank @ 2022-06-15 = 76 rows in 12.1s (cold).

```
python scripts/run_parameter_oos_100t.py build     # daily, checkpointed
python scripts/run_parameter_oos_100t.py analyze
<!-- FILL: sampling, pooled/holdout/xsec rho + date-clustered CIs, S34 recon, E3 -->
pytest tests/test_parameter_oos_100t.py -m "not backtest_regression" -> <!-- FILL -->
```

## Unresolved / handoff

- This PR **stacks on #484** (`claude/parameter-oos-gate`) to reuse the harness;
  retarget its base to `main` after #484 merges. Review-only — do not merge.
- <!-- FILL: any coarsening of cadence, any not-established labels -->
