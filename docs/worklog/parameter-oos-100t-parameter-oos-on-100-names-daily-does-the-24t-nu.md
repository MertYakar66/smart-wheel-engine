---
id: parameter-oos-100t
title: Parameter-OOS on 100 names (daily) — does the 24t null generalize?
kind: verification
status: in-flight
terminal: builder
pr: 485
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

**The answer: the 24-name null does NOT blanket-generalize — a real, out-of-window
rank edge SURVIVES at the tradeable top tier, and it is breadth.**

- **All-candidate ρ ≈ 0 (reconciles with 24t):** 100-name pooled ρ −0.014
  (block-CI [−0.08, +0.05]); holdout −0.077 [−0.19, +0.02]. Measured across all
  ~48 candidates/day, the null holds.
- **Top-tier edge survives OOS.** HOLDOUT top-5 ρ **+0.597** [0.48, 0.70], top-15
  **+0.371** [0.25, 0.49] — block-bootstrap CIs (~20 effective blocks) EXCLUDE
  zero, and *higher* than TRAIN (+0.560 / +0.354) → stable, not decaying.
- **Reproduces S34 exactly:** S34-window top-15 ρ +0.316 vs S34's reported +0.313
  (identical n=10,896) — the pipeline is faithful, so the discrepancy with the
  all-candidate number is real signal, not a bug.
- **Breadth, not one name (E3):** holdout top-15 drop-BKNG no-op (0.371→0.370),
  LOO [0.351, 0.404] across 97 names, BKNG P&L share 0.1%.
- **Edge lives in `ev_raw`, not the tuned overlay:** holdout top-15 by `ev_raw`
  = +0.378 ≈ `ev_dollars` +0.371 → robust to the entire E5 static-parameter
  surface (the regime overlay #484 showed doesn't generalize is not where the
  edge is; 100t optimism gap is still +0.066).
- **The reconciliation:** ρ decays monotonically with the menu width (+0.60 top-5
  → ≈0 all). 24 names surface only ~15/day, so there is no broad field to select
  the best from — its all-candidate ρ *is* its top tier (≈0). 100 names make the
  selective top-tier edge measurable. Same monotone curve, different subsets.

## What didn't

- **A naive z on the ~63k pooled rows would be dishonestly tight.** Daily
  sampling overlaps forward windows and recurs names. Fixed with a **moving-block
  bootstrap** over ~25-date blocks (≈ the horizon): on the partial it widened the
  pooled CI ~3× (block_len=1 [0.14, 0.25] → block_len=25 [0.01, 0.39], ~6
  effective blocks vs 131 dates). Every reported CI is the block interval; the
  naive block_len=1 holdout CI is recorded beside it for contrast.
- **Full-daily is slow (~2.9 h).** Ran in the background, checkpointed per
  6-month batch so a kill keeps completed batches (never silently sub-sampled).
- **First analyze looked like a flat null** (all-candidate holdout −0.077) and
  contradicted S34's 0.313 — until the top-N sweep showed S34's edge is entirely
  in the top of the EV distribution (a `top_n` artifact of how S34 logged rows).

## How we fixed it

Reused the shared `backtests/parameter_oos.py` verbatim and ADDED (additive — the
24t #484 recompute is byte-unchanged, its test stays green): `per_date_cross_sectional_rho`,
`cluster_bootstrap_ci(block_len=…)` (moving-block), `restrict_top_n_per_date`,
`top_n_tier_scores`, `dominant_name_robustness`. New driver
`scripts/run_parameter_oos_100t.py` (checkpointed daily build + independence-corrected
analyze), `tests/test_parameter_oos_100t.py` (unit + fixture↔snapshot recompute +
surviving-edge lock: holdout top-5/top-15 CIs must exclude zero), snapshot
`param_oos_regime_100t.json`, fixture `rank_table_100t.csv`, `docs/PARAMETER_OOS.md` §7.

## Evidence

Provider + smoke: `MarketDataConnector` (bloomberg); 5-ticker smoke @ 2026-06-02
= 5 rows, 0 nulls. 100-name rank @ 2022-06-15 = 76 rows in 12.1s.

```
python scripts/run_parameter_oos_100t.py build     # daily, checkpointed
  -> 63,187 rows, all resolved, 1326 daily as_of dates, 2020-06-01..2025-06-30
python scripts/run_parameter_oos_100t.py analyze
  all-candidate: pooled -0.014 [-0.08,+0.05]  holdout -0.077 [-0.19,+0.02]
  TOP-N tier (pooled rho, block-CI95):
    train    top5 +0.560[.46,.64] top15 +0.354[.25,.45] top50 +0.221 all +0.022
    HOLDOUT  top5 +0.597[.48,.70] top15 +0.371[.25,.49] top50 +0.102 all -0.077
    s34_win  top5 +0.525[.43,.62] top15 +0.316[.22,.41] (=S34 0.313) all -0.062
  edge source: holdout top15 ev_raw +0.378 ~ ev_dollars +0.371 (edge in core EV)
  E3 holdout top-15: full +0.371 drop-BKNG +0.370 LOO [0.351,0.404] 97 names
pytest tests/test_parameter_oos_100t.py -m "not backtest_regression" -> 9 passed
git diff HEAD -- engine/ -> empty (trio + all engine untouched)
```

## Unresolved / handoff

- This PR **stacks on #484** (`claude/parameter-oos-gate`) to reuse the harness;
  retarget its base to `main` after #484 merges. Review-only — do not merge.
- **Not a profitability claim** — ρ is top-tier RANK skill, not dollar edge (E1/E3
  stand). A strict per-fold re-fit of the forward-distribution knobs (block size,
  cascade thresholds) is the natural next parameter-OOS step; the overlay re-fit
  is already covered and adds no OOS value.
- Natural follow-ups: does the top-tier edge translate to P&L after friction (run
  the tracker on top-N-only opens); is the top-5 vs top-15 gradient a usable
  sizing signal.
