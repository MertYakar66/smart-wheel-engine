---
id: parameter-oos-gate
title: Parameter-OOS validation gate (E5)
kind: verification
status: complete
terminal: builder
pr:
decisions: []
date: 2026-07-06
headline: Built a committed, snapshot-locked parameter-OOS gate that re-selects the engine's regime overlay on a leakage-certified training partition and measures the in-sample→out-of-parameter optimism gap. Finding — the re-fittable regime edge is ~86% in-sample optimism (train ρ +0.11 → holdout ρ +0.02), and 24-name rank-ρ is unstable out-of-window (per-fold −0.14..+0.12, pooled ≈0). Reporting-only, off §2; production defaults unchanged.
surface:
  - backtests/parameter_oos.py
  - scripts/run_parameter_oos.py
  - tests/test_parameter_oos.py
  - tests/fixtures/param_oos/rank_table_24t.csv
  - backtests/regression/snapshots/param_oos_regime_24t.json
  - docs/PARAMETER_OOS.md
---

## Goal

Every locked backtest (S27/S32/S34/S35) is parameter-**in-sample** (caveat E5):
the HMM state weights, F4 trigger, POT-GPD threshold, dealer clamp, R11 cutoffs
and heavy-tail penalty were all hand-set with full-history visibility. S35 is
out-of-*window*, not out-of-*parameter*. So "is the edge real, or fit to the
sample?" is unanswered. Quantify how much reported edge survives
**out-of-parameter** and commit it as a snapshot-locked gate — without changing
any production default, without touching the decision-layer trio, and with an
explicit no-leakage proof for every OOS number (a subtly-leaky OOS result is
worse than none).

## What we tried

1. **Phase 0 inventory** of the overfitting surface (verified file:line for
   regime weights `regime_hmm.py:302-307`, clamp `ev_engine.py:552-553`, F4
   `forward_distribution.py:~529`, POT-GPD `tail_risk.py:58/85/225`, dealer
   `dealer_positioning.py:738-739`, R11 `candidate_dossier.py:64-65`, heavy-tail
   `ev_engine.py:260`). Classified each online-fit (PIT-clean) vs static-hand-set.
2. **Probed what is actually testable on Bloomberg.** `dealer_multiplier ≡ 1.0`
   (no GEX/flow), `skew/news/credit_multiplier ≡ 1.0`, so `ev_dollars ≈ ev_raw ×
   clamp(hmm_multiplier)` — the **regime overlay is the only active tuned
   surface**, and it is the marquee E5 risk (i9's leave-one-crisis-out story).
3. **One engine pass, offline re-weighting.** `rank_candidates_by_ev(..,
   include_diagnostic_fields=True)` exposes `ev_raw`, `hmm_regime`,
   `regime_multiplier` per row — so a single production pass captures a per-row
   table and the entire parameter re-fit runs offline in numpy (no re-running the
   engine per grid point, and no engine mutation).
4. **Leakage-certified train/holdout split** with per-row proof, plus rolling
   walk-forward folds and a regime-conditional holdout breakdown.

## What worked

- **The optimism gap is the clean, robust deliverable.** Re-fitting per-regime
  scalars on TRAIN (2020-06..2023-06, n=1170) yields ρ **+0.112** (z≈3.8,
  significant), which collapses to **+0.016** (z≈0.4) on the disjoint holdout —
  gap **+0.096**, ~86% of the apparent edge gone. Train-optimal weights
  `{crisis 2.0, bear 0.1, bull_quiet 2.0}` **invert** the shipped
  `{0.2,0.5,1,1.25}` — the signature of a sample-fit optimum. The shipped
  constant does not rescue it (holdout ρ −0.024 ≈ ev_raw −0.026 ≈ 0).
- **Out-of-window instability** is visible and consistent with the locked set:
  per-fold ρ spans −0.14..+0.12; pooled 2020–2025 ρ = +0.005 (z=0.21). S27 itself
  decays 0.36 (2022) → 0.06 (2024); S35 is 0.50 (2018–2020). Rank-ρ is strongly
  window/universe-dependent, not a steady edge.
- **Snapshot+pytest lock mirrors the S27/S34 pattern**: committed rank-table
  fixture + snapshot; fast fixture↔snapshot recompute lock + fixture-independent
  leakage/identity unit tests; slow `backtest_regression`-marked engine
  regeneration lock. 10 fast tests green.

## What didn't

- **Sparse sampling (every 8 b-days) makes per-year ρ noisy (SE ≈ 0.05)** and my
  calendar-year ρ (2022 +0.25, 2023 −0.03, 2024 −0.11) diverges from S27's
  daily-sampled per-year (0.36/0.19/0.06) by up to ~3 SE in 2023–24. So the
  **absolute ρ level** is sampling-sensitive — I do NOT claim "the engine's true
  ρ is 0." The optimism-gap and instability findings are within-methodology
  differences and are robust to this; the absolute-level reconciliation is
  disclosed as a caveat, and a daily-sampled + 100-name replication is speced.
- **A calendar leave-one-crisis-out could not be re-derived in-harness**: the
  sampled span (2020-06 → 2025-06) contains only the 2022 crisis (COVID predates
  it). A cross-time crisis-label holdout would be temporally leaky, so I did NOT
  ship it. Instead: cite the committed i9 study and report the **leakage-clean**
  regime-conditional holdout breakdown (crisis ρ −0.07, bull_quiet ρ −0.22).
- **Dealer clamp / POT-GPD / F4 / R11 are not scored** by rank-ρ (inert on
  Bloomberg, or they act on tails/prob_profit/verdict) — marked "not established"
  with a per-parameter re-fit spec, not fudged.

## How we fixed it

Shipped (trio + production defaults untouched, `git diff` proves both):
- `backtests/parameter_oos.py` — pure-numpy analysis library + the one-time
  engine pass (`build_rank_table`); offline `apply_regime_scalars` /
  `apply_tilt_exponent`; `refit_regime_scalars` / `refit_tilt_exponent`;
  `assert_no_leakage` (per-row certificate); `rolling_folds`,
  `parameter_holdout_report`.
- `scripts/run_parameter_oos.py` — `build` (engine → committed fixture) /
  `analyze` (fixture → committed snapshot), split so an analysis bug never
  forces an engine re-run.
- `tests/fixtures/param_oos/rank_table_24t.csv` (1957 rows) +
  `backtests/regression/snapshots/param_oos_regime_24t.json` +
  `tests/test_parameter_oos.py` (10 fast + 1 slow) + `docs/PARAMETER_OOS.md`.

## Evidence

Provider + smoke: `MarketDataConnector` (bloomberg); 5-ticker smoke @ 2026-06-02
= 5 rows, 0 nulls.

```
python scripts/run_parameter_oos.py build     # 1957 rows, all resolved (~5.5 min)
python scripts/run_parameter_oos.py analyze
  leakage: max_train_exp=2023-07-31 < min_holdout_rank=2023-08-21  (gap 21d, leakage_free=True)
  train_n=1170  holdout_n=753
  ev_raw_no_overlay     train_rho=+0.0204  holdout_rho=-0.0260
  shipped               train_rho=+0.0370  holdout_rho=-0.0237
  refit_regime_scalars  train_rho=+0.1117  holdout_rho=+0.0158   optimism_gap=+0.096
  refit_tilt_exponent   train_rho=+0.0642  holdout_rho=+0.0015
  walk-forward folds ρ: -0.014 / +0.045 / +0.116 / -0.138 / +0.020  (pooled +0.005, n=1957)
  holdout-by-regime ρ:  crisis -0.067  bear +0.026  normal +0.091  bull_quiet -0.222
pytest tests/test_parameter_oos.py -m "not backtest_regression" -> 10 passed
```

## Unresolved / handoff

- **Denser + 100-name replication** (daily sampling, `UNIVERSE_100`) to pin the
  absolute-ρ level against S34 and shrink per-year SE — the natural robustness
  follow-up; reuses the same harness (change `CONFIG`).
- **Per-parameter hold-out for POT-GPD / F4 / heavy-tail / R11** needs an
  engine-per-grid re-run (monkeypatch the constant in the harness — permitted by
  invariant 3) with a parameter-matched metric (tail coverage / Brier / P&L);
  speced in `docs/PARAMETER_OOS.md` §6, not run.
- **Full per-fold rolling re-fit** (re-select the overlay in each walk-forward
  fold) tightens the parameter-stability estimate; reuses `refit_regime_scalars`.
- **Portfolio NAV per fold** (vs the `mean_realized` per-contract proxy used
  here) needs the full `_common.run_backtest` tracker per fold — deferred.
- **Review-only PR** — do not merge; a re-selected value differing from the
  shipped one is a finding, not a change to ship.
