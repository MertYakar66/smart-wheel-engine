---
id: vnv-campaign-2026-06-01
title: Engine V&V sweep — efficiency / realism / reliability
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-01
headline: Read-only V&V sweep of origin/main — funnel is transparent (423/503 survive, all drops auditable), Wilson-CI coverage 98.6% on the put ranker, prob_profit top-bin over-confidence is real + regime-dependent, ev_dollars SIGN predicts realized direction, IV has zero skew (100%), and connector ticker-filtering dominates a universe scan.
surface:
  - docs/VNV_CAMPAIGN_2026-06-01.md
  - scripts/vnv_funnel_tier_report.py
  - scripts/vnv_prob_profit_calibration.py
  - docs/verification_artifacts/vnv_2026-06-01/funnel_2026-03-20.json
  - docs/verification_artifacts/vnv_2026-06-01/calibration_full.json
  - docs/verification_artifacts/vnv_2026-06-01/profile_150ticker_2026-03-20.txt
---

## Goal
8-hour autonomous mandate: make the product "efficient, realistic, reliable;
run all sorts of tests/scenarios." Stress the production engine's outputs for
realism + reliability, profile efficiency, ship reproducible diagnostics + a
findings record. Hard constraint: no trio edits without operator consent — so
this campaign is strictly read-only measurement + off-trio diagnostic tooling.

## What we tried
1. **Theta real-data realism audit** — wanted to check engine inputs (spot/IV/
   premium) against real historical option prices. ABANDONED: the Theta MCP was
   unavailable (Terminal busy pulling data). Pivoted to internal checks against
   the committed Bloomberg reference files.
2. **Funnel + tier-coverage** (`vnv_funnel_tier_report.py`) — full-universe scan
   reading `df.attrs["drops"]` + `distribution_source`.
3. **prob_profit calibration + ev_dollars realism** (`vnv_prob_profit_calibration.py`)
   — regime-spanning as_of grid, realized hold-to-expiry from forward OHLCV.
4. **cProfile** of a 150-ticker scan for efficiency hot spots.

## What worked
- The funnel is transparent + healthy: 423/503 survive; 80 drops all auditable
  (68 earnings event-lockout, 11 history<504d, 1 thin-premium).
- Wilson-CI coverage 98.6% on the put ranker — the PR #317 tier-gate suppresses
  only the 1.4% overlapping-tier rows (a real, narrow fix).
- Calibration measurement reproduced the I1 top-bin over-confidence on current
  main AND its regime-dependence (calibrated in 2020 recovery, −0.38 in 2022
  rate-bear) — empirical support for the gated calibration-band decision.
- ev_dollars SIGN cleanly separates realized winners (ev>0) from losers (ev<=0)
  — a positive result for the §2 negative-EV gate.

## What didn't
- The naive "expired OTM" realized rule would mis-state calibration by ~12pp
  (HT-B's artifact); used the engine-EXACT breakeven `S_expiry > strike - prem`.
- The limit-25 SMOKE gave a misleading `corr(ev_dollars, realized_pnl)` = +0.24;
  the full universe (n=9,612) is −0.018 (~0). Lesson: validate small-universe
  smokes against the full run before quoting a correlation — the alphabetical
  first-25 tickers are a biased subset. The robust signals are the monotonic
  quintile lift + sign separation, NOT the linear correlation magnitude.
- Calibration-script stdout is block-buffered to its log on Windows (only stderr
  HMM warnings stream live) — progress invisible mid-run; results land at the end.

## How we fixed it
N/A — read-only campaign. No engine change. Findings + drivers shipped; trio /
behaviour-changing follow-ups (connector perf, RA-2 earnings vintage, cockpit
copy reconciliation) recorded for operator decision.

## Evidence
- `funnel_2026-03-20.json`: 503→423; drops {event:68, history:11, premium:1};
  tiers {empirical_non_overlapping:417, empirical_overlapping:6}; CI coverage 98.6%.
- IV no-skew: `hist_put_imp_vol == hist_call_imp_vol` in 100.0000% of 1,353,901
  non-null rows (mean abs diff 0).
- Profile (150t): `comp_method_OBJECT_ARRAY` 10.1s/742 calls (per-ticker object
  scans); `normalize_ticker` 2.4M calls (`_load` `.apply`); HMM E-step 7.9s.
- Calibration full universe (n=9,612): top bins [0.90,0.95) gap −0.113,
  [0.95,1.00) −0.163 (mild unconditionally); 2022 rate-bear top bin realized
  **0.577** (gap −0.345) == the cockpit's 0.57 crisis ghost; calm/recovery
  regimes calibrated (2020 +0.008, 2023 +0.034).
- ev_dollars realism (n=9,612): Pearson corr = **−0.018 (~0)** → confirms the
  cockpit's "~0 linear corr" (the limit-25 smoke's +0.24 was a small-sample
  artifact — full sample is authoritative); monotonic EV-quintile lift
  (Q1 −$43 → Q5 +$107); ev>0 +$85 vs ev<=0 −$9.45. Ranking/sign signal valid,
  linear corr ~0 — both cockpit halves hold.

## Unresolved / handoff
- **Efficiency PR (separate):** memoize/unique-map `normalize_ticker` in `_load`;
  cached-groupby `_filter_ticker` (the 10s scan) — non-trio, ship with an
  output-equivalence test.
- **RA-2 earnings PIT:** `get_next_earnings` uses realized announcement dates
  (no as-of vintage) — stamp a calendar vintage / diagnostic.
- **§4 cockpit-copy reconciliation:** this measurement (single short-puts to
  expiry) shows ev_dollars sign-predicts realized; the cockpit's "~0 correlation"
  is the full-wheel-with-costs claim. Reconcile before editing the cockpit copy.
- **HMM numerical warnings:** guard `logaddexp.reduce` against all-`-inf` rows.
