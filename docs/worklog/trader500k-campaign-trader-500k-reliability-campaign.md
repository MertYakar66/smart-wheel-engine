---
id: trader500k-campaign
title: TRADER-500K reliability campaign
kind: backtest
status: shipped
terminal: MACBOOK
pr: 518
decisions: []
date: 2026-07-22
headline: $500k mechanical wheel over 11×18mo windows + 2 rail re-runs — 9/11 profitable, beats SPX only 2/11, prob_profit compressed with regime-dependent top-bin gap, EV rank-quality dispersion-conditional
surface: [backtests/regression/trader500k_campaign.py, docs/TRADER_500K_RELIABILITY_2026-07.md, docs/verification_artifacts/trader500k/]
---

## Goal
Measure whether the engine's forecasts are reliable (calibration: prob_profit /
ev_dollars vs realized) and economical ($500k book vs SPX / EW-passive, net of
frictions) when a fully mechanical trader executes ONLY ranked output — eleven
18-month windows 2020→2026, weekly rank / daily marks, hold-to-expiry, R10
armed, rail pinned off (Phase A/B) + W05/W09 rail-ON re-runs (Phase C).
Two-terminal protocol: SANDBOX authored the spec + final report, MACBOOK
executed; full audit trail on issue #517.

## What we tried
- Thin driver `backtests/regression/trader500k_campaign.py` wrapping the
  `_common.py` building blocks verbatim (`_tracker_step`, `_tracker_try_opens`,
  rail pin-off CM, `_compute_metrics`) — the harness has no weekly-cadence /
  entry-cutoff / full-universe hooks, so the driver owns the loop.
- 34-agent adversarial review of the driver BEFORE the pilot (4 lenses × 2
  refuting verifiers per finding).
- Pilot W05 (2022 bear) gated by SANDBOX ACK before the 10-window grid;
  sequential Phase B (~45 min/window); Phase C rail-ON with per-row
  premium_source provenance.

## What worked
- Capture-wide/execute-narrow (SANDBOX FIX-REQUEST): rank `top_n=len(universe)`
  for the calibration pool, execute `frame.head(20)` — byte-equivalent book,
  full-spectrum calibration (~180k Phase-A/B rows).
- Real exchange calendar (OHLCV distinct dates, not `pd.bdate_range`) killed
  the holiday-Monday t+1 mark leak the review found, and fixed CAGR/Sharpe
  denominators in one move.
- Post-settlement EOD re-mark: committed daily NAV includes expiry-day
  intrinsics (harness marks pre-settle; maxDD was being shaved at assignment
  clusters).
- Determinism: W01 re-run after the NaN-guard fix was byte-identical away from
  the guarded day.

## What didn't
- `pd.bdate_range` as a trading calendar (holiday rank days + fabricated
  zero-return rows + next-session close leaking into decision-day R10 NAV).
- Trusting any same-day mark for the committed curve: one transient NaN mark
  (2020-11-06, all books) poisoned maxDD → guard non-finite marks via counted
  carry-forward (`01f0351`).
- Assuming full premium-larder coverage for Phase C: the Theta larder holds
  154 tickers total; missing executed names are unavailable without Theta
  Terminal. Partial-coverage rail-ON is well-defined via per-row
  premium_source (26-28% pool, 32-44% executed at market_mid).

## How we fixed it
Driver-level fixes only — decision trio untouched, every candidate flowed
through `rank_candidates_by_ev` → `EVEngine.evaluate` (§2). Bundles committed
per window; SANDBOX reviewed on-thread (ACK per window; one FIX-REQUEST
pre-pilot, incorporated in full).

## Evidence
- Report: `docs/TRADER_500K_RELIABILITY_2026-07.md` (SANDBOX-authored, committed verbatim).
- Bundles: `docs/verification_artifacts/trader500k/{W01..W11,W05_rail,W09_rail}/`
  (summary.json + equity_curve/trades/calibration_rows csv.gz each).
- Headline economics (full friction): 9/11 windows positive, mean +15.4%/18mo;
  beats SPX only W05/W11; mean maxDD 16.4%.
- Headline calibration: pooled Brier 0.161 / ECE 4.3pp; top-bin gap
  regime-dependent (bear −13.5/−14.2pp, mixed −6.7/−10.1pp, bull −3.9..+0.6pp);
  pooled Spearman(ev, realized) ≈ 0 with dispersion-conditional sign split.
- Rail deltas: W05_rail halves bear-window return (+3.3/+2.7/+2.2 vs
  +6.0/+2.3/+4.8), W09_rail raises it (+23.4/+22.3/+21.5 vs ~+15) —
  synthetic-BSM premium bias is window-dependent; calibration
  provenance-robust in both.
- Full protocol trail: issue #517 (BRINGUP → PILOT → 11× WINDOW-DONE →
  ALL-DONE → FINAL-REPORT).

## Unresolved / handoff
- Survivorship (2026-monolith universe) remains the campaign's largest
  future-information channel — carried in every summary.json caveats block and
  the report's limitations; a PIT-membership universe needs
  `sp500_index_membership.csv` wiring.
- Profit-taking / rolling exits are documented follow-ups (hold-to-expiry was
  a locked modeling choice).
- The dispersion-conditional EV rank-quality split (crisis +0.15 / quiet-bull
  −0.12) is the report's sharpest open question for engine work.
