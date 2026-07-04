---
id: sim200k
title: SIM-200K eight-window reliability campaign at 200k
kind: backtest
status: done
terminal:
pr:
decisions: []
date: 2026-06-12
headline: "8 one-year $200k wheel campaigns from regime-diverse start dates: 7/8 positive (mean +14.3%, worst -3.2% in the 2022 bear where it beat EW B&H by +7.8pp); bear-alpha/bull-lag confirmed; top-bin over-confidence confirmed in all 8 windows"
surface: [backtests/regression/sim200k_reliability.py, scripts/sim200k_analysis.py, scripts/sim200k_tier_probe.py, docs/SIM_200K_RELIABILITY_2026-06.md]
---

## Goal

Operator-commissioned overnight campaign (2026-06-11, "act as professional
quant trader… pick multiple points in the past… $200,000… invest for a year…
do not cheat"): measure the *distribution* of one-year outcomes if the engine
had gone live at eight different past dates, at retail ($200k) scale, with
honest PIT discipline, and document reliability.

## What we tried

Canonical harness only — `run_backtest_multi_friction` (S38/S43 lineage),
cloned the `s43_rolling_multiwindow.py` wrapper into
`sim200k_reliability.py` with one knob changed (capital $1M → $200k) and
eight pre-registered one-year windows (crash-entry 2020-02 … recent
2025-03). 3-week smoke first (11 trades, +$1,044, 7.4 s/day), then three
parallel BelowNormal lanes (live Theta pull on the same box was never
disturbed; pull worker stayed Normal-priority and healthy throughout).
Opus adversarial design review BEFORE accepting results; Sonnet built the
post-processor; Opus re-derivation pass on the artifacts after.

## What worked

- All 8 windows completed in ~1h50m wall (3 lanes, ~0.13 day/s each).
- Opus design review: APPROVE_WITH_CHANGES — decision path PIT-clean
  (OHLCV/IV/HMM/forward-dist all sliced ≤ as_of, quoted line evidence);
  required changes were disclosure items, all adopted into the doc.
- FRED/dealer/news overlays probed inert in this environment → runs are
  deterministic, no live revised data contaminated PIT.
- Results: 7/8 positive, mean +14.3%, median +12.9%, worst −3.2% (2022
  bear, +7.8pp over EW B&H there); COVID-entry window survived to +2.1%
  through a −34% mid-window drawdown; w8 (2025) +44.5% is part
  premium-richness, part BP-saturation timing luck — NOT ranking skill
  (its top-EV decile inverted: predicted +$858 → realized −$2,505 mean).
- Calibration: ρ(EV→realized) positive in all 8 (0.23–0.52, p≈0); top
  prob_profit bin over-confident in all 8 (−1.3 … −18.8pp) — confirms the
  known defect that Block-B recalibration targets, now on 8 windows.
- R9/R10 would-fire audit (running-peak): single names hit 23–54% NAV,
  R10 would have fired 126–249 days/window — caps-off canonical config is
  materially un-production-like at $200k; flagged prominently.

## What didn't

- The campaign rank logs don't persist `distribution_source`; needed a
  separate re-rank probe (`sim200k_tier_probe.py`) to show w1 ran mostly on
  the degraded `empirical_overlapping` tier (w2 start too; w3-w8 clean).
- `gh`-style sequential lanes: first plan was 8 sequential runs (~6-12h);
  parallel-3 BelowNormal cut it to <2h with zero pull impact.
- Smoke concentration already showed 22.9% single-name peaks — early
  warning that caps-off + $200k diverges from production-armed behavior.
- First benchmark cut computed EW B&H on the raw CSV "close" column — but
  the CSV's labels are rotated one slot (the connector repairs this on
  read; raw close ≈ open). Verifier caught it; benchmarks recomputed on
  the connector's true close (systematic shift, direction varies by
  window; engine NAVs never affected — the harness settles via the
  connector).
- w8 calibration tail was poisoned by the 2026-03-23 BKNG/CVNA
  unadjusted-split seam (−$1.46M fake hypothetical rank-log P&L; zero
  book impact — neither name ever executed). The "decile-9 inversion"
  was this artifact, not a real calibration finding.

## How we fixed it

Disclosure-first: every flattering bias (survivorship universe, caps-off,
synthetic premiums, D21 horizon, dateless fundamentals, tier degradation)
is in the doc's §3 register; the §10 verdict claims only what the design
supports (cross-regime shape, not absolute alpha vs buy-and-hold).

## Evidence

- Driver: `python -m backtests.regression.sim200k_reliability all|one <id>`
- Artifacts: `%TEMP%\sim200k_backtest\<window>\{none,bid_ask,full}\
  {rank_log.csv, metrics.json, tracker_state.json}` + `summary.json` +
  `analysis.json`; combined `campaign_table.md`; `tier_probe.json`.
- Analysis: `py -3.12 scripts/sim200k_analysis.py --root %TEMP%\sim200k_backtest --all`
- Full writeup: `docs/SIM_200K_RELIABILITY_2026-06.md` (results §4-§7,
  realism §9, adversarial verification §8, verdict §10).
- Engine SHA: branch base `83eacdd` (post #407/#408 brain-audit fixes).

## Unresolved / handoff

- Top-bin over-confidence + EV-level miscalibration (bottom-decile
  pessimism, bear-regime optimism) all route to the existing Block-B
  recalibration item — this campaign adds 8-window evidence, no new fix.
- A production-armed (R9/R10 on) re-run would characterize the deployable
  config; cheap to add as a second campaign if the operator wants it.
- PIT-membership universe (kills survivorship bias) needs the index-
  membership history wiring — Block A adjacency.
- NEW data-layer defect for Block A: 2026-03-23 reconstitution seam left
  BKNG (×0.041) and CVNA (×0.212) with unadjusted split jumps in
  sp500_ohlcv.csv — the only two such seams since 2020 (verifier-scanned).
  Fix split-adjustment continuity when refreshing.
- NEW engineering finding (route with #372): `WheelTracker._compute_live_nav`
  marks D17 cap NAV at `date.today()` / dataset-latest closes — correct
  live, LOOKAHEAD if `make_live_book_tracker` is ever backtested directly.
  The §11 armed companion therefore emulated R10 at the harness layer
  (same production gate fn, PIT NAV); default path regression-checked to
  8 decimals. Armed result: mean +10.0%, 7/8 positive, σ 9.4pp vs 14.3pp
  caps-off — the deployable config keeps the reliability shape.
